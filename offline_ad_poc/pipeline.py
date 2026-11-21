#!/usr/bin/env python3
"""Offline, CPU/GPU-portable audio-description prototype.

This is a research prototype.  The heuristic backend validates the full media
pipeline but does not claim useful visual-language quality.  The qwen25 backend
uses only a local model directory and performs no API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np

from .knowledge_db import EvidenceStore
from .visual_events import apply_ad_policy, validate_visual_event


LOGGER = logging.getLogger("offline_ad_poc")


def configure_logging(work_dir: Path) -> Path:
    """Write readable runtime decisions to both the console and work directory."""
    log_path = work_dir / "runtime.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)
    LOGGER.info("runtime_log path=%s", log_path)
    return log_path


def collect_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info.update(
            torch_version=torch.__version__,
            torch_cuda_build=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
        if torch.cuda.is_available():
            info["cuda_devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_mb": round(
                        torch.cuda.get_device_properties(index).total_memory / (1024**2),
                        3,
                    ),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    LOGGER.info("hardware %s", json.dumps(info, ensure_ascii=False, sort_keys=True))
    return info


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _process_peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return value / divisor


def _cuda_stage_start(compute_device: str) -> dict[str, Any]:
    if compute_device != "cuda":
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_telemetry_error": "CUDA stage requested but torch.cuda is unavailable"}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_allocated_start_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_reserved_start_mb": torch.cuda.memory_reserved() / (1024**2),
        }
    except Exception as exc:
        return {"gpu_telemetry_error": f"{type(exc).__name__}: {exc}"}


def _cuda_stage_end(compute_device: str) -> dict[str, Any]:
    if compute_device != "cuda":
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_telemetry_error": "CUDA stage requested but torch.cuda is unavailable"}
        torch.cuda.synchronize()
        return {
            "gpu_memory_allocated_end_mb": round(torch.cuda.memory_allocated() / (1024**2), 3),
            "gpu_memory_reserved_end_mb": round(torch.cuda.memory_reserved() / (1024**2), 3),
            "gpu_memory_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 3),
            "gpu_memory_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 3),
        }
    except Exception as exc:
        return {"gpu_telemetry_error": f"{type(exc).__name__}: {exc}"}


class RuntimeProfiler:
    """Persist wall-clock stage timings as the pipeline runs."""

    def __init__(self, work_dir: Path) -> None:
        self.events_path = work_dir / "runtime-events.jsonl"
        self.summary_path = work_dir / "runtime-summary.json"
        self.events_path.write_text("", encoding="utf-8")
        self.events: list[dict[str, Any]] = []
        self.started_perf = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self.hardware: dict[str, Any] = {}

    def set_hardware(self, hardware: dict[str, Any]) -> None:
        self.hardware = hardware

    @contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[dict[str, Any]]:
        stage_details = dict(details)
        compute_device = str(stage_details.setdefault("compute_device", "cpu"))
        started = time.perf_counter()
        started_cpu = time.process_time()
        started_children_cpu = _children_cpu_seconds()
        gpu_start = _cuda_stage_start(compute_device)
        offset = started - self.started_perf
        LOGGER.info("stage_start stage=%s compute_device=%s", name, compute_device)
        status = "completed"
        error: BaseException | None = None
        try:
            yield stage_details
        except BaseException as exc:
            status = "failed"
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            cpu_seconds = (
                time.process_time()
                - started_cpu
                + _children_cpu_seconds()
                - started_children_cpu
            )
            gpu_end = _cuda_stage_end(compute_device)
            event: dict[str, Any] = {
                "index": len(self.events) + 1,
                "stage": name,
                "status": status,
                "started_offset_seconds": round(offset, 6),
                "elapsed_seconds": round(elapsed, 6),
                "cpu_seconds": round(cpu_seconds, 6),
                "average_cpu_cores": round(cpu_seconds / elapsed, 4) if elapsed > 0 else 0.0,
                "process_peak_rss_mb": round(_process_peak_rss_mb(), 3),
                "uses_cpu": True,
                "uses_gpu": compute_device == "cuda",
                **stage_details,
                **gpu_start,
                **gpu_end,
            }
            if error is not None:
                event["error_type"] = type(error).__name__
                event["error"] = str(error)
            self.events.append(event)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            LOGGER.log(
                logging.INFO if status == "completed" else logging.ERROR,
                "stage_end stage=%s status=%s compute_device=%s elapsed_seconds=%.6f "
                "cpu_seconds=%.6f average_cpu_cores=%.4f gpu_peak_allocated_mb=%s",
                name,
                status,
                compute_device,
                elapsed,
                cpu_seconds,
                cpu_seconds / elapsed if elapsed > 0 else 0.0,
                gpu_end.get("gpu_memory_peak_allocated_mb", "n/a"),
            )

    def finalize(self) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        totals_by_device: dict[str, dict[str, float | int]] = {}
        for event in self.events:
            device = str(event.get("compute_device", "unknown"))
            totals = totals_by_device.setdefault(
                device,
                {"stages": 0, "elapsed_seconds": 0.0, "cpu_seconds": 0.0},
            )
            totals["stages"] += 1
            totals["elapsed_seconds"] += float(event["elapsed_seconds"])
            totals["cpu_seconds"] += float(event["cpu_seconds"])
        for totals in totals_by_device.values():
            totals["elapsed_seconds"] = round(float(totals["elapsed_seconds"]), 6)
            totals["cpu_seconds"] = round(float(totals["cpu_seconds"]), 6)
        summary = {
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total_elapsed_seconds": round(time.perf_counter() - self.started_perf, 6),
            "event_log": str(self.events_path),
            "hardware": self.hardware,
            "totals_by_compute_device": totals_by_device,
            "slowest_stages": sorted(
                self.events,
                key=lambda event: float(event["elapsed_seconds"]),
                reverse=True,
            )[:5],
            "stages": self.events,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "runtime_summary path=%s total_elapsed_seconds=%.6f stages=%d",
            self.summary_path,
            summary["total_elapsed_seconds"],
            len(self.events),
        )
        return summary


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Gap(Interval):
    word_budget: int = 0


@dataclass
class Shot:
    shot_id: int
    start: float
    end: float
    keyframe: str = ""
    brightness: float = 0.0
    motion: float = 0.0
    audio_rms: float = 0.0
    audio_centroid: float = 0.0
    transcript: str = ""
    boundary_score: float = 0.0


@dataclass
class Scene:
    scene_id: int
    start: float
    end: float
    shot_ids: list[int]
    boundary_confidence: float


@dataclass
class EvidenceFrame:
    path: str
    timestamp: float
    requested_timestamp: float
    shot_id: int | None
    scene_id: int | None
    salience: float = 0.0
    role: str = "gap"


@dataclass
class Decision:
    gap_start: float
    gap_end: float
    decision: str
    description: str
    reason: str
    importance: float
    word_budget: int
    evidence_frames: list[str]
    evidence_timestamps: list[float] = field(default_factory=list)
    event_start: float | None = None
    event_end: float | None = None
    event_timing_source: str = "none"
    preferred_cue_start: float | None = None
    placement_method: str = "unplaced"
    tts_path: str = ""
    tts_duration: float = 0.0
    cue_start: float = 0.0
    cue_end: float = 0.0
    description_ready_at: float | None = None
    visual_proposals: list[dict[str, Any]] = field(default_factory=list)
    selected_window: int | None = None
    gap_index: int | None = None
    visual_followups: list[dict[str, Any]] = field(default_factory=list)


def run_command(args: Sequence[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(map(str, args)),
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    )


def resolve_ffmpeg_tools() -> tuple[str, str]:
    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        LOGGER.info("method stage=media_tools ffmpeg=%s ffprobe=%s", ffmpeg, ffprobe)
        return ffmpeg, ffprobe
    raise RuntimeError(
        "FFmpeg/ffprobe not found. Install the Conda ffmpeg package or set "
        "FFMPEG_BINARY and FFPROBE_BINARY."
    )


def parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    return float(Fraction(value))


def probe_media(path: Path, ffprobe: str) -> MediaInfo:
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    video_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError(f"No video stream found in {path}")
    duration = float(payload.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"Unable to determine media duration for {path}")
    return MediaInfo(
        duration=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        has_audio=any(s.get("codec_type") == "audio" for s in payload.get("streams", [])),
    )


def extract_analysis_audio(input_video: Path, output_wav: Path, ffmpeg: str) -> None:
    run_command(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def choose_device(requested: str) -> str:
    if requested == "cpu":
        LOGGER.info("method stage=device_selection requested=cpu selected=cpu reason=explicit_request")
        return "cpu"
    cuda_available = False
    detection_method = "none"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            cuda_available = True
            detection_method = "ctranslate2"
    except Exception:
        pass
    if not cuda_available:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            detection_method = "torch" if cuda_available else "torch_no_cuda"
        except Exception:
            pass
    if requested == "cuda" and not cuda_available:
        LOGGER.warning(
            "fallback stage=device_selection requested=cuda selected=cpu "
            "reason=cuda_unavailable detection_method=%s",
            detection_method,
        )
        return "cpu"
    selected = "cuda" if cuda_available else "cpu"
    LOGGER.info(
        "method stage=device_selection requested=%s selected=%s detection_method=%s",
        requested,
        selected,
        detection_method,
    )
    return selected


def _segments_from_words(words: Sequence[Word], maximum_pause: float = 0.8) -> list[TranscriptSegment]:
    if not words:
        return []
    groups: list[list[Word]] = [[words[0]]]
    for word in words[1:]:
        current = groups[-1]
        if word.start - current[-1].end > maximum_pause or word.end - current[0].start > 12.0:
            groups.append([word])
        else:
            current.append(word)
    return [
        TranscriptSegment(group[0].start, group[-1].end, " ".join(item.text for item in group))
        for group in groups
    ]


def _whisper_generate_kwargs(generation_config: Any) -> dict[str, str]:
    if bool(getattr(generation_config, "is_multilingual", False)):
        return {"language": "english", "task": "transcribe"}
    return {}


def _transcribe_faster_whisper(
    audio_path: Path,
    model_name_or_path: str,
    device: str,
    allow_model_download: bool,
) -> tuple[list[TranscriptSegment], list[Word], dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to run ASR.") from exc

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        model_name_or_path,
        device=device,
        compute_type=compute_type,
        local_files_only=not allow_model_download,
    )
    generated, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    segments: list[TranscriptSegment] = []
    words: list[Word] = []
    for segment in generated:
        text = " ".join(segment.text.strip().split())
        if text:
            segments.append(TranscriptSegment(float(segment.start), float(segment.end), text))
        for item in segment.words or []:
            token = item.word.strip()
            if token and item.start is not None and item.end is not None:
                words.append(Word(float(item.start), float(item.end), token))
    report = {
        "backend": "faster-whisper",
        "model": model_name_or_path,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "segments": len(segments),
        "words": len(words),
    }
    return segments, words, report


def _transcribe_transformers_whisper(
    audio_path: Path,
    model_name_or_path: str,
    device: str,
    allow_model_download: bool,
) -> tuple[list[TranscriptSegment], list[Word], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The Transformers ASR backend requires the Conda pytorch and transformers packages."
        ) from exc

    dtype = torch.float16 if device == "cuda" else torch.float32
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": not allow_model_download,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=not allow_model_download,
    )
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=28,
        stride_length_s=(4, 2),
        torch_dtype=dtype,
        device=0 if device == "cuda" else -1,
    )
    is_multilingual = bool(getattr(model.generation_config, "is_multilingual", False))
    generate_kwargs = _whisper_generate_kwargs(model.generation_config)
    result = asr(
        str(audio_path),
        return_timestamps="word",
        generate_kwargs=generate_kwargs,
    )
    words: list[Word] = []
    for item in result.get("chunks", []):
        timestamp = item.get("timestamp") or (None, None)
        start, end = timestamp
        token = " ".join(str(item.get("text", "")).strip().split())
        if token and start is not None and end is not None:
            words.append(Word(float(start), float(end), token))
    segments = _segments_from_words(words)
    report = {
        "backend": "transformers",
        "model": model_name_or_path,
        "device": device,
        "compute_type": str(dtype).replace("torch.", ""),
        "language": "en",
        "is_multilingual": is_multilingual,
        "segments": len(segments),
        "words": len(words),
    }
    return segments, words, report


def transcribe_audio(
    audio_path: Path,
    model_name_or_path: str,
    device: str,
    allow_model_download: bool,
    backend: str = "auto",
) -> tuple[list[TranscriptSegment], list[Word], dict[str, Any]]:
    if backend in {"auto", "faster-whisper"}:
        try:
            result = _transcribe_faster_whisper(
                audio_path,
                model_name_or_path,
                device,
                allow_model_download,
            )
            LOGGER.info("method stage=asr backend=faster-whisper device=%s", device)
            return result
        except (ImportError, RuntimeError) as exc:
            if backend == "faster-whisper" or "Install faster-whisper" not in str(exc):
                raise
            LOGGER.warning(
                "fallback stage=asr from=faster-whisper to=transformers reason=%s",
                str(exc).replace("\n", " "),
            )
    result = _transcribe_transformers_whisper(
        audio_path,
        model_name_or_path,
        device,
        allow_model_download,
    )
    LOGGER.info("method stage=asr backend=transformers-whisper device=%s", device)
    return result


def merge_intervals(intervals: Sequence[Interval], maximum_gap: float = 0.45) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for item in ordered:
        if not merged or item.start - merged[-1].end > maximum_gap:
            merged.append(Interval(item.start, item.end))
        else:
            merged[-1].end = max(merged[-1].end, item.end)
    return merged


def speech_intervals_from_words(
    words: Sequence[Word],
    duration: float,
    padding: float = 0.25,
    merge_gap: float = 0.45,
) -> list[Interval]:
    raw = [
        Interval(max(0.0, word.start - padding), min(duration, word.end + padding))
        for word in words
    ]
    return merge_intervals(raw, maximum_gap=merge_gap)


def validate_whisper_word_timestamps(
    words: Sequence[Word],
    minimum_duration: float = 0.03,
    maximum_duration: float = 2.50,
) -> tuple[list[Word], dict[str, Any]]:
    """Remove unusable Whisper alignments before they can become dialogue intervals."""
    validated: list[Word] = []
    rejected_nonpositive = 0
    rejected_stretched = 0
    for word in words:
        duration = word.end - word.start
        if duration < minimum_duration:
            rejected_nonpositive += 1
            continue
        if duration > maximum_duration:
            rejected_stretched += 1
            continue
        validated.append(word)
    report = {
        "method": "whisper_word_duration_validation",
        "minimum_word_duration": minimum_duration,
        "maximum_word_duration": maximum_duration,
        "raw_words": len(words),
        "validated_words": len(validated),
        "rejected_words": len(words) - len(validated),
        "rejected_nonpositive_duration": rejected_nonpositive,
        "rejected_stretched_duration": rejected_stretched,
    }
    LOGGER.info(
        "method stage=dialogue_timestamp_validation backend=whisper_word_duration "
        "raw_words=%d validated_words=%d rejected_nonpositive=%d rejected_stretched=%d",
        len(words),
        len(validated),
        rejected_nonpositive,
        rejected_stretched,
    )
    return validated, report


def build_gaps(
    speech: Sequence[Interval],
    duration: float,
    minimum_duration: float = 1.5,
    safety_margin: float = 0.20,
    words_per_second: float = 2.5,
) -> list[Gap]:
    gaps: list[Gap] = []
    cursor = 0.0
    for interval in speech:
        if interval.start - cursor >= minimum_duration:
            usable_start = cursor + safety_margin
            usable_end = interval.start - safety_margin
            if usable_end - usable_start >= minimum_duration:
                budget = max(1, math.floor((usable_end - usable_start) * words_per_second))
                gaps.append(Gap(usable_start, usable_end, budget))
        cursor = max(cursor, interval.end)
    if duration - cursor >= minimum_duration:
        usable_start = cursor + safety_margin
        usable_end = max(usable_start, duration - safety_margin)
        if usable_end - usable_start >= minimum_duration:
            budget = max(1, math.floor((usable_end - usable_start) * words_per_second))
            gaps.append(Gap(usable_start, usable_end, budget))
    return gaps


def _detect_shots_opencv(
    video_path: Path,
    duration: float,
    threshold: float,
    runtime_details: dict[str, Any] | None = None,
) -> list[Shot]:
    """Fallback content-difference detector that requires only OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    boundaries = [0.0]
    previous: np.ndarray | None = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                score = float(np.mean(cv2.absdiff(previous, gray)))
                timestamp = frame_index / fps
                if score >= threshold and timestamp - boundaries[-1] >= 0.3:
                    boundaries.append(timestamp)
            previous = gray
            frame_index += 1
    finally:
        capture.release()
    boundaries.append(duration)
    shots = [
        Shot(index, start, end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1)
        if end - start > 0.05
    ] or [Shot(1, 0.0, duration)]
    if runtime_details is not None:
        runtime_details.update(
            method="opencv_frame_difference",
            threshold=threshold,
            shots=len(shots),
        )
    LOGGER.info(
        "method stage=shot_detection backend=opencv_frame_difference threshold=%.3f shots=%d",
        threshold,
        len(shots),
    )
    return shots


def detect_shots(
    video_path: Path,
    duration: float,
    threshold: float = 27.0,
    runtime_details: dict[str, Any] | None = None,
) -> list[Shot]:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        LOGGER.warning(
            "fallback stage=shot_detection from=scenedetect.ContentDetector "
            "to=opencv_frame_difference reason=%s",
            str(exc).replace("\n", " "),
        )
        if runtime_details is not None:
            runtime_details["fallback_reason"] = str(exc)
        return _detect_shots_opencv(video_path, duration, threshold, runtime_details)

    try:
        scene_list = detect(str(video_path), ContentDetector(threshold=threshold), show_progress=False)
    except Exception as exc:
        LOGGER.warning(
            "fallback stage=shot_detection from=scenedetect.ContentDetector "
            "to=opencv_frame_difference reason=%s:%s",
            type(exc).__name__,
            str(exc).replace("\n", " "),
        )
        if runtime_details is not None:
            runtime_details["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return _detect_shots_opencv(video_path, duration, threshold, runtime_details)
    else:
        if not scene_list:
            if runtime_details is not None:
                runtime_details.update(
                    method="scenedetect.ContentDetector",
                    threshold=threshold,
                    shots=1,
                    no_boundaries=True,
                )
            LOGGER.info(
                "method stage=shot_detection backend=scenedetect.ContentDetector "
                "threshold=%.3f shots=1 no_boundaries=true",
                threshold,
            )
            return [Shot(1, 0.0, duration)]
        shots = [
            Shot(index, float(start.get_seconds()), min(duration, float(end.get_seconds())))
            for index, (start, end) in enumerate(scene_list, start=1)
        ]
        if shots and shots[-1].end < duration:
            shots[-1].end = duration
        if runtime_details is not None:
            runtime_details.update(
                method="scenedetect.ContentDetector",
                threshold=threshold,
                shots=len(shots),
            )
        LOGGER.info(
            "method stage=shot_detection backend=scenedetect.ContentDetector threshold=%.3f shots=%d",
            threshold,
            len(shots),
        )
        return shots


def frame_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def normalized_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return hist / norm if norm else hist


def validate_shot_timeline(shots: Sequence[Shot], duration: float) -> dict[str, Any]:
    warnings: list[str] = []
    if not shots:
        warnings.append("no_shots")
    else:
        if abs(shots[0].start) > 0.05:
            warnings.append("timeline_does_not_start_at_zero")
        if abs(shots[-1].end - duration) > 0.05:
            warnings.append("timeline_does_not_end_at_media_duration")
        for previous, current in zip(shots, shots[1:]):
            difference = current.start - previous.end
            if difference > 0.05:
                warnings.append(f"gap_before_shot_{current.shot_id}")
            elif difference < -0.05:
                warnings.append(f"overlap_before_shot_{current.shot_id}")
        for shot in shots:
            if shot.end <= shot.start:
                warnings.append(f"nonpositive_duration_shot_{shot.shot_id}")
    durations = [shot.end - shot.start for shot in shots]
    return {
        "valid_timeline": not warnings,
        "warnings": warnings,
        "shots": len(shots),
        "boundaries": max(0, len(shots) - 1),
        "minimum_shot_duration": round(min(durations), 6) if durations else None,
        "maximum_shot_duration": round(max(durations), 6) if durations else None,
    }


def create_shot_boundary_evidence(
    video_path: Path,
    shots: Sequence[Shot],
    output_dir: Path,
    duration: float,
    offset_seconds: float = 0.15,
) -> dict[str, Any]:
    """Save independent before/after evidence for every detected boundary."""
    report = validate_shot_timeline(shots, duration)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    evidence: list[dict[str, Any]] = []
    try:
        for index, shot in enumerate(shots[1:], start=1):
            boundary = shot.start
            before_time = max(0.0, boundary - offset_seconds)
            after_time = min(duration, boundary + offset_seconds)
            before = frame_at(capture, before_time)
            after = frame_at(capture, after_time)
            if before is None or after is None:
                evidence.append(
                    {
                        "boundary_index": index,
                        "timestamp": boundary,
                        "error": "frame_decode_failed",
                    }
                )
                continue
            before_small = cv2.resize(before, (320, 240))
            after_small = cv2.resize(after, (320, 240))
            before_gray = cv2.cvtColor(before_small, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_small, cv2.COLOR_BGR2GRAY)
            pixel_difference = float(np.mean(cv2.absdiff(before_gray, after_gray)))
            histogram_difference = float(
                1.0 - np.clip(np.dot(normalized_histogram(before_small), normalized_histogram(after_small)), 0, 1)
            )
            cv2.putText(
                before_small,
                f"BEFORE {before_time:.2f}s",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                after_small,
                f"AFTER {after_time:.2f}s",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            image_path = output_dir / f"boundary-{index:03d}-{boundary:.3f}.jpg"
            cv2.imwrite(str(image_path), cv2.hconcat([before_small, after_small]))
            evidence.append(
                {
                    "boundary_index": index,
                    "timestamp": round(boundary, 6),
                    "before": round(before_time, 6),
                    "after": round(after_time, 6),
                    "pixel_difference": round(pixel_difference, 4),
                    "histogram_difference": round(histogram_difference, 6),
                    "evidence_image": str(image_path),
                }
            )
    finally:
        capture.release()
    soft_capture = cv2.VideoCapture(str(video_path))
    if not soft_capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    fps = soft_capture.get(cv2.CAP_PROP_FPS) or 25.0
    previous: np.ndarray | None = None
    frame_index = 0
    scored_candidates: list[tuple[float, float]] = []
    try:
        while True:
            ok, frame = soft_capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                score = float(np.mean(cv2.absdiff(previous, gray)))
                timestamp = frame_index / fps
                if score >= 10.0:
                    scored_candidates.append((score, timestamp))
            previous = gray
            frame_index += 1
    finally:
        soft_capture.release()
    detected_boundaries = [shot.start for shot in shots[1:]]
    selected_candidates: list[tuple[float, float]] = []
    for score, timestamp in sorted(scored_candidates, reverse=True):
        if any(abs(timestamp - boundary) < 0.45 for boundary in detected_boundaries):
            continue
        if any(abs(timestamp - selected_time) < 0.60 for _, selected_time in selected_candidates):
            continue
        selected_candidates.append((score, timestamp))
        if len(selected_candidates) >= 24:
            break
    candidate_dir = output_dir / "soft-candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_capture = cv2.VideoCapture(str(video_path))
    if not candidate_capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    soft_evidence: list[dict[str, Any]] = []
    try:
        for index, (score, timestamp) in enumerate(
            sorted(selected_candidates, key=lambda item: item[1]),
            start=1,
        ):
            before_time = max(0.0, timestamp - offset_seconds)
            after_time = min(duration, timestamp + offset_seconds)
            before = frame_at(candidate_capture, before_time)
            after = frame_at(candidate_capture, after_time)
            item: dict[str, Any] = {
                "candidate_index": index,
                "timestamp": round(timestamp, 6),
                "frame_difference_score": round(score, 4),
            }
            if before is not None and after is not None:
                before_small = cv2.resize(before, (320, 240))
                after_small = cv2.resize(after, (320, 240))
                image_path = candidate_dir / f"candidate-{index:03d}-{timestamp:.3f}.jpg"
                cv2.imwrite(str(image_path), cv2.hconcat([before_small, after_small]))
                item["evidence_image"] = str(image_path)
            else:
                item["error"] = "frame_decode_failed"
            soft_evidence.append(item)
    finally:
        candidate_capture.release()
    report["evidence"] = evidence
    report["decoded_boundaries"] = sum("error" not in item for item in evidence)
    report["evidence_complete"] = report["decoded_boundaries"] == report["boundaries"]
    report["validation_passed"] = report["valid_timeline"] and report["evidence_complete"]
    report["soft_candidates_requiring_review"] = soft_evidence
    report["soft_candidate_count"] = len(soft_evidence)
    LOGGER.info(
        "method stage=shot_boundary_validation backend=before_after_evidence "
        "boundaries=%d decoded=%d valid_timeline=%s evidence_complete=%s "
        "soft_candidates=%d",
        report["boundaries"],
        report["decoded_boundaries"],
        report["valid_timeline"],
        report["evidence_complete"],
        report["soft_candidate_count"],
    )
    return report


def sample_shot_features(video_path: Path, shots: list[Shot], keyframe_dir: Path) -> list[np.ndarray]:
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    histograms: list[np.ndarray] = []
    try:
        for shot in shots:
            duration = max(0.01, shot.end - shot.start)
            sample_times = [shot.start + duration * ratio for ratio in (0.2, 0.5, 0.8)]
            frames = [frame for frame in (frame_at(capture, t) for t in sample_times) if frame is not None]
            if not frames:
                histograms.append(np.zeros(24 * 24, dtype=np.float32))
                continue
            middle = frames[len(frames) // 2]
            keyframe = keyframe_dir / f"shot-{shot.shot_id:04d}.jpg"
            cv2.imwrite(str(keyframe), middle, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            shot.keyframe = str(keyframe)
            shot.brightness = float(np.mean(cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)) / 255.0)
            if len(frames) > 1:
                changes = [
                    np.mean(
                        cv2.absdiff(
                            cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                        )
                    )
                    / 255.0
                    for i in range(1, len(frames))
                ]
                shot.motion = float(np.mean(changes))
            histograms.append(np.mean([normalized_histogram(frame) for frame in frames], axis=0))
    finally:
        capture.release()
    return histograms


def wav_features_by_shot(wav_path: Path, shots: list[Shot]) -> None:
    with wave.open(str(wav_path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if width != 2:
        return
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    for shot in shots:
        start = max(0, int(shot.start * sample_rate))
        end = min(len(audio), int(shot.end * sample_rate))
        chunk = audio[start:end]
        if chunk.size < 32:
            continue
        shot.audio_rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        window = chunk[: min(chunk.size, sample_rate * 4)]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        frequencies = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)
        total = float(spectrum.sum())
        shot.audio_centroid = float((spectrum * frequencies).sum() / total) if total > 0 else 0.0


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def attach_transcripts(shots: list[Shot], segments: Sequence[TranscriptSegment]) -> None:
    for shot in shots:
        shot.transcript = " ".join(
            segment.text
            for segment in segments
            if overlap(shot.start, shot.end, segment.start, segment.end) > 0
        ).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0 else float(np.clip(1.0 - np.dot(a, b) / denominator, 0.0, 1.0))


def jaccard_distance(a: str, b: str) -> float:
    left, right = token_set(a), token_set(b)
    if not left and not right:
        return 0.25
    if not left or not right:
        return 0.65
    return 1.0 - len(left & right) / len(left | right)


def segment_scenes(
    shots: list[Shot],
    histograms: Sequence[np.ndarray],
    boundary_threshold: float = 0.52,
) -> list[Scene]:
    if not shots:
        return []
    boundaries = [0.0]
    for index in range(1, len(shots)):
        visual = cosine_distance(histograms[index - 1], histograms[index])
        prev, current = shots[index - 1], shots[index]
        rms_distance = min(1.0, abs(math.log10(prev.audio_rms + 1e-6) - math.log10(current.audio_rms + 1e-6)) / 2.0)
        centroid_distance = min(1.0, abs(prev.audio_centroid - current.audio_centroid) / 5000.0)
        audio = 0.6 * rms_distance + 0.4 * centroid_distance
        text = jaccard_distance(prev.transcript, current.transcript)
        score = 0.55 * visual + 0.25 * audio + 0.20 * text
        current.boundary_score = round(score, 4)
        boundaries.append(score)

    scenes: list[Scene] = []
    current_ids = [shots[0].shot_id]
    scene_start = shots[0].start
    confidence = 0.0
    for index in range(1, len(shots)):
        if boundaries[index] >= boundary_threshold:
            scenes.append(
                Scene(len(scenes) + 1, scene_start, shots[index - 1].end, current_ids, round(confidence, 4))
            )
            current_ids = [shots[index].shot_id]
            scene_start = shots[index].start
            confidence = boundaries[index]
        else:
            current_ids.append(shots[index].shot_id)
    scenes.append(Scene(len(scenes) + 1, scene_start, shots[-1].end, current_ids, round(confidence, 4)))
    return scenes


def resize_frame_max_edge(frame: np.ndarray, max_edge: int | None) -> np.ndarray:
    if max_edge is None or max_edge <= 0:
        return frame
    height, width = frame.shape[:2]
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return frame
    scale = max_edge / longest_edge
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def sample_window_frames(
    video_path: Path,
    start: float,
    end: float,
    output_dir: Path,
    count: int = 4,
    max_edge: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    paths: list[Path] = []
    try:
        times = np.linspace(start, max(start, end - 0.05), num=max(1, count))
        for index, timestamp in enumerate(times):
            frame = frame_at(capture, float(timestamp))
            if frame is None:
                continue
            frame = resize_frame_max_edge(frame, max_edge)
            path = output_dir / f"frame-{index:02d}-{timestamp:.2f}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths.append(path)
    finally:
        capture.release()
    return paths


def _shot_scene_at(
    timestamp: float,
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
) -> tuple[int | None, int | None]:
    shot = next(
        (item for item in shots if item.start <= timestamp < item.end),
        shots[-1] if shots and math.isclose(timestamp, shots[-1].end, abs_tol=0.1) else None,
    )
    scene = next(
        (item for item in scenes if shot and shot.shot_id in item.shot_ids),
        None,
    )
    return (shot.shot_id if shot else None, scene.scene_id if scene else None)


def _unique_timestamps(values: Iterable[float], minimum_spacing: float = 0.04) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or value - result[-1] >= minimum_spacing:
            result.append(value)
    return result


def evidence_candidate_timestamps(
    gap: Gap,
    shots: Sequence[Shot],
    soft_candidate_timestamps: Sequence[float],
    maximum_candidates: int = 24,
) -> list[float]:
    """Build a low-cost time grid enriched around local transition evidence."""
    inside_end = max(gap.start, gap.end - 0.05)
    uniform_count = min(maximum_candidates, max(5, math.ceil(gap.duration) + 1))
    uniform_values = list(np.linspace(gap.start, inside_end, num=uniform_count))
    anchors: list[float] = []
    if gap.start > 0.05:
        anchors.append(max(0.0, gap.start - min(1.0, gap.start)))
    for shot in shots[1:]:
        if gap.start <= shot.start <= gap.end:
            anchors.extend((shot.start, min(inside_end, shot.start + 0.15)))
    for timestamp in soft_candidate_timestamps:
        if gap.start <= timestamp <= gap.end:
            anchors.extend((timestamp, min(inside_end, timestamp + 0.15)))
    selected = _unique_timestamps(anchors)
    selected.extend(
        value
        for value in uniform_values
        if all(abs(value - anchor) >= 0.04 for anchor in selected)
    )
    return sorted(selected)


def sample_event_evidence_frames(
    video_path: Path,
    gap: Gap,
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
    soft_candidate_timestamps: Sequence[float],
    output_dir: Path,
    maximum_frames: int = 6,
    max_edge: int | None = None,
    coverage_first: bool = False,
) -> list[EvidenceFrame]:
    """Select timestamped frames using local change and boundary evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    hard_boundaries = [shot.start for shot in shots[1:]]
    decoded: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_histogram: np.ndarray | None = None
    try:
        for requested in evidence_candidate_timestamps(gap, shots, soft_candidate_timestamps):
            frame = frame_at(capture, requested)
            if frame is None:
                continue
            reported = float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            timestamp = reported if reported > 0 and abs(reported - requested) <= 1.0 else requested
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            histogram = normalized_histogram(small)
            if previous_gray is None or previous_histogram is None:
                change = 0.0
            else:
                pixel = float(np.mean(cv2.absdiff(previous_gray, gray)) / 255.0)
                histogram_change = float(
                    1.0 - np.clip(np.dot(previous_histogram, histogram), 0.0, 1.0)
                )
                change = 0.65 * pixel + 0.35 * histogram_change
            hard_distance = min((abs(requested - value) for value in hard_boundaries), default=999.0)
            soft_distance = min(
                (abs(requested - value) for value in soft_candidate_timestamps),
                default=999.0,
            )
            boundary_bonus = 0.35 if hard_distance <= 0.20 else 0.18 if soft_distance <= 0.20 else 0.0
            role = (
                "context_before"
                if requested < gap.start
                else "confirmed_cut"
                if hard_distance <= 0.20
                else "soft_transition"
                if soft_distance <= 0.20
                else "gap"
            )
            decoded.append(
                {
                    "frame": frame,
                    "requested": requested,
                    "timestamp": timestamp,
                    "salience": change + boundary_bonus,
                    "role": role,
                }
            )
            previous_gray = gray
            previous_histogram = histogram
    finally:
        capture.release()
    if not decoded:
        return []

    if coverage_first:
        # Reserve one frame in each time bin before filling remaining slots by salience.
        bin_count = min(4, maximum_frames - 1)
        bins: list[list[int]] = [[] for _ in range(bin_count)]
        for index, item in enumerate(decoded):
            if gap.start <= item["timestamp"] < gap.end:
                fraction = (item["timestamp"] - gap.start) / gap.duration
                bins[min(bin_count - 1, int(fraction * bin_count))].append(index)
        required_indices = {max(group, key=lambda i: decoded[i]["salience"]) for group in bins if group}
        if decoded[0]["timestamp"] < gap.start:
            required_indices.add(0)
        if bins[-1] and len(required_indices) < maximum_frames:
            required_indices.add(bins[-1][-1])
    else:
        required_indices = {0, len(decoded) - 1}
        first_inside = next(
            (index for index, item in enumerate(decoded) if item["requested"] >= gap.start),
            0,
        )
        required_indices.add(first_inside)
    ranked = sorted(
        range(len(decoded)),
        key=lambda index: (decoded[index]["salience"], -decoded[index]["requested"]),
        reverse=True,
    )
    selected_indices = set(required_indices)
    for index in ranked:
        if len(selected_indices) >= maximum_frames:
            break
        timestamp = decoded[index]["requested"]
        if any(abs(timestamp - decoded[other]["requested"]) < 0.30 for other in selected_indices):
            continue
        selected_indices.add(index)
    if len(selected_indices) < maximum_frames:
        for index in ranked:
            selected_indices.add(index)
            if len(selected_indices) >= maximum_frames:
                break

    evidence: list[EvidenceFrame] = []
    for output_index, index in enumerate(sorted(selected_indices), start=1):
        item = decoded[index]
        frame = resize_frame_max_edge(item["frame"], max_edge)
        timestamp = float(item["timestamp"])
        path = output_dir / f"frame-{output_index:02d}-{timestamp:.3f}.jpg"
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
            raise OSError(f"Could not write evidence frame {path}")
        shot_id, scene_id = _shot_scene_at(timestamp, shots, scenes)
        evidence.append(
            EvidenceFrame(
                path=str(path),
                timestamp=round(timestamp, 6),
                requested_timestamp=round(float(item["requested"]), 6),
                shot_id=shot_id,
                scene_id=scene_id,
                salience=round(float(item["salience"]), 6),
                role=str(item["role"]),
            )
        )
    return evidence


def json_object_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object")
    return json.loads(match.group(0))


class VisualBackend:
    uses_event_windows = False

    def decide(
        self,
        frames: Sequence[EvidenceFrame],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def runtime_metrics(self) -> dict[str, Any]:
        return {}


class HeuristicBackend(VisualBackend):
    """A deterministic media-pipeline test, not a semantic AD model."""

    def decide(
        self,
        frames: Sequence[EvidenceFrame],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        boundary_score = float(context.get("boundary_score", 0.0))
        motion = float(context.get("motion", 0.0))
        if not frames:
            return {"decision": "WAIT", "description": "", "reason": "no visual evidence", "importance": 0.0}
        if boundary_score < 0.12 and motion < 0.015 and previous_ad:
            return {"decision": "WAIT", "description": "", "reason": "low visual change", "importance": 0.15}
        if boundary_score >= 0.52:
            description = "The scene shifts as movement continues."
            reason = "multimodal scene change candidate"
            importance = min(1.0, 0.5 + boundary_score)
        elif motion >= 0.03:
            description = "Movement continues through the scene."
            reason = "visible motion during a speech gap"
            importance = min(1.0, 0.4 + motion * 5)
        else:
            description = "The view holds during a quiet moment."
            reason = "pipeline validation cue"
            importance = 0.25
        if previous_ad and previous_ad[-1] == description:
            return {
                "decision": "WAIT",
                "description": "",
                "reason": "same heuristic description was already emitted",
                "importance": 0.0,
            }
        most_salient = max(
            (frame for frame in frames if gap.start <= frame.timestamp <= gap.end),
            key=lambda frame: frame.salience,
            default=None,
        )
        return {
            "decision": "AD",
            "description": description,
            "reason": reason,
            "importance": round(importance, 3),
            "event_start": most_salient.timestamp if most_salient else gap.start,
            "event_end": most_salient.timestamp if most_salient else gap.start,
            "evidence_timestamps": [most_salient.timestamp] if most_salient else [],
        }


def expand_qwen_image_tokens(
    chat_text: str,
    image_token: str,
    image_grid_thw: Sequence[Sequence[int]],
    merge_size: int,
) -> str:
    parts = chat_text.split(image_token)
    if len(parts) - 1 != len(image_grid_thw):
        raise ValueError("Qwen image placeholder count does not match the image grid count")
    expanded = [parts[0]]
    for grid, suffix in zip(image_grid_thw, parts[1:]):
        token_count = math.prod(int(value) for value in grid) // (merge_size**2)
        expanded.extend((image_token * token_count, suffix))
    return "".join(expanded)


class Qwen25Backend(VisualBackend):
    uses_event_windows = True

    def __init__(self, model_path: str, device: str) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen backend requires the Conda pytorch, transformers, accelerate, and Pillow packages."
            ) from exc
        self.torch = torch
        self.image_type = Image
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.processed_image_grids: list[list[list[int]]] = []
        self.vision_tokens = 0

    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "qwen_image_grid_thw": self.processed_image_grids,
            "qwen_vision_tokens": self.vision_tokens,
        }

    def decide(
        self,
        frames: Sequence[EvidenceFrame],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not any(gap.start <= frame.timestamp < gap.end for frame in frames):
            return {"decision": "WAIT", "reason": "no_in_window_evidence", "attempts": []}
        word_limit = min(18, gap.word_budget)
        prompt = (
            "Identify the strongest new visible event or informative detail in the numbered film images. "
            "A cut alone is not an event. Describe only supported facts in the present tense. "
            "Set important=true when the detail helps someone follow the film. "
            "Set already_audible=true when the dialogue communicates the same information. "
            "Use earlier proposed narration to recognize repeated content. "
            "Return one JSON object using this example and no additional text: "
            '{"kind":"action","important":true,"already_audible":false,"uncertain":false,'
            '"frame_ids":[2,3],"ready_frame_id":3,"description":"A woman stands."} '
            "kind is one of none, action, setting, text, detail. "
            "Use actual integer image IDs marked inside the window. "
            "ready_frame_id is the first supporting image where the complete description is apparent. "
            "For kind=none use important=false, frame_ids=[], ready_frame_id=null, description=\"\". "
            f"Keep description to {word_limit} words or fewer. "
            f"Recent dialogue: {transcript or 'none'}. "
            f"Earlier proposed narration: {' | '.join(previous_ad[-3:]) or 'none'}."
        )
        content: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, start=1):
            content.append({"type": "image", "image": str(Path(frame.path).resolve())})
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Image ID {index} ("
                        + ("inside window" if gap.start <= frame.timestamp < gap.end else "context only")
                        + ")."
                    ),
                }
            )
        timestamps = [frame.timestamp for frame in frames]
        attempts: list[dict[str, Any]] = []
        instruction = prompt
        if context.get("refine_description"):
            prompt += (
                "\nRecheck this specific event: " + json.dumps(context["refine_description"])
                + ". Find a shorter description of the same action that the earliest available frame "
                "clearly supports. Return kind=none if there is no such evidence."
            )
            instruction = prompt
        for _ in range(1 if context.get("refine_description") else 2):
            try:
                decoded = self._generate(frames, [*content, {"type": "text", "text": instruction}])
            except Exception as exc:
                LOGGER.exception("vlm_generation_failure window=%.3f-%.3f", gap.start, gap.end)
                attempts.append({"generation_error": f"{type(exc).__name__}: {exc}"})
                return {"decision": "WAIT", "reason": "vlm_generation_failure", "attempts": attempts}
            try:
                proposal = validate_visual_event(
                    json_object_from_text(decoded), timestamps, gap.start, gap.end
                )
                if len(proposal["description"].split()) > word_limit:
                    raise ValueError(f"description exceeds the {word_limit}-word budget")
            except ValueError as exc:
                attempts.append({"raw_output": decoded, "validation_error": str(exc)})
                LOGGER.warning("vlm_event_invalid window=%.3f-%.3f attempt=%d error=%s",
                               gap.start, gap.end, len(attempts), exc)
                instruction = f"{prompt}\nThe previous answer failed validation: {exc}. Correct it."
                continue
            attempts.append({"raw_output": decoded})
            return {**apply_ad_policy(proposal, timestamps), "proposal": proposal, "attempts": attempts}
        return {"decision": "WAIT", "reason": "invalid_vlm_output", "attempts": attempts}

    def compare_events(
        self, frames: Sequence[EvidenceFrame], split: int, left: str, right: str
    ) -> dict[str, Any]:
        content = []
        for index, frame in enumerate(frames):
            group = "A" if index < split else "B"
            content.extend([
                {"type": "text", "text": f"Event {group}."},
                {"type": "image", "image": str(Path(frame.path).resolve())},
            ])
        content.append({
            "type": "text",
            "text": (
                "Compare the two groups of film images. Do they show the same occurrence "
                "of an action or two separate events? A new camera angle can continue an action. "
                "Repeated actions at separate times are distinct. "
                f"Candidate A: {json.dumps(left)}. Candidate B: {json.dumps(right)}. "
                'Return only {"relation":"same"}, {"relation":"distinct"}, or '
                '{"relation":"uncertain"} if the images cannot settle it.'
            ),
        })
        decoded = self._generate(frames, content)
        try:
            relation = json_object_from_text(decoded).get("relation")
            if relation not in ("same", "distinct", "uncertain"):
                raise ValueError("Expected a valid event relation")
        except ValueError as exc:
            return {"relation": "uncertain", "raw_output": decoded, "validation_error": str(exc)}
        return {"relation": relation, "raw_output": decoded}

    def _generate(self, frames: Sequence[EvidenceFrame], content: list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": content}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs = []
        for frame in frames:
            with self.image_type.open(frame.path) as image:
                image_inputs.append(image.convert("RGB"))
        image_features = self.image_processor(images=image_inputs, return_tensors="pt")
        image_grid_thw = image_features["image_grid_thw"]
        merge_size = int(
            getattr(
                self.image_processor,
                "merge_size",
                self.model.config.vision_config.spatial_merge_size,
            )
        )
        image_grid_list = image_grid_thw.tolist()
        self.processed_image_grids.append(image_grid_list)
        self.vision_tokens += sum(
            math.prod(int(value) for value in grid) // (merge_size**2)
            for grid in image_grid_list
        )
        image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        chat_text = expand_qwen_image_tokens(
            chat_text,
            image_token,
            image_grid_list,
            merge_size,
        )
        text_features = self.tokenizer([chat_text], padding=True, return_tensors="pt")
        inputs = {**text_features, **image_features}
        model_device = next(self.model.parameters()).device
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs["input_ids"], generated)]
        return self.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def context_for_gap(
    gap: Gap,
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
    soft_candidate_timestamps: Sequence[float] = (),
) -> dict[str, Any]:
    window_start = max(0.0, gap.start - 8.0)
    relevant = [shot for shot in shots if overlap(window_start, gap.end, shot.start, shot.end) > 0]
    current = max(relevant, key=lambda shot: overlap(window_start, gap.end, shot.start, shot.end)) if relevant else None
    scene = next((item for item in scenes if current and current.shot_id in item.shot_ids), None)
    return {
        "window_start": window_start,
        "window_end": gap.end,
        "shot_ids": [shot.shot_id for shot in relevant],
        "scene_id": scene.scene_id if scene else None,
        "boundary_score": max((shot.boundary_score for shot in relevant), default=0.0),
        "motion": max((shot.motion for shot in relevant), default=0.0),
        "confirmed_boundaries": [
            shot.start for shot in shots[1:] if window_start <= shot.start <= gap.end
        ],
        "soft_candidate_timestamps": [
            value for value in soft_candidate_timestamps if window_start <= value <= gap.end
        ],
    }


def recent_transcript(gap: Gap, segments: Sequence[TranscriptSegment], context_seconds: float = 30.0) -> str:
    start = max(0.0, gap.start - context_seconds)
    return " ".join(
        segment.text
        for segment in segments
        if overlap(start, gap.end, segment.start, segment.end) > 0
    ).strip()


def limit_words(text: str, maximum: int) -> str:
    words = text.strip().split()
    if len(words) <= maximum:
        return " ".join(words)
    shortened = " ".join(words[:maximum]).rstrip(" ,;:-")
    return shortened.rstrip(".!?") + "."


def choose_backend(name: str, model_path: str | None, device: str) -> VisualBackend:
    if name == "heuristic":
        LOGGER.info("method stage=visual_backend backend=heuristic device=cpu")
        return HeuristicBackend()
    if name == "qwen25":
        if not model_path:
            raise ValueError("--vlm-model is required for --visual-backend qwen25")
        LOGGER.info("method stage=visual_backend backend=qwen2.5-vl device=%s", device)
        return Qwen25Backend(model_path, device)
    raise ValueError(f"Unknown visual backend: {name}")


def _coerce_timestamp(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_event_payload(
    payload: dict[str, Any],
    frames: Sequence[EvidenceFrame],
    gap: Gap,
) -> dict[str, Any]:
    """Ground model-provided event times in the frames that were actually decoded."""
    normalized = dict(payload)
    if str(payload.get("decision", "WAIT")).upper() != "AD" or not frames:
        normalized.update(
            event_start=None,
            event_end=None,
            evidence_timestamps=[],
            event_timing_source="none",
        )
        return normalized
    available = sorted({frame.timestamp for frame in frames})
    inside = [value for value in available if gap.start <= value <= gap.end]
    permitted = inside or available

    def snap(value: Any) -> float | None:
        timestamp = _coerce_timestamp(value)
        if timestamp is None:
            return None
        return min(permitted, key=lambda candidate: abs(candidate - timestamp))

    requested_evidence = payload.get("evidence_timestamps", [])
    if not isinstance(requested_evidence, list):
        requested_evidence = []
    evidence = [value for value in (snap(item) for item in requested_evidence) if value is not None]
    raw_start = snap(payload.get("event_start"))
    raw_end = snap(payload.get("event_end"))
    if raw_start is not None:
        evidence.append(raw_start)
    if raw_end is not None:
        evidence.append(raw_end)
    if not evidence:
        fallback = max(
            (frame for frame in frames if frame.timestamp in permitted),
            key=lambda frame: frame.salience,
        )
        evidence = [fallback.timestamp]
        normalized["event_timing_source"] = "highest_salience_evidence_frame"
    else:
        normalized["event_timing_source"] = "model_timestamps_grounded_to_frames"
    evidence = sorted(set(evidence))
    normalized["evidence_timestamps"] = evidence
    normalized["event_start"] = max(gap.start, evidence[0])
    normalized["event_end"] = min(gap.end, max(evidence[-1], normalized["event_start"]))
    return normalized


def split_evidence_windows(gap: Gap, maximum_seconds: float = 12.0) -> list[Gap]:
    """Cover the entire speech gap with contiguous, bounded visual windows."""
    count = max(1, math.ceil(gap.duration / maximum_seconds))
    edges = [gap.start + gap.duration * index / count for index in range(count)] + [gap.end]
    return [Gap(start, end, gap.word_budget) for start, end in zip(edges, edges[1:])]


def plan_qwen_calls(
    gaps: Sequence[Gap], maximum_gaps: int, followups: int, seconds_per_call: float | None
) -> dict[str, Any]:
    initial = sum(len(split_evidence_windows(gap)) for gap in gaps[:maximum_gaps])
    extra = followups if initial else 0
    plan = {
        "initial_windows": initial,
        "max_followup_calls": extra,
        "max_calls_including_initial_retries": 2 * initial + extra,
    }
    if seconds_per_call is not None:
        plan.update(
            seconds_per_call=seconds_per_call,
            estimated_initial_seconds=round(initial * seconds_per_call, 3),
            estimated_seconds_with_all_followups=round((initial + extra) * seconds_per_call, 3),
        )
    return plan


def refine_gap_events(
    candidates: list[dict[str, Any]],
    gap: Gap,
    backend: VisualBackend,
    inspect_window: Callable[[Gap, str], dict[str, Any]],
    followup_budget: int,
    frame_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Review only nearby potential duplicates or events with a tight narration slot."""
    kept: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for candidate in candidates:
        if kept:
            previous = kept[-1]
            distance = candidate["event_start"] - previous["event_end"]
            before = re.findall(r"\w+", previous["description"].casefold())
            after = re.findall(r"\w+", candidate["description"].casefold())
            similarity = SequenceMatcher(None, before, after, autojunk=False).ratio()
            if distance <= 0.75 and before == after:
                reviews.append({
                    "kind": "duplicate_text", "kept_window": previous["window_index"],
                    "discarded_window": candidate["window_index"], "calls": 0,
                })
                continue
            if (
                distance <= 4.0
                and similarity >= 0.5
                and sum(item["calls"] for item in reviews) < followup_budget
            ):
                groups = [
                    [frame for frame in item["frames"] if frame["timestamp"] in item["evidence_timestamps"]]
                    for item in (previous, candidate)
                ]
                left, right = [group[:max(1, frame_limit // 2)] for group in groups]
                if left and right:
                    frames = [EvidenceFrame(**{k: v for k, v in frame.items() if k != "id"}) for frame in left + right]
                    started = time.perf_counter()
                    try:
                        result = backend.compare_events(
                            frames, len(left), previous["description"], candidate["description"]
                        )
                    except Exception as exc:
                        LOGGER.exception("vlm_event_comparison_failed")
                        result = {"relation": "uncertain", "generation_error": str(exc)}
                    reviews.append({
                        "kind": "compare_events", "calls": 1,
                        "elapsed_seconds": round(time.perf_counter() - started, 6),
                        "windows": [previous["window_index"], candidate["window_index"]],
                        "frame_paths": [frame.path for frame in frames],
                        **result,
                    })
                    if result["relation"] == "same":
                        continue
        kept.append(candidate)

    for index, candidate in enumerate(kept):
        if sum(item["calls"] for item in reviews) >= followup_budget:
            break
        ready = candidate["description_ready_at"]
        estimated = len(candidate["description"].split()) / 2.5
        room = gap.end - ready - 0.075
        earlier = max(gap.start, ready - 3.0)
        if room >= estimated or ready - earlier < 0.5:
            continue
        budget = max(1, min(len(candidate["description"].split()), int((gap.end - earlier - 0.15) * 2.5)))
        review_window = Gap(earlier, min(gap.end, ready + 0.25), budget)
        started = time.perf_counter()
        refined = inspect_window(review_window, candidate["description"])
        use_refinement = (
            refined.get("decision") == "AD"
            and refined["description_ready_at"] < ready
            and len(refined["description"].split()) / 2.5 <= gap.end - refined["description_ready_at"] - 0.075
        )
        reviews.append({
            "kind": "earlier_support", "calls": 1, "source_window": candidate["window_index"],
            "review_window": refined["window_index"], "used": use_refinement,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        })
        if use_refinement:
            kept[index] = refined
    return kept, reviews


def make_decisions(
    video_path: Path,
    gaps: Sequence[Gap],
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
    segments: Sequence[TranscriptSegment],
    backend: VisualBackend,
    frame_root: Path,
    maximum_gaps: int,
    evidence_max_edge: int | None = None,
    evidence_frames_per_gap: int = 6,
    soft_candidate_timestamps: Sequence[float] = (),
    knowledge_store: EvidenceStore | None = None,
    multi_event: bool = False,
    followup_budget: int = 0,
) -> list[Decision]:
    decisions: list[Decision] = []
    previous_ad: list[str] = []
    for index, gap in enumerate(gaps[:maximum_gaps], start=1):
        windows = split_evidence_windows(gap) if backend.uses_event_windows else [gap]
        proposals: list[dict[str, Any]] = []
        frames: list[EvidenceFrame] = []
        gap_dir = frame_root / f"gap-{index:03d}"

        def inspect_window(window: Gap, refine_description: str = "") -> dict[str, Any]:
            window_index = len(proposals) + 1
            context = context_for_gap(window, shots, scenes, soft_candidate_timestamps)
            if refine_description:
                context["refine_description"] = refine_description
            evidence = sample_event_evidence_frames(
                video_path,
                window,
                shots,
                scenes,
                soft_candidate_timestamps,
                gap_dir / f"window-{window_index:03d}" if backend.uses_event_windows else gap_dir,
                maximum_frames=evidence_frames_per_gap,
                max_edge=evidence_max_edge,
                coverage_first=backend.uses_event_windows,
            )
            LOGGER.info(
                "evidence gap=%d window=%d range=%.3f-%.3f timestamps=%s",
                index, window_index, window.start, window.end,
                [frame.timestamp for frame in evidence],
            )
            if knowledge_store is not None:
                knowledge_store.record_frames(index, evidence)
                context["knowledge"] = knowledge_store.query_gap_context(window.start, window.end)
            try:
                proposal = backend.decide(
                    evidence, window, recent_transcript(window, segments), previous_ad, context
                )
            except Exception as exc:
                proposal = {
                    "decision": "WAIT",
                    "reason": f"visual backend failure: {type(exc).__name__}: {exc}",
                }
            proposals.append({
                "window_index": window_index,
                "scan": "timing_followup" if refine_description else "initial",
                "start": window.start,
                "end": window.end,
                "frames": [{"id": i, **asdict(frame)} for i, frame in enumerate(evidence, start=1)],
                **proposal,
            })
            frames.extend(evidence)
            return proposals[-1]

        eligible = [item for item in map(inspect_window, windows) if item.get("decision") == "AD"]
        reviews = []
        if multi_event and backend.uses_event_windows:
            eligible, reviews = refine_gap_events(
                eligible, gap, backend, inspect_window, followup_budget, evidence_frames_per_gap
            )
            followup_budget -= sum(item["calls"] for item in reviews)
        else:
            eligible = eligible[:1]
        if not eligible:
            eligible = [proposals[0] if len(proposals) == 1 else {"decision": "WAIT", "reason": "no_eligible_event_in_gap"}]
        for position, payload in enumerate(eligible):
            decision = decision_from_proposal(gap, payload, frames, index, backend.uses_event_windows)
            if position == 0:
                decision.visual_proposals = proposals
                decision.visual_followups = reviews
            decisions.append(decision)
            LOGGER.info(
                "ad_decision gap=%d window=%s decision=%s timing_source=%s ready_at=%s reason=%s",
                index, decision.selected_window, decision.decision, decision.event_timing_source,
                decision.description_ready_at, decision.reason,
            )
            if knowledge_store is not None:
                knowledge_store.record_decision(len(decisions), decision)
            if decision.decision == "AD":
                previous_ad.append(decision.description)
    return decisions


def decision_from_proposal(
    gap: Gap, proposal: dict[str, Any], frames: Sequence[EvidenceFrame], gap_index: int, grounded: bool
) -> Decision:
    payload = proposal if grounded else normalize_event_payload(proposal, frames, gap)
    description = ""
    if str(payload.get("decision", "WAIT")).upper() == "AD":
        description = limit_words(str(payload.get("description", "")).strip(), gap.word_budget)
    is_ad = bool(description)
    ready_at = _coerce_timestamp(payload.get("description_ready_at")) if is_ad else None
    event_start = _coerce_timestamp(payload.get("event_start")) if is_ad else None
    anchor = ready_at if ready_at is not None else event_start
    return Decision(
        gap_start=gap.start,
        gap_end=gap.end,
        decision="AD" if is_ad else "WAIT",
        description=description,
        reason=str(payload.get("reason", "")),
        importance=float(payload.get("importance", 0.0)),
        word_budget=gap.word_budget,
        evidence_frames=(
            [frame["path"] for frame in payload.get("frames", [])]
            if is_ad and grounded else [frame.path for frame in frames]
        ),
        evidence_timestamps=list(payload.get("evidence_timestamps", [])) if is_ad else [],
        event_start=event_start,
        event_end=_coerce_timestamp(payload.get("event_end")) if is_ad else None,
        event_timing_source=str(payload.get("event_timing_source", "none")) if is_ad else "none",
        preferred_cue_start=anchor + 0.15 if anchor is not None else None,
        description_ready_at=ready_at,
        selected_window=payload.get("window_index") if is_ad else None,
        gap_index=gap_index,
    )


def resolve_tts_backend(
    requested: str,
    piper_model: str | None,
    transformers_model: str | None,
) -> tuple[str, str]:
    candidates = [requested] if requested != "auto" else ["transformers", "piper", "espeak-ng", "say"]
    for candidate in candidates:
        if candidate == "transformers" and transformers_model:
            LOGGER.info("method stage=tts backend=transformers-vits device=pipeline_device")
            return candidate, transformers_model
        executable = shutil.which(candidate)
        if candidate == "piper" and executable and piper_model:
            LOGGER.info("method stage=tts backend=piper device=cpu")
            return candidate, executable
        if candidate in {"espeak-ng", "say"} and executable:
            LOGGER.info("method stage=tts backend=%s device=cpu", candidate)
            return candidate, executable
    raise RuntimeError(
        "No offline TTS backend found. Supply --tts-model for Conda-native Transformers TTS, "
        "or provide piper, espeak-ng, or macOS say."
    )


_TRANSFORMERS_TTS_CACHE: dict[str, tuple[Any, Any, Any, int, str]] = {}


def synthesize_transformers_tts(text: str, output: Path, model_path: str, device: str) -> Path:
    cache_key = f"{model_path}:{device}"
    if cache_key not in _TRANSFORMERS_TTS_CACHE:
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise RuntimeError("Transformers TTS requires the Conda pytorch and transformers packages.") from exc
        dtype = torch.float32
        model = VitsModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _TRANSFORMERS_TTS_CACHE[cache_key] = (
            torch,
            tokenizer,
            model,
            int(model.config.sampling_rate),
            device,
        )
    torch, tokenizer, model, sample_rate, model_device = _TRANSFORMERS_TTS_CACHE[cache_key]
    inputs = tokenizer(text, return_tensors="pt").to(model_device)
    with torch.inference_mode():
        waveform = model(**inputs).waveform.squeeze().detach().float().cpu().numpy()
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return output


def synthesize_text(
    text: str,
    output_stem: Path,
    backend: str,
    executable: str,
    piper_model: str | None,
    device: str,
) -> Path:
    if backend == "transformers":
        return synthesize_transformers_tts(
            text,
            output_stem.with_suffix(".wav"),
            executable,
            device,
        )
    if backend == "piper":
        output = output_stem.with_suffix(".wav")
        run_command([executable, "--model", str(piper_model), "--output_file", str(output)], input_text=text)
        return output
    if backend == "espeak-ng":
        output = output_stem.with_suffix(".wav")
        run_command([executable, "-s", "175", "-w", str(output), text])
        return output
    if backend == "say":
        output = output_stem.with_suffix(".aiff")
        run_command([executable, "-r", "190", "-o", str(output), text])
        return output
    raise ValueError(backend)


def probe_duration(path: Path, ffprobe: str) -> float:
    result = run_command(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def choose_event_aware_cue_start(
    decision: Decision,
    narration_duration: float,
    safety_margin: float = 0.075,
    maximum_anticipation: float = 0.35,
    occupied: Sequence[Interval] = (),
) -> tuple[float | None, str]:
    """Choose a safe cue near the visual event without narrating it too early."""
    earliest = decision.gap_start + safety_margin
    latest = decision.gap_end - safety_margin - narration_duration
    if latest < earliest:
        return None, "no_gap_fit"
    if decision.event_start is None:
        centered = decision.gap_start + (decision.gap_end - decision.gap_start - narration_duration) / 2.0
        return min(max(centered, earliest), latest), "gap_center_fallback"
    preferred = decision.preferred_cue_start
    ready_at = decision.description_ready_at
    if preferred is None:
        preferred = (ready_at if ready_at is not None else decision.event_start) + 0.15
    minimum_start = ready_at if ready_at is not None else decision.event_start - maximum_anticipation
    earliest = max(earliest, minimum_start)
    if latest < earliest:
        return None, "no_post_event_fit"
    cue_start = min(max(preferred, earliest), latest)
    if occupied:
        # A second cue can move slightly to fit, while staying near its evidence.
        latest = min(latest, preferred + 2.5)
        candidates = []
        for prior in sorted(occupied, key=lambda item: item.start):
            end_before_prior = min(latest, prior.start - safety_margin - narration_duration)
            if earliest <= end_before_prior:
                candidates.append(min(max(preferred, earliest), end_before_prior))
            earliest = max(earliest, prior.end + safety_margin)
        if earliest <= latest:
            candidates.append(min(max(preferred, earliest), latest))
        if not candidates:
            return None, "cue_conflict"
        new_start = min(candidates, key=lambda start: (abs(start - preferred), start))
        if not math.isclose(new_start, cue_start, abs_tol=0.001):
            return new_start, "event_shifted_around_other_cue"
    if math.isclose(cue_start, preferred, abs_tol=0.001):
        method = "event_aligned"
    elif cue_start < preferred:
        method = "event_shifted_earlier_to_fit"
    else:
        method = "event_shifted_later_for_gap_margin"
    return cue_start, method


def synthesize_decisions(
    decisions: list[Decision],
    output_dir: Path,
    requested_backend: str,
    piper_model: str | None,
    transformers_model: str | None,
    device: str,
    ffprobe: str,
) -> tuple[str, list[Decision]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend, executable = resolve_tts_backend(requested_backend, piper_model, transformers_model)
    occupied_by_gap: dict[tuple[float, float], list[Interval]] = {}
    priority = sorted(
        enumerate(decisions, start=1),
        key=lambda item: (
            item[1].gap_start, -item[1].importance,
            item[1].preferred_cue_start if item[1].preferred_cue_start is not None else item[1].gap_start,
        ),
    )
    for index, decision in priority:
        if decision.decision != "AD":
            continue
        text = decision.description
        occupied = occupied_by_gap.setdefault((decision.gap_start, decision.gap_end), [])
        available = max(0.25, decision.gap_end - decision.gap_start - 0.15)
        for attempt in range(4):
            audio_path = synthesize_text(
                text,
                output_dir / f"ad-{index:03d}-try-{attempt + 1}",
                backend,
                executable,
                piper_model,
                device,
            )
            duration = probe_duration(audio_path, ffprobe)
            cue_start, placement_method = choose_event_aware_cue_start(decision, duration, occupied=occupied)
            if duration <= available and cue_start is not None:
                decision.description = text
                decision.tts_path = str(audio_path)
                decision.tts_duration = duration
                decision.cue_start = cue_start
                decision.cue_end = decision.cue_start + duration
                decision.placement_method = placement_method
                occupied.append(Interval(decision.cue_start, decision.cue_end))
                LOGGER.info(
                    "ad_placement decision=%d gap=%.3f-%.3f event=%.3f-%.3f "
                    "cue=%.3f-%.3f method=%s tts_duration=%.3f",
                    index,
                    decision.gap_start,
                    decision.gap_end,
                    decision.event_start if decision.event_start is not None else -1.0,
                    decision.event_end if decision.event_end is not None else -1.0,
                    decision.cue_start,
                    decision.cue_end,
                    placement_method,
                    duration,
                )
                break
            words = text.split()
            if decision.description_ready_at is not None or len(words) <= 2:
                break
            event_available = available
            if decision.event_start is not None:
                event_available = max(
                    0.25,
                    decision.gap_end
                    - 0.075
                    - max(decision.gap_start + 0.075, decision.event_start - 0.35),
                )
            target = max(
                2,
                math.floor(len(words) * min(available, event_available) / max(duration, 0.01) * 0.88),
            )
            text = limit_words(text, max(2, min(target, len(words) - 1)))
        if not decision.tts_path:
            decision.decision = "WAIT"
            decision.reason = (
                "TTS conflicts with a higher-priority cue"
                if placement_method == "cue_conflict" else "TTS could not fit a safe post-event slot"
            )
            decision.description = ""
            decision.placement_method = (
                "rejected_cue_conflict" if placement_method == "cue_conflict" else "rejected_no_safe_event_slot"
            )
    return backend, decisions


def vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def write_vtt(decisions: Sequence[Decision], output: Path) -> None:
    lines = ["WEBVTT", ""]
    for decision in sorted(decisions, key=lambda item: item.cue_start):
        if decision.decision != "AD" or not decision.tts_path:
            continue
        lines.extend(
            [
                f"{vtt_timestamp(decision.cue_start)} --> {vtt_timestamp(decision.cue_end)}",
                decision.description,
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def mix_described_video(
    input_video: Path,
    output_video: Path,
    decisions: Sequence[Decision],
    ffmpeg: str,
) -> str:
    active = [decision for decision in decisions if decision.decision == "AD" and decision.tts_path]
    if not active:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(input_video),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
        )
        LOGGER.info(
            "method stage=video_mix backend=ffmpeg_stream_copy reason=no_accepted_ad_events"
        )
        return "ffmpeg_stream_copy_no_ad"
    args: list[str] = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(input_video)]
    for decision in active:
        args.extend(["-i", decision.tts_path])
    intervals = "+".join(f"between(t,{d.cue_start:.3f},{d.cue_end:.3f})" for d in active)
    filters = [
        "[0:a:0]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume='if(gt({intervals},0),0.35,1)':eval=frame[base]"
    ]
    labels = ["[base]"]
    for index, decision in enumerate(active, start=1):
        delay = max(0, round(decision.cue_start * 1000))
        label = f"ad{index}"
        filters.append(
            f"[{index}:a:0]aresample=48000,aformat=channel_layouts=stereo,"
            f"adelay={delay}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.95[mix]"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map",
            "[mix]",
            "-c:v",
            "copy",
            "-c:a:0",
            "aac",
            "-b:a:0",
            "192k",
            "-c:a:1",
            "aac",
            "-b:a:1",
            "192k",
            "-ar",
            "48000",
            "-metadata:s:a:0",
            "title=Original Audio",
            "-metadata:s:a:0",
            "handler_name=Original Audio",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "title=Audio Description",
            "-metadata:s:a:1",
            "handler_name=Audio Description",
            "-metadata:s:a:1",
            "language=eng",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "visual_impaired",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    run_command(args)
    return "ffmpeg_filter_complex_audio_mix"


def validate_decode(output_video: Path, ffmpeg: str) -> None:
    run_command([ffmpeg, "-nostdin", "-v", "error", "-i", str(output_video), "-f", "null", "-"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate(
    media: MediaInfo,
    speech: Sequence[Interval],
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
    gaps: Sequence[Gap],
    decisions: Sequence[Decision],
    elapsed: float,
) -> dict[str, Any]:
    active = [item for item in decisions if item.decision == "AD" and item.tts_path]
    overlap_seconds = sum(
        overlap(item.cue_start, item.cue_end, interval.start, interval.end)
        for item in active
        for interval in speech
    )
    event_offsets = [
        item.cue_start - item.event_start
        for item in active
        if item.event_start is not None
    ]
    windows = [window for decision in decisions for window in decision.visual_proposals]
    reviews = [review for decision in decisions for review in decision.visual_followups]
    cue_counts: dict[tuple[float, float], int] = {}
    for item in active:
        key = (item.gap_start, item.gap_end)
        cue_counts[key] = cue_counts.get(key, 0) + 1
    return {
        "duration_seconds": round(media.duration, 3),
        "shots": len(shots),
        "scenes": len(scenes),
        "candidate_gaps": len(gaps),
        "processed_gaps": sum(bool(item.visual_proposals) for item in decisions),
        "accepted_ad_events": len(active),
        "gaps_with_multiple_ad_events": sum(count > 1 for count in cue_counts.values()),
        "wait_events": sum(item.decision == "WAIT" for item in decisions),
        "event_grounded_ad_events": sum(item.event_start is not None for item in active),
        "frame_id_grounded_ad_events": sum(item.event_timing_source == "model_frame_ids" for item in active),
        "visual_windows_evaluated": sum(window["scan"] == "initial" for window in windows),
        "first_pass_qwen_attempts": sum(
            len(window.get("attempts", [])) for window in windows if window["scan"] == "initial"
        ),
        "followup_qwen_attempts": sum(review["calls"] for review in reviews),
        "followup_elapsed_seconds": round(sum(review.get("elapsed_seconds", 0.0) for review in reviews), 3),
        "duplicate_proposals_removed": sum(
            review["kind"] == "duplicate_text" or review.get("relation") == "same"
            for review in reviews
        ),
        "invalid_vlm_windows": sum(window["reason"] == "invalid_vlm_output" for window in windows),
        "vlm_generation_failed_windows": sum(window["reason"] == "vlm_generation_failure" for window in windows),
        "vlm_retried_windows": sum(len(window.get("attempts", [])) > 1 for window in windows),
        "event_aligned_ad_events": sum(
            item.placement_method.startswith("event_") for item in active
        ),
        "rejected_no_safe_event_slot": sum(
            item.placement_method == "rejected_no_safe_event_slot" for item in decisions
        ),
        "rejected_cue_conflict": sum(item.placement_method == "rejected_cue_conflict" for item in decisions),
        "mean_cue_start_offset_from_event_seconds": (
            round(sum(event_offsets) / len(event_offsets), 4) if event_offsets else None
        ),
        "speech_overlap_seconds": round(overlap_seconds, 4),
        "tts_fit_rate": round(
            sum(item.tts_duration <= item.gap_end - item.gap_start for item in active) / len(active), 4
        )
        if active
        else None,
        "processing_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / media.duration, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--asr-backend",
        choices=["auto", "transformers", "faster-whisper"],
        default="auto",
        help="auto uses faster-whisper when installed, otherwise Conda-native Transformers",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-small.en",
        help="Local Whisper model directory; model downloading is disabled by default",
    )
    parser.add_argument("--allow-model-download", action="store_true", help="Disabled by default for offline runs")
    parser.add_argument("--visual-backend", choices=["heuristic", "qwen25"], default="heuristic")
    parser.add_argument("--vlm-model", help="Local Qwen2.5-VL model directory")
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "transformers", "piper", "espeak-ng", "say"],
        default="auto",
    )
    parser.add_argument("--tts-model", help="Local Transformers VITS TTS model directory")
    parser.add_argument("--piper-model", help="Local Piper .onnx voice model")
    parser.add_argument("--minimum-gap", type=float, default=1.5)
    parser.add_argument("--maximum-gaps", type=int, default=12)
    parser.add_argument("--ad-plan", choices=["single", "multi"], default="multi",
                        help="Choose one proposal or schedule multiple events per speech gap")
    parser.add_argument("--event-followups", type=int, default=2,
                        help="Maximum extra Qwen calls per video for multi-event refinement")
    parser.add_argument("--qwen-seconds-per-call", type=float,
                        help="Optional measured per-call time for a planning estimate")
    parser.add_argument(
        "--dialogue-filter",
        choices=["timestamp", "none"],
        default="timestamp",
        help="Validate Whisper word durations before treating timestamps as dialogue",
    )
    parser.add_argument("--maximum-word-duration", type=float, default=2.5)
    parser.add_argument(
        "--evidence-max-edge",
        type=int,
        default=0,
        help=(
            "Downscale VLM evidence frames so their longest edge is at most this many pixels; "
            "0 preserves the source resolution"
        ),
    )
    parser.add_argument(
        "--evidence-frames",
        type=int,
        default=6,
        help="Maximum event-selected frames supplied per visual window",
    )
    parser.add_argument("--shot-threshold", type=float, default=27.0)
    parser.add_argument("--scene-threshold", type=float, default=0.52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evidence_max_edge < 0:
        raise ValueError("--evidence-max-edge must be zero or a positive integer")
    if args.evidence_frames < 3:
        raise ValueError("--evidence-frames must be at least 3")
    if args.event_followups < 0:
        raise ValueError("--event-followups must be zero or a positive integer")
    if args.qwen_seconds_per_call is not None and (
        not math.isfinite(args.qwen_seconds_per_call) or args.qwen_seconds_per_call <= 0
    ):
        raise ValueError("--qwen-seconds-per-call must be finite and positive")
    started = time.perf_counter()
    input_video = args.input.resolve()
    output_video = args.output_video.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(work_dir)
    LOGGER.info(
        "pipeline_start input=%s output=%s requested_device=%s asr_backend=%s "
        "visual_backend=%s tts_backend=%s",
        input_video,
        output_video,
        args.device,
        args.asr_backend,
        args.visual_backend,
        args.tts_backend,
    )
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    profiler = RuntimeProfiler(work_dir)

    with profiler.stage("media_tools", method="environment_or_path_lookup", compute_device="cpu"):
        ffmpeg, ffprobe = resolve_ffmpeg_tools()
    with profiler.stage("media_probe", method="ffprobe", compute_device="cpu") as stage:
        media = probe_media(input_video, ffprobe)
        stage.update(
            duration_seconds=round(media.duration, 6),
            width=media.width,
            height=media.height,
            fps=round(media.fps, 6),
            has_audio=media.has_audio,
        )
    if not media.has_audio:
        raise ValueError("Prototype currently requires an input audio stream")
    with profiler.stage("device_selection", method="cuda_capability_check", compute_device="cpu") as stage:
        device = choose_device(args.device)
        stage.update(requested_device=args.device, selected_device=device)
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    with profiler.stage("hardware_probe", method="python_platform_and_torch", compute_device="cpu"):
        profiler.set_hardware(collect_hardware_info())

    analysis_wav = work_dir / "analysis-audio.wav"
    with profiler.stage(
        "audio_extraction",
        method="ffmpeg_pcm_s16le_16khz_mono",
        compute_device="cpu",
    ):
        extract_analysis_audio(input_video, analysis_wav, ffmpeg)
    with profiler.stage("asr", requested_method=args.asr_backend, compute_device=device) as stage:
        raw_segments, raw_words, asr_report = transcribe_audio(
            analysis_wav,
            args.asr_model,
            device,
            args.allow_model_download,
            args.asr_backend,
        )
        stage.update(
            method=asr_report["backend"],
            model=asr_report["model"],
            segments=len(raw_segments),
            words=len(raw_words),
        )
    if args.dialogue_filter == "timestamp":
        with profiler.stage(
            "dialogue_timestamp_validation",
            method="whisper_word_duration_validation",
            compute_device="cpu",
        ) as stage:
            words, dialogue_validation_report = validate_whisper_word_timestamps(
                raw_words,
                maximum_duration=args.maximum_word_duration,
            )
            segments = _segments_from_words(words)
            stage.update(**dialogue_validation_report)
    else:
        dialogue_validation_report = {
            "method": "none",
            "raw_words": len(raw_words),
            "validated_words": len(raw_words),
            "rejected_words": 0,
        }
        words = list(raw_words)
        segments = list(raw_segments)
        LOGGER.warning(
            "fallback stage=dialogue_timestamp_validation backend=none "
            "reason=unvalidated_whisper_word_timestamps_requested"
        )
    with profiler.stage(
        "speech_gap_detection",
        method="validated_word_intervals_padding_merge",
        compute_device="cpu",
    ) as stage:
        speech = speech_intervals_from_words(words, media.duration)
        gaps = build_gaps(speech, media.duration, minimum_duration=args.minimum_gap)
        stage.update(speech_intervals=len(speech), candidate_gaps=len(gaps))

    with profiler.stage("shot_detection", compute_device="cpu") as stage:
        shots = detect_shots(
            input_video,
            media.duration,
            threshold=args.shot_threshold,
            runtime_details=stage,
        )
    with profiler.stage(
        "shot_boundary_validation",
        method="before_after_evidence",
        compute_device="cpu",
    ) as stage:
        shot_validation = create_shot_boundary_evidence(
            input_video,
            shots,
            work_dir / "shot-boundary-evidence",
            media.duration,
        )
        stage.update(
            boundaries=shot_validation["boundaries"],
            decoded_boundaries=shot_validation["decoded_boundaries"],
            valid_timeline=shot_validation["valid_timeline"],
            evidence_complete=shot_validation["evidence_complete"],
            validation_passed=shot_validation["validation_passed"],
            warnings=len(shot_validation["warnings"]),
            soft_candidates=shot_validation["soft_candidate_count"],
        )
    with profiler.stage(
        "shot_visual_features",
        method="opencv_hsv_histogram_and_motion",
        compute_device="cpu",
    ):
        histograms = sample_shot_features(input_video, shots, work_dir / "keyframes")
    with profiler.stage(
        "shot_audio_features",
        method="numpy_rms_and_fft_centroid",
        compute_device="cpu",
    ):
        wav_features_by_shot(analysis_wav, shots)
    with profiler.stage(
        "transcript_shot_alignment",
        method="timestamp_overlap",
        compute_device="cpu",
    ):
        attach_transcripts(shots, segments)
    with profiler.stage(
        "scene_segmentation",
        method="weighted_visual_audio_text_distance",
        compute_device="cpu",
    ) as stage:
        scenes = segment_scenes(shots, histograms, boundary_threshold=args.scene_threshold)
        stage.update(boundary_threshold=args.scene_threshold, scenes=len(scenes))

    soft_candidate_timestamps = [
        float(item["timestamp"])
        for item in shot_validation.get("soft_candidates_requiring_review", [])
        if "timestamp" in item
    ]
    knowledge_store = EvidenceStore(work_dir / "evidence-knowledge.sqlite3")
    with profiler.stage(
        "knowledge_index",
        method="sqlite_timestamp_evidence_store",
        compute_device="cpu",
    ) as stage:
        knowledge_store.index_timeline(
            input_video,
            media.duration,
            segments,
            shots,
            scenes,
            shot_validation,
        )
        stage.update(
            transcript_segments=len(segments),
            shots=len(shots),
            scenes=len(scenes),
            confirmed_boundaries=max(0, len(shots) - 1),
            soft_candidates=len(soft_candidate_timestamps),
        )

    qwen_plan = None
    if args.visual_backend == "qwen25":
        qwen_plan = plan_qwen_calls(
            gaps, args.maximum_gaps, args.event_followups if args.ad_plan == "multi" else 0,
            args.qwen_seconds_per_call,
        )
        LOGGER.info("qwen_call_plan %s", json.dumps(qwen_plan))
    visual_compute_device = device if args.visual_backend == "qwen25" else "cpu"
    with profiler.stage(
        "visual_backend_initialization",
        method=args.visual_backend,
        compute_device=visual_compute_device,
    ):
        backend = choose_backend(args.visual_backend, args.vlm_model, device)
    with profiler.stage(
        "ad_decisions",
        method=args.visual_backend,
        compute_device=visual_compute_device,
        evidence_max_edge=args.evidence_max_edge or "source",
        evidence_frames_per_window=args.evidence_frames,
        ad_plan=args.ad_plan,
        event_followup_budget=args.event_followups if args.ad_plan == "multi" and backend.uses_event_windows else 0,
        evidence_sampling="temporal_windows" if backend.uses_event_windows else "adaptive_visual_change_and_boundaries",
    ) as stage:
        decisions = make_decisions(
            input_video,
            gaps,
            shots,
            scenes,
            segments,
            backend,
            work_dir / "evidence-frames",
            args.maximum_gaps,
            args.evidence_max_edge or None,
            args.evidence_frames,
            soft_candidate_timestamps,
            knowledge_store,
            multi_event=args.ad_plan == "multi",
            followup_budget=args.event_followups,
        )
        stage.update(
            processed_gaps=sum(bool(item.visual_proposals) for item in decisions),
            followup_qwen_attempts=sum(
                review["calls"] for item in decisions for review in item.visual_followups
            ),
            ad_decisions=sum(item.decision == "AD" for item in decisions),
            wait_decisions=sum(item.decision == "WAIT" for item in decisions),
            **backend.runtime_metrics(),
        )
    with profiler.stage("visual_proposal_write", method="json", compute_device="cpu"):
        plan_by_gap: dict[int | None, dict[str, Any]] = {}
        for item in decisions:
            if item.visual_proposals:
                plan_by_gap[item.gap_index] = {
                    "gap_index": item.gap_index,
                    "gap_start": item.gap_start,
                    "gap_end": item.gap_end,
                    "selected_window": item.selected_window,
                    "selected_windows": [],
                    "windows": item.visual_proposals,
                    "followups": item.visual_followups,
                }
            if item.selected_window is not None:
                plan_by_gap[item.gap_index]["selected_windows"].append(item.selected_window)
        write_json(work_dir / "ad-proposals.json", list(plan_by_gap.values()))
    tts_compute_device = (
        device
        if args.tts_backend == "transformers" or (args.tts_backend == "auto" and args.tts_model)
        else "cpu"
    )
    with profiler.stage(
        "tts",
        requested_method=args.tts_backend,
        compute_device=tts_compute_device,
    ) as stage:
        tts_backend, decisions = synthesize_decisions(
            decisions,
            work_dir / "narration",
            args.tts_backend,
            args.piper_model,
            args.tts_model,
            device,
            ffprobe,
        )
        stage.update(
            method=tts_backend,
            compute_device=device if tts_backend == "transformers" else "cpu",
            generated_cues=sum(bool(item.tts_path) for item in decisions),
            event_aligned_cues=sum(
                item.placement_method.startswith("event_") for item in decisions if item.tts_path
            ),
        )

    with profiler.stage(
        "knowledge_finalize",
        method="sqlite_decision_upsert",
        compute_device="cpu",
    ):
        for index, decision in enumerate(decisions, start=1):
            knowledge_store.record_decision(index, decision)
        knowledge_store.close()

    with profiler.stage("artifact_write", method="json_and_webvtt", compute_device="cpu"):
        write_vtt(decisions, work_dir / "descriptions.vtt")
        write_json(work_dir / "media.json", asdict(media))
        write_json(
            work_dir / "transcript.json",
            {
                "report": {
                    **asr_report,
                    "raw_segments": len(raw_segments),
                    "raw_words": len(raw_words),
                    "validated_segments": len(segments),
                    "validated_words": len(words),
                },
                "segments": [asdict(x) for x in segments],
                "words": [asdict(x) for x in words],
                "raw_segments": [asdict(x) for x in raw_segments],
                "raw_words": [asdict(x) for x in raw_words],
            },
        )
        write_json(work_dir / "dialogue-validation.json", dialogue_validation_report)
        write_json(
            work_dir / "speech-and-gaps.json",
            {"speech": [asdict(x) for x in speech], "gaps": [asdict(x) for x in gaps]},
        )
        write_json(work_dir / "shots.json", [asdict(x) for x in shots])
        write_json(work_dir / "shot-boundary-validation.json", shot_validation)
        write_json(work_dir / "scenes.json", [asdict(x) for x in scenes])
        write_json(work_dir / "ad-decisions.json", [
            {key: value for key, value in vars(item).items() if key not in {"visual_proposals", "visual_followups"}}
            for item in decisions
        ])

    with profiler.stage("video_mix", compute_device="cpu") as stage:
        stage["method"] = mix_described_video(input_video, output_video, decisions, ffmpeg)
    with profiler.stage("output_validation", method="ffmpeg_full_decode", compute_device="cpu"):
        validate_decode(output_video, ffmpeg)
    elapsed = time.perf_counter() - started
    with profiler.stage("evaluation", method="internal_metrics", compute_device="cpu"):
        report = {
            "input": str(input_video),
            "output": str(output_video),
            "device": device,
            "visual_backend": args.visual_backend,
            "qwen_call_plan": qwen_plan,
            "tts_backend": tts_backend,
            "offline": not args.allow_model_download,
            "runtime_log": str(work_dir / "runtime.log"),
            "runtime_events": str(profiler.events_path),
            "runtime_summary": str(profiler.summary_path),
            "metrics": evaluate(media, speech, shots, scenes, gaps, decisions, elapsed),
        }
        write_json(work_dir / "evaluation-report.json", report)
    profiler.finalize()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

