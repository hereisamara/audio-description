#!/usr/bin/env python3
"""Offline, CPU/GPU-portable audio-description prototype.

This is a research prototype.  The heuristic backend validates the full media
pipeline but does not claim useful visual-language quality. The Qwen backends
use local checkpoints and perform no API calls.
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

from .ad_preparer import DEFAULT_REASONING_TOKENS, build_qwen_cue_input, prepare_ad_cues, retry_prepared_cue, run_qwen_cue
from .knowledge_db import EvidenceStore
from .smolvlm import DEFAULT_MAX_NEW_TOKENS
from .tts_client import synthesize as synthesize_remote_tts
from .tts_model import synthesize_transformers_tts
from .visual_events import apply_ad_policy, description_sentences, direct_ad_text, validate_visual_event


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
    near_blank: bool = False


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
    tts_attempts: list[dict[str, Any]] = field(default_factory=list)
    cue_start: float = 0.0
    cue_end: float = 0.0
    description_ready_at: float | None = None
    visual_proposals: list[dict[str, Any]] = field(default_factory=list)
    selected_window: int | None = None
    gap_index: int | None = None
    visual_followups: list[dict[str, Any]] = field(default_factory=list)
    fallback_scene_id: int | None = None
    overlaps_dialogue: bool = False
    dialogue_overlap_seconds: float = 0.0


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


def analysis_audio_is_silent(wav_path: Path) -> bool:
    """Check the extracted 16-bit PCM for digital silence in bounded reads."""
    with wave.open(str(wav_path), "rb") as source:
        if source.getsampwidth() != 2:
            return False
        while frames := source.readframes(65536):
            if np.frombuffer(frames, dtype="<i2").any():
                return False
    return True


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
    *,
    audit: list[dict[str, Any]] | None = None,
) -> list[Gap]:
    gaps: list[Gap] = []

    def scan(start: float, end: float) -> None:
        span = end - start
        if span <= 0:
            return
        usable_start = start + safety_margin
        usable_end = end - safety_margin
        usable_seconds = max(0.0, usable_end - usable_start)
        gap_index = None
        reason = "gap_too_short" if span < minimum_duration else "gap_too_short_after_padding"
        if span >= minimum_duration and usable_seconds >= minimum_duration:
            budget = max(1, math.floor(usable_seconds * words_per_second))
            gaps.append(Gap(usable_start, usable_end, budget))
            gap_index = len(gaps)
            reason = "eligible_gap"
        if audit is not None:
            audit.append({
                "start": start, "end": end, "gap_index": gap_index,
                "usable_seconds": usable_seconds, "minimum_seconds": minimum_duration,
                "status": "eligible" if gap_index is not None else "not_scanned",
                "reason": reason,
            })

    cursor = 0.0
    for interval in speech:
        scan(cursor, interval.start)
        cursor = max(cursor, interval.end)
    scan(cursor, duration)
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


def is_near_blank(gray: np.ndarray) -> bool:
    """Flag frames with very low brightness and almost no brighter detail."""
    return bool(
        gray.mean() < 12
        and cv2.countNonZero(cv2.compare(gray, 24, cv2.CMP_GT)) <= 0.05 * gray.size
    )


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
    near_blank_intervals: list[dict[str, float]] = []
    dark_start: int | None = None
    minimum_dark_frames = math.ceil(0.15 * fps)

    def close_dark_interval(end: int) -> None:
        if dark_start is not None and end - dark_start >= minimum_dark_frames:
            near_blank_intervals.append({
                "start": round(dark_start / fps, 6),
                "end": round(min(duration, end / fps), 6),
            })

    try:
        while True:
            ok, frame = soft_capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if is_near_blank(gray):
                if dark_start is None:
                    dark_start = frame_index
            elif dark_start is not None:
                close_dark_interval(frame_index)
                dark_start = None
            if previous is not None:
                score = float(np.mean(cv2.absdiff(previous, gray)))
                timestamp = frame_index / fps
                if score >= 10.0:
                    scored_candidates.append((score, timestamp))
            previous = gray
            frame_index += 1
    finally:
        soft_capture.release()
    close_dark_interval(frame_index)
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
    report["near_blank_intervals"] = near_blank_intervals
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
                    "near_blank": is_near_blank(gray),
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
                near_blank=bool(item["near_blank"]),
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


class QwenEventBackend(VisualBackend):
    uses_event_windows = True

    def decide(
        self,
        frames: Sequence[EvidenceFrame],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_ids = [index for index, frame in enumerate(frames, 1) if gap.start <= frame.timestamp < gap.end]
        if not evidence_ids:
            return {"decision": "WAIT", "reason": "no_in_window_evidence", "attempts": []}
        action_ready_ids = [index for index in evidence_ids if not frames[index - 1].near_blank]
        expected_ids = ", ".join(map(str, evidence_ids))
        action_ids = ", ".join(map(str, action_ready_ids)) or "none"
        defer_word_budget = context.get("defer_word_budget") and not context.get("refine_description")
        word_limit = 32 if defer_word_budget else min(18, gap.word_budget)
        description_instruction = (
            "In description, write one short sentence about the main visible action or detail. "
            "Use a second sentence only if it adds a separate detail essential to following the scene. "
            "Skip lists of clothing, objects, or background details unless they matter to the event. "
            f"Use at most two sentences and {word_limit} words total. "
        )
        if context.get("scene_fallback"):
            description_instruction += "Describe the main new visible detail of this scene in one concise line. "
        history_instruction = (
            f"Earlier proposed narration: {' | '.join(previous_ad[-3:]) or 'none'}."
            if not context.get("defer_word_budget") else ""
        )
        prompt = (
            "Identify the strongest new visible event or informative detail in the numbered film images. "
            "A cut alone is not an event. Describe only supported facts in the present tense. "
            "Set important=true when the detail helps someone follow the film. "
            "Set already_audible=true when the dialogue communicates the same information. "
            "Identify each object from the current numbered images. Dialogue may help identify a visible object. "
            "Return one JSON object using this example and no additional text: "
            '{"kind":"action","important":true,"already_audible":false,"uncertain":false,'
            '"frame_ids":[2,3],"ready_frame_id":3,"description":"A woman stands."} '
            "kind is one of none, action, setting, text, detail. "
            f"Use only these integer image IDs as evidence: {expected_ids}. "
            f"For an action ready_frame_id, use only these IDs: {action_ids}. "
            "Near-blank images may show a transition but cannot anchor a visible action. "
            "ready_frame_id is the first supporting image where the complete description is apparent. "
            "For kind=none use important=false, frame_ids=[], ready_frame_id=null, description=\"\". "
            f"{description_instruction}"
            f"Recent dialogue: {transcript or 'none'}. "
            f"{history_instruction}"
        )
        content: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, start=1):
            role = "context only"
            if index in evidence_ids:
                role = "inside window, near-blank" if frame.near_blank else "inside window"
            content.append({"type": "image", "image": str(Path(frame.path).resolve())})
            content.append({"type": "text", "text": f"Image ID {index} ({role})."})
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
            attempt = {"prompt": instruction, "recent_dialogue": transcript}
            try:
                decoded = self._generate(frames, [*content, {"type": "text", "text": instruction}])
            except Exception as exc:
                LOGGER.exception("vlm_generation_failure window=%.3f-%.3f", gap.start, gap.end)
                attempts.append({**attempt, "generation_error": f"{type(exc).__name__}: {exc}"})
                return {"decision": "WAIT", "reason": "vlm_generation_failure", "attempts": attempts}
            try:
                proposal = validate_visual_event(
                    json_object_from_text(decoded), timestamps, gap.start, gap.end,
                    action_ready_ids=action_ready_ids,
                )
                description = proposal["description"]
                if len(description_sentences(description)) > 2:
                    raise ValueError("description exceeds the two-sentence limit")
                if len(description.split()) > word_limit:
                    raise ValueError(f"description exceeds the {word_limit}-word budget")
            except ValueError as exc:
                attempts.append({**attempt, "raw_output": decoded, "validation_error": str(exc)})
                LOGGER.warning("vlm_event_invalid window=%.3f-%.3f attempt=%d error=%s",
                               gap.start, gap.end, len(attempts), exc)
                instruction = f"{prompt}\nThe previous answer failed validation: {exc}. Correct it."
                continue
            attempts.append({**attempt, "raw_output": decoded})
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
        raise NotImplementedError


class QwenTransformersBackend(QwenEventBackend):
    """Shared image-only Transformers path for Qwen2.5-VL and Qwen3-VL."""

    def __init__(self, model_path: str, model_class: Any, **model_kwargs: Any) -> None:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoTokenizer

        self.torch = torch
        self.image_type = Image
        self.model = model_class.from_pretrained(
            model_path, local_files_only=True, low_cpu_mem_usage=True, **model_kwargs,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=False,
        )
        self.processed_image_grids: list[list[list[int]]] = []
        self.vision_tokens = 0

    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "qwen_image_grid_thw": self.processed_image_grids,
            "qwen_vision_tokens": self.vision_tokens,
        }

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


class Qwen25Backend(QwenTransformersBackend):
    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration

        kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        if device == "cuda":
            kwargs["device_map"] = "auto"
        super().__init__(model_path, Qwen2_5_VLForConditionalGeneration, **kwargs)


class Qwen3VL8BBackend(QwenTransformersBackend):
    """Run the local Transformers checkpoint on the selected NVIDIA GPU."""

    def __init__(self, model_path: str, device: str) -> None:
        import torch

        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("Qwen3-VL-8B requires CUDA. Use the Qwen3-VL-2B profile for CPU.")
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-8B requires Transformers 4.57.6 or newer in the GPU Conda environment.") from exc
        gpu = torch.cuda.current_device()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(gpu)[0] >= 8 else torch.float16
        LOGGER.info("model_load backend=qwen3vl8b checkpoint=%s dtype=%s cuda_device=%d",
                    model_path, dtype, gpu)
        super().__init__(
            model_path, Qwen3VLForConditionalGeneration, dtype=dtype,
            device_map={"": gpu}, attn_implementation="sdpa", use_safetensors=True,
        )


class Qwen3VLBackend(QwenEventBackend):
    """Run Qwen3-VL GGUF on CPU with the same events and prompt as Qwen2.5-VL."""

    def __init__(self, model_path: str, mmproj: Path, executable: str) -> None:
        self.model_path = Path(model_path).resolve()
        self.mmproj = mmproj.resolve()
        resolved = shutil.which(executable)
        if not resolved or not self.model_path.is_file() or not self.mmproj.is_file():
            raise ValueError("Qwen3-VL requires local GGUF weights, a vision projector, and llama-mtmd-cli")
        self.executable = resolved
        self.calls = 0
        self.images = 0

    def runtime_metrics(self) -> dict[str, Any]:
        return {"qwen3vl_calls": self.calls, "qwen3vl_images": self.images,
                "qwen3vl_max_new_tokens": 160, "qwen3vl_cpu_threads": 4}

    def _generate(self, frames: Sequence[EvidenceFrame], content: list[dict[str, Any]]) -> str:
        marker = "<__media__>"
        prompt = "\n".join(
            marker if part["type"] == "image" else part["text"].replace(marker, "[media]")
            for part in content
        )
        paths = [str(Path(frame.path).resolve()) for frame in frames]
        # The mtmd CLI accepts a single comma-separated image list. Pass image
        # paths relative to the window directory, which may itself contain commas.
        common_dir = Path(os.path.commonpath([str(Path(path).parent) for path in paths]))
        images = ",".join(os.path.relpath(path, common_dir) for path in paths)
        self.calls += 1
        self.images += len(paths)
        result = subprocess.run(
            [
                self.executable, "--offline", "--device", "none", "-ngl", "0",
                "--no-mmproj-offload", "--no-op-offload", "--no-warmup",
                "-t", "4", "-tb", "4", "-c", "4096", "-n", "160",
                "--temp", "0", "--seed", "42", "-m", str(self.model_path),
                "--mmproj", str(self.mmproj), "--image", images,
                "--json-schema", '{"type":"object"}', "-p", prompt,
            ],
            cwd=common_dir, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip()
                               else f"llama-mtmd-cli exited with status {result.returncode}")
        return result.stdout.strip()



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


def choose_backend(
    name: str, model_path: str | None, device: str, *, smolvlm_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    vlm_mmproj: Path | None = None, vlm_executable: str = "llama-mtmd-cli",
) -> VisualBackend:
    if name == "smolvlm":
        if not model_path:
            raise ValueError("--vlm-model is required for --visual-backend smolvlm")
        from .smolvlm import SmolVLMBackend

        LOGGER.info("method stage=visual_backend backend=smolvlm device=%s", device)
        return SmolVLMBackend(model_path, device, smolvlm_max_new_tokens)
    if name == "heuristic":
        LOGGER.info("method stage=visual_backend backend=heuristic device=cpu")
        return HeuristicBackend()
    if name == "qwen3vl":
        if not model_path or not vlm_mmproj:
            raise ValueError("--vlm-model and --vlm-mmproj are required for --visual-backend qwen3vl")
        LOGGER.info("method stage=visual_backend backend=qwen3-vl-2b device=cpu runtime=llama-mtmd-cli")
        return Qwen3VLBackend(model_path, vlm_mmproj, vlm_executable)
    if name == "qwen3vl8b":
        if not model_path:
            raise ValueError("--vlm-model is required for --visual-backend qwen3vl8b")
        LOGGER.info("method stage=visual_backend backend=qwen3-vl-8b device=%s", device)
        return Qwen3VL8BBackend(model_path, device)
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
        event_start = candidate.get("event_start", ready)
        earlier = max(gap.start, ready - 6.0, min(ready - 3.0, event_start))
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
    defer_word_budget: bool = False,
    gap_index_offset: int = 0,
    scene_fallback: bool = False,
) -> list[Decision]:
    decisions: list[Decision] = []
    previous_ad: list[str] = []
    for index, gap in enumerate(gaps[:maximum_gaps], start=gap_index_offset + 1):
        windows = split_evidence_windows(gap) if backend.uses_event_windows and not scene_fallback else [gap]
        proposals: list[dict[str, Any]] = []
        frames: list[EvidenceFrame] = []
        gap_dir = frame_root / f"gap-{index:03d}"

        def inspect_window(window: Gap, refine_description: str = "") -> dict[str, Any]:
            window_index = len(proposals) + 1
            context = context_for_gap(window, shots, scenes, soft_candidate_timestamps)
            if defer_word_budget:
                context["defer_word_budget"] = True
            if scene_fallback:
                context["scene_fallback"] = True
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
                "scan": "scene_fallback" if scene_fallback else "timing_followup" if refine_description else "initial",
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
            decision = decision_from_proposal(
                gap, payload, frames, index, backend.uses_event_windows, defer_word_budget=defer_word_budget,
            )
            if payload.get("decision") == "AD" and decision.decision == "WAIT":
                payload["selection_reason"] = "no_usable_description"
            if position == 0:
                decision.visual_proposals = proposals
                decision.visual_followups = reviews
            decisions.append(decision)
            LOGGER.info(
                "ad_decision gap=%d window=%s decision=%s timing_source=%s ready_at=%s reason=%s",
                index, decision.selected_window, decision.decision, decision.event_timing_source,
                decision.description_ready_at, decision.reason,
            )
            if knowledge_store is not None and not scene_fallback:
                knowledge_store.record_decision(len(decisions), decision)
            if decision.decision == "AD":
                previous_ad.append(decision.description)
    return decisions


def record_ad_outcomes(
    gap_audit: list[dict[str, Any]], decisions: Sequence[Decision], *, multi_event: bool,
) -> list[dict[str, Any]]:
    """Link each inspected window to its final cue or the stage that stopped it."""
    groups: dict[int, dict[str, Any]] = {}
    selected = {}
    for index, item in enumerate(decisions, start=1):
        if item.gap_index is None:
            continue
        if item.visual_proposals:
            groups[item.gap_index] = {
                "gap_index": item.gap_index, "gap_start": item.gap_start, "gap_end": item.gap_end,
                "selected_window": item.selected_window, "selected_windows": [],
                "windows": item.visual_proposals, "followups": item.visual_followups,
            }
        if item.selected_window is not None:
            selected[item.gap_index, item.selected_window] = (index, item)
            groups[item.gap_index]["selected_windows"].append(item.selected_window)

    for gap in gap_audit:
        if gap["gap_index"] is None:
            continue
        group = groups.get(gap["gap_index"])
        if group is None:
            gap.update(status="not_scanned", reason="gap_scan_limit")
            continue
        discarded = {}
        for review in group["followups"]:
            if review["kind"] == "duplicate_text":
                discarded[review["discarded_window"]] = "duplicate_visual_event"
            elif review["kind"] == "compare_events" and review["relation"] == "same":
                discarded[review["windows"][1]] = "duplicate_visual_event"
            elif review["kind"] == "earlier_support":
                if review["used"]:
                    discarded[review["source_window"]] = "replaced_by_earlier_evidence"
                else:
                    discarded[review["review_window"]] = "timing_followup_not_selected"

        cue_count = 0
        for window in group["windows"]:
            window_index = window["window_index"]
            outcome = {"status": "no_ad", "stage": "visual_review", "reason": window.get("reason") or "no_vlm_output"}
            if outcome["reason"] in {"invalid_vlm_output", "empty_or_incomplete_caption"}:
                replies = [window.get("raw_caption"), *(attempt.get("raw_output") for attempt in window.get("attempts", []))]
                if not any(isinstance(reply, str) and reply.strip() for reply in replies):
                    outcome["reason"] = "empty_vlm_output"
            if window.get("decision") == "AD":
                result = selected.get((gap["gap_index"], window_index))
                if result is None:
                    reason = "window_not_selected" if multi_event else "single_event_plan"
                    outcome.update(stage="event_selection", reason=window.get("selection_reason") or discarded.get(window_index, reason))
                else:
                    index, item = result
                    outcome["decision_index"] = index
                    if item.decision == "AD" and item.tts_path:
                        cue_count += 1
                        outcome.update(status="ad_added", stage="placement", reason="cue_placed",
                                       cue_start=item.cue_start, cue_end=item.cue_end)
                        if item.fallback_scene_id is not None:
                            outcome.update(
                                reason=item.placement_method, scene_id=item.fallback_scene_id,
                                overlaps_dialogue=item.overlaps_dialogue,
                                dialogue_overlap_seconds=item.dialogue_overlap_seconds,
                            )
                    elif item.placement_method.startswith("rejected_"):
                        outcome.update(stage="tts_and_placement", reason=item.reason)
                    else:
                        outcome.update(stage="ad_preparation", reason=item.reason or "ad_preparer_wait")
                    if item.tts_attempts:
                        outcome["tts_attempts"] = item.tts_attempts
            window["outcome"] = outcome
        flagged = any(window["outcome"].get("overlaps_dialogue") for window in group["windows"])
        gap.update(status="ad_added" if cue_count else "no_ad", overlaps_dialogue=flagged,
                   reason="cue_placed" if cue_count else "no_cue_from_windows",
                   cue_count=cue_count)
    return list(groups.values())


def decision_from_proposal(
    gap: Gap, proposal: dict[str, Any], frames: Sequence[EvidenceFrame], gap_index: int, grounded: bool,
    *, defer_word_budget: bool = False,
) -> Decision:
    payload = proposal if grounded else normalize_event_payload(proposal, frames, gap)
    description = ""
    if str(payload.get("decision", "WAIT")).upper() == "AD":
        description = direct_ad_text(str(payload.get("description", "")))
        if not defer_word_budget:
            description = limit_words(description, gap.word_budget)
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
    tts_url: str | None = None,
) -> tuple[str, str]:
    if requested == "http":
        if not tts_url:
            raise ValueError("--tts-backend http requires --tts-url.")
        LOGGER.info("method stage=tts backend=http device=cpu service=tts")
        return requested, tts_url
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


def synthesize_text(
    text: str,
    output_stem: Path,
    backend: str,
    executable: str,
    piper_model: str | None,
    device: str,
) -> Path:
    if backend == "http":
        output = output_stem.with_suffix(".wav")
        synthesize_remote_tts(executable, text, "en", output)
        return output
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


NARRATION_MARGIN = 0.075


def uncovered_scenes(scenes: Sequence[Scene], decisions: Sequence[Decision]) -> list[Scene]:
    """Cover a scene when it has a rendered cue grounded in that scene."""
    anchors = []
    for decision in decisions:
        if decision.decision == "AD" and decision.tts_path:
            anchor = decision.description_ready_at
            if anchor is None:
                anchor = decision.event_start if decision.event_start is not None else decision.cue_start
            anchors.append(anchor)
    return [scene for scene in scenes if not any(scene.start <= anchor < scene.end for anchor in anchors)]


def scene_fallback_windows(scenes: Sequence[Scene]) -> list[Gap]:
    """Sample the opening of each scene while reserving some room for speech at its end."""
    return [
        Gap(scene.start, min(scene.start + 12.0, scene.end - min(3.0, (scene.end - scene.start) / 3)),
            min(12, max(2, math.floor((scene.end - scene.start) * 2.5))))
        for scene in scenes
    ]


def earliest_narration_start(decision: Decision, safety_margin: float = NARRATION_MARGIN,
                             maximum_anticipation: float = 0.35) -> float:
    earliest = decision.gap_start + safety_margin
    if decision.event_start is not None:
        ready = decision.description_ready_at
        earliest = max(earliest, ready if ready is not None else decision.event_start - maximum_anticipation)
    return earliest


def narration_seconds_after_anchor(decision: Decision) -> float:
    return max(0.0, decision.gap_end - NARRATION_MARGIN - earliest_narration_start(decision))


def choose_event_aware_cue_start(
    decision: Decision,
    narration_duration: float,
    safety_margin: float = NARRATION_MARGIN,
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
    earliest = earliest_narration_start(decision, safety_margin, maximum_anticipation)
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


def choose_scene_fallback_start(
    decision: Decision, duration: float, speech: Sequence[Interval], occupied: Sequence[Interval],
) -> tuple[float | None, str, float]:
    earliest = earliest_narration_start(decision)
    latest = decision.gap_end - NARRATION_MARGIN - duration
    if latest < earliest:
        return None, "no_post_event_fit", 0.0
    preferred = decision.preferred_cue_start if decision.preferred_cue_start is not None else earliest
    latest = min(latest, max(earliest, preferred + 3.0))
    preferred = min(latest, max(earliest, preferred))
    speech = [item for item in speech if item.end > earliest and item.start < latest + duration]
    blockers = [
        Interval(item.start - NARRATION_MARGIN, item.end + NARRATION_MARGIN)
        for item in occupied
        if item.end + NARRATION_MARGIN > earliest and item.start - NARRATION_MARGIN < latest + duration
    ]
    # The overlap score is piecewise linear. Its minimum is at a speech boundary
    # entering or leaving either end of the cue, or an allowed range endpoint.
    edges = {earliest, latest, preferred}
    for item in speech:
        edges.update((item.start, item.end, item.start - duration, item.end - duration))
    for item in blockers:
        edges.update((item.start - duration, item.end))
    best: tuple[float, float, float] | None = None
    for start in edges:
        end = start + duration
        if not earliest <= start <= latest or any(
            overlap(start, end, item.start, item.end) > 1e-6 for item in blockers
        ):
            continue
        seconds = sum(overlap(start, end, item.start, item.end) for item in speech)
        score = (round(seconds, 6), abs(start - preferred), start)
        if best is None or score < best:
            best = score
    if best is None:
        return None, "cue_conflict", 0.0
    seconds, _, start = best
    return start, "scene_fallback_overlap" if seconds > 0 else "scene_fallback_safe", seconds


def synthesize_decisions(
    decisions: list[Decision],
    output_dir: Path,
    requested_backend: str,
    piper_model: str | None,
    transformers_model: str | None,
    device: str,
    ffprobe: str,
    *,
    tts_url: str | None = None,
    rewrite_for_time: Callable[[int, str, float, float], str | None] | None = None,
    decision_index_offset: int = 0,
    fallback_speech: Sequence[Interval] = (),
    existing_cues: Sequence[Decision] = (),
) -> tuple[str, list[Decision]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend, executable = resolve_tts_backend(requested_backend, piper_model, transformers_model, tts_url)
    occupied_by_gap: dict[tuple[float, float], list[Interval]] = {}
    priority = sorted(
        enumerate(decisions, start=decision_index_offset + 1),
        key=lambda item: (
            item[1].gap_start, -item[1].importance,
            item[1].preferred_cue_start if item[1].preferred_cue_start is not None else item[1].gap_start,
        ),
    )
    globally_occupied = [Interval(item.cue_start, item.cue_end) for item in existing_cues
                         if item.decision == "AD" and item.tts_path]
    for index, decision in priority:
        if decision.decision != "AD":
            continue
        text = decision.description
        occupied = occupied_by_gap.setdefault((decision.gap_start, decision.gap_end), [])
        anchor_seconds = narration_seconds_after_anchor(decision)
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
            dialogue_overlap = 0.0
            if decision.fallback_scene_id is not None:
                cue_start, placement_method, dialogue_overlap = choose_scene_fallback_start(
                    decision, duration, fallback_speech, globally_occupied,
                )
            else:
                cue_start, placement_method = choose_event_aware_cue_start(decision, duration, occupied=occupied)
            decision.tts_attempts.append({
                "description": text, "duration_seconds": round(duration, 3),
                "available_seconds": round(anchor_seconds, 3), "result": placement_method,
            })
            if cue_start is not None:
                decision.description = text
                decision.tts_path = str(audio_path)
                decision.tts_duration = duration
                decision.cue_start = cue_start
                decision.cue_end = decision.cue_start + duration
                decision.placement_method = placement_method
                decision.overlaps_dialogue = dialogue_overlap > 0
                decision.dialogue_overlap_seconds = dialogue_overlap
                occupied.append(Interval(decision.cue_start, decision.cue_end))
                globally_occupied.append(Interval(decision.cue_start, decision.cue_end))
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
                if decision.fallback_scene_id is not None:
                    LOGGER.info("scene_fallback_ad scene=%d dialogue_overlap_seconds=%.3f",
                                decision.fallback_scene_id, dialogue_overlap)
                break
            words = text.split()
            if decision.description_ready_at is not None:
                if (attempt == 0 and rewrite_for_time is not None and duration > 0
                        and placement_method in {"no_gap_fit", "no_post_event_fit"}):
                    shorter = rewrite_for_time(index, text, anchor_seconds, duration)
                    if shorter and len(shorter.split()) < len(words):
                        text = shorter
                        continue
                break
            if len(words) <= 2:
                break
            target = max(
                2,
                math.floor(len(words) * anchor_seconds / max(duration, 0.01) * 0.88),
            )
            text = limit_words(text, max(2, min(target, len(words) - 1)))
        if not decision.tts_path:
            decision.decision = "WAIT"
            decision.reason = {
                "cue_conflict": "TTS conflicts with a higher-priority cue",
                "no_gap_fit": "TTS narration is too long for the usable gap",
                "no_post_event_fit": "TTS could not fit a safe post-event slot",
            }.get(placement_method, "TTS could not fit a safe post-event slot")
            if decision.fallback_scene_id is not None and placement_method == "no_post_event_fit":
                decision.reason = "TTS could not finish inside the scene fallback window"
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
        if decision.overlaps_dialogue:
            lines.extend([f"NOTE Dialogue overlap: {decision.dialogue_overlap_seconds:.3f} s", ""])
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
    tts_candidates = sum(bool(item.tts_attempts or item.tts_path) for item in decisions)
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
        "processed_gaps": sum(bool(item.visual_proposals) and item.fallback_scene_id is None for item in decisions),
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
        "scene_fallback_windows": sum(window["scan"] == "scene_fallback" for window in windows),
        "scene_fallback_cues": sum(item.fallback_scene_id is not None for item in active),
        "scene_fallback_overlap_cues": sum(item.overlaps_dialogue for item in active),
        "tts_candidates": tts_candidates,
        "tts_fit_rate": round(len(active) / tts_candidates, 4) if tts_candidates else None,
        "processing_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / media.duration, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, help="Described MP4 (required for AD generation)")
    dialogue = parser.add_mutually_exclusive_group()
    dialogue.add_argument("--dialogue-only", action="store_true", help="Export dialogue-result.json and the analysis WAV")
    dialogue.add_argument("--dialogue-input", type=Path, help="Resume AD generation from a saved dialogue result")
    parser.add_argument("--analysis-audio", type=Path, help="Reuse the dialogue worker's analysis WAV")
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
    parser.add_argument("--visual-backend", choices=["heuristic", "qwen25", "qwen3vl", "qwen3vl8b", "smolvlm"], default="heuristic")
    parser.add_argument("--smolvlm-max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Maximum generated tokens per SmolVLM frame")
    parser.add_argument("--vlm-model", help="Local VLM model directory or Qwen3-VL GGUF")
    parser.add_argument("--vlm-mmproj", type=Path, help="Local Qwen3-VL vision projector GGUF")
    parser.add_argument("--vlm-executable", default="llama-mtmd-cli", help="Local Qwen3-VL CPU executable")
    parser.add_argument("--ad-preparer-model", type=Path, help="Optional local Qwen3-1.7B GGUF for cue editing")
    parser.add_argument("--ad-preparer-executable", default="llama-completion", help="Path to llama-completion")
    parser.add_argument("--ad-preparer-reasoning-tokens", type=int,
                        help=f"CPU thinking budget when the editor is enabled (default: {DEFAULT_REASONING_TOKENS}; 0 disables it)")
    parser.add_argument(
        "--tts-backend",
        choices=["auto", "transformers", "piper", "espeak-ng", "say", "http"],
        default="auto",
    )
    parser.add_argument("--tts-url", help="Internal speech service URL for the http backend")
    parser.add_argument("--tts-model", help="Local Transformers VITS TTS model directory")
    parser.add_argument("--piper-model", help="Local Piper .onnx voice model")
    parser.add_argument("--minimum-gap", type=float, default=1.5)
    parser.add_argument("--maximum-gaps", type=int, default=12)
    parser.add_argument("--scene-fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Describe uncovered scenes and flag any dialogue overlap")
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
    args = parser.parse_args()
    if not args.dialogue_only and args.output_video is None:
        parser.error("--output-video is required for AD generation")
    if args.analysis_audio and not args.dialogue_input:
        parser.error("--analysis-audio requires --dialogue-input")
    return args


def analyze_dialogue(
    args: argparse.Namespace, media: MediaInfo, analysis_wav: Path,
    device: str, profiler: RuntimeProfiler,
) -> dict:
    """Run the same dialogue analysis for standalone and combined jobs."""
    with profiler.stage("audio_silence_check", method="pcm_exact_zero", compute_device="cpu") as stage:
        silent_audio = analysis_audio_is_silent(analysis_wav)
        stage.update(silent=silent_audio)
    with profiler.stage(
        "asr", requested_method=args.asr_backend, compute_device="cpu" if silent_audio else device,
    ) as stage:
        if silent_audio:
            raw_segments, raw_words = [], []
            asr_report = {
                "backend": "silence_skip",
                "model": args.asr_model,
                "device": "cpu",
                "skipped": True,
                "skip_reason": "silent_pcm",
                "segments": 0,
                "words": 0,
            }
            stage.update(skip_reason=asr_report["skip_reason"])
            LOGGER.info("method stage=asr backend=silence_skip device=cpu reason=silent_pcm")
        else:
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
            skipped=silent_audio,
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
        gap_audit: list[dict[str, Any]] = []
        gaps = build_gaps(speech, media.duration, minimum_duration=args.minimum_gap, audit=gap_audit)
        stage.update(speech_intervals=len(speech), candidate_gaps=len(gaps))
    return {
        "schema_version": 1,
        "media": asdict(media),
        "transcript": {
            "report": {
                **asr_report,
                "raw_segments": len(raw_segments),
                "raw_words": len(raw_words),
                "validated_segments": len(segments),
                "validated_words": len(words),
            },
            "segments": [asdict(row) for row in segments],
            "words": [asdict(row) for row in words],
            "raw_segments": [asdict(row) for row in raw_segments],
            "raw_words": [asdict(row) for row in raw_words],
        },
        "dialogue_validation": dialogue_validation_report,
        "speech_and_gaps": {
            "speech": [asdict(row) for row in speech],
            "gaps": [asdict(row) for row in gaps],
            "gap_audit": gap_audit,
        },
    }


def write_dialogue_artifacts(work_dir: Path, dialogue: dict) -> None:
    write_json(work_dir / "media.json", dialogue["media"])
    write_json(work_dir / "transcript.json", dialogue["transcript"])
    write_json(work_dir / "dialogue-validation.json", dialogue["dialogue_validation"])
    write_json(work_dir / "speech-and-gaps.json", dialogue["speech_and_gaps"])


def main() -> None:
    args = parse_args()
    if args.tts_backend == "http" and not args.tts_url:
        raise ValueError("--tts-backend http requires --tts-url.")
    if args.smolvlm_max_new_tokens < 1:
        raise ValueError("--smolvlm-max-new-tokens must be positive")
    if args.ad_preparer_reasoning_tokens is None:
        args.ad_preparer_reasoning_tokens = DEFAULT_REASONING_TOKENS if args.ad_preparer_model else 0
    if not 0 <= args.ad_preparer_reasoning_tokens <= 512:
        raise ValueError("--ad-preparer-reasoning-tokens must be between 0 and 512")
    if args.ad_preparer_reasoning_tokens and not args.ad_preparer_model:
        raise ValueError("--ad-preparer-reasoning-tokens requires --ad-preparer-model")
    generate_cue = None
    if args.ad_preparer_model:
        model_path = args.ad_preparer_model.resolve()
        executable = shutil.which(args.ad_preparer_executable)
        if not model_path.is_file() or not executable:
            raise ValueError("AD preparation requires a local GGUF file and an executable llama-completion")
        generate_cue = lambda llm_input: run_qwen_cue(executable, model_path, llm_input)
    if args.evidence_max_edge < 0:
        raise ValueError("--evidence-max-edge must be zero or a positive integer")
    minimum_evidence_frames = 2 if args.visual_backend == "qwen3vl" else 3
    if args.evidence_frames < minimum_evidence_frames:
        raise ValueError(f"--evidence-frames must be at least {minimum_evidence_frames}")
    if args.event_followups < 0:
        raise ValueError("--event-followups must be zero or a positive integer")
    if args.qwen_seconds_per_call is not None and (
        not math.isfinite(args.qwen_seconds_per_call) or args.qwen_seconds_per_call <= 0
    ):
        raise ValueError("--qwen-seconds-per-call must be finite and positive")
    started = time.perf_counter()
    input_video = args.input.resolve()
    output_video = args.output_video.resolve() if args.output_video else None
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

    analysis_wav = args.analysis_audio.resolve() if args.analysis_audio else work_dir / "analysis-audio.wav"
    if not args.analysis_audio:
        with profiler.stage(
            "audio_extraction",
            method="ffmpeg_pcm_s16le_16khz_mono",
            compute_device="cpu",
        ):
            extract_analysis_audio(input_video, analysis_wav, ffmpeg)
    elif not analysis_wav.is_file():
        raise FileNotFoundError(analysis_wav)

    if args.dialogue_input:
        with profiler.stage("dialogue_import", method="saved_dialogue_result", compute_device="cpu") as stage:
            dialogue = json.loads(args.dialogue_input.read_text(encoding="utf-8"))
            if dialogue.get("schema_version") != 1 or dialogue.get("media") != asdict(media):
                raise ValueError("Dialogue result version or media metadata does not match the input video")
            stage.update(candidate_gaps=len(dialogue["speech_and_gaps"]["gaps"]))
    else:
        dialogue = analyze_dialogue(args, media, analysis_wav, device, profiler)

    if args.dialogue_only:
        write_json(work_dir / "dialogue-result.json", dialogue)
        write_dialogue_artifacts(work_dir, dialogue)
        profiler.finalize()
        return

    segments = [TranscriptSegment(**row) for row in dialogue["transcript"]["segments"]]
    words = [Word(**row) for row in dialogue["transcript"]["words"]]
    speech = [Interval(**row) for row in dialogue["speech_and_gaps"]["speech"]]
    gaps = [Gap(**row) for row in dialogue["speech_and_gaps"]["gaps"]]
    gap_audit = dialogue["speech_and_gaps"]["gap_audit"]

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
            near_blank_intervals=len(shot_validation["near_blank_intervals"]),
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
    qwen_event_backend = args.visual_backend in {"qwen25", "qwen3vl", "qwen3vl8b"}
    if qwen_event_backend:
        qwen_plan = plan_qwen_calls(
            gaps, args.maximum_gaps, args.event_followups if args.ad_plan == "multi" else 0,
            args.qwen_seconds_per_call,
        )
        LOGGER.info("qwen_call_plan %s", json.dumps(qwen_plan))
    visual_compute_device = device if args.visual_backend in {"qwen25", "qwen3vl8b", "smolvlm"} else "cpu"
    followup_budget = args.event_followups if qwen_event_backend else 0
    defer_word_budget = qwen_event_backend and generate_cue is not None
    with profiler.stage(
        "visual_backend_initialization",
        method=args.visual_backend,
        compute_device=visual_compute_device,
        model=args.vlm_model,
    ):
        backend = choose_backend(
            args.visual_backend, args.vlm_model, device, smolvlm_max_new_tokens=args.smolvlm_max_new_tokens,
            vlm_mmproj=args.vlm_mmproj, vlm_executable=args.vlm_executable,
        )
    with profiler.stage(
        "ad_decisions",
        method=args.visual_backend,
        compute_device=visual_compute_device,
        evidence_max_edge=args.evidence_max_edge or "source",
        evidence_frames_per_window=args.evidence_frames,
        ad_plan=args.ad_plan,
        event_followup_budget=followup_budget if args.ad_plan == "multi" else 0,
        evidence_sampling="temporal_windows" if backend.uses_event_windows else "adaptive_visual_change_and_boundaries",
    ) as stage:
        if qwen_event_backend:
            stage["word_budget_stage"] = "ad_preparation" if defer_word_budget else "visual_review"
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
            followup_budget=followup_budget,
            defer_word_budget=defer_word_budget,
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
    with profiler.stage(
        "ad_preparation", compute_device="cpu",
        method="qwen3_llama_cpp" if generate_cue else "incomplete_description_validation",
    ) as stage:
        def build_cue_input(context: dict) -> dict:
            return build_qwen_cue_input(context, reasoning_tokens=args.ad_preparer_reasoning_tokens)

        prepared = prepare_ad_cues(
            decisions, shots, words, generate_cue, scenes=scenes,
            build_input=build_cue_input, narration_seconds=narration_seconds_after_anchor,
        )
        write_json(work_dir / "ad-preparation.json", prepared)
        counts = {status: sum(row["status"] == status for row in prepared)
                  for status in ("prepared", "fallback", "wait")}
        counts["contextual_identifications"] = sum(bool(row["dialogue_reference"]) for row in prepared)
        stage.update(reasoning_tokens=args.ad_preparer_reasoning_tokens, **counts)
        LOGGER.info("ad_preparation model=%s results=%s", args.ad_preparer_model, counts)
    prepared_by_index = {row["decision_index"]: row for row in prepared}

    def rewrite_for_time(index: int, text: str, seconds: float, measured: float) -> str | None:
        if generate_cue is None or index not in prepared_by_index:
            return None
        return retry_prepared_cue(prepared_by_index[index], text, seconds, measured, generate_cue, build_cue_input)

    tts_compute_device = (
        device
        if args.tts_backend == "transformers" or (args.tts_backend == "auto" and args.tts_model)
        else "cpu"
    )
    tts_service = {"synthesis_service": "tts"} if args.tts_backend == "http" else {}
    with profiler.stage(
        "tts",
        requested_method=args.tts_backend,
        compute_device=tts_compute_device,
        **tts_service,
    ) as stage:
        tts_backend, decisions = synthesize_decisions(
            decisions,
            work_dir / "narration",
            args.tts_backend,
            args.piper_model,
            args.tts_model,
            device,
            ffprobe,
            tts_url=args.tts_url, rewrite_for_time=rewrite_for_time if generate_cue else None,
        )
        stage.update(
            method=tts_backend,
            compute_device=device if tts_backend == "transformers" else "cpu",
            generated_cues=sum(bool(item.tts_path) for item in decisions),
            event_aligned_cues=sum(
                item.placement_method.startswith("event_") for item in decisions if item.tts_path
            ),
            cue_rewrite_attempts=sum("timing_retry" in row for row in prepared),
            cue_rewrite_elapsed_seconds=round(sum(
                row.get("timing_retry", {}).get("elapsed_seconds", 0) for row in prepared
            ), 6),
        )

    missing_scenes = uncovered_scenes(scenes, decisions) if args.scene_fallback else []
    if qwen_plan is not None:
        qwen_plan["max_scene_fallback_calls"] = 2 * len(missing_scenes)
    if missing_scenes:
        fallback_windows = scene_fallback_windows(missing_scenes)
        with profiler.stage(
            "scene_fallback_visual", method=args.visual_backend, compute_device=visual_compute_device,
        ) as stage:
            fallback = make_decisions(
                input_video, fallback_windows, shots, scenes, segments, backend,
                work_dir / "evidence-frames", len(fallback_windows),
                args.evidence_max_edge or None, args.evidence_frames, soft_candidate_timestamps, knowledge_store,
                defer_word_budget=defer_word_budget, gap_index_offset=len(gaps), scene_fallback=True,
            )
            stage["uncovered_scenes"] = len(missing_scenes)
        for scene, window, item in zip(missing_scenes, fallback_windows, fallback, strict=True):
            item.fallback_scene_id = scene.scene_id
            item.gap_end = min(scene.end, window.end + 4.0)
            gap_audit.append({
                "kind": "scene_fallback", "scene_id": scene.scene_id, "gap_index": item.gap_index,
                "start": item.gap_start, "end": item.gap_end, "status": "eligible",
            })
        offset = len(decisions)
        with profiler.stage("scene_fallback_preparation", method="ad_preparation", compute_device="cpu"):
            extra_prepared = prepare_ad_cues(
                fallback, shots, words, generate_cue, scenes=scenes, build_input=build_cue_input,
                narration_seconds=narration_seconds_after_anchor, decision_index_offset=offset,
            )
            prepared.extend(extra_prepared)
            prepared_by_index.update((row["decision_index"], row) for row in extra_prepared)
        with profiler.stage("scene_fallback_tts", method=tts_backend, compute_device=tts_compute_device,
                            **tts_service) as stage:
            synthesize_decisions(
                fallback, work_dir / "narration", args.tts_backend, args.piper_model, args.tts_model, device, ffprobe,
                tts_url=args.tts_url, rewrite_for_time=rewrite_for_time if generate_cue else None,
                decision_index_offset=offset,
                fallback_speech=speech, existing_cues=decisions,
            )
            stage.update(
                generated_cues=sum(bool(item.tts_path) for item in fallback),
                dialogue_overlap_cues=sum(item.overlaps_dialogue for item in fallback),
                compute_device=device if tts_backend == "transformers" else "cpu",
            )
        decisions.extend(fallback)

    with profiler.stage("visual_proposal_write", method="json", compute_device="cpu"):
        proposals = record_ad_outcomes(gap_audit, decisions, multi_event=args.ad_plan == "multi")
        write_json(work_dir / "ad-proposals.json", proposals)

    with profiler.stage(
        "knowledge_finalize",
        method="sqlite_decision_upsert",
        compute_device="cpu",
    ):
        for index, decision in enumerate(decisions, start=1):
            knowledge_store.record_decision(index, decision)
        knowledge_store.close()

    with profiler.stage("artifact_write", method="json_and_webvtt", compute_device="cpu"):
        write_json(work_dir / "ad-preparation.json", prepared)
        write_vtt(decisions, work_dir / "descriptions.vtt")
        write_dialogue_artifacts(work_dir, dialogue)
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
            "scene_fallback": args.scene_fallback,
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

