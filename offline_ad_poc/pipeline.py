#!/usr/bin/env python3
"""Offline, CPU/GPU-portable audio-description prototype.

This is a research prototype.  The heuristic backend validates the full media
pipeline but does not claim useful visual-language quality.  The qwen25 backend
uses only a local model directory and performs no API calls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


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
class Decision:
    gap_start: float
    gap_end: float
    decision: str
    description: str
    reason: str
    importance: float
    word_budget: int
    evidence_frames: list[str]
    tts_path: str = ""
    tts_duration: float = 0.0
    cue_start: float = 0.0
    cue_end: float = 0.0


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
        return "cpu"
    cuda_available = False
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            cuda_available = True
    except Exception:
        pass
    if not cuda_available:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            pass
    if requested == "cuda" and not cuda_available:
        return "cpu"
    return "cuda" if cuda_available else "cpu"


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
            return _transcribe_faster_whisper(
                audio_path,
                model_name_or_path,
                device,
                allow_model_download,
            )
        except (ImportError, RuntimeError) as exc:
            if backend == "faster-whisper" or "Install faster-whisper" not in str(exc):
                raise
    return _transcribe_transformers_whisper(
        audio_path,
        model_name_or_path,
        device,
        allow_model_download,
    )


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


def detect_shots(video_path: Path, duration: float, threshold: float = 27.0) -> list[Shot]:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        ContentDetector = None
        detect = None

    if ContentDetector is not None and detect is not None:
        scene_list = detect(str(video_path), ContentDetector(threshold=threshold), show_progress=False)
        if not scene_list:
            return [Shot(1, 0.0, duration)]
        shots = [
            Shot(index, float(start.get_seconds()), min(duration, float(end.get_seconds())))
            for index, (start, end) in enumerate(scene_list, start=1)
        ]
        if shots and shots[-1].end < duration:
            shots[-1].end = duration
        return shots

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
    return [
        Shot(index, start, end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1)
        if end - start > 0.05
    ] or [Shot(1, 0.0, duration)]


def frame_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def normalized_histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return hist / norm if norm else hist


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


def sample_window_frames(
    video_path: Path,
    start: float,
    end: float,
    output_dir: Path,
    count: int = 4,
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
            path = output_dir / f"frame-{index:02d}-{timestamp:.2f}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths.append(path)
    finally:
        capture.release()
    return paths


def json_object_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object")
    return json.loads(match.group(0))


class VisualBackend:
    def decide(
        self,
        frames: Sequence[Path],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicBackend(VisualBackend):
    """A deterministic media-pipeline test, not a semantic AD model."""

    def decide(
        self,
        frames: Sequence[Path],
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
        return {
            "decision": "AD",
            "description": description,
            "reason": reason,
            "importance": round(importance, 3),
        }


def expand_qwen_image_tokens(
    chat_text: str,
    image_token: str,
    image_grid_thw: Sequence[Sequence[int]],
    merge_size: int,
) -> str:
    expanded = chat_text
    for grid in image_grid_thw:
        token_count = math.prod(int(value) for value in grid) // (merge_size**2)
        if image_token not in expanded:
            raise ValueError("Qwen chat template did not emit enough image placeholder tokens")
        expanded = expanded.replace(image_token, image_token * token_count, 1)
    return expanded


class Qwen25Backend(VisualBackend):
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

    def decide(
        self,
        frames: Sequence[Path],
        gap: Gap,
        transcript: str,
        previous_ad: Sequence[str],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are deciding whether an audio description is needed in a speech-free film interval. "
            "Use only visible evidence from the ordered frames. Return WAIT for static, redundant, "
            "incomplete, uncertain, or unimportant content. Return AD only for meaningful new action, "
            "location change, character behaviour, or an important visual detail not conveyed by audio. "
            f"The safe gap is {gap.duration:.2f} seconds and the maximum is {gap.word_budget} words. "
            f"Recent dialogue: {transcript or 'none'}. "
            f"Previous AD: {' | '.join(previous_ad[-3:]) or 'none'}. "
            "Never mention the camera or invent names, emotions, causes, or off-screen facts. "
            "Return exactly one JSON object with keys decision, description, reason, importance. "
            "decision must be AD or WAIT. description must be empty for WAIT. importance is 0 to 1."
        )
        content = [{"type": "image", "image": str(path.resolve())} for path in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs = []
        for path in frames:
            with self.image_type.open(path) as image:
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
        image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        chat_text = expand_qwen_image_tokens(
            chat_text,
            image_token,
            image_grid_thw.tolist(),
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
        decoded = self.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return json_object_from_text(decoded)


def context_for_gap(gap: Gap, shots: Sequence[Shot], scenes: Sequence[Scene]) -> dict[str, Any]:
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
        return HeuristicBackend()
    if name == "qwen25":
        if not model_path:
            raise ValueError("--vlm-model is required for --visual-backend qwen25")
        return Qwen25Backend(model_path, device)
    raise ValueError(f"Unknown visual backend: {name}")


def make_decisions(
    video_path: Path,
    gaps: Sequence[Gap],
    shots: Sequence[Shot],
    scenes: Sequence[Scene],
    segments: Sequence[TranscriptSegment],
    backend: VisualBackend,
    frame_root: Path,
    maximum_gaps: int,
) -> list[Decision]:
    decisions: list[Decision] = []
    previous_ad: list[str] = []
    for index, gap in enumerate(gaps[:maximum_gaps], start=1):
        context = context_for_gap(gap, shots, scenes)
        frames = sample_window_frames(
            video_path,
            float(context["window_start"]),
            float(context["window_end"]),
            frame_root / f"gap-{index:03d}",
            count=4,
        )
        try:
            payload = backend.decide(
                frames,
                gap,
                recent_transcript(gap, segments),
                previous_ad,
                context,
            )
        except Exception as exc:
            payload = {
                "decision": "WAIT",
                "description": "",
                "reason": f"visual backend failure: {type(exc).__name__}: {exc}",
                "importance": 0.0,
            }
        decision_name = str(payload.get("decision", "WAIT")).upper()
        if decision_name not in {"AD", "WAIT"}:
            decision_name = "WAIT"
        description = str(payload.get("description", "")).strip() if decision_name == "AD" else ""
        description = limit_words(description, gap.word_budget) if description else ""
        if decision_name == "AD" and not description:
            decision_name = "WAIT"
        decision = Decision(
            gap_start=gap.start,
            gap_end=gap.end,
            decision=decision_name,
            description=description,
            reason=str(payload.get("reason", "")),
            importance=float(payload.get("importance", 0.0)),
            word_budget=gap.word_budget,
            evidence_frames=[str(path) for path in frames],
        )
        decisions.append(decision)
        if decision.decision == "AD":
            previous_ad.append(decision.description)
    return decisions


def resolve_tts_backend(
    requested: str,
    piper_model: str | None,
    transformers_model: str | None,
) -> tuple[str, str]:
    candidates = [requested] if requested != "auto" else ["transformers", "piper", "espeak-ng", "say"]
    for candidate in candidates:
        if candidate == "transformers" and transformers_model:
            return candidate, transformers_model
        executable = shutil.which(candidate)
        if candidate == "piper" and executable and piper_model:
            return candidate, executable
        if candidate in {"espeak-ng", "say"} and executable:
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
    for index, decision in enumerate(decisions, start=1):
        if decision.decision != "AD":
            continue
        text = decision.description
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
            if duration <= available:
                decision.description = text
                decision.tts_path = str(audio_path)
                decision.tts_duration = duration
                decision.cue_start = decision.gap_start + (decision.gap_end - decision.gap_start - duration) / 2.0
                decision.cue_end = decision.cue_start + duration
                break
            words = text.split()
            target = max(2, math.floor(len(words) * available / max(duration, 0.01) * 0.88))
            text = limit_words(text, min(target, len(words) - 1))
        if not decision.tts_path:
            decision.decision = "WAIT"
            decision.reason = "TTS could not fit the safe gap after shortening"
            decision.description = ""
    return backend, decisions


def vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def write_vtt(decisions: Sequence[Decision], output: Path) -> None:
    lines = ["WEBVTT", ""]
    for decision in decisions:
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
) -> None:
    active = [decision for decision in decisions if decision.decision == "AD" and decision.tts_path]
    if not active:
        raise RuntimeError("No accepted AD events to mix")
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
    return {
        "duration_seconds": round(media.duration, 3),
        "shots": len(shots),
        "scenes": len(scenes),
        "candidate_gaps": len(gaps),
        "processed_gaps": len(decisions),
        "accepted_ad_events": len(active),
        "wait_events": sum(item.decision == "WAIT" for item in decisions),
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
    parser.add_argument("--shot-threshold", type=float, default=27.0)
    parser.add_argument("--scene-threshold", type=float, default=0.52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    input_video = args.input.resolve()
    output_video = args.output_video.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not input_video.exists():
        raise FileNotFoundError(input_video)

    ffmpeg, ffprobe = resolve_ffmpeg_tools()
    media = probe_media(input_video, ffprobe)
    if not media.has_audio:
        raise ValueError("Prototype currently requires an input audio stream")
    device = choose_device(args.device)
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("CUDA was requested but is not available")

    analysis_wav = work_dir / "analysis-audio.wav"
    extract_analysis_audio(input_video, analysis_wav, ffmpeg)
    segments, words, asr_report = transcribe_audio(
        analysis_wav,
        args.asr_model,
        device,
        args.allow_model_download,
        args.asr_backend,
    )
    speech = speech_intervals_from_words(words, media.duration)
    gaps = build_gaps(speech, media.duration, minimum_duration=args.minimum_gap)

    shots = detect_shots(input_video, media.duration, threshold=args.shot_threshold)
    histograms = sample_shot_features(input_video, shots, work_dir / "keyframes")
    wav_features_by_shot(analysis_wav, shots)
    attach_transcripts(shots, segments)
    scenes = segment_scenes(shots, histograms, boundary_threshold=args.scene_threshold)

    backend = choose_backend(args.visual_backend, args.vlm_model, device)
    decisions = make_decisions(
        input_video,
        gaps,
        shots,
        scenes,
        segments,
        backend,
        work_dir / "evidence-frames",
        args.maximum_gaps,
    )
    tts_backend, decisions = synthesize_decisions(
        decisions,
        work_dir / "narration",
        args.tts_backend,
        args.piper_model,
        args.tts_model,
        device,
        ffprobe,
    )

    write_vtt(decisions, work_dir / "descriptions.vtt")
    write_json(work_dir / "media.json", asdict(media))
    write_json(work_dir / "transcript.json", {"report": asr_report, "segments": [asdict(x) for x in segments], "words": [asdict(x) for x in words]})
    write_json(work_dir / "speech-and-gaps.json", {"speech": [asdict(x) for x in speech], "gaps": [asdict(x) for x in gaps]})
    write_json(work_dir / "shots.json", [asdict(x) for x in shots])
    write_json(work_dir / "scenes.json", [asdict(x) for x in scenes])
    write_json(work_dir / "ad-decisions.json", [asdict(x) for x in decisions])

    mix_described_video(input_video, output_video, decisions, ffmpeg)
    validate_decode(output_video, ffmpeg)
    elapsed = time.perf_counter() - started
    report = {
        "input": str(input_video),
        "output": str(output_video),
        "device": device,
        "visual_backend": args.visual_backend,
        "tts_backend": tts_backend,
        "offline": not args.allow_model_download,
        "metrics": evaluate(media, speech, shots, scenes, gaps, decisions, elapsed),
    }
    write_json(work_dir / "evaluation-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

