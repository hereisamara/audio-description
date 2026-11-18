"""Compare local ASR models against timestamped human dialogue reference data."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .pipeline import (
    Interval,
    TranscriptSegment,
    Word,
    _segments_from_words,
    choose_device,
    extract_analysis_audio,
    merge_intervals,
    probe_media,
    speech_intervals_from_words,
    transcribe_audio,
    validate_whisper_word_timestamps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local ASR models sequentially and compare timestamp coverage and text "
            "against a human dialogue reference."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video with audio")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=LOCAL_PATH",
        help="Repeat once per local Transformers model directory",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--maximum-word-duration", type=float, default=2.5)
    parser.add_argument("--speech-padding", type=float, default=0.25)
    parser.add_argument("--merge-gap", type=float, default=0.45)
    return parser.parse_args()


def parse_model_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"Invalid --model value {value!r}; expected NAME=LOCAL_PATH")
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {path}")
        names.add(name)
        specs.append((name, path))
    return specs


def load_reference(path: Path) -> tuple[list[Interval], str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("Reference must contain a non-empty segments list")
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        start = float(item["start"])
        end = float(item["end"])
        text = " ".join(str(item.get("text", "")).split())
        if start < 0 or end <= start:
            raise ValueError(f"Invalid reference interval at item {index}: {start}–{end}")
        segments.append({"id": item.get("id", f"D{index}"), "start": start, "end": end, "text": text})
    intervals = merge_intervals(
        [Interval(item["start"], item["end"]) for item in segments],
        maximum_gap=0.001,
    )
    transcript = " ".join(item["text"] for item in segments if item["text"])
    return intervals, transcript, segments


def interval_duration(intervals: Sequence[Interval]) -> float:
    return sum(max(0.0, item.end - item.start) for item in intervals)


def interval_overlap(first: Sequence[Interval], second: Sequence[Interval]) -> float:
    left = sorted(first, key=lambda item: item.start)
    right = sorted(second, key=lambda item: item.start)
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i].end, right[j].end) - max(left[i].start, right[j].start))
        if left[i].end <= right[j].end:
            i += 1
        else:
            j += 1
    return total


def temporal_metrics(reference: Sequence[Interval], predicted: Sequence[Interval]) -> dict[str, float]:
    reference_seconds = interval_duration(reference)
    predicted_seconds = interval_duration(predicted)
    overlap_seconds = interval_overlap(reference, predicted)
    precision = overlap_seconds / predicted_seconds if predicted_seconds else 0.0
    recall = overlap_seconds / reference_seconds if reference_seconds else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_reference_start = min(item.start for item in reference)
    intro = [Interval(0.0, first_reference_start)]
    return {
        "reference_dialogue_seconds": round(reference_seconds, 6),
        "predicted_speech_seconds": round(predicted_seconds, 6),
        "overlap_seconds": round(overlap_seconds, 6),
        "false_positive_seconds": round(max(0.0, predicted_seconds - overlap_seconds), 6),
        "missed_dialogue_seconds": round(max(0.0, reference_seconds - overlap_seconds), 6),
        "false_positive_intro_seconds": round(interval_overlap(predicted, intro), 6),
        "temporal_precision": round(precision, 6),
        "temporal_recall": round(recall, 6),
        "temporal_f1": round(f1, 6),
    }


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_tokens = normalized_tokens(reference)
    hypothesis_tokens = normalized_tokens(hypothesis)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else math.inf
    return edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def transcribe_ctc(
    audio_path: Path,
    model_path: Path,
    device: str,
) -> tuple[list[TranscriptSegment], list[Word], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError("CTC ASR requires the Conda pytorch and transformers packages") from exc

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(
        model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=20,
        stride_length_s=(4, 2),
        torch_dtype=dtype,
        device=0 if device == "cuda" else -1,
    )
    result = asr(str(audio_path), return_timestamps="word")
    words: list[Word] = []
    for item in result.get("chunks", []):
        timestamp = item.get("timestamp") or (None, None)
        start, end = timestamp
        token = " ".join(str(item.get("text", "")).strip().split())
        if token and start is not None and end is not None:
            words.append(Word(float(start), float(end), token))
    if not words:
        raise RuntimeError("CTC model returned no word timestamps")
    segments = _segments_from_words(words)
    return segments, words, {
        "backend": "transformers-ctc",
        "model": str(model_path),
        "device": device,
        "compute_type": str(dtype).replace("torch.", ""),
        "segments": len(segments),
        "words": len(words),
    }


def model_type(model_path: Path) -> str:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return str(config.model_type)


def transcribe_model(
    audio_path: Path,
    model_path: Path,
    device: str,
) -> tuple[list[TranscriptSegment], list[Word], dict[str, Any]]:
    architecture = model_type(model_path)
    if architecture == "whisper":
        segments, words, report = transcribe_audio(
            audio_path,
            str(model_path),
            device,
            allow_model_download=False,
            backend="transformers",
        )
    else:
        segments, words, report = transcribe_ctc(audio_path, model_path, device)
    report["model_type"] = architecture
    return segments, words, report


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "name",
        "status",
        "model_type",
        "backend",
        "elapsed_seconds",
        "real_time_factor",
        "gpu_peak_allocated_mb",
        "raw_words",
        "validated_words",
        "rejected_words",
        "word_error_rate",
        "temporal_precision",
        "temporal_recall",
        "temporal_f1",
        "false_positive_seconds",
        "missed_dialogue_seconds",
        "false_positive_intro_seconds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.maximum_word_duration <= 0:
        raise ValueError("--maximum-word-duration must be positive")
    if args.speech_padding < 0 or args.merge_gap < 0:
        raise ValueError("--speech-padding and --merge-gap must be non-negative")

    input_video = args.input.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    models = parse_model_specs(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available in the Conda environment")
    media = probe_media(input_video, ffprobe)
    audio_path = output_dir / "analysis-audio.wav"
    extract_analysis_audio(input_video, audio_path, ffmpeg)
    reference_intervals, reference_text, reference_segments = load_reference(reference_path)
    device = choose_device(args.device)

    import torch

    results: list[dict[str, Any]] = []
    for index, (name, path) in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] ASR model={name} path={path} device={device}", flush=True)
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            segments, raw_words, asr_report = transcribe_model(audio_path, path, device)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            validated_words, validation = validate_whisper_word_timestamps(
                raw_words,
                maximum_duration=args.maximum_word_duration,
            )
            validated_segments = _segments_from_words(validated_words)
            speech = speech_intervals_from_words(
                validated_words,
                duration=media.duration,
                padding=args.speech_padding,
                merge_gap=args.merge_gap,
            )
            metrics = temporal_metrics(reference_intervals, speech)
            raw_text = " ".join(item.text for item in raw_words)
            validated_text = " ".join(item.text for item in validated_words)
            result: dict[str, Any] = {
                "name": name,
                "status": "completed",
                "model_path": str(path),
                "model_type": asr_report.get("model_type"),
                "backend": asr_report.get("backend"),
                "device": device,
                "elapsed_seconds": round(elapsed, 6),
                "real_time_factor": round(elapsed / media.duration, 6),
                "gpu_peak_allocated_mb": (
                    round(torch.cuda.max_memory_allocated() / (1024**2), 3)
                    if device == "cuda"
                    else None
                ),
                "raw_words": len(raw_words),
                "validated_words": len(validated_words),
                "rejected_words": validation["rejected_words"],
                "raw_word_error_rate": round(word_error_rate(reference_text, raw_text), 6),
                "word_error_rate": round(word_error_rate(reference_text, validated_text), 6),
                **metrics,
            }
            transcript = {
                "report": asr_report,
                "validation": validation,
                "raw_segments": [asdict(item) for item in segments],
                "raw_words": [asdict(item) for item in raw_words],
                "segments": [asdict(item) for item in validated_segments],
                "words": [asdict(item) for item in validated_words],
                "speech": [asdict(item) for item in speech],
                "metrics": metrics,
            }
            (model_dir / "transcript.json").write_text(
                json.dumps(transcript, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result = {
                "name": name,
                "status": "failed",
                "model_path": str(path),
                "device": device,
                "elapsed_seconds": round(elapsed, 6),
                "real_time_factor": round(elapsed / media.duration, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        (model_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    completed = [item for item in results if item["status"] == "completed"]
    ranking = sorted(
        completed,
        key=lambda item: (-item["temporal_f1"], item["word_error_rate"], item["elapsed_seconds"]),
    )
    report = {
        "input": str(input_video),
        "duration_seconds": media.duration,
        "reference": str(reference_path),
        "reference_segments": reference_segments,
        "reference_dialogue_seconds": round(interval_duration(reference_intervals), 6),
        "device": device,
        "models_run_sequentially": True,
        "results": results,
        "ranking": [item["name"] for item in ranking],
    }
    (output_dir / "asr-comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "asr-comparison.csv", results)
    print(f"Finished ASR comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
