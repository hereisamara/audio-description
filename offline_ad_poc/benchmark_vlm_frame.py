from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2

from .pipeline import Gap, Qwen25Backend, frame_at, resize_frame_max_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one fixed video frame with Qwen2.5-VL at multiple resolutions."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vlm-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp", type=float, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[160, 320, 480, 640])
    parser.add_argument("--gap-start", type=float, default=25.75)
    parser.add_argument("--gap-end", type=float, default=38.75)
    parser.add_argument("--word-budget", type=int, default=32)
    parser.add_argument("--recent-dialogue", default="none")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def extract_source_frame(video_path: Path, timestamp: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    try:
        frame = frame_at(capture, timestamp)
    finally:
        capture.release()
    if frame is None:
        raise RuntimeError(f"Could not decode a frame at {timestamp:.3f} seconds")
    return frame


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    json_path = output_dir / "vlm-frame-resolution-results.json"
    csv_path = output_dir / "vlm-frame-resolution-results.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "requested_max_edge",
        "actual_width",
        "actual_height",
        "elapsed_seconds",
        "minimum_elapsed_seconds",
        "maximum_elapsed_seconds",
        "qwen_vision_tokens",
        "gpu_allocated_start_mb",
        "gpu_peak_allocated_mb",
        "gpu_incremental_peak_mb",
        "decision",
        "description",
        "reason",
        "importance",
        "valid_output",
        "outputs_consistent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])


def main() -> None:
    args = parse_args()
    if args.timestamp < 0:
        raise ValueError("--timestamp must be non-negative")
    if args.gap_end <= args.gap_start:
        raise ValueError("--gap-end must be greater than --gap-start")
    if any(value <= 0 for value in args.resolutions):
        raise ValueError("--resolutions must contain only positive integers")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This Lanta benchmark requires an allocated CUDA GPU")

    input_video = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_frame = extract_source_frame(input_video, args.timestamp)
    source_height, source_width = source_frame.shape[:2]

    torch.cuda.synchronize()
    initialization_started = time.perf_counter()
    backend = Qwen25Backend(args.vlm_model, "cuda")
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_started

    gap = Gap(args.gap_start, args.gap_end, args.word_budget)
    dialogue = "" if args.recent_dialogue.lower() == "none" else args.recent_dialogue
    prepared_frames: dict[int, dict[str, Any]] = {}
    for max_edge in args.resolutions:
        run_dir = output_dir / f"edge-{max_edge}"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"frame-{args.timestamp:.2f}.jpg"
        frame = resize_frame_max_edge(source_frame, max_edge)
        actual_height, actual_width = frame.shape[:2]
        if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Could not write {image_path}")
        prepared_frames[max_edge] = {
            "path": image_path,
            "width": actual_width,
            "height": actual_height,
        }

    runs: list[dict[str, Any]] = []
    for max_edge in args.resolutions:
        prepared = prepared_frames[max_edge]
        image_path = prepared["path"]
        for warmup_index in range(args.warmup_runs):
            print(
                f"Warm-up {warmup_index + 1}/{args.warmup_runs} at max edge {max_edge}; excluded from results",
                flush=True,
            )
            backend.decide([image_path], gap, dialogue, [], {})
            torch.cuda.synchronize()
        samples: list[dict[str, Any]] = []
        for repeat_index in range(args.repeats):
            vision_tokens_before = backend.vision_tokens
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            allocated_start = torch.cuda.memory_allocated() / (1024**2)
            started = time.perf_counter()
            payload = backend.decide([image_path], gap, dialogue, [], {})
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            sample = {
                "repeat": repeat_index + 1,
                "elapsed_seconds": round(elapsed, 6),
                "qwen_vision_tokens": backend.vision_tokens - vision_tokens_before,
                "gpu_allocated_start_mb": round(allocated_start, 3),
                "gpu_peak_allocated_mb": round(peak_allocated, 3),
                "gpu_incremental_peak_mb": round(max(0.0, peak_allocated - allocated_start), 3),
                "payload": payload,
            }
            samples.append(sample)
            print(
                json.dumps(
                    {"requested_max_edge": max_edge, **sample},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        elapsed_values = [sample["elapsed_seconds"] for sample in samples]
        payloads = [sample["payload"] for sample in samples]
        payload = payloads[0]
        description = str(payload.get("description", "")).strip()
        decision = str(payload.get("decision", "WAIT")).upper()
        row = {
            "requested_max_edge": max_edge,
            "actual_width": prepared["width"],
            "actual_height": prepared["height"],
            "elapsed_seconds": round(statistics.median(elapsed_values), 6),
            "minimum_elapsed_seconds": min(elapsed_values),
            "maximum_elapsed_seconds": max(elapsed_values),
            "qwen_vision_tokens": samples[0]["qwen_vision_tokens"],
            "qwen_image_grid_thw": backend.processed_image_grids[-1],
            "gpu_allocated_start_mb": samples[0]["gpu_allocated_start_mb"],
            "gpu_peak_allocated_mb": max(sample["gpu_peak_allocated_mb"] for sample in samples),
            "gpu_incremental_peak_mb": max(
                sample["gpu_incremental_peak_mb"] for sample in samples
            ),
            "decision": decision,
            "description": description,
            "reason": str(payload.get("reason", "")),
            "importance": float(payload.get("importance", 0.0)),
            "valid_output": decision == "WAIT" or bool(description),
            "outputs_consistent": all(item == payload for item in payloads),
            "image": str(image_path),
            "samples": samples,
        }
        runs.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "samples"}, ensure_ascii=False), flush=True)

    report = {
        "input": str(input_video),
        "timestamp": args.timestamp,
        "source_width": source_width,
        "source_height": source_height,
        "model": args.vlm_model,
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(),
        "model_initialization_seconds": round(initialization_seconds, 6),
        "frames_per_inference": 1,
        "runs_are_sequential": True,
        "warmup_runs_excluded_per_resolution": args.warmup_runs,
        "repeats_per_resolution": args.repeats,
        "runs": runs,
    }
    write_reports(output_dir, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
