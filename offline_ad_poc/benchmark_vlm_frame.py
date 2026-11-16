from __future__ import annotations

import argparse
import csv
import json
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
        "qwen_vision_tokens",
        "gpu_allocated_start_mb",
        "gpu_peak_allocated_mb",
        "gpu_incremental_peak_mb",
        "decision",
        "description",
        "reason",
        "importance",
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
    runs: list[dict[str, Any]] = []
    for max_edge in args.resolutions:
        run_dir = output_dir / f"edge-{max_edge}"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / f"frame-{args.timestamp:.2f}.jpg"
        frame = resize_frame_max_edge(source_frame, max_edge)
        actual_height, actual_width = frame.shape[:2]
        if not cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Could not write {image_path}")

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
        vision_tokens = backend.vision_tokens - vision_tokens_before
        row = {
            "requested_max_edge": max_edge,
            "actual_width": actual_width,
            "actual_height": actual_height,
            "elapsed_seconds": round(elapsed, 6),
            "qwen_vision_tokens": vision_tokens,
            "qwen_image_grid_thw": backend.processed_image_grids[-1],
            "gpu_allocated_start_mb": round(allocated_start, 3),
            "gpu_peak_allocated_mb": round(peak_allocated, 3),
            "gpu_incremental_peak_mb": round(max(0.0, peak_allocated - allocated_start), 3),
            "decision": str(payload.get("decision", "WAIT")),
            "description": str(payload.get("description", "")),
            "reason": str(payload.get("reason", "")),
            "importance": float(payload.get("importance", 0.0)),
            "image": str(image_path),
        }
        runs.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

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
        "runs": runs,
    }
    write_reports(output_dir, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
