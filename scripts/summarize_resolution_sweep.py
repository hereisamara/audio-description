#!/usr/bin/env python3
"""Summarize timing and output stability across VLM evidence resolutions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_by_name(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return next((stage for stage in summary.get("stages", []) if stage.get("stage") == name), {})


def normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = normalized_tokens(candidate)
    reference_tokens = normalized_tokens(reference)
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_run(run_dir: Path) -> dict[str, Any]:
    work_dir = run_dir / "work"
    summary = read_json(work_dir / "runtime-summary.json")
    media = read_json(work_dir / "media.json")
    decisions = read_json(work_dir / "ad-decisions.json")
    visual_init = stage_by_name(summary, "visual_backend_initialization")
    visual_inference = stage_by_name(summary, "ad_decisions")
    tts = stage_by_name(summary, "tts")
    match = re.fullmatch(r"edge-(\d+)", run_dir.name)
    if not match:
        raise ValueError(f"Unexpected resolution run directory: {run_dir}")
    edge = int(match.group(1))
    descriptions = [item.get("description", "") for item in decisions if item.get("decision") == "AD"]
    decision_sequence = [item.get("decision", "WAIT") for item in decisions]
    return {
        "evidence_max_edge": edge,
        "source_resolution": f'{media.get("width", 0)}x{media.get("height", 0)}',
        "total_seconds": float(summary.get("total_elapsed_seconds", 0.0)),
        "vlm_initialization_seconds": float(visual_init.get("elapsed_seconds", 0.0)),
        "vlm_decision_seconds": float(visual_inference.get("elapsed_seconds", 0.0)),
        "vlm_gpu_peak_mb": visual_inference.get("gpu_memory_peak_allocated_mb"),
        "qwen_vision_tokens": int(visual_inference.get("qwen_vision_tokens", 0)),
        "tts_seconds": float(tts.get("elapsed_seconds", 0.0)),
        "decision_sequence": decision_sequence,
        "ad_descriptions": descriptions,
        "description_text": " | ".join(descriptions),
        "manual_quality_score_1_to_5": "",
        "run_directory": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    args = parser.parse_args()
    root = args.sweep_root.resolve()
    runs = [load_run(path) for path in sorted(root.glob("edge-*")) if (path / "work/runtime-summary.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No completed edge-*/work/runtime-summary.json runs found under {root}")
    runs.sort(key=lambda item: item["evidence_max_edge"])
    baseline = runs[-1]
    baseline_time = baseline["vlm_decision_seconds"]
    baseline_text = baseline["description_text"]
    baseline_decisions = baseline["decision_sequence"]
    for run in runs:
        current_time = run["vlm_decision_seconds"]
        run["vlm_speedup_vs_largest"] = round(baseline_time / current_time, 4) if current_time else None
        run["decision_match_vs_largest"] = run["decision_sequence"] == baseline_decisions
        run["description_token_f1_vs_largest"] = round(token_f1(run["description_text"], baseline_text), 4)

    json_path = root / "resolution-comparison.json"
    csv_path = root / "resolution-comparison.csv"
    json_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = [
        "evidence_max_edge",
        "source_resolution",
        "total_seconds",
        "vlm_initialization_seconds",
        "vlm_decision_seconds",
        "vlm_speedup_vs_largest",
        "vlm_gpu_peak_mb",
        "qwen_vision_tokens",
        "tts_seconds",
        "decision_match_vs_largest",
        "description_token_f1_vs_largest",
        "decision_sequence",
        "description_text",
        "manual_quality_score_1_to_5",
        "run_directory",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    print(
        f'{"edge":>6} {"VLM sec":>9} {"speedup":>8} {"GPU MB":>10} '
        f'{"tokens":>9} {"match":>7} {"text F1":>8}  description'
    )
    for run in runs:
        gpu = run["vlm_gpu_peak_mb"]
        gpu_text = f"{float(gpu):.1f}" if gpu is not None else "n/a"
        print(
            f'{run["evidence_max_edge"]:6d} {run["vlm_decision_seconds"]:9.3f} '
            f'{run["vlm_speedup_vs_largest"]:8.3f} {gpu_text:>10} '
            f'{run["qwen_vision_tokens"]:9d} {str(run["decision_match_vs_largest"]):>7} '
            f'{run["description_token_f1_vs_largest"]:8.3f}  {run["description_text"]}'
        )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("The text-F1 column measures wording stability against the largest resolution, not ground-truth quality.")


if __name__ == "__main__":
    main()
