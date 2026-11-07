#!/usr/bin/env python3
"""Prepare and document a real-video scene for offline AD experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .pipeline import probe_media, resolve_ffmpeg_tools, run_command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Lawfully obtained source video")
    parser.add_argument("--output", type=Path, required=True, help="Normalized scene MP4")
    parser.add_argument("--start", required=True, help="FFmpeg start time, for example 00:05:30")
    parser.add_argument("--duration", type=float, required=True, help="Scene duration in seconds")
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="user-provided; verify before redistribution")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    ffmpeg, ffprobe = resolve_ffmpeg_tools()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            args.start,
            "-i",
            str(source),
            "-t",
            str(args.duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    media = probe_media(output, ffprobe)
    if not media.has_audio:
        raise RuntimeError("Prepared scene has no audio stream")
    run_command([ffmpeg, "-nostdin", "-v", "error", "-i", str(output), "-f", "null", "-"])

    manifest = {
        "title": args.title,
        "source_file": str(source),
        "source_sha256": sha256(source),
        "source_url": args.source_url,
        "license": args.license,
        "clip_start": args.start,
        "requested_duration_seconds": args.duration,
        "prepared_file": str(output),
        "prepared_sha256": sha256(output),
        "media": {
            "duration": media.duration,
            "width": media.width,
            "height": media.height,
            "fps": media.fps,
            "has_audio": media.has_audio,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

