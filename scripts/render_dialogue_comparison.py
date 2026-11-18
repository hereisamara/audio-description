#!/usr/bin/env python3
"""Render ground-truth and validated Whisper dialogue over the source video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline_ad_poc.pipeline import Word, _segments_from_words, validate_whisper_word_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--whisper-transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--maximum-word-duration", type=float, default=2.5)
    return parser.parse_args()


def run_json(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe(path: Path, ffprobe: str) -> dict[str, Any]:
    return run_json(
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


def load_reference(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", payload)
    return [
        {
            "id": str(item.get("id", f"D{index}")),
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": " ".join(str(item["text"]).split()),
        }
        for index, item in enumerate(segments, start=1)
    ]


def load_predicted(path: Path, maximum_word_duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload.get("raw_words") or payload.get("words") or []
    raw_words = [Word(float(item["start"]), float(item["end"]), str(item["text"])) for item in raw_items]
    words, report = validate_whisper_word_timestamps(
        raw_words,
        maximum_duration=maximum_word_duration,
    )
    segments = _segments_from_words(words)
    return [
        {
            "id": f"W{index}",
            "start": item.start,
            "end": item.end,
            "text": item.text,
        }
        for index, item in enumerate(segments, start=1)
    ], report


def active_segment(segments: Sequence[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    return next(
        (item for item in segments if item["start"] <= timestamp < item["end"]),
        None,
    )


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    bounds = draw.textbbox((0, 0), text, font=font)
    return float(bounds[2] - bounds[0])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    maximum_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return ["—"]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= maximum_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    maximum_width: int,
    maximum_lines: int,
) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    for size in range(20, 11, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        lines = wrap_text(draw, text, font, maximum_width)
        if len(lines) <= maximum_lines:
            return font, lines
    font = ImageFont.truetype(str(font_path), size=11)
    return font, wrap_text(draw, text, font, maximum_width)


def timecode(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_value, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{milliseconds:03d}"


def write_srt(path: Path, segments: Sequence[dict[str, Any]]) -> None:
    lines: list[str] = []
    for index, item in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{timecode(item['start'])} --> {timecode(item['end'])}",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    bottom: int,
    label: str,
    segment: dict[str, Any] | None,
    label_font: ImageFont.FreeTypeFont,
    font_path: Path,
    accent: tuple[int, int, int],
) -> None:
    width, _ = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (12, top, width - 12, bottom),
        radius=10,
        fill=(8, 10, 14, 205),
    )
    overlay_draw.rounded_rectangle(
        (22, top + 9, 190, top + 35),
        radius=6,
        fill=(*accent, 235),
    )
    image.alpha_composite(overlay)
    draw.text((32, top + 13), label, font=label_font, fill=(255, 255, 255, 255))
    if segment:
        id_time = f"{segment['id']}  {segment['start']:.3f}–{segment['end']:.3f}s"
        draw.text((205, top + 14), id_time, font=label_font, fill=(*accent, 255))
        text = segment["text"]
    else:
        text = "—"
    font, lines = fit_text(draw, text, font_path, width - 56, maximum_lines=3)
    line_height = max(16, font.size + 4)
    y = top + 43
    for line in lines:
        draw.text(
            (28, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_height


def draw_frame(
    frame,
    timestamp: float,
    original: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    font_path: Path,
    bold_font_path: Path,
):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.truetype(str(bold_font_path), size=14)
    clock_font = ImageFont.truetype(str(bold_font_path), size=15)
    width, height = image.size
    panel_height = min(130, max(112, round(height * 0.27)))
    draw_panel(
        image,
        draw,
        top=10,
        bottom=10 + panel_height,
        label="ORIGINAL DIALOGUE",
        segment=original,
        label_font=label_font,
        font_path=font_path,
        accent=(44, 170, 104),
    )
    draw_panel(
        image,
        draw,
        top=height - panel_height - 10,
        bottom=height - 10,
        label="WHISPER VALIDATED",
        segment=predicted,
        label_font=label_font,
        font_path=font_path,
        accent=(65, 132, 224),
    )
    if original and predicted:
        status = "BOTH ACTIVE"
        status_color = (44, 170, 104)
    elif original:
        status = "MISSED BY WHISPER"
        status_color = (225, 126, 45)
    elif predicted:
        status = "WHISPER ONLY"
        status_color = (208, 66, 73)
    else:
        status = "NO DIALOGUE"
        status_color = (90, 96, 108)
    status_box = draw.textbbox((0, 0), status, font=label_font)
    status_width = status_box[2] - status_box[0]
    status_y = height // 2 - 15
    draw.rounded_rectangle(
        (width // 2 - status_width // 2 - 12, status_y, width // 2 + status_width // 2 + 12, status_y + 28),
        radius=7,
        fill=(*status_color, 225),
    )
    draw.text(
        (width // 2 - status_width // 2, status_y + 6),
        status,
        font=label_font,
        fill=(255, 255, 255, 255),
    )
    clock = f"SOURCE {timestamp:07.3f}s"
    clock_box = draw.textbbox((0, 0), clock, font=clock_font)
    draw.text(
        (width - (clock_box[2] - clock_box[0]) - 22, height // 2 + 20),
        clock,
        font=clock_font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    input_video = args.input.resolve()
    reference_path = args.reference.resolve()
    transcript_path = args.whisper_transcript.resolve()
    output_path = args.output.resolve()
    font_path = args.font.resolve()
    bold_font_path = args.bold_font.resolve()
    for path in (input_video, reference_path, transcript_path, font_path, bold_font_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = load_reference(reference_path)
    predicted, validation = load_predicted(transcript_path, args.maximum_word_duration)
    write_srt(output_path.with_name("original-dialogue.srt"), original)
    write_srt(output_path.with_name("whisper-validated.srt"), predicted)
    cue_sheet = {
        "input": str(input_video),
        "reference": str(reference_path),
        "whisper_transcript": str(transcript_path),
        "validation": validation,
        "original": original,
        "predicted": predicted,
        "style": {
            "original_position": "top",
            "predicted_position": "bottom",
            "audio": "original source audio copied unchanged",
            "playback_speed": 1.0,
        },
    }
    output_path.with_name("dialogue-comparison-cues.json").write_text(
        json.dumps(cue_sheet, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata = probe(input_video, args.ffprobe)
    video_stream = next(item for item in metadata["streams"] if item.get("codec_type") == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    rate = Fraction(video_stream.get("avg_frame_rate") or video_stream["r_frame_rate"])
    fps_text = f"{rate.numerator}/{rate.denominator}"
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {input_video}")

    command = [
        args.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps_text,
        "-i",
        "pipe:0",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("FFmpeg stdin was not created")
    qa_times = [0.0, 53.8, 60.5, 64.0, 74.8, 88.5, 126.0, 160.5, 168.0]
    qa_dir = output_path.parent / "qa-frames"
    qa_dir.mkdir(exist_ok=True)
    qa_index = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / float(rate)
            rendered = draw_frame(
                frame,
                timestamp,
                active_segment(original, timestamp),
                active_segment(predicted, timestamp),
                font_path,
                bold_font_path,
            )
            process.stdin.write(rendered.tobytes())
            if qa_index < len(qa_times) and timestamp >= qa_times[qa_index]:
                cv2.imwrite(str(qa_dir / f"frame-{qa_index + 1:02d}-{timestamp:.3f}.jpg"), rendered)
                qa_index += 1
            frame_index += 1
    finally:
        capture.release()
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {stderr}")

    subprocess.run(
        [
            args.ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    rendered_metadata = probe(output_path, args.ffprobe)
    report = {
        "output": str(output_path),
        "frames_rendered": frame_index,
        "source_duration_seconds": float(metadata["format"]["duration"]),
        "output_duration_seconds": float(rendered_metadata["format"]["duration"]),
        "width": width,
        "height": height,
        "frame_rate": fps_text,
        "original_segments": len(original),
        "predicted_segments": len(predicted),
        "validation": validation,
        "full_decode_validation": "passed",
        "listening_review": "not_performed",
    }
    output_path.with_name("render-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
