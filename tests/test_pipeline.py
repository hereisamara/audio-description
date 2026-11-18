from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from offline_ad_poc.pipeline import (
    Gap,
    Interval,
    Decision,
    RuntimeProfiler,
    Shot,
    Word,
    _detect_shots_opencv,
    _segments_from_words,
    _whisper_generate_kwargs,
    build_gaps,
    expand_qwen_image_tokens,
    limit_words,
    merge_intervals,
    mix_described_video,
    resize_frame_max_edge,
    segment_scenes,
    speech_intervals_from_words,
    validate_whisper_word_timestamps,
    validate_shot_timeline,
)


class PipelineUnitTests(unittest.TestCase):
    def test_real_scene_helper_is_present(self) -> None:
        self.assertTrue((Path(__file__).parents[1] / "offline_ad_poc" / "prepare_real_scene.py").exists())

    def test_word_intervals_are_padded_and_merged(self) -> None:
        words = [Word(1.0, 1.3, "hello"), Word(1.5, 1.8, "there")]
        intervals = speech_intervals_from_words(words, duration=5.0, padding=0.1, merge_gap=0.3)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].start, 0.9)
        self.assertAlmostEqual(intervals[0].end, 1.9)

    def test_safe_gaps_do_not_overlap_speech(self) -> None:
        speech = [Interval(0.0, 2.0), Interval(6.0, 8.0)]
        gaps = build_gaps(speech, duration=10.0, minimum_duration=1.5, safety_margin=0.2)
        self.assertEqual(len(gaps), 2)
        self.assertGreater(gaps[0].start, 2.0)
        self.assertLess(gaps[0].end, 6.0)
        self.assertGreater(gaps[0].word_budget, 0)

    def test_invalid_whisper_word_durations_are_rejected(self) -> None:
        words = [
            Word(1.0, 1.4, "real"),
            Word(2.0, 2.0, "zero"),
            Word(5.0, 15.0, "stretched"),
        ]
        validated, report = validate_whisper_word_timestamps(words)

        self.assertEqual(validated, [words[0]])
        self.assertEqual(report["rejected_nonpositive_duration"], 1)
        self.assertEqual(report["rejected_stretched_duration"], 1)

    def test_limit_words_respects_budget(self) -> None:
        text = limit_words("one two three four five six", 4)
        self.assertEqual(len(text.split()), 4)

    def test_scene_split_uses_combined_boundary_score(self) -> None:
        shots = [Shot(1, 0, 2, transcript="same words"), Shot(2, 2, 4, transcript="different topic")]
        first = np.zeros(4, dtype=np.float32)
        first[0] = 1
        second = np.zeros(4, dtype=np.float32)
        second[1] = 1
        scenes = segment_scenes(shots, [first, second], boundary_threshold=0.5)
        self.assertEqual(len(scenes), 2)

    def test_shot_timeline_validation_checks_coverage_and_continuity(self) -> None:
        valid = validate_shot_timeline([Shot(1, 0.0, 2.0), Shot(2, 2.0, 5.0)], 5.0)
        invalid = validate_shot_timeline([Shot(1, 0.2, 2.0), Shot(2, 2.4, 4.0)], 5.0)

        self.assertTrue(valid["valid_timeline"])
        self.assertFalse(invalid["valid_timeline"])
        self.assertIn("timeline_does_not_start_at_zero", invalid["warnings"])
        self.assertIn("gap_before_shot_2", invalid["warnings"])

    def test_opencv_shot_detector_finds_a_hard_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "hard-cut.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (0,) * 10 + (255,) * 10:
                    writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            finally:
                writer.release()

            shots = _detect_shots_opencv(video_path, duration=2.0, threshold=20.0)

        self.assertEqual(len(shots), 2)
        self.assertAlmostEqual(shots[0].end, 1.0, places=1)
        self.assertAlmostEqual(shots[1].start, 1.0, places=1)

    def test_merge_intervals_keeps_distant_regions_separate(self) -> None:
        merged = merge_intervals([Interval(0, 1), Interval(1.2, 2), Interval(4, 5)], maximum_gap=0.3)
        self.assertEqual(len(merged), 2)

    def test_transformers_words_are_grouped_into_segments(self) -> None:
        words = [Word(0.0, 0.3, "hello"), Word(0.4, 0.8, "there"), Word(2.0, 2.3, "again")]
        segments = _segments_from_words(words)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "hello there")

    def test_english_only_whisper_omits_language_and_task(self) -> None:
        self.assertEqual(_whisper_generate_kwargs(SimpleNamespace(is_multilingual=False)), {})
        self.assertEqual(
            _whisper_generate_kwargs(SimpleNamespace(is_multilingual=True)),
            {"language": "english", "task": "transcribe"},
        )

    def test_qwen_image_placeholders_expand_from_grid(self) -> None:
        expanded = expand_qwen_image_tokens(
            "start <|image_pad|> middle <|image_pad|> end",
            "<|image_pad|>",
            [[1, 4, 4], [1, 2, 4]],
            merge_size=2,
        )
        self.assertEqual(expanded.count("<|image_pad|>"), 6)

    def test_evidence_frame_resize_preserves_aspect_ratio_without_upscaling(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_frame_max_edge(frame, 320)
        unchanged = resize_frame_max_edge(frame, 1280)

        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertIs(unchanged, frame)

    @patch("offline_ad_poc.pipeline.run_command")
    def test_wait_only_output_copies_original_streams(self, run_command) -> None:
        decision = Decision(
            gap_start=1.0,
            gap_end=3.0,
            decision="WAIT",
            description="",
            reason="no important visual change",
            importance=0.1,
            word_budget=4,
            evidence_frames=[],
        )
        method = mix_described_video(
            Path("input.mp4"),
            Path("output.mp4"),
            [decision],
            "ffmpeg",
        )

        self.assertEqual(method, "ffmpeg_stream_copy_no_ad")
        command = run_command.call_args.args[0]
        self.assertIn("copy", command)
        self.assertNotIn("-filter_complex", command)

    def test_runtime_profiler_writes_stage_events_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profiler = RuntimeProfiler(Path(directory))
            profiler.set_hardware({"cpu_count_logical": 4, "cuda_available": False})
            with profiler.stage("example", method="unit-test", compute_device="cpu") as details:
                details["items"] = 3
            summary = profiler.finalize()

            event = json.loads(profiler.events_path.read_text(encoding="utf-8"))
            self.assertEqual(event["stage"], "example")
            self.assertEqual(event["method"], "unit-test")
            self.assertEqual(event["items"], 3)
            self.assertEqual(event["status"], "completed")
            self.assertGreaterEqual(event["elapsed_seconds"], 0.0)
            self.assertGreaterEqual(event["cpu_seconds"], 0.0)
            self.assertEqual(event["compute_device"], "cpu")
            self.assertTrue(event["uses_cpu"])
            self.assertFalse(event["uses_gpu"])
            self.assertGreater(event["process_peak_rss_mb"], 0.0)
            self.assertEqual(summary["stages"], [event])
            self.assertEqual(summary["hardware"]["cpu_count_logical"], 4)
            self.assertEqual(summary["totals_by_compute_device"]["cpu"]["stages"], 1)
            self.assertTrue(profiler.summary_path.exists())


if __name__ == "__main__":
    unittest.main()

