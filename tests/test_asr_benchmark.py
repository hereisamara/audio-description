from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_ad_poc.asr_benchmark import (
    load_reference,
    parse_model_specs,
    temporal_metrics,
    word_error_rate,
)
from offline_ad_poc.pipeline import Interval


class AsrBenchmarkTests(unittest.TestCase):
    def test_model_specs_require_unique_names_and_existing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first.mkdir()
            second.mkdir()

            specs = parse_model_specs([f"small={first}", f"turbo={second}"])

            self.assertEqual([name for name, _ in specs], ["small", "turbo"])
            with self.assertRaises(ValueError):
                parse_model_specs([f"same={first}", f"same={second}"])

    def test_temporal_metrics_separate_false_positive_and_missed_seconds(self) -> None:
        metrics = temporal_metrics(
            [Interval(10.0, 20.0)],
            [Interval(0.0, 15.0)],
        )

        self.assertAlmostEqual(metrics["temporal_precision"], 1 / 3, places=5)
        self.assertAlmostEqual(metrics["temporal_recall"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_seconds"], 10.0)
        self.assertAlmostEqual(metrics["missed_dialogue_seconds"], 5.0)
        self.assertAlmostEqual(metrics["false_positive_intro_seconds"], 10.0)

    def test_word_error_rate_is_zero_for_normalized_match(self) -> None:
        self.assertEqual(word_error_rate("Hello, Miss Wilson!", "hello miss wilson"), 0.0)
        self.assertAlmostEqual(word_error_rate("one two three", "one four three"), 1 / 3)

    def test_real_scene_reference_contains_sixteen_dialogue_subtitles(self) -> None:
        reference = Path(__file__).parents[1] / "references" / "real-scene-01-dialogue.json"
        intervals, transcript, segments = load_reference(reference)

        self.assertEqual(len(segments), 16)
        self.assertEqual(len(intervals), 6)
        self.assertAlmostEqual(sum(item.end - item.start for item in intervals), 87.424)
        self.assertIn("Miss Wilson", transcript)


if __name__ == "__main__":
    unittest.main()
