from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from offline_ad_poc.pipeline import (
    Gap,
    Interval,
    Shot,
    Word,
    _segments_from_words,
    _whisper_generate_kwargs,
    build_gaps,
    expand_qwen_image_tokens,
    limit_words,
    merge_intervals,
    segment_scenes,
    speech_intervals_from_words,
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


if __name__ == "__main__":
    unittest.main()

