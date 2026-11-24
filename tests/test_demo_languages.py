import json
from pathlib import Path
import unittest
from unittest.mock import patch

from demo_app.server import combined_ad_rows, subtitle_text
from offline_ad_poc.translate_ad import translate


class DemoLanguageTests(unittest.TestCase):
    def test_bilingual_subtitles_preserve_independent_cue_timing(self):
        english = [{"start": 2, "end": 4, "text": "A frog sits."}]
        thai = [{"start": 3, "end": 5, "text": "กบนั่งอยู่"}]
        rows = combined_ad_rows(english, thai)
        self.assertEqual([(row["start"], row["end"]) for row in rows], [(2, 3), (3, 4), (4, 5)])
        self.assertIn("A frog sits.\nกบนั่งอยู่", subtitle_text(rows))
        self.assertEqual(combined_ad_rows(english, []), english)

    def test_thai_translation_preserves_time_and_checks_key_visible_terms(self):
        source = [{"start": 2, "end": 4, "text": "A small frog sits."}]
        with patch("offline_ad_poc.translate_ad.run_qwen_cue", return_value='{"1":"กบตัวเล็กนั่งอยู่"}') as run:
            [cue] = translate(source, "llama-completion", Path("model.gguf"))
            self.assertEqual(cue, {"start": 2, "end": 4, "text": "กบตัวเล็กนั่งอยู่"})
            prompt = json.loads(run.call_args.args[2]["user_prompt"])
            self.assertEqual(prompt["required_thai_terms"]["1"]["frog"], "กบ")
        with patch("offline_ad_poc.translate_ad.run_qwen_cue", return_value='{"1":"มีแมลงตัวเล็ก"}'):
            with self.assertRaisesRegex(ValueError, "frog"):
                translate(source, "llama-completion", Path("model.gguf"))
