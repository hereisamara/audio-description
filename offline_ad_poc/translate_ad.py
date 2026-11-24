"""Translate timed English audio-description cues with the local Qwen3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .ad_preparer import run_qwen_cue


GLOSSARY = {
    "egg mass": "กลุ่มไข่",
    "frog": "กบ",
    "tadpole": "ลูกอ๊อด",
    "chandelier": "โคมระย้า",
    "small": "เล็ก",
    "black": "ดำ",
    "u.s. fish & wildlife service": "หน่วยงานประมงและสัตว์ป่าแห่งสหรัฐอเมริกา",
}


def translate(rows: list[dict], executable: str, model: Path) -> list[dict]:
    translated = []
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset + 4]
        texts = {str(offset + index + 1): row["text"] for index, row in enumerate(batch)}
        terms = {
            key: {en: th for en, th in GLOSSARY.items() if re.search(rf"\b{re.escape(en)}(?:s|es)?\b", text, re.I)}
            for key, text in texts.items()
        }
        response = run_qwen_cue(executable, model, {
            "system_prompt": (
                "Translate each English audio-description cue into natural, concise Thai for speech. "
                "Keep each cue separate, preserve visible facts, actions and names, and use simple Thai. "
                "Do not add directions, labels or detail. Return the same numbered keys with Thai strings. "
                "Translate only the values in cues. Include the exact provided Thai terms for each numbered cue. "
                "Preserve grammatical number, such as a single person rather than a crowd."
            ),
            "user_prompt": json.dumps({"cues": texts, "required_thai_terms": terms}, ensure_ascii=False).replace("<", "\\u003c"),
            "response_schema": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in texts},
                "required": list(texts),
                "additionalProperties": False,
            },
            "generation": {"context_tokens": 4096, "final_tokens": 128 * len(batch)},
        })
        result = json.loads(response)
        if not isinstance(result, dict) or set(result) != set(texts):
            raise ValueError("Translation did not return all of the requested cues.")
        for row, key in zip(batch, texts):
            text = result[key]
            if not isinstance(text, str) or not re.search(r"[\u0e01-\u0e5b]", text):
                raise ValueError(f"Cue {key} did not return Thai text. Try translating again.")
            missing = [term for term, thai in terms[key].items() if thai not in text]
            if missing:
                raise ValueError(f"Cue {key} missed the Thai term for {', '.join(missing)}. Review the cue before speaking it.")
            translated.append({**row, "text": text.strip()})
    return translated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(translate(json.load(sys.stdin), args.executable, args.model), ensure_ascii=False))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
