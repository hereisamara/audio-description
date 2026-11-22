"""Prepare narration from shot-local evidence before TTS."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from .visual_events import direct_ad_text

if TYPE_CHECKING:
    from .pipeline import Decision, Scene, Shot, Word


UNFINISHED_ENDINGS = {"a", "an", "the", "to", "of", "in", "on", "at", "with", "and", "or"}
DEMONSTRATIVE_REFERENCE = re.compile(
    r"\b(?:(?:here(?:'s| is| are)|this is|these are)(?: an example of)?|an example of|look at) "
    r"(?:(?:a|an|the|some) )?"
)
HANDLING_WORDS = {"hand", "hands", "palm", "hold", "holds", "holding", "show", "shows", "showing",
                  "display", "displays", "displaying", "pointing"}
CONTEXT_CUE_WORDS = HANDLING_WORDS | {
    "a", "an", "the", "some", "of", "in", "on", "at", "with", "is", "are", "person", "their", "someone",
}
GENERIC_SUBJECT = re.compile(r"^(?:(?:a|the) person|he|she|they|i|we) ")
CUE_INFLECTIONS = {
    "a": "the", "an": "the", "are": "is",
    "holding": "hold", "holds": "hold",
    "revealing": "reveal", "revealed": "reveal",
}
CUE_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}, "dialogue_reference": {"type": "string"}},
    "required": ["description", "dialogue_reference"],
    "additionalProperties": False,
}


def text_words(text: str) -> list[str]:
    return re.findall(r"\b[\w’'-]+\b", text.casefold())


def incomplete_cue(text: str) -> bool:
    words = text_words(text)
    return not words or words[-1] in UNFINISHED_ENDINGS


def repeats_spoken_action(draft: str, dialogue: str) -> bool:
    """Match an identical action and all its details across generic speakers."""
    text = " ".join(text_words(draft))
    action = GENERIC_SUBJECT.sub("", text)
    if action == text or len(action.split()) < 3:
        return False
    return any(
        GENERIC_SUBJECT.sub("", " ".join(text_words(sentence))) == action
        for sentence in re.split(r"(?<=[.!?])\s+", dialogue)
    )


def contextual_identification(draft: str, cue: str, dialogue: str, reference: str) -> str:
    """Verify a cited noun phrase, or recover it from a direct spoken identification."""
    draft_words = set(text_words(draft))
    cue_words = text_words(cue)
    cue_vocabulary = set(cue_words)
    if not draft_words & HANDLING_WORDS or not cue_vocabulary & HANDLING_WORDS:
        return ""
    cue_text = f" {' '.join(cue_words)} "
    cited = " ".join(text_words(reference))
    for match in DEMONSTRATIVE_REFERENCE.finditer(dialogue):
        spoken = dialogue[match.end():].split()[:6]
        for length in range(len(spoken), 0, -1):
            words = spoken[:length]
            phrase = " ".join(words)
            if (not cited or phrase == cited) and f" {phrase} " in cue_text:
                if phrase not in {"it", "one", "them", "something"} and cue_vocabulary <= (
                    draft_words | set(words) | CONTEXT_CUE_WORDS
                ):
                    return phrase
    return ""


def supports_visual_rewrite(draft: str, cue: str) -> bool:
    draft_words = text_words(draft)
    cue_words = text_words(cue)
    vocabulary = {CUE_INFLECTIONS.get(word, word) for word in draft_words}
    if all(CUE_INFLECTIONS.get(word, word) in vocabulary for word in cue_words):
        return True
    # A visible hand holding an explicit object also supports a person holding that same object.
    holding = re.fullmatch(r"(?:a|the) person (?:holds|is holding) (?:a|an|the) (.+)", " ".join(cue_words))
    return bool(holding and re.search(
        rf"\bhands? holding (?:a|an|the) {re.escape(holding[1])}\b", " ".join(draft_words),
    ))


def validate_prepared_reply(context: dict, suggestion: str, reference: str) -> tuple[str | None, str]:
    if not suggestion:
        return "ad_preparer_no_reply", ""
    if suggestion.casefold() == "wait" and not reference:
        return "ad_preparer_wait", ""
    if "\n" in suggestion or incomplete_cue(suggestion):
        return "incomplete_visual_description", ""
    if len(suggestion.split()) > context["max_words"]:
        return "ad_preparer_word_limit", ""
    if context["incomplete"]:
        valid = not reference and suggestion in context["allowed_completions"]
        return (None if valid else "ad_preparer_ungrounded_rewrite"), ""
    dialogue = " ".join(text_words(context["dialogue"])).replace("’", "'")
    grounded = contextual_identification(context["draft"], suggestion, dialogue, reference)
    if grounded or (not reference and supports_visual_rewrite(context["draft"], suggestion)):
        return None, grounded
    return "ad_preparer_ungrounded_rewrite", ""


def build_qwen_cue_input(context: dict, *, reasoning_tokens: int = 0) -> dict:
    budget = context["max_words"]
    instruction = (
        "You are the final editor of one audio-description cue. Read the input JSON as evidence. "
        "Describe the most useful visible detail that dialogue leaves out. If the same action and object "
        "are already described in dialogue, put WAIT in description. Unrelated dialogue leaves the visual draft usable. "
        "Write a direct, present-tense line in third person. Keep the main action and drop secondary details. "
        f"The description may contain at most {budget} words. Return a JSON object.\n"
    )
    if "available_seconds" in context:
        instruction += (
            f"The safe narration slot is {context['available_seconds']:.2f} seconds. "
            "Keep just the main visible action when time is tight. If it cannot fit, put WAIT in description.\n"
        )
    if context.get("timing_retry"):
        instruction += "The previous cue was too long when spoken. Make this rewrite shorter.\n"
    if context.get("previous_ad") or context.get("previous_dialogue") or context.get("scene_previous_ads"):
        instruction += (
            "Use previous_dialogue and earlier prepared AD to follow the scene and prefer a new visible detail. "
            "The current draft and identifying_dialogue establish what is visible now. Set an object name "
            "from current identifying_dialogue only. Earlier prepared cues are candidates for narration.\n"
        )
    schema = CUE_SCHEMA
    if context["incomplete"]:
        instruction += (
            "The draft is incomplete. Choose an allowed completion verbatim if it fits the word limit, "
            "or put WAIT in description. Leave dialogue_reference empty."
        )
    elif context["identifying_dialogue"]:
        instruction += (
            "If identifying_dialogue names the object visibly held or shown, use that name in place of a "
            "vague or mistaken label. Describe what the person does with it and copy the exact noun phrase "
            "without its article into dialogue_reference. A spoken name leaves the visible action worth describing. "
            "Otherwise use words from the draft and leave dialogue_reference empty.\n"
            'Draft: A hand is visible with dirt and debris on it. '
            'Identifying dialogue: an example of a cluster of tiny seeds that we collected. '
            'Output: {"description":"A person shows a cluster of tiny seeds.","dialogue_reference":"cluster of tiny seeds"}'
        )
    else:
        instruction += (
            "Use the draft's evidence and preserve who does the action. "
            "You may change verb inflections. Leave dialogue_reference empty and check for repetition.\n"
            'Draft: A person sets a bowl on the table. Dialogue: She sets a bowl on the table. '
            'Output: {"description":"WAIT","dialogue_reference":""}\n'
            'Draft: A person lifts a blue bowl from the counter. Dialogue: none. Max words: 5. '
            'Output: {"description":"A person lifts a bowl.","dialogue_reference":""}'
        )
        schema = {**CUE_SCHEMA, "properties": {**CUE_SCHEMA["properties"], "dialogue_reference": {"const": ""}}}
    data = json.dumps(context, ensure_ascii=True).replace("<", "\\u003c")
    return {
        "system_prompt": instruction, "user_prompt": data, "response_schema": schema,
        "generation": {"context_tokens": 2048, "reasoning_tokens": reasoning_tokens, "final_tokens": 72},
    }


def request_prepared_cue(context: dict, generate: Callable[[dict], str],
                         build_input: Callable[[dict], dict] | None) -> dict:
    llm_input = build_input(context) if build_input is not None else None
    raw = suggestion = grounded = ""
    error = None
    try:
        raw = generate(llm_input if llm_input is not None else context)
        response = json.loads(raw)
        if (not isinstance(response, dict) or set(response) != set(CUE_SCHEMA["required"])
                or not all(isinstance(value, str) for value in response.values())):
            raise ValueError("Invalid cue response")
        suggestion = direct_ad_text(response["description"])
        rejection, grounded = validate_prepared_reply(context, suggestion, response["dialogue_reference"].strip())
    except (OSError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__
        rejection = "ad_preparer_execution_failed"
    except ValueError:
        error = rejection = "invalid_cue_response"
    return {"llm_input": llm_input, "raw_output": raw, "suggestion": suggestion,
            "rewrite_rejection": rejection, "dialogue_reference": grounded, "error": error}


def retry_prepared_cue(record: dict, text: str, seconds: float, measured: float,
                       generate: Callable[[dict], str], build_input: Callable[[dict], dict]) -> str | None:
    words = len(text.split())
    budget = min(words - 1, math.floor(words * seconds / measured * 0.9))
    if budget < 2:
        return None
    started = time.perf_counter()
    context = {**record["context"], "max_words": budget, "available_seconds": round(seconds, 3),
               "timing_retry": {"previous_cue": text, "measured_seconds": round(measured, 3)}}
    reply = request_prepared_cue(context, generate, build_input)
    record["timing_retry"] = {
        "context": context, **reply, "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    return reply["suggestion"] if reply["rewrite_rejection"] is None else None


def run_qwen_cue(executable: str, model: Path, llm_input: dict) -> str:
    generation = llm_input.get("generation", {})
    reasoning_tokens = generation.get("reasoning_tokens", 0)
    prompt = (
        f"<|im_start|>system\n{llm_input['system_prompt']}<|im_end|>\n"
        f"<|im_start|>user\n{llm_input['user_prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    command = [executable, "-m", str(model), "--device", "none", "-ngl", "0", "-t", "4",
               "-c", str(generation.get("context_tokens", 2048)), "--no-conversation", "--no-display-prompt"]

    def complete(prompt_text: str, options: list[str]) -> str:
        result = subprocess.run(
            [*command, *options, "-p", prompt_text],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout.replace("[end of text]", "").strip()

    scratch = ""
    if reasoning_tokens > 0:
        scratch = complete(prompt, [
            "-n", str(reasoning_tokens), "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
            "--min-p", "0", "--seed", "42", "--special", "--reverse-prompt", "</think>",
        ])
        scratch = re.split(r"</think>|<\|", scratch, maxsplit=1)[0].removeprefix("<think>").strip()
    return complete(prompt + scratch + "\n</think>\n\n", [
        "-n", str(generation.get("final_tokens", 72)), "--temp", "0",
        "--json-schema", json.dumps(llm_input["response_schema"]),
    ])


def generate_qwen_cue(executable: str, model: Path, context: dict) -> str:
    return run_qwen_cue(executable, model, build_qwen_cue_input(context))


def prepare_ad_cues(
    decisions: Sequence[Decision], shots: Sequence[Shot], words: Sequence[Word],
    generate: Callable[[dict], str] | None = None,
    *,
    build_input: Callable[[dict], dict] | None = None,
    scenes: Sequence[Scene] = (),
    narration_seconds: Callable[[Decision], float] | None = None,
) -> list[dict]:
    """Use prior evidence from the anchor shot and leave cue timing to TTS."""
    proposals = {
        (decision.gap_index, proposal["window_index"]): proposal
        for decision in decisions for proposal in decision.visual_proposals
    }
    audit = []
    history: list[tuple[float, int | None, str]] = []
    for index, decision in enumerate(decisions, 1):
        if decision.decision != "AD":
            continue
        proposal = proposals.get((decision.gap_index, decision.selected_window), {})
        draft = direct_ad_text(proposal.get("visual_evidence") or proposal.get("description") or decision.description)
        incomplete = incomplete_cue(draft)
        fallback = decision.description if not incomplete and not incomplete_cue(decision.description) else ""
        if generate is None and fallback:
            continue

        anchor = decision.description_ready_at
        if anchor is None:
            anchor = decision.event_start
        shot = next((item for item in shots if anchor is not None and item.start <= anchor < item.end), None)
        scene = next((item for item in scenes if anchor is not None and item.start <= anchor < item.end), None)
        start = max(shot.start, anchor - 8.0) if shot is not None else anchor
        dialogue = []
        completions = []
        budget = min(18, decision.word_budget)
        available = narration_seconds(decision) if generate is not None and narration_seconds is not None else None
        if available is not None:
            budget = min(budget, math.floor(available * 2.5))
        if generate is not None and len(fallback.split()) > budget:
            fallback = ""
        if shot is not None:
            dialogue = [word for word in words if start <= word.start and word.end <= anchor][-60:]
            if incomplete:
                prefix = text_words(draft)
                for observation in proposals.values():
                    timestamp = observation.get("description_ready_at")
                    text = direct_ad_text(observation.get("description") or "")
                    if (observation.get("decision") == "AD" and prefix
                            and timestamp is not None and start <= timestamp < anchor
                            and not incomplete_cue(text) and text_words(text)[:len(prefix)] == prefix
                            and len(text.split()) <= budget):
                        completions.append({"timestamp": timestamp, "cue": text})
                completions = sorted(completions, key=lambda item: item["timestamp"], reverse=True)[:3]

        dialogue_text = " ".join(word.text for word in dialogue)
        normalized_dialogue = " ".join(text_words(dialogue_text)).replace("’", "'")
        previous_dialogue = [
            word.text for word in words if anchor is not None and anchor - 30.0 <= word.start and word.end <= start
        ][-36:]
        earlier_ads = sorted((item for item in history if anchor is not None and item[0] < anchor),
                             key=lambda item: item[0])
        scene_ads = [
            {"time": round(timestamp, 3), "text": text}
            for timestamp, scene_id, text in earlier_ads if scene is not None and scene_id == scene.scene_id
        ][-3:]
        context = {
            "draft": draft, "max_words": budget, "incomplete": incomplete,
            "dialogue": dialogue_text,
            "previous_dialogue": " ".join(previous_dialogue),
            "previous_ad": earlier_ads[-1][2] if earlier_ads else "",
            "current_scene": {"id": scene.scene_id, "start": round(scene.start, 3),
                              "end": round(scene.end, 3)} if scene is not None else None,
            "scene_previous_ads": scene_ads,
            "identifying_dialogue": [
                " ".join(normalized_dialogue[match.start():].split()[:14])
                for match in DEMONSTRATIVE_REFERENCE.finditer(normalized_dialogue)
            ][:3],
            "allowed_completions": [item["cue"] for item in completions],
        }
        if available is not None:
            context["available_seconds"] = round(available, 3)
        started = time.perf_counter()
        reply = {"llm_input": None, "raw_output": "", "suggestion": "", "rewrite_rejection": None,
                 "dialogue_reference": "", "error": None}
        already_spoken = generate is not None and not incomplete and repeats_spoken_action(draft, dialogue_text)
        hold = None
        if budget < 2:
            hold = "no_safe_post_event_slot"
        elif already_spoken:
            hold = "already_communicated_by_dialogue"
        elif incomplete and not completions:
            hold = "incomplete_visual_description"
        elif generate is not None:
            reply = request_prepared_cue(context, generate, build_input)
            if reply["rewrite_rejection"] == "ad_preparer_wait":
                hold = "ad_preparer_wait"
        valid = bool(reply["suggestion"]) and reply["rewrite_rejection"] is None
        decision.description = "" if hold else reply["suggestion"] if valid else fallback
        status = "prepared" if valid else "fallback" if decision.description else "wait"
        if status == "wait":
            decision.decision = "WAIT"
            decision.reason = hold or reply["rewrite_rejection"] or "no_complete_description_within_word_budget"
        elif anchor is not None:
            history.append((anchor, scene.scene_id if scene is not None else None, decision.description))
        audit.append({
            "decision_index": index, "shot_id": shot.shot_id if shot else None, "anchor": anchor,
            "dialogue_start": dialogue[0].start if dialogue else None,
            "dialogue_end": dialogue[-1].end if dialogue else None,
            "context": context, "completion_evidence": completions, **reply,
            "description": decision.description, "status": status,
            "hold_reason": decision.reason if status == "wait" else None,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        })
    return audit
