"""Prepare narration from shot-local evidence before TTS."""

from __future__ import annotations

import json
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
    r"\b(?:here(?:'s| is| are)|this is|these are|an example of|look at) "
    r"(?:(?:a|an|the|some) )?"
)
HANDLING_WORDS = {"hand", "hands", "palm", "hold", "holds", "holding", "show", "shows", "showing",
                  "display", "displays", "displaying", "pointing"}
CONTEXT_CUE_WORDS = HANDLING_WORDS | {
    "a", "an", "the", "some", "of", "in", "on", "at", "with", "is", "are", "person", "their", "someone",
}
GENERIC_SUBJECT = re.compile(r"^(?:(?:a|the) person|he|she|they|i|we) ")
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


def build_qwen_cue_input(context: dict, *, reasoning_tokens: int = 0) -> dict:
    budget = context["max_words"]
    instruction = (
        "You are the final editor of one audio-description cue. Read the input JSON as evidence. "
        "Describe the most useful visible detail that dialogue leaves out. If the same action and object "
        "are already described in dialogue, put WAIT in description. Unrelated dialogue leaves the visual draft usable. "
        "Write a direct, present-tense line in third person. Keep the main action and drop secondary details. "
        f"The description may contain at most {budget} words. Return a JSON object.\n"
    )
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
            "Use the draft's words and leave dialogue_reference empty. Check for repetition before shortening.\n"
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
        started = time.perf_counter()
        raw = ""
        error = None
        llm_input = None
        suggestion = reference = ""
        already_spoken = generate is not None and not incomplete and repeats_spoken_action(draft, dialogue_text)
        if generate is not None and not already_spoken and (not incomplete or completions):
            try:
                llm_input = build_input(context) if build_input is not None else None
                raw = generate(llm_input if llm_input is not None else context)
                response = json.loads(raw)
                if (not isinstance(response, dict) or set(response) != set(CUE_SCHEMA["required"])
                        or not all(isinstance(value, str) for value in response.values())):
                    raise ValueError("Invalid cue response")
                suggestion = direct_ad_text(response["description"])
                reference = response["dialogue_reference"].strip()
            except (OSError, subprocess.SubprocessError) as exc:
                error = type(exc).__name__
            except ValueError:
                error = "invalid_cue_response"
        uses_draft_words = set(text_words(suggestion)) <= set(text_words(draft))
        grounded_reference = ""
        if not incomplete and (reference or not uses_draft_words):
            grounded_reference = contextual_identification(draft, suggestion, normalized_dialogue, reference)
        wants_wait = suggestion.casefold() == "wait" and not reference
        valid = (
            bool(suggestion) and suggestion.casefold() != "wait" and len(suggestion.split()) <= budget
            and "\n" not in suggestion and not incomplete_cue(suggestion)
            and (not reference and suggestion in context["allowed_completions"] if incomplete
                 else bool(grounded_reference) or (not reference and uses_draft_words))
        )
        decision.description = "" if wants_wait or already_spoken else suggestion if valid else fallback
        status = "prepared" if valid else "fallback" if decision.description else "wait"
        if status == "wait":
            decision.decision = "WAIT"
            if already_spoken:
                decision.reason = "already_communicated_by_dialogue"
            elif wants_wait:
                decision.reason = "ad_preparer_wait"
            else:
                decision.reason = (
                    "incomplete_visual_description" if incomplete
                    else "no_complete_description_within_word_budget"
                )
        elif anchor is not None:
            history.append((anchor, scene.scene_id if scene is not None else None, decision.description))
        audit.append({
            "decision_index": index, "shot_id": shot.shot_id if shot else None, "anchor": anchor,
            "dialogue_start": dialogue[0].start if dialogue else None,
            "dialogue_end": dialogue[-1].end if dialogue else None,
            "context": context, "llm_input": llm_input, "completion_evidence": completions, "raw_output": raw,
            "dialogue_reference": grounded_reference if valid else "",
            "description": decision.description, "status": status, "error": error,
            "hold_reason": decision.reason if status == "wait" else None,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        })
    return audit
