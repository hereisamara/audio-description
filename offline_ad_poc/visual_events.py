"""Validate visual observations and derive narration decisions."""

from __future__ import annotations

import re
from typing import Any, Sequence


EVENT_KINDS = {"none", "action", "setting", "text", "detail"}
BOOLEAN_FIELDS = ("important", "already_audible", "uncertain")
NARRATOR_LEAD_IN = re.compile(
    r"^(?:(?:in (?:this|the) (?:image|picture|frame|shot)[,:\s]+)?"
    r"(?:i|we|you) (?:can )?(?:see|notice)(?:\s+that\b)?"
    r"|(?:this|the) (?:image|picture|frame|shot) (?:shows|depicts))(?=$|[\s,.!?;:])",
    re.IGNORECASE,
)


def direct_ad_text(text: str) -> str:
    """Keep the visual description and remove a preceding narrator phrase."""
    text = text.strip()
    remainder = NARRATOR_LEAD_IN.sub("", text, count=1)
    if remainder == text:
        return text
    remainder = remainder.lstrip(" \t,:;").strip()
    if not remainder.strip(".!?"):
        return ""
    return remainder[0].upper() + remainder[1:]


def validate_visual_event(
    proposal: dict[str, Any],
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    *,
    action_ready_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    required = {"kind", *BOOLEAN_FIELDS, "frame_ids", "ready_frame_id", "description"}
    if proposal.keys() != required:
        raise ValueError("Use exactly the seven fields in the event schema")
    kind = proposal["kind"]
    if not isinstance(kind, str) or kind not in EVENT_KINDS:
        raise ValueError("kind must be none, action, setting, text, or detail")
    if any(type(proposal[key]) is not bool for key in BOOLEAN_FIELDS):
        raise ValueError("important, already_audible and uncertain must be JSON booleans")
    ids = proposal["frame_ids"]
    ready_id = proposal["ready_frame_id"]
    description = proposal["description"]
    if not isinstance(ids, list) or not isinstance(description, str):
        raise ValueError("Use an array for frame_ids and a string for description")
    description = description.strip()
    if kind == "none":
        if ids or ready_id is not None or description or proposal["important"]:
            raise ValueError("A none event requires empty evidence, empty description and important=false")
        return proposal
    description = direct_ad_text(description)
    if not ids or not description:
        raise ValueError("An event requires supporting frame IDs and a description")
    if any(type(i) is not int or not 1 <= i <= len(timestamps) for i in ids):
        raise ValueError("frame_ids must contain only supplied integer image IDs")
    if type(ready_id) is not int or ready_id not in ids:
        raise ValueError("ready_frame_id must be one of the supporting IDs")
    if any(not window_start <= timestamps[i - 1] < window_end for i in ids):
        raise ValueError("Only images marked inside the current window can support an event")
    if kind == "action" and action_ready_ids is not None and ready_id not in action_ready_ids:
        allowed_ids = ", ".join(map(str, action_ready_ids)) or "none"
        raise ValueError(f"Near-blank image {ready_id} cannot anchor an action. Allowed IDs: {allowed_ids}")
    return {**proposal, "frame_ids": sorted(set(ids)), "description": description}


def apply_ad_policy(proposal: dict[str, Any], timestamps: Sequence[float]) -> dict[str, Any]:
    """Convert a validated observation into a deterministic AD/WAIT decision."""
    reason = next(
        (
            label for blocked, label in (
                (proposal["kind"] == "none", "no_visual_event"),
                (proposal["uncertain"], "uncertain_visual_evidence"),
                (proposal["already_audible"], "already_communicated_by_dialogue"),
                (not proposal["important"], "low_visual_importance"),
            )
            if blocked
        ),
        None,
    )
    if reason:
        return {"decision": "WAIT", "reason": reason, "importance": float(proposal["important"])}
    evidence = sorted({timestamps[i - 1] for i in proposal["frame_ids"]})
    return {
        "decision": "AD",
        "description": proposal["description"],
        "reason": f"important_visual_{proposal['kind']}",
        "importance": 1.0,
        "event_start": evidence[0],
        "event_end": evidence[-1],
        "event_timing_source": "model_frame_ids",
        "evidence_timestamps": evidence,
        "description_ready_at": timestamps[proposal["ready_frame_id"] - 1],
    }
