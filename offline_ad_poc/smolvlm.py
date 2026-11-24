"""Small offline visual captioner for CPU smoke runs."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from .visual_events import direct_ad_text

if TYPE_CHECKING:
    from .pipeline import EvidenceFrame, Gap


DEFAULT_MAX_NEW_TOKENS = 48


def caption_texts(raw_caption: str, reached_token_limit: bool) -> tuple[str, str]:
    """Keep full evidence for the editor and a short draft for direct narration."""
    caption = direct_ad_text(" ".join(raw_caption.split()))
    sentences = re.split(r"(?<=[.!?])\s+", caption)
    if reached_token_limit and len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?")):
        sentences.pop()
    return " ".join(sentences), sentences[0]


class SmolVLMBackend:
    uses_event_windows = True

    def __init__(self, model_path: str, device: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> None:
        import torch
        from PIL import Image, ImageOps
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.image = Image
        self.image_ops = ImageOps
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        # Send one global image to the vision encoder rather than a grid of crops.
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.do_resize = False
        self.image_edge = self.processor.image_processor.max_image_size["longest_edge"]
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype,
        ).to(device).eval()
        self.caption_calls = 0
        self.vision_crops = 0
        self.output_tokens = 0
        self.token_limit_hits = 0

    def runtime_metrics(self) -> dict:
        return {"caption_inferences": self.caption_calls, "images_per_inference": 1,
                "total_vision_crops": self.vision_crops, "caption_max_new_tokens": self.max_new_tokens,
                "caption_output_tokens": self.output_tokens, "caption_token_limit_hits": self.token_limit_hits}

    def decide(
        self, frames: list[EvidenceFrame], gap: Gap, transcript: str,
        previous_ad: list[str], context: dict,
    ) -> dict:
        eligible = [frame for frame in frames if gap.start <= frame.timestamp < gap.end and not frame.near_blank]
        if not eligible:
            return {"decision": "WAIT", "reason": "no_clear_frame_inside_window"}
        # Prefer strong evidence early enough to speak even in a short gap.
        early_limit = gap.end - min(2.0, gap.duration / 2)
        sample = max(
            [frame for frame in eligible if frame.timestamp <= early_limit] or eligible,
            key=lambda frame: (frame.salience, frame.timestamp),
        )
        with self.image.open(sample.path) as raw:
            image = self.image_ops.pad(raw.convert("RGB"), (self.image_edge, self.image_edge))
        prompt = self.processor.apply_chat_template([{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": (
                "Describe the visible evidence in this frame for a later audio-description editor. "
                "Write one breif paragraph of one to two sentences. "
            )}],
        }], add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=[[image]], return_tensors="pt").to(self.device)
        self.caption_calls += 1
        self.vision_crops += int(inputs.pixel_values.shape[1])
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = result[:, inputs.input_ids.shape[-1]:]
        output_tokens = generated.shape[-1]
        reached_limit = output_tokens >= self.max_new_tokens
        self.output_tokens += output_tokens
        self.token_limit_hits += int(reached_limit)
        raw_caption = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        visual_evidence, caption = caption_texts(raw_caption, reached_limit)
        audit = {"raw_caption": raw_caption, "caption_frame_timestamp": sample.timestamp,
                 "visual_evidence": visual_evidence, "caption_output_tokens": output_tokens,
                 "caption_token_limit_reached": reached_limit}
        if len(caption.split()) < 3:
            return {**audit, "decision": "WAIT", "reason": "empty_or_incomplete_caption"}
        if any(SequenceMatcher(None, caption.casefold(), prior.casefold()).ratio() >= .85
               for prior in previous_ad[-3:]):
            return {**audit, "decision": "WAIT", "reason": "repeated_caption"}
        return {**audit, "decision": "AD", "description": caption,
                "reason": "compact_single_frame_caption", "importance": 0.5,
                "event_start": sample.timestamp, "event_end": sample.timestamp,
                "evidence_timestamps": [sample.timestamp],
                "description_ready_at": sample.timestamp,
                "event_timing_source": "local_caption_frame"}
