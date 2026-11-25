"""Offline Transformers VITS synthesis shared by the service and the CLI."""

import wave
from pathlib import Path
from typing import Any

import numpy as np


_TRANSFORMERS_TTS_CACHE: dict[str, tuple[Any, Any, Any, int, str]] = {}


def synthesize_transformers_tts(text: str, output: Path, model_path: str, device: str) -> Path:
    cache_key = f"{model_path}:{device}"
    if cache_key not in _TRANSFORMERS_TTS_CACHE:
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise RuntimeError("Transformers TTS requires the Conda pytorch and transformers packages.") from exc
        dtype = torch.float32
        model = VitsModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _TRANSFORMERS_TTS_CACHE[cache_key] = (
            torch,
            tokenizer,
            model,
            int(model.config.sampling_rate),
            device,
        )
    torch, tokenizer, model, sample_rate, model_device = _TRANSFORMERS_TTS_CACHE[cache_key]
    inputs = tokenizer(text, return_tensors="pt").to(model_device)
    with torch.inference_mode():
        waveform = model(**inputs).waveform.squeeze().detach().float().cpu().numpy()
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return output
