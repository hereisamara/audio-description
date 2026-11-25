"""Serve English and Thai speech synthesis using local MMS-TTS checkpoints."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time

from .tts_client import LANGUAGES, MAX_REQUEST_BYTES, MAX_TEXT_LENGTH, MAX_WAV_BYTES, TtsError, wav_duration
from .tts_model import synthesize_transformers_tts


VOICES = {"en": "mms-tts-eng", "th": "mms-tts-tha"}
MODEL_FILES = ("config.json", "tokenizer_config.json", "vocab.json", "model.safetensors")
LOGGER = logging.getLogger(__name__)


class SpeechServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], models: dict[str, Path]) -> None:
        super().__init__(address, SpeechHandler)
        self.models = models
        self.synthesis_lock = threading.Lock()

    def voices(self) -> dict[str, str]:
        return {
            language: VOICES[language]
            for language, model in self.models.items()
            if all((model / name).is_file() for name in MODEL_FILES)
        }


class SpeechHandler(BaseHTTPRequestHandler):
    def reply(self, status: int, data: bytes, content_type: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def json_reply(self, status: int, payload: dict) -> None:
        self.reply(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        if self.path == "/v1/voices":
            self.json_reply(200, {"voices": self.server.voices()})
        else:
            self.json_reply(404, {"error": "Unknown TTS route."})

    def do_POST(self) -> None:
        if self.path != "/v1/synthesize":
            self.json_reply(404, {"error": "Unknown TTS route."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES or self.headers.get_content_type() != "application/json":
                raise ValueError
            payload = json.loads(self.rfile.read(size))
            language = payload["language"]
            text = payload["text"]
            if language not in LANGUAGES or not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            self.json_reply(400, {"error": "Supply 1–3000 characters of text and language 'en' or 'th' as JSON."})
            return

        voice = self.server.voices().get(language)
        if voice is None:
            self.json_reply(503, {"error": f"The {language} voice checkpoint is unavailable."})
            return
        try:
            started = time.monotonic()
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "speech.wav"
                with self.server.synthesis_lock:
                    synthesize_transformers_tts(text.strip(), output, str(self.server.models[language]), "cpu")
                with output.open("rb") as sound:
                    data = sound.read(MAX_WAV_BYTES + 1)
            if len(data) > MAX_WAV_BYTES:
                raise TtsError("Synthesized audio exceeds the response limit.")
            duration = wav_duration(data)
        except Exception:
            LOGGER.exception("Speech synthesis failed for %s", language)
            self.json_reply(500, {"error": f"Speech synthesis failed for language '{language}'."})
            return
        LOGGER.info("voice=%s compute_device=cpu elapsed_seconds=%.3f audio_seconds=%.3f",
                    voice, time.monotonic() - started, duration)
        self.reply(200, data, "audio/wav", **{"X-TTS-Voice": voice, "X-TTS-Duration-Seconds": f"{duration:.6f}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--english-model", type=Path,
                        default=os.environ.get("TTS_MODEL_EN") or "/models/mms-tts-eng-local")
    parser.add_argument("--thai-model", type=Path,
                        default=os.environ.get("TTS_MODEL_TH") or "/models/mms-tts-tha")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with SpeechServer((args.host, args.port), {"en": args.english_model, "th": args.thai_model}) as server:
        LOGGER.info("Speech service ready on %s:%d (staged voices: %s)",
                    args.host, server.server_port, sorted(server.voices()))
        server.serve_forever()


if __name__ == "__main__":
    main()
