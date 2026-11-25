"""Small, dependency-free client for the internal speech synthesis API."""

from __future__ import annotations

from http.client import HTTPException
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import wave


LANGUAGES = ("en", "th")
MAX_TEXT_LENGTH = 3000
MAX_REQUEST_BYTES = 64 * 1024
MAX_WAV_BYTES = 16 * 1024 * 1024


class TtsError(ValueError):
    """The speech synthesis service could not complete the request."""


def _request(url: str, route: str, *, body: dict | None = None, timeout: int, limit: int) -> bytes:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = Request(f"{url.rstrip('/')}{route}", data=data, headers=headers)
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            result = response.read(limit + 1)
    except HTTPError as error:
        try:
            message = json.loads(error.read(4096))["error"]
        except (ValueError, KeyError, TypeError, OSError, HTTPException):
            message = f"HTTP {error.code}"
        finally:
            error.close()
        raise TtsError(f"TTS service: {message}") from error
    except (URLError, OSError, HTTPException) as error:
        raise TtsError(f"TTS service is unreachable: {error}") from error
    if len(result) > limit:
        raise TtsError("TTS service response is too large.")
    return result


def available_voices(url: str) -> dict[str, str]:
    try:
        voices = json.loads(_request(url, "/v1/voices", timeout=2, limit=4096))["voices"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
        raise TtsError("TTS service returned an invalid voice list.") from error
    if not isinstance(voices, dict) or any(
        language not in LANGUAGES or not isinstance(voice, str) or not voice
        for language, voice in voices.items()
    ):
        raise TtsError("TTS service returned an invalid voice list.")
    return voices


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as sound:
            frames = sound.getnframes()
            rate = sound.getframerate()
            if (
                sound.getnchannels() != 1 or sound.getsampwidth() != 2 or not frames or not rate
                or len(sound.readframes(frames)) != frames * 2
            ):
                raise TtsError("TTS service must return a nonempty mono 16-bit PCM WAV.")
            return frames / rate
    except (EOFError, wave.Error) as error:
        raise TtsError("TTS service returned an invalid WAV.") from error


def synthesize(url: str, text: str, language: str, output: Path) -> float:
    data = _request(url, "/v1/synthesize", body={"text": text, "language": language},
                    timeout=120, limit=MAX_WAV_BYTES)
    duration = wav_duration(data)
    output.write_bytes(data)
    return duration
