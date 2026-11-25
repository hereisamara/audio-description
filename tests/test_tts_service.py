import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from demo_app.server import narration_mix
from offline_ad_poc.pipeline import resolve_tts_backend, synthesize_text
from offline_ad_poc.profiles import inference_profile
from offline_ad_poc.tts_client import TtsError, synthesize


def speech_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(32000))
    return buffer.getvalue()


class TtsServiceTests(unittest.TestCase):
    def test_ad_profile_uses_service_without_local_tts_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            vlm = Path(temporary) / "vlm"
            vlm.mkdir()
            (vlm / "config.json").touch()
            url = "http://tts:7861"
            profile = inference_profile("auto", "cuda")
            args = profile.ad_args(Path(temporary), vlm=vlm, tts_url=url)
            self.assertNotIn("--tts-model", args)
            backend, endpoint = resolve_tts_backend(args[args.index("--tts-backend") + 1], None, None, url)
            with patch("offline_ad_poc.tts_client._request", return_value=speech_wav()) as request:
                output = synthesize_text("A frog sits.", Path(temporary) / "narration", backend, endpoint, None, "cuda")
                self.assertEqual(output.read_bytes(), speech_wav())
                self.assertEqual(request.call_args.kwargs["body"], {"language": "en", "text": "A frog sits."})

    def test_review_sends_thai_cues_to_service_without_system_voice(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("demo_app.server.voice_engine", side_effect=AssertionError("System voice was used")), \
             patch("offline_ad_poc.tts_client._request", return_value=speech_wav()) as request:
            cue = {"start": 2, "end": 4, "text": "กบนั่งอยู่"}
            inputs, filters = narration_mix([cue], Path(temporary), 0, 1, "th", tts_url="http://tts:7861")
            self.assertEqual(request.call_args.kwargs["body"], {"language": "th", "text": cue["text"]})
            self.assertEqual(inputs[1].read_bytes(), speech_wav())
            self.assertIn("adelay=2000:all=1", filters)

    def test_invalid_wav_fails_before_publishing_a_file(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("offline_ad_poc.tts_client._request", return_value=speech_wav()[:-2]):
            path = Path(temporary) / "speech.wav"
            with self.assertRaises(TtsError):
                synthesize("http://tts:7861", "A frog sits.", "en", path)
            self.assertFalse(path.exists())
