import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from offline_ad_poc.job_store import JobStore
from offline_ad_poc.job_worker import write_narration_plan
from offline_ad_poc.profiles import inference_profile


class SplitJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = JobStore(self.root)
        self.video = self.root / "source.mp4"
        self.video.touch()
        self.store.register_media("media", self.video)

    def test_whisper_publishes_dialogue_before_ad_is_claimable(self):
        job = self.store.enqueue("media", "Frog", "Studio", "smolvlm", "auto", False)
        self.assertIsNone(self.store.claim("ad", False, "ad-host"))
        whisper = self.store.claim("whisper", True, "whisper-host")
        self.assertEqual(whisper["selected_device"], "cuda")
        self.assertIsNone(self.store.claim("whisper", True, "second-worker"))

        payload = {"schema_version": 1, "transcript": {"segments": []}, "speech_and_gaps": {"gaps": []}}
        self.assertTrue(self.store.finish_dialogue(whisper, payload))
        reopened = JobStore(self.root)
        self.assertEqual(reopened.dialogue(job["id"])[1], payload)
        self.assertEqual(reopened.source_path("media"), self.video.resolve())

        ad = reopened.claim("ad", False, "ad-host")
        self.assertEqual(ad["selected_device"], "cpu")
        self.assertTrue(reopened.finish_ad(ad))
        self.assertEqual(reopened.get(job["id"])["state"], "finished")
        self.assertEqual(reopened.get(job["id"])["result_dir"], ad["attempt_dir"])

    def test_expired_attempt_is_retried_with_separate_files(self):
        row = self.store.enqueue("media", "Frog", "Studio", "smolvlm", "cpu", False)
        old = self.store.claim("whisper", False, "old-host")
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (row["id"],))
        current = self.store.claim("whisper", False, "new-host")
        self.assertNotEqual(old["attempt_dir"], current["attempt_dir"])
        self.assertFalse(self.store.renew(old))
        self.assertFalse(self.store.finish_dialogue(old, {}))
        self.assertIsNone(self.store.dialogue(row["id"]))
        self.assertTrue(self.store.finish_dialogue(current, {"schema_version": 1}))

    def test_multiple_services_can_initialize_the_same_new_database(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            stores = list(workers.map(JobStore, [self.root / "concurrent"] * 8))
        self.assertTrue(all(store.recent() == [] for store in stores))

    def test_cuda_without_fallback_fails_on_a_cpu_worker(self):
        row = self.store.enqueue("media", "Frog", "Studio", "smolvlm", "cuda", False)
        self.assertIsNone(self.store.claim("whisper", False, "cpu-host"))
        self.assertEqual(self.store.get(row["id"])["state"], "failed")
        self.assertIn("CUDA", self.store.get(row["id"])["error"])

    def test_auto_uses_cpu_qwen3vl_and_lanta_profile_for_cuda(self):
        profiles = {device: inference_profile("auto", device).name for device in ("cpu", "cuda")}
        cpu = inference_profile("auto", "cpu")
        self.assertIn("Qwen3VL-2B-Instruct", cpu.vlm)
        self.assertEqual((cpu.evidence_frames, cpu.evidence_max_edge, cpu.followups), (2, 384, 0))
        cuda = inference_profile("auto", "cuda")
        self.assertEqual((cuda.asr, cuda.vlm, cuda.tts),
                         ("whisper-small.en-transformers", "Qwen3-VL-8B-Instruct", "mms-tts-eng"))
        self.assertEqual((cuda.evidence_frames, cuda.evidence_max_edge, cuda.followups), (6, 0, 2))
        for has_cuda in (True, False):
            job = self.store.enqueue("media", "Frog", "Studio", "auto", "cuda", True)
            speech_claim = self.store.claim("whisper", has_cuda, "asr-host", profiles)
            self.assertTrue(self.store.finish_dialogue(speech_claim, {"schema_version": 1}))
            ad_claim = self.store.claim("ad", has_cuda, "ad-host", profiles)
            expected = "qwen3vl8b" if has_cuda else "qwen3vl"
            self.assertEqual(self.store.get(job["id"])["profile"], expected)
            self.assertTrue(self.store.finish_ad(ad_claim))

    def test_narration_plan_reuses_timed_cues_and_portable_wav_refs(self):
        (self.root / "ad-decisions.json").write_text(json.dumps([
            {"decision": "WAIT", "description": "Ignore."},
            {"decision": "AD", "description": "A frog sits.", "cue_start": 1, "cue_end": 2.4,
             "tts_path": str(self.root / "narration/ad-001-try-1.wav"),
             "overlaps_dialogue": False, "dialogue_overlap_seconds": 0},
        ]))
        write_narration_plan(self.root)
        plan = json.loads((self.root / "narration-plan.json").read_text())
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["cues"], [{
            "start": 1, "end": 2.4, "text": "A frog sits.", "audio": "narration/ad-001-try-1.wav",
            "overlaps_dialogue": False, "dialogue_overlap_seconds": 0,
        }])
