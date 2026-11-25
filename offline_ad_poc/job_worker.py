"""Run one container role against Studio's shared job volume."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .job_store import JobStore
from .profiles import PROFILE_OPTIONS, inference_profile, staged_ad_preparer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("whisper", "ad"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_OPTIONS, default=os.environ.get("AD_PROFILE", "auto"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, default=os.environ.get("ASR_MODEL") or None)
    parser.add_argument("--vlm-model", type=Path, default=os.environ.get("VLM_MODEL") or None)
    parser.add_argument("--vlm-mmproj", type=Path, default=os.environ.get("VLM_MMPROJ") or None)
    parser.add_argument("--vlm-executable", default=os.environ.get("VLM_EXECUTABLE") or None)
    parser.add_argument("--tts-url", default=os.environ.get("TTS_URL") or None)
    parser.add_argument("--tts-model", type=Path, default=os.environ.get("TTS_MODEL") or None)
    parser.add_argument("--ad-preparer-model", type=Path, default=os.environ.get("AD_PREPARER_MODEL") or None)
    parser.add_argument("--ad-preparer-executable", default=os.environ.get("AD_PREPARER_EXECUTABLE") or "llama-completion")
    parser.add_argument("--ad-preparer-reasoning-tokens", type=int,
                        default=os.environ.get("AD_PREPARER_REASONING_TOKENS") or None)
    parser.add_argument("--evidence-frames", type=int, default=os.environ.get("EVIDENCE_FRAMES") or None)
    parser.add_argument("--evidence-max-edge", type=int, default=os.environ.get("EVIDENCE_MAX_EDGE") or None)
    parser.add_argument("--event-followups", type=int, default=os.environ.get("EVENT_FOLLOWUPS") or None)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.role == "ad":
        mode = os.environ.get("AD_PREPARER", "auto")
        if mode not in {"auto", "0", "1"}:
            parser.error("AD_PREPARER must be auto, 0, or 1.")
        if mode == "0":
            args.ad_preparer_model = None
        elif not args.ad_preparer_model:
            available = staged_ad_preparer(args.models, args.ad_preparer_executable)
            if available:
                args.ad_preparer_model, args.ad_preparer_executable = available
        if args.ad_preparer_model:
            if not args.ad_preparer_model.is_file() or not shutil.which(args.ad_preparer_executable):
                parser.error("AD preparation requires a local GGUF and llama-completion.")
        elif mode == "1" or (mode != "0" and args.ad_preparer_reasoning_tokens):
            parser.error("Stage the AD preparer checkpoint and llama-completion or set AD_PREPARER=0.")
    return args


def write_narration_plan(work: Path) -> None:
    decisions = json.loads((work / "ad-decisions.json").read_text(encoding="utf-8"))
    cues = []
    for decision in decisions:
        if decision["decision"] != "AD" or not decision.get("tts_path"):
            continue
        audio = Path(decision["tts_path"]).resolve().relative_to(work.resolve())
        cues.append({
            "start": decision["cue_start"],
            "end": decision["cue_end"],
            "text": decision["description"],
            "audio": audio.as_posix(),
            "overlaps_dialogue": decision["overlaps_dialogue"],
            "dialogue_overlap_seconds": decision["dialogue_overlap_seconds"],
        })
    cues.sort(key=lambda cue: cue["start"])
    (work / "narration-plan.json").write_text(
        json.dumps({"schema_version": 1, "language": "en", "cues": cues}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.store = JobStore(args.data)
        self.host = socket.gethostname()
        self.cuda = not args.cpu_only and args.profile != "qwen3vl" and subprocess.check_output(
            [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
            text=True, timeout=60,
        ).strip() == "1"
        self.profile_names = {device: inference_profile(args.profile, device).name for device in ("cpu", "cuda")}
        self.model_args("cuda" if self.cuda else "cpu")
        self.stopping = False
        self.last_heartbeat = float("-inf")

    def heartbeat(self) -> None:
        now = time.monotonic()
        if now - self.last_heartbeat < 4:
            return
        args = self.args
        self.store.report_worker(args.role, self.host, args.profile, self.cuda, bool(args.ad_preparer_model))
        self.last_heartbeat = now

    def stop(self, *_signal) -> None:
        self.stopping = True

    def model_args(self, device: str) -> list:
        args = self.args
        profile = inference_profile(args.profile, device)
        if args.role == "whisper":
            return profile.asr_args(args.models, args.asr_model)
        return profile.ad_args(
            args.models, vlm=args.vlm_model, tts=args.tts_model, tts_url=args.tts_url, mmproj=args.vlm_mmproj,
            executable=args.vlm_executable, evidence_frames=args.evidence_frames,
            evidence_max_edge=args.evidence_max_edge, followups=args.event_followups,
        )

    def pipeline_args(self, job: dict, directory: Path) -> list:
        args = self.args
        video = self.store.source_path(job["media_id"])
        if video is None or not video.is_file():
            raise ValueError("The job source video is missing from the shared volume.")
        command = [
            sys.executable, "-m", "offline_ad_poc.pipeline", "--input", video,
            "--work-dir", directory / "work", "--device", job["selected_device"],
        ]

        command.extend(self.model_args(job["selected_device"]))
        if args.role == "whisper":
            return [*command, "--dialogue-only"]

        saved = self.store.dialogue(job["id"])
        if saved is None:
            raise ValueError("The Whisper result is missing from the job database.")
        dialogue_work, payload = saved
        # The database is the handoff. Materialize its versioned payload for the pipeline CLI.
        checkpoint = directory / "dialogue-input.json"
        checkpoint.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        command.extend([
            "--dialogue-input", checkpoint, "--analysis-audio", dialogue_work / "analysis-audio.wav",
            "--output-video", directory / "video-ad.mp4",
        ])
        if args.ad_preparer_model:
            command.extend([
                "--ad-preparer-model", args.ad_preparer_model, "--ad-preparer-executable", args.ad_preparer_executable,
            ])
            if args.ad_preparer_reasoning_tokens is not None:
                command.extend(["--ad-preparer-reasoning-tokens", str(args.ad_preparer_reasoning_tokens)])
        return command

    def run_job(self, job: dict) -> None:
        directory = self.store.root / job["attempt_dir"]
        log = directory / "job.log"
        try:
            directory.mkdir(parents=True)
            command = self.pipeline_args(job, directory)
            with log.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                         "TOKENIZERS_PARALLELISM": "false"},
                )
                deadline = time.monotonic() + 3600
                try:
                    while not self.stopping:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            self.heartbeat()
                            if not self.store.renew(job):
                                return
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Pipeline worker exceeded its one-hour limit.")
                    else:
                        # Keep the persisted claim available for retry after the lease expires.
                        return
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
            if code:
                raise ValueError(f"{self.args.role} process exited with code {code}. See the job log.")
            work = directory / "work"
            if self.args.role == "whisper":
                self.store.finish_dialogue(job, json.loads((work / "dialogue-result.json").read_text(encoding="utf-8")))
            else:
                write_narration_plan(work)
                self.store.finish_ad(job)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.store.fail(job, str(error))
            print(f"{job['id']}: {error}", file=sys.stderr, flush=True)

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        print(f"{self.args.role} worker ready on {self.host} (CUDA={self.cuda})", flush=True)
        while not self.stopping:
            self.heartbeat()
            if job := self.store.claim(self.args.role, self.cuda, self.host, self.profile_names):
                self.run_job(job)
            else:
                time.sleep(1)


if __name__ == "__main__":
    Worker(parse_args()).run()
