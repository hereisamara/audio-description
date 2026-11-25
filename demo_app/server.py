"""Local review and job server for the offline audio-description demo."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit
from uuid import uuid4

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline_ad_poc.tts_client import TtsError, available_voices, synthesize as synthesize_remote_tts


WEB = Path(__file__).resolve().parent / "web"
ARTIFACTS = {
    "ad-decisions.json", "ad-proposals.json", "ad-preparation.json", "descriptions.vtt", "ad-th.vtt", "ad-edits.json",
    "speech-and-gaps.json", "transcript.json", "shots.json", "scenes.json",
    "shot-boundary-validation.json", "runtime-summary.json", "dialogue-runtime-summary.json",
    "evaluation-report.json", "media.json", "source.json", "narration-plan.json",
}
IDENTIFIER = re.compile(r"[a-f0-9]{32}")
UPLOAD_LIMIT = 512 * 1024 * 1024


def command(*args: str | Path) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=120)
    if result.returncode:
        raise ValueError(result.stderr.strip()[-500:] or "Media command failed.")
    return result.stdout


def audio_streams(video: Path) -> list[dict]:
    result = json.loads(command(
        "ffprobe", "-v", "error", "-f", "mov", "-select_streams", "a", "-show_entries",
        "stream_tags=title,handler_name,language:stream_disposition=default,visual_impaired",
        "-of", "json", video,
    ))
    tracks = []
    for index, stream in enumerate(result.get("streams", [])):
        tags = stream.get("tags", {})
        label = tags.get("title") or tags.get("handler_name") or f"Audio {index + 1}"
        tracks.append({
            "index": index, "label": label, "language": tags.get("language"),
            "ad": bool(stream.get("disposition", {}).get("visual_impaired")) or "audio description" in label.lower(),
        })
    return tracks


def srt_timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    minutes, tail = divmod(ms, 60_000)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:{tail // 1000:02d},{tail % 1000:03d}"


def subtitle_text(rows: list) -> str:
    if not isinstance(rows, list) or len(rows) > 5000:
        raise ValueError("Invalid subtitle list.")
    subtitles = []
    for index, row in enumerate(rows, 1):
        start, end, text = row["start"], row["end"], row["text"]
        if (
            not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or isinstance(start, bool) or isinstance(end, bool)
            or not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= 86400 or not isinstance(text, str) or len(text) > 3000
        ):
            raise ValueError(f"Invalid cue {index}.")
        clean = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
        if clean:
            subtitles.append(f"{index}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{clean}\n")
    return "\n".join(subtitles)


@lru_cache(maxsize=1)
def mac_voices() -> dict[str, str]:
    if sys.platform != "darwin" or not Path("/usr/bin/say").is_file():
        return {}
    voices = {}
    for line in command("/usr/bin/say", "-v", "?").splitlines():
        match = re.match(r"(.+?)\s+(en_US|th_TH)\s+#", line)
        if match:
            voice, locale = match.groups()
            language = "th" if locale == "th_TH" else "en"
            if language not in voices or voice in {"Samantha", "Kanya"}:
                voices[language] = voice
    return voices


def voice_engine(language: str = "en") -> str | None:
    if sys.platform == "darwin" and Path("/usr/bin/say").is_file():
        voice = mac_voices().get(language)
        if voice:
            return "macOS " + voice
    if language == "en" and shutil.which("espeak-ng"):
        return "eSpeak NG"
    return None


def narration_mix(rows: list, directory: Path, audio: int, first_input: int,
                  language: str = "en", *, fit_speed: bool = False, tts_url: str | None = None) -> tuple[list, str]:
    engine = voice_engine(language) if not tts_url else None
    if not tts_url and not engine:
        raise ValueError("Install a Thai system voice in macOS Read & Speak." if language == "th"
                         else "Revoicing needs an English macOS voice or eSpeak NG.")
    if not 0 < len(rows) <= 100:
        raise ValueError("Revoicing needs between 1 and 100 AD cues.")
    inputs = []
    filters = []
    intervals = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["start"])):
        if intervals and row["start"] < intervals[-1][1]:
            raise ValueError(f"AD cue {index + 1} overlaps an earlier cue.")
        text = " ".join(row["text"].split())
        if not text:
            raise ValueError(f"AD cue {index + 1} is empty.")
        stem = directory / f"cue-{index:03d}.wav"
        if tts_url:
            duration = synthesize_remote_tts(tts_url, text, language, stem)
        else:
            if engine.startswith("macOS "):
                stem = stem.with_suffix(".aiff")
                cue_file = directory / f"cue-{index:03d}.txt"
                cue_file.write_text(text, encoding="utf-8")
                command("/usr/bin/say", "-v", engine.removeprefix("macOS "), "-o", stem, "-f", cue_file)
            else:
                command("espeak-ng", "-v", "en-us", "-w", stem, "--", text)
            duration = float(command("ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=nw=1:nk=1", stem))
        slot = row["end"] - row["start"]
        speed = max(1.0, duration / slot)
        if speed > (1.5 if fit_speed else 1.0):
            raise ValueError(f"AD cue {index + 1} needs {duration:.2f}s. Shorten its text or extend the cue.")
        start = row["start"]
        intervals.append((start, start + duration / speed))
        inputs.extend(("-i", stem))
        filters.append(
            f"[{first_input + index}:a:0]aresample=48000,aformat=channel_layouts=stereo,"
            f"atempo={speed:.7f},atrim=duration={slot:.6f},adelay={round(start * 1000)}:all=1[voice{index}]"
        )
    duck = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in intervals)
    filters.insert(0, (
        f"[0:a:{audio}]aresample=48000,aformat=channel_layouts=stereo,"
        f"volume='if(gt({duck},0),0.35,1)':eval=frame[bed]"
    ))
    voices = "".join(f"[voice{i}]" for i in range(len(rows)))
    filters.append(f"[bed]{voices}amix=inputs={len(rows) + 1}:duration=first:normalize=0,"
                   "alimiter=limit=0.84:level=false:latency=1[mix]")
    return inputs, ";".join(filters)


def render_narration(video: Path, destination: Path, audio: int, rows: list, language: str,
                     *, tts_url: str | None = None) -> None:
    subtitle_text(rows)
    tracks = audio_streams(video)
    if type(audio) is not int or not 0 <= audio < len(tracks) or tracks[audio]["ad"]:
        raise ValueError("Narration needs the original soundtrack.")
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        directory = Path(temporary)
        inputs, graph = narration_mix(rows, directory, audio, 1, language, fit_speed=True, tts_url=tts_url)
        output = directory / "mix.m4a"
        command("ffmpeg", "-nostdin", "-v", "error", "-i", video, *inputs,
                "-filter_complex", graph, "-map", "[mix]", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", output)
        output.replace(destination)


def combined_ad_rows(english: list, thai: list) -> list:
    subtitle_text(english)
    subtitle_text(thai)
    rows = [*english, *thai]
    boundaries = sorted({row[key] for row in rows for key in ("start", "end")})
    combined = []
    for start, end in zip(boundaries, boundaries[1:]):
        texts = [row["text"] for row in rows if row["start"] <= start < row["end"]]
        if texts:
            combined.append({"start": start, "end": end, "text": "\n".join(texts)})
    return combined


def render(video: Path, destination: Path, audio: int, captions: dict, revoice: bool = False,
           narration: Path | None = None, language: str | None = None,
           alternate_narration: tuple[Path, str] | None = None, *, tts_url: str | None = None) -> None:
    tracks = audio_streams(video)
    if type(audio) is not int or not 0 <= audio < len(tracks):
        raise ValueError("Select an available audio track.")
    with tempfile.TemporaryDirectory() as temporary:
        inputs = []
        labels = []
        english, thai = captions.get("ad", []), captions.get("adThai", [])
        subtitle_tracks = [
            ("ad", "AD captions (English)", "eng", english),
            ("ad-th", "AD captions (Thai)", "tha", thai),
            ("dialogue", "Dialogue", "eng", captions.get("dialogue", [])),
        ]
        if english and thai:
            subtitle_tracks.insert(2, ("ad-both", "AD captions (English + Thai)", "mul", combined_ad_rows(english, thai)))
        for name, label, code, rows in subtitle_tracks:
            if subtitles := subtitle_text(rows):
                path = Path(temporary) / f"{name}.srt"
                path.write_text(subtitles, encoding="utf-8")
                inputs.extend(("-i", path))
                labels.append((label, code))
        args = ["ffmpeg", "-nostdin", "-v", "error", "-f", "mov", "-i", video, *inputs]
        selected_audio = f"0:a:{audio}"
        if narration:
            args.extend(("-i", narration))
            selected_audio = f"{len(labels) + 1}:a:0"
        if alternate_narration:
            args.extend(("-i", alternate_narration[0]))
        if revoice:
            if tracks[audio]["ad"]:
                raise ValueError("Select the original soundtrack to revoice the edited AD.")
            speech_inputs, graph = narration_mix(
                captions.get("ad", []), Path(temporary), audio, len(labels) + 1, tts_url=tts_url,
            )
            args.extend((*speech_inputs, "-filter_complex", graph))
            selected_audio = "[mix]"
        args.extend(("-map", "0:v:0", "-map", selected_audio))
        if alternate_narration:
            args.extend(("-map", f"{len(labels) + 2}:a:0"))
        default_subtitle = next((index for index, (_, code) in enumerate(labels) if code == "mul"), 0)
        for index, (label, code) in enumerate(labels):
            args.extend(("-map", f"{index + 1}:s:0", f"-metadata:s:s:{index}", f"handler_name={label}",
                         f"-metadata:s:s:{index}", f"language={code}",
                         f"-disposition:s:{index}", "default" if index == default_subtitle else "0"))
        if narration or revoice:
            languages = [language, alternate_narration[1]] if alternate_narration else [language]
            for index, track_language in enumerate(languages):
                code, label = ("tha", "Thai") if track_language == "th" else ("eng", "English")
                args.extend((f"-metadata:s:a:{index}", f"language={code}",
                             f"-metadata:s:a:{index}", f"handler_name=Audio Description ({label})",
                             f"-disposition:a:{index}", "default+visual_impaired" if index == 0 else "visual_impaired"))
        else:
            args.extend(("-disposition:a:0", "default"))
        command(*args, "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
                "-movflags", "+faststart", destination)


class Demo(ThreadingHTTPServer):
    def __init__(self, address, data: Path, models: Path | None, repository: Path, example: Path | None,
                 profile: str = "auto", preparer_args: tuple[str | Path, ...] = (),
                 translator: tuple[Path, str] | None = None, split_workers: bool = False,
                 allowed_hosts: tuple[str, ...] = (), tts_url: str | None = None):
        super().__init__(address, Handler)
        self.data = data
        self.models = models
        self.profile = profile
        self.preparer_args = preparer_args
        self.translator = translator
        self.tts_url = tts_url
        self.repository = repository
        self.example = example
        self.allowed_hosts = {"localhost", "127.0.0.1", *(host.casefold() for host in allowed_hosts)}
        self.jobs: dict[str, dict] = {}
        self.media_sources: dict[str, Path] = {}
        self.lock = threading.Lock()
        self.worker = threading.Semaphore(1)
        self.data.mkdir(parents=True, exist_ok=True)
        self.job_store = None
        if split_workers:
            from offline_ad_poc.job_store import JobStore

            self.job_store = JobStore(data)

    def local_model_args(self, device: str) -> list:
        from offline_ad_poc.profiles import inference_profile

        profile = inference_profile(self.profile, device)
        overrides = {key: Path(value) if (value := os.environ.get(env)) else None
                     for key, env in (("vlm", "VLM_MODEL"), ("mmproj", "VLM_MMPROJ"), ("tts", "TTS_MODEL"))}
        asr = Path(value) if (value := os.environ.get("ASR_MODEL")) else None
        return [*profile.asr_args(self.models, asr), *profile.ad_args(
            self.models, **overrides, tts_url=self.tts_url, executable=os.environ.get("VLM_EXECUTABLE") or None,
        )]

    def voices(self) -> dict[str, str]:
        if self.tts_url:
            return available_voices(self.tts_url)
        return {language: voice for language in ("en", "th") if (voice := voice_engine(language))}

    def capabilities(self) -> dict:
        tts_detail = None
        try:
            voices = self.voices()
        except TtsError as error:
            voices = {}
            tts_detail = str(error)
        if self.tts_url and "en" not in voices and not tts_detail:
            tts_detail = "Stage the English checkpoint in the TTS container."
        report = {"host": socket.gethostname(), "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                  "voice": voices.get("en"), "thai_voice": voices.get("th"),
                  "tts_service": bool(self.tts_url), "tts_detail": tts_detail,
                  "thai_translation": bool(self.translator),
                  "models": False, "cuda": False, "worker": False, "example": self.example is not None,
                  "profile": self.profile, "ad_preparer": bool(self.preparer_args)}
        if self.job_store is not None:
            roles = self.job_store.active_workers()
            ready = {"whisper", "ad"} <= roles.keys()
            report.update(
                split_workers=True, worker=ready and report["ffmpeg"] and not tts_detail,
                models=ready, cuda=any(worker["cuda"] for worker in roles.values()),
                workers={role: {"host": worker["host"], "cuda": bool(worker["cuda"])}
                         for role, worker in roles.items()},
                profile=roles.get("ad", {}).get("profile", self.profile),
                ad_preparer=bool(roles.get("ad", {}).get("preparer")),
                detail=tts_detail or ("Split workers ready." if ready else "Waiting for Whisper and AD workers."),
            )
            return report
        if not self.models:
            report["detail"] = "Start the app with --models to enable generation."
            return report
        try:
            output = command(sys.executable, "-c", "import cv2, numpy, torch, transformers; "
                             "print('cuda' if torch.cuda.is_available() else 'cpu')")
        except (ValueError, subprocess.TimeoutExpired):
            report["detail"] = "Start the app in a Conda environment with the pipeline dependencies."
            return report
        cuda = output.strip().splitlines()[-1] == "cuda" and self.profile != "qwen3vl"
        errors = {}
        for device in ("cpu", "cuda") if cuda else ("cpu",):
            try:
                self.local_model_args(device)
            except ValueError as error:
                errors[device] = str(error)
        report["cuda"] = cuda and "cuda" not in errors
        report["models"] = report["cuda"] or "cpu" not in errors
        report["worker"] = report["models"] and report["ffmpeg"] and not tts_detail
        report["detail"] = tts_detail or "; ".join(errors.values()) or "Local checkpoints ready."
        return report

    def run_job(self, job_id: str, video: Path, device: str) -> None:
        directory = self.data / job_id
        with self.worker:
            self.jobs[job_id]["state"] = "running"
            args = [
                sys.executable, "-m", "offline_ad_poc.pipeline", "--input", video,
                "--output-video", directory / "video-ad.mp4", "--work-dir", directory / "work",
                "--device", device,
            ]
            try:
                args.extend(self.local_model_args(device))
                args.extend(self.preparer_args)
                with (directory / "job.log").open("w") as log:
                    result = subprocess.run(
                        args, cwd=self.repository, stdout=log, stderr=subprocess.STDOUT, timeout=3600,
                    )
                self.jobs[job_id]["state"] = "finished" if result.returncode == 0 else "failed"
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                self.jobs[job_id].update(state="failed", error=str(error))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        host = self.headers.get("Host", "").split(":", 1)[0].casefold()
        if host not in self.server.allowed_hosts:
            self.send_error(403, "Open the app on localhost or through an SSH tunnel.")
            return False
        return True

    def reply(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        length = path.stat().st_size
        first, last = 0, length - 1
        if requested := self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            suffix = re.fullmatch(r"bytes=-(\d+)", requested)
            if match:
                first = int(match[1])
                last = min(last, int(match[2])) if match[2] else last
            elif suffix:
                first = max(0, length - int(suffix[1]))
            if first > last or (not match and not suffix):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{length}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {first}-{last}/{length}")
        self.end_headers()
        try:
            with path.open("rb") as source:
                source.seek(first)
                remaining = last - first + 1
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parts = urlsplit(self.path).path.strip("/").split("/")
        store = self.server.job_store
        if parts == ["api", "capabilities"]:
            self.reply(self.server.capabilities())
        elif store is not None and parts == ["api", "jobs"]:
            self.reply({"jobs": [self.public_job(row) for row in store.recent()]})
        elif store is not None and len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
            self.serve_split_job(parts[2:])
        elif len(parts) == 3 and parts[:2] == ["api", "jobs"] and parts[2] in self.server.jobs:
            job = dict(self.server.jobs[parts[2]])
            directory = self.server.data / parts[2]
            log = directory / "job.log"
            if log.is_file():
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 5000))
                    job["log"] = handle.read().decode(errors="replace")
            if job["state"] == "finished":
                job["files"] = self.manifest(directory, f"/result/{parts[2]}")
            self.reply(job)
        elif parts == ["api", "example"] and self.server.example:
            self.reply({"files": self.manifest(self.server.example, "/sample")})
        elif len(parts) == 2 and parts[0] == "sample" and self.server.example:
            self.serve_package(self.server.example, parts[1])
        elif (store is not None and len(parts) in {3, 4} and parts[0] == "result"
              and (job := store.get(parts[1]))):
            self.serve_split_result(job, parts[2:])
        elif len(parts) == 3 and parts[0] == "result" and IDENTIFIER.fullmatch(parts[1]):
            self.serve_package(self.server.data / parts[1], parts[2])
        elif parts[0] in {"api", "sample", "result"}:
            self.send_error(404)
        else:
            super().do_GET()

    @staticmethod
    def public_job(row: dict) -> dict:
        devices = {
            role: {"device": row[f"{role}_device"], "host": row[f"{role}_host"]}
            for role in ("whisper", "ad") if row[f"{role}_device"]
        }
        return {
            **{key: row[key] for key in ("id", "title", "state", "stage", "host", "requested", "profile", "error")},
            "device": row["ad_device"] or row["whisper_device"] or "pending",
            "devices": devices,
            "fallback": row["requested"] == "cuda" and any(x["device"] == "cpu" for x in devices.values()),
            "dialogue_ready": row["stage"] != "whisper",
            "original": {"name": "original.mp4", "url": f"/result/{row['id']}/original.mp4"},
        }

    def serve_split_job(self, parts: list[str]) -> None:
        store = self.server.job_store
        row = store.get(parts[0])
        if row is None:
            self.send_error(404)
            return
        if parts[1:] == ["dialogue"]:
            saved = store.dialogue(row["id"])
            if saved is None:
                self.reply({"error": "Dialogue is not ready."}, 409)
            else:
                self.reply(saved[1])
            return
        if len(parts) != 1:
            self.send_error(404)
            return
        job = self.public_job(row)
        if row["attempt_dir"]:
            log = self.server.data / row["attempt_dir"] / "job.log"
            if log.is_file():
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 5000))
                    job["log"] = handle.read().decode(errors="replace")
        if row["state"] == "finished":
            prefix = f"/result/{row['id']}"
            job["files"] = [*self.manifest(self.server.data / row["result_dir"], prefix), job["original"]]
            saved = store.dialogue(row["id"])
            if saved and (saved[0] / "runtime-summary.json").is_file():
                job["files"].append({
                    "name": "dialogue-runtime-summary.json",
                    "url": f"{prefix}/dialogue-runtime-summary.json",
                })
        self.reply(job)

    def serve_split_result(self, row: dict, parts: list[str]) -> None:
        if parts == ["original.mp4"]:
            source = self.server.job_store.source_path(row["media_id"])
            if source is None:
                self.send_error(404)
                return
            self.file(source, "video/mp4")
        elif parts == ["dialogue-runtime-summary.json"]:
            saved = self.server.job_store.dialogue(row["id"])
            if saved is None:
                self.send_error(404)
                return
            self.file(saved[0] / "runtime-summary.json", "application/json")
        elif row["state"] != "finished":
            self.send_error(404)
        else:
            directory = self.server.data / row["result_dir"]
            if len(parts) == 1:
                self.serve_package(directory, parts[0])
            elif parts[0] == "narration" and re.fullmatch(r"ad-\d+-try-\d+\.wav", parts[1]):
                self.file(directory / "work" / "narration" / parts[1], "audio/wav")
            else:
                self.send_error(404)

    @staticmethod
    def manifest(directory: Path, prefix: str) -> list[dict]:
        return [
            {"name": name, "url": f"{prefix}/{name}"}
            for name in sorted(ARTIFACTS | {"video-ad.mp4", "original.mp4", "dialogue.vtt", "reference-ad.vtt"})
            if (directory / name).is_file() or (directory / "work" / name).is_file()
        ]

    def serve_package(self, directory: Path, name: str) -> None:
        fixed = {"video-ad.mp4", "original.mp4", "dialogue.vtt", "reference-ad.vtt"}
        if name not in ARTIFACTS | fixed and not re.fullmatch(
            r"render-[a-f0-9]{32}\.mp4|narration-(?:en|th)\.m4a", name,
        ):
            self.send_error(404)
            return
        path = directory / name
        if not path.is_file():
            path = directory / "work" / name
        self.file(path, "audio/mp4" if name.endswith(".m4a") else self.guess_type(name))

    def translate_ad(self, rows: list) -> None:
        if not self.server.translator:
            raise ValueError("Start Studio with a local Qwen3 translation model.")
        subtitle_text(rows)
        if not 0 < len(rows) <= 100:
            raise ValueError("Select between 1 and 100 English AD cues to translate.")
        model, executable = self.server.translator
        result = subprocess.run(
            [sys.executable, "-m", "offline_ad_poc.translate_ad",
             "--model", model, "--executable", executable],
            cwd=self.server.repository, input=json.dumps(rows), text=True, capture_output=True, timeout=900,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip().splitlines()[-1] if result.stderr else "Thai translation failed.")
        self.reply({"cues": json.loads(result.stdout)})

    def packaged_video(self, payload: dict) -> Path:
        path = payload["url"]
        if self.server.example and path in {"/sample/video-ad.mp4", "/sample/original.mp4"}:
            return self.server.example / path.split("/")[-1]
        match = re.fullmatch(r"/result/([a-f0-9]{32})/(video-ad\.mp4|original\.mp4)", path)
        if match and self.server.job_store is not None:
            row = self.server.job_store.get(match[1])
            if row:
                if match[2] == "original.mp4":
                    return self.server.job_store.source_path(row["media_id"])
                if row["state"] == "finished":
                    return self.server.data / row["result_dir"] / match[2]
        if match and self.server.jobs.get(match[1], {}).get("state") == "finished":
            return self.server.data / match[1] / match[2]
        raise ValueError("Select a video from an imported sample or completed job.")

    def do_POST(self) -> None:
        expected_origin = f"http://{self.headers.get('Host', '')}"
        origin = self.headers.get("Origin")
        if origin and origin != expected_origin:
            self.reply({"error": "Cross-origin requests are disabled."}, 403)
            return
        try:
            parts = urlsplit(self.path).path.strip("/").split("/")
            size = int(self.headers.get("Content-Length", "0"))
            if parts == ["api", "translate-ad"]:
                if not 0 < size <= 4 * 1024 * 1024:
                    raise ValueError("Invalid translation request.")
                self.translate_ad(json.loads(self.rfile.read(size))["cues"])
            elif parts == ["api", "media"]:
                if not 0 < size <= UPLOAD_LIMIT:
                    raise ValueError("Video uploads must be between 1 byte and 512 MiB.")
                media_id = uuid4().hex
                directory = self.server.data / media_id
                directory.mkdir()
                video = directory / "source.mp4"
                try:
                    if self.headers.get_content_type() == "application/json":
                        if size > 2048:
                            raise ValueError("Invalid media reference.")
                        video = self.packaged_video(json.loads(self.rfile.read(size)))
                    else:
                        with video.open("wb") as handle:
                            remaining = size
                            while remaining:
                                data = self.rfile.read(min(1024 * 1024, remaining))
                                if not data:
                                    raise ValueError("Incomplete upload.")
                                handle.write(data)
                                remaining -= len(data)
                    tracks = audio_streams(video)
                    if not tracks:
                        raise ValueError("An audio stream is required.")
                    if self.server.job_store is not None:
                        self.server.job_store.register_media(media_id, video)
                except Exception:
                    (directory / "source.mp4").unlink(missing_ok=True)
                    directory.rmdir()
                    raise
                if self.server.job_store is None:
                    self.server.media_sources[media_id] = video
                self.reply({"id": media_id, "tracks": tracks})
            elif len(parts) == 4 and parts[:2] == ["api", "media"] and IDENTIFIER.fullmatch(parts[2]):
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("Invalid request size.")
                options = json.loads(self.rfile.read(size))
                directory = self.server.data / parts[2]
                video = (self.server.job_store.source_path(parts[2]) if self.server.job_store is not None
                         else self.server.media_sources.get(parts[2], directory / "source.mp4"))
                if video is None or not video.is_file():
                    self.reply({"error": "Upload the video again."}, 404)
                elif parts[3] == "render":
                    name = f"render-{uuid4().hex}.mp4"
                    language = options.get("narration_language")
                    narration = None
                    alternate = None
                    if language is not None:
                        if language not in {"en", "th"} or options.get("revoice"):
                            raise ValueError("Select one AD audio language.")
                        narration = directory / f"narration-{language}.m4a"
                        if not narration.is_file():
                            raise ValueError("Generate the selected AD audio before exporting.")
                        alternate_language = options.get("alternate_narration_language")
                        if alternate_language in {"en", "th"} and language != alternate_language:
                            other = directory / f"narration-{alternate_language}.m4a"
                            if not other.is_file():
                                raise ValueError("Generate both AD audio languages before exporting both tracks.")
                            alternate = (other, alternate_language)
                    render(video, directory / name, options["audio"], options.get("captions", {}),
                           revoice=options.get("revoice") is True, narration=narration, language=language,
                           alternate_narration=alternate, tts_url=self.server.tts_url)
                    self.reply({"url": f"/result/{parts[2]}/{name}"})
                elif parts[3] == "narration":
                    language = options["language"]
                    if language not in {"en", "th"}:
                        raise ValueError("Select English or Thai.")
                    name = f"narration-{language}.m4a"
                    voice = self.server.voices().get(language)
                    if self.server.tts_url and not voice:
                        raise TtsError(f"The {language} voice checkpoint is unavailable.")
                    render_narration(video, directory / name, options["audio"], options["cues"], language,
                                     tts_url=self.server.tts_url)
                    self.reply({"url": f"/result/{parts[2]}/{name}?{uuid4().hex}", "voice": voice})
                elif parts[3] == "generate":
                    self.submit(video, options, parts[2])
                else:
                    self.send_error(404)
            else:
                self.send_error(404)
        except TtsError as error:
            self.reply({"error": str(error)}, 503)
        except (ValueError, TypeError, KeyError, AttributeError, OSError, subprocess.TimeoutExpired) as error:
            self.reply({"error": str(error)}, 400)

    def submit(self, video: Path, options: dict, media_id: str) -> None:
        requested = options.get("device", "auto")
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("Select Auto, CPU, or CUDA.")
        caps = self.server.capabilities()
        if not caps["worker"]:
            raise ValueError(caps["detail"])
        if self.server.job_store is not None:
            fallback = options.get("fallback") is True
            if requested == "cuda" and not fallback and not all(role["cuda"] for role in caps["workers"].values()):
                raise ValueError("One of the workers requires CPU fallback.")
            title = options.get("title", "")
            title = title.strip()[:140] if isinstance(title, str) else ""
            job = self.server.job_store.enqueue(
                media_id, title or video.name, caps["host"], caps["profile"], requested, fallback,
            )
            self.reply(self.public_job(job), 202)
            return
        device = requested
        if requested == "auto":
            device = "cuda" if caps["cuda"] else "cpu"
        elif requested == "cuda" and not caps["cuda"]:
            if options.get("fallback") is not True:
                raise ValueError("CUDA is unavailable on the app host. Select CPU or allow fallback.")
            device = "cpu"
        from offline_ad_poc.profiles import inference_profile

        self.server.local_model_args(device)
        profile = inference_profile(self.server.profile, device).name
        with self.server.lock:
            if any(job["state"] in {"queued", "running"} for job in self.server.jobs.values()):
                raise ValueError("A generation job is already active.")
            job_id = uuid4().hex
            (self.server.data / job_id).mkdir()
            job = {"state": "queued", "host": caps["host"], "requested": requested, "device": device,
                   "profile": profile,
                   "fallback": device == "cpu" and requested == "cuda"}
            self.server.jobs[job_id] = job
            threading.Thread(target=self.server.run_job, args=(job_id, video, device), daemon=True).start()
        self.reply({"id": job_id, **job}, 202)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data", type=Path, default=Path(tempfile.gettempdir()) / "offline-ad-demo")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--profile", choices=["auto", "qwen3vl", "qwen3vl8b", "qwen25", "smolvlm"], default="auto")
    parser.add_argument("--ad-preparer-model", type=Path)
    parser.add_argument("--ad-preparer-executable", default=os.environ.get("AD_PREPARER_EXECUTABLE") or "llama-completion")
    parser.add_argument("--ad-preparer-reasoning-tokens", type=int,
                        default=os.environ.get("AD_PREPARER_REASONING_TOKENS") or None)
    parser.add_argument("--translation-model", type=Path, help="Local Qwen3 GGUF for English-to-Thai captions")
    parser.add_argument("--translation-executable", help="Local llama-completion for translation")
    parser.add_argument("--repository", type=Path, default=WEB.parents[1])
    parser.add_argument("--example", type=Path)
    parser.add_argument("--split-workers", action="store_true", help="Use the shared Whisper and AD job database")
    parser.add_argument("--tts-url", default=os.environ.get("TTS_URL") or None, help="Internal speech service URL")
    parser.add_argument("--allowed-host", action="append", default=[], help="Additional trusted HTTP host name")
    args = parser.parse_args()
    if not __package__ and (args.models or args.split_workers):
        sys.path.insert(0, str(args.repository.resolve()))
    preparer_args: tuple[str | Path, ...] = ()
    if args.models and not args.split_workers and not args.ad_preparer_model and os.environ.get("AD_PREPARER") != "0":
        from offline_ad_poc.profiles import staged_ad_preparer

        available = staged_ad_preparer(args.models.resolve(), args.ad_preparer_executable)
        if available:
            args.ad_preparer_model, args.ad_preparer_executable = available
    if args.ad_preparer_reasoning_tokens is not None and not 0 <= args.ad_preparer_reasoning_tokens <= 512:
        parser.error("--ad-preparer-reasoning-tokens must be between 0 and 512")
    if (not args.split_workers and args.ad_preparer_reasoning_tokens and not args.ad_preparer_model
            and os.environ.get("AD_PREPARER") != "0"):
        parser.error("AD reasoning requires a staged editor or --ad-preparer-model")
    if args.ad_preparer_model:
        executable = shutil.which(args.ad_preparer_executable)
        model = args.ad_preparer_model.resolve()
        if not args.models or not model.is_file() or not executable:
            parser.error("AD preparation requires --models, a local GGUF, and an executable llama-completion")
        preparer_args = (
            "--ad-preparer-model", model, "--ad-preparer-executable", executable,
        )
        if args.ad_preparer_reasoning_tokens is not None:
            preparer_args += ("--ad-preparer-reasoning-tokens", str(args.ad_preparer_reasoning_tokens))
    translation_model = args.translation_model or args.ad_preparer_model
    translator = None
    if translation_model:
        executable = shutil.which(args.translation_executable or args.ad_preparer_executable)
        if not translation_model.is_file() or not executable:
            parser.error("Thai translation requires a local GGUF and an executable llama-completion")
        translator = (translation_model.resolve(), executable)
    server = Demo(
        (args.host, args.port), args.data.resolve(), args.models.resolve() if args.models else None,
        args.repository.resolve(), args.example.resolve() if args.example else None, args.profile,
        preparer_args, translator, args.split_workers, tuple(args.allowed_host), args.tts_url,
    )
    print(f"AD Review Studio at http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()
