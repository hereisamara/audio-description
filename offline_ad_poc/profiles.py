"""Local inference profiles shared by the Studio and worker launchers."""

from dataclasses import dataclass
from pathlib import Path
import shutil


PROFILE_OPTIONS = ("auto", "qwen3vl", "qwen3vl8b", "qwen25", "smolvlm")
QWEN3_MODEL = "qwen3-vl-2b-instruct-gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
AD_PREPARER_CHECKPOINT = "Qwen3-1.7B-Q4_K_M.gguf"
QWEN3_MMPROJ = "qwen3-vl-2b-instruct-gguf/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"


@dataclass(frozen=True)
class Profile:
    name: str
    asr: str
    vlm: str
    tts: str
    evidence_frames: int
    evidence_max_edge: int
    followups: int

    def asr_args(self, models: Path, override: Path | None = None) -> list:
        model = override or models / self.asr
        require_file(model / "config.json")
        return ["--asr-backend", "transformers", "--asr-model", model]

    def ad_args(
        self, models: Path, *, vlm: Path | None = None, tts: Path | None = None,
        tts_url: str | None = None, mmproj: Path | None = None, executable: str | None = None,
        evidence_frames: int | None = None, evidence_max_edge: int | None = None, followups: int | None = None,
    ) -> list:
        vlm = vlm or models / self.vlm
        args = ["--visual-backend", self.name, "--vlm-model", vlm,
                "--evidence-frames", str(self.evidence_frames if evidence_frames is None else evidence_frames),
                "--evidence-max-edge", str(self.evidence_max_edge if evidence_max_edge is None else evidence_max_edge),
                "--event-followups", str(self.followups if followups is None else followups)]
        if tts_url:
            args.extend(["--tts-backend", "http", "--tts-url", tts_url])
        else:
            tts = tts or models / self.tts
            require_file(tts / "config.json")
            args.extend(["--tts-backend", "transformers", "--tts-model", tts])
        if self.name == "qwen3vl":
            mmproj = mmproj or models / QWEN3_MMPROJ
            require_file(vlm)
            require_file(mmproj)
            executable = find_vlm_executable(models, executable)
            args.extend(["--vlm-mmproj", mmproj, "--vlm-executable", executable])
        else:
            require_file(vlm / "config.json")
        return args


PROFILES = {
    "qwen3vl": Profile("qwen3vl", "whisper-tiny.en", QWEN3_MODEL, "mms-tts-eng-local", 2, 384, 0),
    "qwen3vl8b": Profile("qwen3vl8b", "whisper-small.en-transformers", "Qwen3-VL-8B-Instruct", "mms-tts-eng", 6, 0, 2),
    "qwen25": Profile("qwen25", "whisper-small.en-transformers", "Qwen2.5-VL-3B-Instruct", "mms-tts-eng", 6, 0, 2),
    "smolvlm": Profile("smolvlm", "whisper-tiny.en", "smolvlm-256m", "mms-tts-eng-local", 4, 384, 0),
}


def inference_profile(name: str, device: str) -> Profile:
    if name in {"auto", "qwen3vl8b"}:
        name = "qwen3vl8b" if device == "cuda" else "qwen3vl"
    return PROFILES[name]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing local file: {path}")


def find_vlm_executable(models: Path, override: str | None) -> str:
    executable = shutil.which(override or "llama-mtmd-cli")
    if executable is None and not override:
        executable = shutil.which(str(models.parent / "outputs/cpu-llm-runtime/bin/llama-mtmd-cli"))
    if executable is None:
        raise ValueError("Qwen3-VL requires the local Conda llama-mtmd-cli. Set VLM_EXECUTABLE to its path.")
    return executable


def staged_ad_preparer(models: Path, executable: str) -> tuple[Path, str] | None:
    model = next(
        (candidate for candidate in (models / "cpu-llm" / AD_PREPARER_CHECKPOINT,
                                     models / AD_PREPARER_CHECKPOINT) if candidate.is_file()),
        None,
    )
    if model is None:
        return None
    resolved = shutil.which(executable)
    if resolved is None and executable == "llama-completion":
        resolved = shutil.which(str(models.parent / "outputs/cpu-llm-runtime/bin/llama-completion"))
    return (model, resolved) if resolved else None
