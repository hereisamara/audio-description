# Lanta Audio Description POC

This branch holds the offline batch pipeline, Slurm launchers, and
benchmarks for Lanta. The Docker app and review UI live on
[`main`](https://github.com/hereisamara/audio-description/tree/main).

## Setup

From the repository root, load the Conda module available on the cluster,
then create or update the project environment.

```bash
module load Mamba/23.11.0-0
bash scripts/create_conda_env.sh gpu
```

On a transfer node with internet access, stage the checkpoints:

```bash
conda run -n offline-ad-gpu hf download openai/whisper-small.en \
  --local-dir models/whisper-small.en-transformers
conda run -n offline-ad-gpu hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir models/Qwen3-VL-8B-Instruct
conda run -n offline-ad-gpu hf download facebook/mms-tts-eng \
  --local-dir models/mms-tts-eng
```

Copy your MP4 into `inputs/`. Supply your own Slurm allocation when you
submit. The compute job uses the staged models offline.

```bash
mkdir -p inputs
sbatch --account=YOUR_SLURM_ACCOUNT scripts/submit_lanta_gpu.sh inputs/video.mp4
```

Results go to `outputs/`. The MP4 contains the original audio and, when
a cue is placed, a selectable audio description track. The work directory
includes WebVTT cues, dialogue, timing, and VLM decisions.

For Studio import, put the source at `inputs/review/original.mp4` and
submit a packaged run:

```bash
sbatch --account=YOUR_SLURM_ACCOUNT scripts/submit_lanta_gpu.sh --studio-package inputs/review
```

Open the extracted result under **Import run** in Studio. To change
frame size, set `EVIDENCE_MAX_EDGE`. Zero keeps the source resolution.
Use `VLM_BACKEND=qwen25` with the older Qwen2.5-VL checkpoint when repeating
a baseline. The single-frame and pipeline resolution sweeps are under
`scripts/`. For the ASR benchmark, set `ASR_REFERENCE` to your own JSON
file containing `segments` with `start`, `end`, and `text` fields.
