# Offline Audio Description POC for Lanta

This repository is the first offline proof of concept for generating a
selectable audio-description track from a real video scene. It supports CPU
and CUDA execution, uses Conda for all software dependencies, and makes no
external API calls.

Large inputs, generated videos, extracted frames, audio, model checkpoints,
and work directories are intentionally excluded from Git. Upload those
directly to Lanta storage instead.

## Pipeline

```text
real input video
  -> FFmpeg media inspection and analysis audio
  -> local Whisper word timestamps (Conda-native Transformers backend)
  -> speech-safe narration gaps
  -> OpenCV content-difference shots
  -> lightweight visual/audio/transcript scene grouping
  -> local visual backend: heuristic or Qwen2.5-VL
  -> AD or WAIT decision
  -> local Transformers VITS TTS
  -> FFmpeg soundtrack ducking and mixing
  -> MP4 with original and audio-description tracks
```

The `heuristic` backend tests the complete media path but does not produce
meaningful semantic descriptions. Use a locally staged Qwen2.5-VL checkpoint
for real visual descriptions.

## Repository contents

```text
offline_ad_poc/
  pipeline.py
  prepare_real_scene.py
scripts/
  create_conda_env.sh
  run_lanta.sh
  submit_lanta_gpu.sh
tests/
  test_pipeline.py
environment-cpu.yml
environment-gpu.yml
```

No input or output video is stored in this repository.

## 1. Create a Conda environment

No `venv` or `pip` command is required. From the repository root, create one
of the supplied environments.

On Lanta, load Conda first:

```bash
module load Mamba/23.11.0-0
```

CPU environment:

```bash
bash scripts/create_conda_env.sh cpu
conda activate offline-ad-cpu
```

GPU environment:

```bash
bash scripts/create_conda_env.sh gpu
conda activate offline-ad-gpu
```

The setup script creates a missing environment or updates the existing one
with `--prune`, so it is safe to run again after pulling dependency changes.

The GPU file uses `pytorch-cuda=12.1`. The cluster's NVIDIA driver must be
new enough for that runtime; no system CUDA toolkit is required inside the
environment. If Lanta requires a different approved CUDA runtime, change only
that version in `environment-gpu.yml` before creating the environment.

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
python -m unittest discover -s tests -v
```

## 2. Stage offline assets on Lanta

Upload these separately from Git:

```text
/path/to/input.mp4
/path/to/whisper-small.en-transformers/
/path/to/Qwen2.5-VL-3B-Instruct/
/path/to/mms-tts-eng/
```

The Whisper and TTS directories must be Hugging Face Transformers-format
checkpoints, not CTranslate2 conversions. The Qwen directory is only needed
for the GPU semantic-description run. A compatible TTS checkpoint is
`facebook/mms-tts-eng`, staged locally before the offline job. Model downloading
is disabled by default during pipeline execution.

## 3. Optional real-scene preparation

```bash
python -m offline_ad_poc.prepare_real_scene \
  --input /data/full-film.mp4 \
  --output inputs/real-scene-01.mp4 \
  --start 00:05:30 \
  --duration 120 \
  --title "Real scene 01" \
  --source-url "SOURCE_PAGE_URL" \
  --license "LICENSE_OR_RESEARCH_ACCESS_NOTE"
```

The generated scene and manifest are ignored by Git.

## 4. CPU smoke test

```bash
bash scripts/run_lanta.sh \
  inputs/real-scene-01.mp4 \
  outputs/real-scene-01-cpu.mp4 \
  outputs/real-scene-01-cpu-work \
  --device cpu \
  --asr-backend transformers \
  --asr-model /models/whisper-small.en-transformers \
  --visual-backend heuristic \
  --tts-backend transformers \
  --tts-model /models/mms-tts-eng \
  --maximum-gaps 6
```

## 5. GPU local-VLM test

```bash
bash scripts/run_lanta.sh \
  inputs/real-scene-01.mp4 \
  outputs/real-scene-01-qwen.mp4 \
  outputs/real-scene-01-qwen-work \
  --device auto \
  --asr-backend transformers \
  --asr-model /models/whisper-small.en-transformers \
  --visual-backend qwen25 \
  --vlm-model /models/Qwen2.5-VL-3B-Instruct \
  --tts-backend transformers \
  --tts-model /models/mms-tts-eng \
  --maximum-gaps 12
```

`--device auto` uses CUDA when detected and CPU otherwise.

## 6. Submit a Lanta GPU job

The supplied Slurm script uses the `gpu` partition, project `lt200246`, one
GPU, one Python task, and 16 CPU cores. It loads `Mamba/23.11.0-0` and runs the
`offline-ad-gpu` Conda environment.

By default, it expects this layout inside the repository:

```text
models/whisper-small.en-transformers/
models/Qwen2.5-VL-3B-Instruct/
models/mms-tts-eng/
inputs/real-scene-01.mp4
```

Submit from the repository root:

```bash
sbatch scripts/submit_lanta_gpu.sh \
  "$PWD/inputs/real-scene-01.mp4"
```

Optionally provide explicit output and work paths:

```bash
sbatch scripts/submit_lanta_gpu.sh \
  /lustrefs/disk/home/khtun/audio-description/inputs/real-scene-01.mp4 \
  /lustrefs/disk/home/khtun/audio-description/outputs/real-scene-01-ad.mp4 \
  /lustrefs/disk/home/khtun/audio-description/outputs/real-scene-01-work
```

For models stored elsewhere, override their locations when submitting:

```bash
ASR_MODEL=/lustrefs/disk/home/khtun/models/whisper-small.en-transformers \
VLM_MODEL=/lustrefs/disk/home/khtun/models/Qwen2.5-VL-3B-Instruct \
TTS_MODEL=/lustrefs/disk/home/khtun/models/mms-tts-eng \
sbatch scripts/submit_lanta_gpu.sh "$PWD/inputs/real-scene-01.mp4"
```

Monitor the job with:

```bash
squeue -u "$USER"
tail -f audio_ad.JOB_ID.out
```

## 7. Tests

```bash
python -m unittest discover -s tests -v
```

## Outputs

The final MP4 contains:

- the original video;
- the original soundtrack;
- a selectable English `Audio Description` track marked `visual_impaired`.

The ignored work directory contains transcript, gap, shot, scene, evidence,
decision, narration, WebVTT, and evaluation files.

## Current limitations

- The Conda Transformers Whisper, Qwen, and VITS paths still need validation on Lanta.
- Piper and system speech tools remain optional, but the strict Conda-only
  examples use a locally staged Transformers VITS model.
- The heuristic backend is for plumbing only.
- Scene segmentation is a lightweight baseline, not EARS4SEE reproduction.
- Character recognition and object tracking are not implemented yet.
- Music and sound-effect preservation requires a future audio-event model.
- Human review is required for factual correctness and AD usefulness.

