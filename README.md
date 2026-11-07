# Offline Audio Description POC for Lanta

This repository is the first offline proof of concept for generating a
selectable audio-description track from a real video scene. It supports CPU
and CUDA execution and makes no external API calls.

Large inputs, generated videos, extracted frames, audio, model checkpoints,
and work directories are intentionally excluded from Git. Upload those
directly to Lanta storage instead.

## Pipeline

```text
real input video
  -> FFmpeg media inspection and analysis audio
  -> faster-whisper word timestamps
  -> speech-safe narration gaps
  -> PySceneDetect shots
  -> lightweight visual/audio/transcript scene grouping
  -> local visual backend: heuristic or Qwen2.5-VL
  -> AD or WAIT decision
  -> offline Piper/espeak TTS
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
  run_lanta.sh
tests/
  test_pipeline.py
requirements-core.txt
requirements-vlm.txt
```

No input or output video is stored in this repository.

## 1. Create the environment

Use Python 3.10 or 3.11.

```bash
python -m venv .venv-lanta
source .venv-lanta/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

For local VLM inference, install a CUDA-compatible PyTorch build for the CUDA
module available in the job, then run:

```bash
python -m pip install -r requirements-vlm.txt
```

## 2. Stage offline assets on Lanta

Upload these separately from Git:

```text
/path/to/input.mp4
/path/to/faster-whisper-small.en/
/path/to/Qwen2.5-VL-3B-Instruct/
/path/to/piper-voice.onnx
/path/to/piper-voice.onnx.json
```

Model downloading is disabled by default during pipeline execution.

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
  --asr-model /models/faster-whisper-small.en \
  --visual-backend heuristic \
  --tts-backend piper \
  --piper-model /models/piper/en_US-voice-medium.onnx \
  --maximum-gaps 6
```

## 5. GPU local-VLM test

```bash
bash scripts/run_lanta.sh \
  inputs/real-scene-01.mp4 \
  outputs/real-scene-01-qwen.mp4 \
  outputs/real-scene-01-qwen-work \
  --device auto \
  --asr-model /models/faster-whisper-small.en \
  --visual-backend qwen25 \
  --vlm-model /models/Qwen2.5-VL-3B-Instruct \
  --tts-backend piper \
  --piper-model /models/piper/en_US-voice-medium.onnx \
  --maximum-gaps 12
```

`--device auto` uses CUDA when detected and CPU otherwise.

## 6. Tests

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

- Qwen and Piper paths are implemented but still need validation on Lanta.
- The heuristic backend is for plumbing only.
- Scene segmentation is a lightweight baseline, not EARS4SEE reproduction.
- Character recognition and object tracking are not implemented yet.
- Music and sound-effect preservation requires a future audio-event model.
- Human review is required for factual correctness and AD usefulness.

