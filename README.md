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
  -> PySceneDetect ContentDetector when available; logged OpenCV fallback otherwise
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
Qwen uses Transformers' PIL image processor, so Torchvision is not required.

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

The pipeline also writes a dedicated live log inside the work directory:

```bash
tail -f outputs/real-scene-01-work/runtime.log
```

The job performs a CUDA preflight before loading the models. It records
`CUDA_VISIBLE_DEVICES`, the PyTorch build, PyTorch's CUDA runtime, availability,
and visible device count in the output log. If this preflight fails after an
environment update, inspect the installed variants with:

```bash
conda list -n offline-ad-gpu | grep -E 'pytorch|cuda|cudnn'
```

## 7. Runtime profiling and optimization logs

Every run records the selected implementation, fallback reason, elapsed time,
and compute resources for each pipeline stage. The work directory contains:

- `runtime.log`: human-readable live events, including method selection,
  fallbacks, stage start/end, elapsed time, CPU time, and CUDA peak memory;
- `runtime-events.jsonl`: one structured record appended immediately after each
  stage, so completed timings remain available if a later stage fails;
- `runtime-summary.json`: completed-run hardware information, totals grouped by
  compute device, the five slowest stages, and all stage records;
- `evaluation-report.json`: quality metrics plus paths to the three runtime
  profiling files.

Method lines make optional fallbacks explicit. For example, because
PySceneDetect is not part of the minimal Conda environments, the normal log is:

```text
fallback stage=shot_detection from=scenedetect.ContentDetector to=opencv_frame_difference reason=...
method stage=shot_detection backend=opencv_frame_difference threshold=27.000 shots=14
```

Each structured stage record includes:

- `stage`, `method`, `status`, and any stage-specific counts;
- `elapsed_seconds`: wall-clock latency;
- `cpu_seconds`: Python plus completed child-process CPU time, including FFmpeg;
- `average_cpu_cores`: `cpu_seconds / elapsed_seconds`, which may be greater
  than one for multithreaded work;
- `compute_device`, `uses_cpu`, and `uses_gpu`;
- `process_peak_rss_mb`: cumulative process peak memory observed by that point;
- CUDA stages additionally record PyTorch allocated/reserved memory at the
  start and end and peak allocated/reserved memory.

Inspect the selected methods and fallback decisions:

```bash
grep -E 'method|fallback' outputs/real-scene-01-work/runtime.log
```

Print the slowest completed stages without installing an extra analysis tool:

```bash
python -c 'import json,sys; d=json.load(open(sys.argv[1])); [print("{:9.3f}s  {:4s}  {}  {}".format(x["elapsed_seconds"], x["compute_device"], x["stage"], x.get("method", ""))) for x in d["slowest_stages"]]' \
  outputs/real-scene-01-work/runtime-summary.json
```

The CUDA fields measure PyTorch memory and synchronize timed CUDA stages so
their wall time includes completed GPU work. They do not continuously sample
GPU utilization; use the Slurm job's monitoring facilities or NVIDIA tooling
if utilization-over-time traces are required later.

## 8. Compare VLM evidence resolutions

For the lowest-risk first experiment, benchmark only Qwen2.5-VL with one fixed
video frame. This path does not load Whisper or TTS, does not create output
videos, and runs every resolution sequentially with one Qwen model on one GPU:

```bash
sbatch scripts/submit_lanta_vlm_frame_sweep.sh \
  "$PWD/inputs/real-scene-01.mp4" \
  "$PWD/outputs/vlm-frame-sweep-real-scene-01-v1"
```

The default frame is at 31.72 seconds and the default long-edge resolutions are
160, 320, 480, and 640 pixels. Override them without editing the script:

```bash
FRAME_TIMESTAMP=31.72 RESOLUTIONS="160 320 480 640" \
sbatch scripts/submit_lanta_vlm_frame_sweep.sh \
  "$PWD/inputs/real-scene-01.mp4" \
  "$PWD/outputs/vlm-frame-sweep-real-scene-01-v2"
```

The output directory contains the exact image tested at every resolution plus
`vlm-frame-resolution-results.csv` and `vlm-frame-resolution-results.json`.
They record model initialization separately from each inference, Qwen visual
tokens, synchronized wall time, peak and incremental GPU allocation, and the
complete AD/WAIT result. One excluded warm-up inference at each resolution
removes first-use and input-shape overhead, and the reported time is the median
of three sequential measured inferences per resolution. The report also flags
an `AD` response with an empty description as invalid and records whether
repeated outputs were consistent. Use this isolated benchmark before the full
pipeline sweep below.

### Full-pipeline resolution sweep

The resolution sweep runs the same video at four evidence-frame sizes while
keeping the model, four-frame sampling, prompt, audio, and pipeline parameters
fixed. For a 640x480 source, the defaults test long edges of 160, 320, 480,
and 640 pixels. The sampler never upscales a source frame, so do not add 720 or
1080 to a 640x480 source: those settings cannot add visual information.

Submit one sequential four-run job:

```bash
sbatch scripts/submit_lanta_resolution_sweep.sh \
  "$PWD/inputs/real-scene-01.mp4" \
  "$PWD/outputs/resolution-sweep-real-scene-01-v1"
```

Override the tested long-edge values when the source video has a different
native resolution:

```bash
RESOLUTIONS="240 360 540 720 1080" \
sbatch scripts/submit_lanta_resolution_sweep.sh \
  "$PWD/inputs/example-1080p.mp4" \
  "$PWD/outputs/resolution-sweep-example-1080p-v1"
```

Each run has its own final video, work artifacts, and runtime logs. The job
also writes `resolution-comparison.csv` and `resolution-comparison.json` with:

- total, VLM initialization, VLM decision, and TTS times;
- VLM-stage CUDA peak allocation and Qwen visual-token count;
- AD/WAIT decisions and generated descriptions;
- speedup, decision agreement, and description-token overlap relative to the
  largest tested resolution;
- an empty `manual_quality_score_1_to_5` column for human review.

Description-token overlap measures output stability, not semantic correctness.
Choose the smallest resolution whose descriptions retain the important visible
actions and details when compared with the source video and any human AD ground
truth. Inspect the report after completion with:

```bash
column -s, -t \
  outputs/resolution-sweep-real-scene-01-v1/resolution-comparison.csv | less -S
```

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

## Outputs

The final MP4 contains:

- the original video;
- the original soundtrack;
- a selectable English `Audio Description` track marked `visual_impaired`.

The ignored work directory contains transcript, gap, shot, scene, evidence,
decision, narration, WebVTT, evaluation, and runtime profiling files.

## Current limitations

- The Conda Transformers Whisper, Qwen, and VITS paths still need validation on Lanta.
- Piper and system speech tools remain optional, but the strict Conda-only
  examples use a locally staged Transformers VITS model.
- The heuristic backend is for plumbing only.
- Scene segmentation is a lightweight baseline, not EARS4SEE reproduction.
- Character recognition and object tracking are not implemented yet.
- Music and sound-effect preservation requires a future audio-event model.
- Human review is required for factual correctness and AD usefulness.

