#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J ad-resolution-sweep
#SBATCH --output=audio_ad_resolution.%j.out
#SBATCH --error=audio_ad_resolution.%j.err

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 INPUT_VIDEO [SWEEP_OUTPUT_DIR]" >&2
  exit 2
fi

module load Mamba/23.11.0-0

repo_dir=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
conda_env=${CONDA_ENV:-offline-ad-gpu}
input_video=$1
input_name=$(basename "$input_video")
input_stem=${input_name%.*}
sweep_root=${2:-"$repo_dir/outputs/resolution-sweep-$input_stem"}
asr_model=${ASR_MODEL:-"$repo_dir/models/whisper-small.en-transformers"}
vlm_model=${VLM_MODEL:-"$repo_dir/models/Qwen2.5-VL-3B-Instruct"}
tts_model=${TTS_MODEL:-"$repo_dir/models/mms-tts-eng"}
maximum_gaps=${MAXIMUM_GAPS:-12}
resolution_values=${RESOLUTIONS:-"160 320 480 640"}
read -r -a resolutions <<< "$resolution_values"

for required_path in "$input_video" "$asr_model" "$vlm_model" "$tts_model"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input or model path does not exist: $required_path" >&2
    exit 1
  fi
done

if [[ ${#resolutions[@]} -lt 2 ]]; then
  echo "RESOLUTIONS must contain at least two positive pixel values" >&2
  exit 2
fi
for resolution in "${resolutions[@]}"; do
  if [[ ! "$resolution" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid resolution in RESOLUTIONS: $resolution" >&2
    exit 2
  fi
done

cd "$repo_dir"
mkdir -p "$sweep_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Repository: $repo_dir"
echo "Input: $input_video"
echo "Sweep output: $sweep_root"
echo "Evidence max-edge values: ${resolutions[*]}"
nvidia-smi

conda run --no-capture-output -n "$conda_env" \
  python -c 'import os, sys, torch; print("PyTorch:", torch.__version__); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("CUDA available:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)'

for resolution in "${resolutions[@]}"; do
  run_dir="$sweep_root/edge-$resolution"
  echo "Starting evidence resolution: max edge $resolution pixels"
  mkdir -p "$run_dir"
  conda run --no-capture-output -n "$conda_env" \
    python -m offline_ad_poc.pipeline \
    --input "$input_video" \
    --output-video "$run_dir/video-ad.mp4" \
    --work-dir "$run_dir/work" \
    --device cuda \
    --asr-backend transformers \
    --asr-model "$asr_model" \
    --visual-backend qwen25 \
    --vlm-model "$vlm_model" \
    --tts-backend transformers \
    --tts-model "$tts_model" \
    --maximum-gaps "$maximum_gaps" \
    --evidence-max-edge "$resolution"
done

conda run --no-capture-output -n "$conda_env" \
  python scripts/summarize_resolution_sweep.py "$sweep_root"

echo "Finished resolution sweep: $sweep_root"
