#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -A lt200246
#SBATCH -J vlm-frame-sweep
#SBATCH --output=vlm_frame_sweep.%j.out
#SBATCH --error=vlm_frame_sweep.%j.err

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 INPUT_VIDEO [OUTPUT_DIR]" >&2
  exit 2
fi

module load Mamba/23.11.0-0

repo_dir=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
conda_env=${CONDA_ENV:-offline-ad-gpu}
input_video=$1
input_name=$(basename "$input_video")
input_stem=${input_name%.*}
output_dir=${2:-"$repo_dir/outputs/vlm-frame-sweep-$input_stem"}
vlm_model=${VLM_MODEL:-"$repo_dir/models/Qwen2.5-VL-3B-Instruct"}
frame_timestamp=${FRAME_TIMESTAMP:-31.72}
resolution_values=${RESOLUTIONS:-"160 320 480 640"}
read -r -a resolutions <<< "$resolution_values"

for required_path in "$input_video" "$vlm_model"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input or model path does not exist: $required_path" >&2
    exit 1
  fi
done

cd "$repo_dir"
mkdir -p "$output_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Input: $input_video"
echo "Frame timestamp: $frame_timestamp"
echo "Evidence max-edge values: ${resolutions[*]}"
echo "Only Qwen2.5-VL is loaded; runs are sequential on one GPU."
nvidia-smi

conda run --no-capture-output -n "$conda_env" \
  python -m offline_ad_poc.benchmark_vlm_frame \
  --input "$input_video" \
  --vlm-model "$vlm_model" \
  --output-dir "$output_dir" \
  --timestamp "$frame_timestamp" \
  --resolutions "${resolutions[@]}"

echo "Finished one-frame VLM resolution sweep: $output_dir"
