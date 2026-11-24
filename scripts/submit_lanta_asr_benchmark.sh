#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -J asr-compare
#SBATCH --output=asr_compare.%j.out
#SBATCH --error=asr_compare.%j.err

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
output_dir=${2:-"$repo_dir/outputs/asr-comparison-$input_stem"}
reference=${ASR_REFERENCE:?Set ASR_REFERENCE to your reference dialogue JSON file}

default_models="whisper-small=$repo_dir/models/whisper-small.en-transformers;whisper-turbo=$repo_dir/models/whisper-large-v3-turbo;distil-whisper=$repo_dir/models/distil-large-v3;wav2vec2-large=$repo_dir/models/wav2vec2-large-960h-lv60-self"
IFS=';' read -r -a model_specs <<< "${ASR_MODELS:-$default_models}"

for required_path in "$input_video" "$reference"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

model_args=()
for spec in "${model_specs[@]}"; do
  model_name=${spec%%=*}
  model_path=${spec#*=}
  if [[ "$model_name" == "$model_path" || ! -d "$model_path" ]]; then
    echo "Missing or invalid ASR model specification: $spec" >&2
    exit 1
  fi
  model_args+=(--model "$model_name=$model_path")
done

cd "$repo_dir"
mkdir -p "$output_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Input: $input_video"
echo "Reference: $reference"
echo "Output: $output_dir"
printf 'ASR model: %s\n' "${model_specs[@]}"
echo "Models run sequentially; only one model is resident on the GPU at a time."
nvidia-smi

conda run --no-capture-output -n "$conda_env" \
  python -m offline_ad_poc.asr_benchmark \
  --input "$input_video" \
  --reference "$reference" \
  --output-dir "$output_dir" \
  --device cuda \
  "${model_args[@]}"

echo "Finished ASR comparison: $output_dir"
