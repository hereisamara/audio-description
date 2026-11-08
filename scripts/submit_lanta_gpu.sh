#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J audio-ad-poc
#SBATCH --output=audio_ad.%j.out
#SBATCH --error=audio_ad.%j.err

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: sbatch $0 INPUT_VIDEO [OUTPUT_VIDEO] [WORK_DIR]" >&2
  exit 2
fi

module load Mamba/23.11.0-0

repo_dir=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
conda_env=${CONDA_ENV:-offline-ad-gpu}
input_video=$1
input_name=$(basename "$input_video")
input_stem=${input_name%.*}
output_video=${2:-"$repo_dir/outputs/${input_stem}-ad.mp4"}
work_dir=${3:-"$repo_dir/outputs/${input_stem}-work"}

asr_model=${ASR_MODEL:-"$repo_dir/models/whisper-small.en-transformers"}
vlm_model=${VLM_MODEL:-"$repo_dir/models/Qwen2.5-VL-3B-Instruct"}
tts_model=${TTS_MODEL:-"$repo_dir/models/mms-tts-eng"}
maximum_gaps=${MAXIMUM_GAPS:-12}

for required_path in "$input_video" "$asr_model" "$vlm_model" "$tts_model"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input or model path does not exist: $required_path" >&2
    exit 1
  fi
done

cd "$repo_dir"
mkdir -p "$(dirname "$output_video")" "$work_dir"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Repository: $repo_dir"
echo "Conda environment: $conda_env"
echo "Input: $input_video"
echo "Output: $output_video"
nvidia-smi

conda run --no-capture-output -n "$conda_env" \
  python -m offline_ad_poc.pipeline \
  --input "$input_video" \
  --output-video "$output_video" \
  --work-dir "$work_dir" \
  --device cuda \
  --asr-backend transformers \
  --asr-model "$asr_model" \
  --visual-backend qwen25 \
  --vlm-model "$vlm_model" \
  --tts-backend transformers \
  --tts-model "$tts_model" \
  --maximum-gaps "$maximum_gaps"

echo "Finished: $output_video"
