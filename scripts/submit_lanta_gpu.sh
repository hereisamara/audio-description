#!/bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -J audio-ad-poc
#SBATCH --output=audio_ad.%j.out
#SBATCH --error=audio_ad.%j.err

set -euo pipefail

usage() {
  echo "Usage: sbatch $0 INPUT_VIDEO [OUTPUT_VIDEO] [WORK_DIR]" >&2
  echo "       sbatch $0 --studio-package INPUT_FOLDER" >&2
  exit 2
}

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
repo_dir=${REPO_DIR:-$submit_dir}
model_root="$repo_dir/models"
conda_env=${CONDA_ENV:-offline-ad-gpu}
package_dir=
if [[ ${1:-} == --studio-package ]]; then
  [[ $# -eq 2 ]] || usage
  input_dir=$(cd -- "$2" && pwd -P)
  input_video="$input_dir/original.mp4"
  if [[ -z ${REPO_DIR:-} && -f "$input_dir/code/offline_ad_poc/pipeline.py" ]]; then
    repo_dir="$input_dir/code"
  fi
  package_name="$(basename "$input_dir")-qwen-gpu-${SLURM_JOB_ID:?Submit this script with sbatch}"
  package_dir="$submit_dir/outputs/studio-runs/$package_name"
  output_video="$package_dir/video-ad.mp4"
  work_dir="$package_dir/work"
else
  [[ $# -ge 1 && $# -le 3 ]] || usage
  input_video=$1
  input_name=$(basename "$input_video")
  input_stem=${input_name%.*}
  output_video=${2:-"$repo_dir/outputs/${input_stem}-ad.mp4"}
  work_dir=${3:-"$repo_dir/outputs/${input_stem}-work"}
fi

asr_model=${ASR_MODEL:-"$model_root/whisper-small.en-transformers"}
vlm_backend=${VLM_BACKEND:-qwen3vl8b}
case "$vlm_backend" in
  qwen3vl8b) default_vlm=Qwen3-VL-8B-Instruct ;;
  qwen25) default_vlm=Qwen2.5-VL-3B-Instruct ;;
  *) echo "VLM_BACKEND must be qwen3vl8b or qwen25." >&2; exit 2 ;;
esac
vlm_model=${VLM_MODEL:-"$model_root/$default_vlm"}
tts_model=${TTS_MODEL:-"$model_root/mms-tts-eng"}
maximum_gaps=${MAXIMUM_GAPS:-12}
evidence_frames=${EVIDENCE_FRAMES:-6}
evidence_max_edge=${EVIDENCE_MAX_EDGE:-0}
ad_plan=${AD_PLAN:-multi}
event_followups=${EVENT_FOLLOWUPS:-2}
ad_preparer=${AD_PREPARER:-auto}
set --
if [[ ${SCENE_FALLBACK:-1} == 0 ]]; then
  set -- --no-scene-fallback
fi
if [[ -n "${QWEN_SECONDS_PER_CALL:-}" ]]; then
  set -- "$@" --qwen-seconds-per-call "$QWEN_SECONDS_PER_CALL"
fi

for required_path in "$input_video" "$asr_model" "$vlm_model" "$tts_model"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input or model path does not exist: $required_path" >&2
    exit 1
  fi
done

module load Mamba/23.11.0-0
default_preparer="$model_root/Qwen3-1.7B-Q4_K_M.gguf"
if [[ -s "$model_root/cpu-llm/Qwen3-1.7B-Q4_K_M.gguf" ]]; then
  default_preparer="$model_root/cpu-llm/Qwen3-1.7B-Q4_K_M.gguf"
fi
preparer_model=${AD_PREPARER_MODEL:-"$default_preparer"}
preparer_executable=${AD_PREPARER_EXECUTABLE:-"$submit_dir/outputs/lanta-ad-preparer/bin/llama-completion"}
if [[ $ad_preparer == auto ]]; then
  if [[ -s "$preparer_model" && -x "$preparer_executable" ]]; then
    ad_preparer=1
  else
    ad_preparer=0
  fi
fi
case $ad_preparer in
  0) ;;
  1)
    if [[ ! -s "$preparer_model" || ! -x "$preparer_executable" ]]; then
      printf 'Missing AD preparer checkpoint or executable:\n  %s\n  %s\n' "$preparer_model" "$preparer_executable" >&2
      exit 1
    fi
    conda run --no-capture-output -n "$conda_env" "$preparer_executable" --version
    set -- "$@" --ad-preparer-model "$preparer_model" --ad-preparer-executable "$preparer_executable"
    if [[ -n ${AD_PREPARER_REASONING_TOKENS:-} ]]; then
      set -- "$@" --ad-preparer-reasoning-tokens "$AD_PREPARER_REASONING_TOKENS"
    fi
    ;;
  *)
    echo "AD_PREPARER must be auto, 0, or 1." >&2
    exit 2
    ;;
esac
cd "$repo_dir"
if [[ -n $package_dir ]]; then
  mkdir -p "$(dirname "$package_dir")"
  mkdir "$package_dir"
  ln -L "$input_video" "$package_dir/original.mp4" || cp "$input_video" "$package_dir/original.mp4"
  for name in source.json dialogue.vtt dialogue.srt reference-ad.vtt; do
    if [[ -f "$input_dir/$name" ]]; then
      cp "$input_dir/$name" "$package_dir/$name"
    fi
  done
fi
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
echo "Visual model: $vlm_backend ($vlm_model)"
echo "Visual evidence: $evidence_frames frames per window, max edge $evidence_max_edge (0 = source)"
echo "AD preparer: $ad_preparer (1 = Qwen3-1.7B on CPU)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi

conda run --no-capture-output -n "$conda_env" \
  python -c 'import os, sys, torch; print("PyTorch:", torch.__version__); print("PyTorch CUDA build:", torch.version.cuda); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("CUDA available:", torch.cuda.is_available()); print("CUDA device count:", torch.cuda.device_count()); sys.exit(0 if torch.cuda.is_available() else 3)'

conda run --no-capture-output -n "$conda_env" \
  python -m offline_ad_poc.pipeline \
  --input "$input_video" \
  --output-video "$output_video" \
  --work-dir "$work_dir" \
  --device cuda \
  --asr-backend transformers \
  --asr-model "$asr_model" \
  --visual-backend "$vlm_backend" \
  --vlm-model "$vlm_model" \
  --tts-backend transformers \
  --tts-model "$tts_model" \
  --evidence-frames "$evidence_frames" \
  --evidence-max-edge "$evidence_max_edge" \
  --ad-plan "$ad_plan" \
  --event-followups "$event_followups" \
  --maximum-gaps "$maximum_gaps" \
  "$@"

echo "Finished: $output_video"
if [[ -n $package_dir ]]; then
  archive="$package_dir.tar"
  tar -cf "$archive.partial" -C "$(dirname "$package_dir")" "$package_name"
  mv "$archive.partial" "$archive"
  echo "Studio package: $archive"
fi
