#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No Conda environment is active. Activate offline-ad-cpu or offline-ad-gpu first." >&2
  exit 1
fi

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 INPUT_VIDEO OUTPUT_VIDEO WORK_DIR [additional pipeline arguments]" >&2
  exit 2
fi

input_video=$1
output_video=$2
work_dir=$3
shift 3

python -m offline_ad_poc.pipeline \
  --input "$input_video" \
  --output-video "$output_video" \
  --work-dir "$work_dir" \
  --device auto \
  "$@"

