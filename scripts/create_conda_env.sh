#!/usr/bin/env bash
set -euo pipefail

variant=${1:-cpu}
case "$variant" in
  cpu)
    environment_file=environment-cpu.yml
    ;;
  gpu)
    environment_file=environment-gpu.yml
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]" >&2
    exit 2
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available. Load the Lanta Conda module first." >&2
  exit 1
fi

conda env create --file "$environment_file"
