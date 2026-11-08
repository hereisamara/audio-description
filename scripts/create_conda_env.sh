#!/usr/bin/env bash
set -euo pipefail

variant=${1:-cpu}
case "$variant" in
  cpu)
    environment_file=environment-cpu.yml
    environment_name=offline-ad-cpu
    ;;
  gpu)
    environment_file=environment-gpu.yml
    environment_name=offline-ad-gpu
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

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fqx "$environment_name"; then
  echo "Updating existing Conda environment: $environment_name"
  conda env update --name "$environment_name" --file "$environment_file" --prune
else
  echo "Creating Conda environment: $environment_name"
  conda env create --file "$environment_file"
fi

echo "Environment is ready. Verify it with:"
echo "  conda run --no-capture-output -n $environment_name python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "  conda run --no-capture-output -n $environment_name python -m unittest discover -s tests -v"
