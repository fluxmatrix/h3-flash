#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 VERSION OUTPUT_ROOT [GPU_LIST]" >&2
  echo "VERSION: OFFICIAL, LOSSLESS, or FLASH" >&2
  exit 2
fi

version=${1^^}
output_root=$2
gpu_list=${3:-0,1,2,3,4,5,6,7}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"
project_root=$h3_flash_project_root
weights_root=$h3_flash_weights_root

case "$version" in
  OFFICIAL|LOSSLESS)
    model_root="$weights_root/official-diffusers"
    ;;
  FLASH)
    model_root="$weights_root/official-diffusers-turbo4-bf16"
    ;;
  *)
    echo "unknown VERSION: $version" >&2
    exit 2
    ;;
esac

exec "$project_root/scripts/benchmark_latency_matrix.sh" \
  "$version" \
  "$h3_flash_venv_root/bin/python" \
  "$h3_flash_diffusers_root" \
  "$model_root" \
  "$output_root" \
  "$gpu_list" \
  "$h3_flash_ffmpeg_root/bin/ffmpeg"
