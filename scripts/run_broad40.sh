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
python_bin="$h3_flash_venv_root/bin/python"
diffusers_checkout=$h3_flash_diffusers_root
ffmpeg_bin="$h3_flash_ffmpeg_root/bin/ffmpeg"

case "$version" in
  OFFICIAL)
    exec "$project_root/scripts/benchmark_official_broad40.sh" \
      "$python_bin" "$diffusers_checkout" \
      "$weights_root/official-diffusers" "$output_root" "$gpu_list"
    ;;
  LOSSLESS)
    exec "$project_root/scripts/benchmark_lossless_broad40.sh" \
      "$python_bin" "$diffusers_checkout" \
      "$weights_root/official-diffusers" "$output_root" "$gpu_list" \
      lossless "$ffmpeg_bin"
    ;;
  FLASH)
    exec "$project_root/scripts/benchmark_fast_turbo4_broad40.sh" \
      "$python_bin" "$diffusers_checkout" \
      "$weights_root/official-diffusers-turbo4-bf16" "$output_root" \
      "$gpu_list" "$ffmpeg_bin"
    ;;
  *)
    echo "unknown VERSION: $version" >&2
    exit 2
    ;;
esac
