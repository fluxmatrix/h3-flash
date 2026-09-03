#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OFFICIAL|LOSSLESS|FLASH [OUTPUT_DIR]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"
project_root=$h3_flash_project_root
version=${1^^}
weights_root=$h3_flash_weights_root
output_root=${2:-$h3_flash_artifacts_root/smoke-${version,,}}
suite="$project_root/configs/evals/h3-smoke1-768p-5s-v1.json"
h3="$project_root/scripts/h3-flash"

[[ ! -e "$output_root" ]] || {
  echo "output already exists; choose a new path: $output_root" >&2
  exit 2
}

case "$version" in
  OFFICIAL)
    "$h3" benchmark official \
      --suite-file "$suite" \
      --model-root "$weights_root/official-diffusers" \
      --output-root "$output_root" \
      --gpus 0 \
      --profile official
    ;;
  LOSSLESS)
    "$h3" benchmark distributed \
      --suite-file "$suite" \
      --model-root "$weights_root/official-diffusers" \
      --output-root "$output_root" \
      --gpus 0,1,2,3,4,5,6,7 \
      --profile lossless
    ;;
  FLASH)
    "$h3" benchmark distributed \
      --suite-file "$suite" \
      --model-root "$weights_root/official-diffusers-turbo4-bf16" \
      --output-root "$output_root" \
      --gpus 0,1,2,3,4,5,6,7 \
      --profile flash
    ;;
  *)
    echo "unknown version: $version (expected OFFICIAL, LOSSLESS, or FLASH)" >&2
    exit 2
    ;;
esac

echo "[h3-flash] video: $output_root/cases/people_mandarin_interview/output.mp4"
echo "[h3-flash] timings: $output_root/summary.json"
