#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 PYTHON OFFICIAL_DIFFUSERS_CHECKOUT DERIVED_TURBO_MODEL_ROOT OUTPUT_ROOT GPU_LIST [FFMPEG_BIN]" >&2
  echo "example GPU_LIST: 0,1,2,3,4,5,6,7" >&2
  exit 2
fi

python_bin=$1
diffusers_checkout=$2
model_root=$3
output_root=$4
gpu_list=$5
ffmpeg_bin=${6:-}
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$diffusers_checkout/src/diffusers/__init__.py" ]]; then
  echo "official Diffusers checkout not found: $diffusers_checkout" >&2
  exit 2
fi

if [[ ! -f "$model_root/turbo-bake.json" ]]; then
  echo "derived Turbo model manifest not found: $model_root/turbo-bake.json" >&2
  echo "prepare it first with scripts/prepare_turbo_model.py" >&2
  exit 2
fi

export H3_FLASH_ROOT="$project_root"
export PYTHONPATH="$project_root/src:$diffusers_checkout/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$ffmpeg_bin" ]]; then
  export H3_FLASH_FFMPEG_BIN="$ffmpeg_bin"
fi

"$python_bin" -m h3_flash.cli benchmark distributed \
  --suite-file "$project_root/configs/evals/h3-broad40-v1.1.json" \
  --model-root "$model_root" \
  --output-root "$output_root" \
  --gpus "$gpu_list" \
  --profile flash
