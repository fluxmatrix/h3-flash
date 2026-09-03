#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 PYTHON OFFICIAL_DIFFUSERS_CHECKOUT MODEL_ROOT OUTPUT_ROOT GPU_LIST [PROFILE]" >&2
  echo "example GPU_LIST: 0,1,2,3,4,5,6,7" >&2
  exit 2
fi

python_bin=$1
diffusers_checkout=$2
model_root=$3
output_root=$4
gpu_list=$5
profile=${6:-official}
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$diffusers_checkout/src/diffusers/__init__.py" ]]; then
  echo "official Diffusers checkout not found: $diffusers_checkout" >&2
  exit 2
fi

export H3_FLASH_ROOT="$project_root"
export PYTHONPATH="$project_root/src:$diffusers_checkout/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -m h3_flash.cli benchmark official \
  --suite-file "$project_root/configs/evals/h3-broad40-v1.1.json" \
  --model-root "$model_root" \
  --output-root "$output_root" \
  --gpus "$gpu_list" \
  --profile "$profile"
