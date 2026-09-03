#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"
project_root=$h3_flash_project_root
mode=${1:-flash}
weights_root=$h3_flash_weights_root
h3="$project_root/scripts/h3-flash"
export PYTHONPATH="$project_root/src:$h3_flash_diffusers_root/src${PYTHONPATH:+:$PYTHONPATH}"

case "${mode,,}" in
  official|lossless)
    prepare_flash=false
    ;;
  flash|all)
    prepare_flash=true
    ;;
  *)
    echo "usage: $0 [official|lossless|flash]" >&2
    exit 2
    ;;
esac

mkdir -p "$weights_root"
echo "[h3-flash] downloading locked official model (~134.13 GiB)"
"$h3" model download --lock models.official --weights-root "$weights_root"

if [[ $prepare_flash == true ]]; then
  echo "[h3-flash] downloading locked LightX2V Turbo LoRA (~1.29 GiB)"
  "$h3" model download --lock models.fast-turbo4-bf16-inputs --weights-root "$weights_root"
  destination="$weights_root/official-diffusers-turbo4-bf16"
  if [[ -f "$destination/turbo-bake.json" ]]; then
    echo "[h3-flash] verifying existing derived FLASH model (~62 GiB read)"
  elif [[ -e "$destination" ]]; then
    echo "incomplete FLASH model destination exists: $destination" >&2
    exit 2
  else
    "$h3_flash_venv_root/bin/python" "$project_root/scripts/prepare_turbo_model.py" \
      --source-model-root "$weights_root/official-diffusers" \
      --destination-model-root "$destination" \
      --lora "$weights_root/turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" \
      --device cuda:0 \
      --compute-dtype bfloat16
  fi
  verification_report=$(mktemp)
  if ! "$h3_flash_venv_root/bin/python" "$project_root/scripts/verify_turbo_model.py" \
    --destination-model-root "$destination" \
    --source-model-root "$weights_root/official-diffusers" \
    --lora "$weights_root/turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" \
    --model-lock models.fast-turbo4-bf16-inputs >"$verification_report"; then
    cat "$verification_report" >&2
    rm -f "$verification_report"
    exit 2
  fi
  rm -f "$verification_report"
  echo "[h3-flash] FLASH model verified"
fi

echo "[h3-flash] model preparation complete: $weights_root"
