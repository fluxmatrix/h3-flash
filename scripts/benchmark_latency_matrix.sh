#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 VERSION PYTHON DIFFUSERS_CHECKOUT MODEL_ROOT OUTPUT_ROOT GPU_LIST [FFMPEG_BIN]" >&2
  echo "VERSION: OFFICIAL, LOSSLESS, or FLASH; GPU_LIST must contain eight GPUs" >&2
  exit 2
fi

version=${1^^}
python_bin=$2
diffusers_checkout=$3
model_root=$4
output_root=$5
gpu_list=$6
ffmpeg_bin=${7:-}
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$diffusers_checkout/src/diffusers/__init__.py" ]]; then
  echo "Diffusers checkout not found: $diffusers_checkout" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<<"$gpu_list"
if [[ ${#gpus[@]} -ne 8 ]]; then
  echo "latency matrix requires exactly eight unique GPUs" >&2
  exit 2
fi
unique_gpu_count=$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)
if [[ $unique_gpu_count -ne 8 ]]; then
  echo "latency matrix requires exactly eight unique GPUs; found: $gpu_list" >&2
  exit 2
fi
[[ -x "$python_bin" ]] || { echo "Python is not executable: $python_bin" >&2; exit 2; }
[[ -d "$model_root" ]] || { echo "model root not found: $model_root" >&2; exit 2; }
if [[ -n "$ffmpeg_bin" && ! -x "$ffmpeg_bin" ]]; then
  echo "FFmpeg is not executable: $ffmpeg_bin" >&2
  exit 2
fi

export H3_FLASH_ROOT="$project_root"
export PYTHONPATH="$project_root/src:$diffusers_checkout/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$ffmpeg_bin" ]]; then
  export H3_FLASH_FFMPEG_BIN="$ffmpeg_bin"
fi

case "$version" in
  OFFICIAL)
    command_name=official
    profile=official
    ;;
  LOSSLESS)
    command_name=distributed
    profile=lossless
    ;;
  FLASH)
    command_name=distributed
    profile=flash
    ;;
  *)
    echo "unknown VERSION: $version" >&2
    exit 2
    ;;
esac

run_suite() {
  local suite_name=$1
  local assigned_gpus=$2
  local suite="$project_root/configs/evals/$suite_name.json"
  local destination="$output_root/$suite_name"
  if [[ -f "$destination/summary.json" ]]; then
    if "$python_bin" -c \
      'import sys; from pathlib import Path; from h3_flash.benchmark import suite_summary_is_complete; raise SystemExit(0 if suite_summary_is_complete(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]) else 1)' \
      "$destination/summary.json" "$suite" "$profile"; then
      echo "[matrix] skip validated complete suite $destination"
      return
    fi
  fi
  if [[ -e "$destination" ]]; then
    echo "refusing incomplete existing destination: $destination" >&2
    return 1
  fi
  "$python_bin" -m h3_flash.cli benchmark "$command_name" \
    --suite-file "$suite" \
    --model-root "$model_root" \
    --output-root "$destination" \
    --gpus "$assigned_gpus" \
    --profile "$profile"
}

durations=(5 10 15)
resolutions=(480 768)
if [[ "$version" == OFFICIAL ]]; then
  left_gpus=$(IFS=,; echo "${gpus[*]:0:4}")
  right_gpus=$(IFS=,; echo "${gpus[*]:4:4}")
  for resolution in "${resolutions[@]}"; do
    for duration in "${durations[@]}"; do
      run_suite "h3-latency4-${resolution}p-${duration}s-v1" "$left_gpus" &
      left_pid=$!
      run_suite "h3-latency4-${resolution}p-portrait-${duration}s-v1" "$right_gpus" &
      right_pid=$!
      wait "$left_pid"
      wait "$right_pid"
    done
  done
else
  for resolution in "${resolutions[@]}"; do
    for duration in "${durations[@]}"; do
      run_suite "h3-latency4-${resolution}p-${duration}s-v1" "$gpu_list"
      run_suite "h3-latency4-${resolution}p-portrait-${duration}s-v1" "$gpu_list"
    done
  done
fi
