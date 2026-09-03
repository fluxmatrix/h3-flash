#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

vllm_bin="$h3_flash_prompt_venv_root/bin/vllm"
model_root=$h3_flash_story_weights_root
gpu=${H3_FLASH_STORY_GPU:-0}
port=${H3_FLASH_STORY_PORT:-8002}
served_model=${H3_FLASH_STORY_MODEL_NAME:-qwen3-8b-story}
cache_root="$h3_flash_runtime_root/cache/vllm-story-director"
cuda_root="$h3_flash_prompt_venv_root/lib/python3.12/site-packages/nvidia/cu13"

[[ -x $vllm_bin ]] || {
  echo "[h3-flash] story runtime is missing; run scripts/setup_story_director.sh" >&2
  exit 2
}
[[ -f "$model_root/config.json" ]] || {
  echo "[h3-flash] story weights are missing; run scripts/setup_story_director.sh" >&2
  exit 2
}

mkdir -p "$cache_root/torch" "$cache_root/triton"
export CUDA_VISIBLE_DEVICES=$gpu
export VLLM_CACHE_ROOT="$cache_root/torch"
export TRITON_CACHE_DIR="$cache_root/triton"
export CUDA_HOME="$cuda_root"
export PATH="$h3_flash_prompt_venv_root/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$vllm_bin" serve "$model_root" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --dtype auto \
  --max-model-len 7168 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 7168 \
  --kv-cache-memory-bytes 1073741824 \
  --gpu-memory-utilization 0.12 \
  --attention-backend TRITON_ATTN \
  --generation-config vllm \
  --disable-log-stats
