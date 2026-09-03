#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

vllm_bin="$h3_flash_prompt_venv_root/bin/vllm"
model_root=$h3_flash_prompt_weights_root
gpu=${H3_FLASH_PROMPT_GPU:-0}
port=${H3_FLASH_PROMPT_PORT:-8001}
served_model=${H3_FLASH_PROMPT_MODEL_NAME:-qwen3.5-4b-prompt}
cache_root="$h3_flash_runtime_root/cache/vllm-prompt-enhancer"
cuda_root="$h3_flash_prompt_venv_root/lib/python3.12/site-packages/nvidia/cu13"

[[ -x $vllm_bin ]] || {
  echo "[h3-flash] prompt runtime is missing; run scripts/setup_prompt_enhancer.sh" >&2
  exit 2
}
[[ -f "$model_root/config.json" ]] || {
  echo "[h3-flash] prompt weights are missing; run scripts/setup_prompt_enhancer.sh" >&2
  exit 2
}

mkdir -p "$cache_root/torch" "$cache_root/triton" "$cache_root/flashinfer"
export CUDA_VISIBLE_DEVICES=$gpu
export VLLM_CACHE_ROOT="$cache_root/torch"
export TRITON_CACHE_DIR="$cache_root/triton"
export FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
export CUDA_HOME="$cuda_root"
export PATH="$h3_flash_prompt_venv_root/bin:$CUDA_HOME/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$vllm_bin" serve "$model_root" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 1073741824 \
  --gpu-memory-utilization 0.18 \
  --limit-mm-per-prompt '{"image":4}' \
  --attention-backend TRITON_ATTN \
  --generation-config vllm \
  --disable-log-stats
