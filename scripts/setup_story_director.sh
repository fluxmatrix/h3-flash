#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

if [[ ! -x "$h3_flash_prompt_venv_root/bin/vllm" ]]; then
  "$script_dir/setup_prompt_enhancer.sh"
fi

echo "[h3-flash] downloading locked Qwen3-8B-FP8 story director (~8.9 GiB)"
"$script_dir/h3-flash" model download \
  --lock models.story-director \
  --weights-root "$h3_flash_weights_root"

echo "[h3-flash] story director ready"
