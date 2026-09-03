#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

uv_bin="$h3_flash_tools_root/uv/uv"
python_bin="$h3_flash_python_root/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12"
requirements="$h3_flash_project_root/requirements/b200-cu130.prompt-enhancer.direct.txt"
uv_cache="$h3_flash_runtime_root/cache/uv-prompt-enhancer"

[[ -x $uv_bin && -x $python_bin ]] || {
  echo "[h3-flash] base runtime is missing; run scripts/bootstrap_b200.sh first" >&2
  exit 2
}

mkdir -p "$uv_cache"
export UV_CACHE_DIR="$uv_cache"
if [[ ! -x "$h3_flash_prompt_venv_root/bin/python" ]]; then
  "$uv_bin" venv "$h3_flash_prompt_venv_root" --python "$python_bin" --managed-python
fi

installed=$(
  "$h3_flash_prompt_venv_root/bin/python" -c \
    'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null || true
)
if [[ $installed != 0.28.0 ]]; then
  echo "[h3-flash] installing isolated vLLM prompt runtime"
  "$uv_bin" pip install \
    --python "$h3_flash_prompt_venv_root/bin/python" \
    --requirements "$requirements" \
    --torch-backend cu130 \
    --link-mode copy \
    --exact \
    --strict
fi

echo "[h3-flash] downloading locked Qwen3.5-4B prompt model (~8.7 GiB)"
"$script_dir/h3-flash" model download \
  --lock models.prompt-enhancer \
  --weights-root "$h3_flash_weights_root"

echo "[h3-flash] prompt enhancer ready"
