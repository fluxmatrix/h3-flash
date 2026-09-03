#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

"$script_dir/bootstrap_b200.sh"
if ! "$script_dir/download_models.sh" flash; then
  echo "[h3-flash] model download failed; if authentication is required, run:" >&2
  echo "  $h3_flash_venv_root/bin/python -m huggingface_hub.cli.hf auth login" >&2
  exit 2
fi
"$script_dir/setup_prompt_enhancer.sh"

echo "[h3-flash] setup complete"
echo "[h3-flash] next: scripts/generate.sh --help"
