#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"
project_root=$h3_flash_project_root
uv_bin=${H3_FLASH_UV_BIN:-$h3_flash_tools_root/uv/uv}

[[ -x "$uv_bin" ]] || {
  echo "uv is missing; run scripts/bootstrap_b200.sh first or set H3_FLASH_UV_BIN" >&2
  exit 2
}

"$uv_bin" pip compile \
  "$project_root/requirements/b200-cu130.official.direct.txt" \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_34 \
  --torch-backend cu130 \
  --exclude-newer 2026-09-03T00:00:00Z \
  --generate-hashes \
  --custom-compile-command scripts/refresh_runtime_lock.sh \
  --output-file "$project_root/requirements/b200-cu130.runtime.lock.txt"
