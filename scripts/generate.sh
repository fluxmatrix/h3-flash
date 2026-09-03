#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

exec "$script_dir/h3-flash" generate \
  --weights-root "$h3_flash_weights_root" \
  "$@"
