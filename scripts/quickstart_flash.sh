#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

output_dir=${1:-$h3_flash_artifacts_root/quickstart-flash}

"$script_dir/setup.sh"
"$script_dir/run_smoke.sh" FLASH "$output_dir"
