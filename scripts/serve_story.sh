#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The story UI uses fixed 480p landscape, 10-second chapters. Warm that shape
# instead of the standard demo's 768p/5s shape so Chapter 1 has no compile tax.
exec "$script_dir/serve.sh" "$@" --warm-story
