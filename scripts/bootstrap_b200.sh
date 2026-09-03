#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"
project_root=$h3_flash_project_root
tools_root=$h3_flash_tools_root
uv_root="$tools_root/uv"
uv_bin="$uv_root/uv"
python_root=$h3_flash_python_root
venv_root=$h3_flash_venv_root
deps_root=$h3_flash_deps_root
diffusers_root=$h3_flash_diffusers_root
ffmpeg_root=$h3_flash_ffmpeg_root
runtime_lock="$project_root/requirements/b200-cu130.runtime.lock.txt"
uv_cache_root="$h3_flash_runtime_root/cache/uv"
export UV_CACHE_DIR="$uv_cache_root"

uv_version=0.12.9
uv_archive=uv-x86_64-unknown-linux-gnu.tar.gz
uv_url="https://github.com/astral-sh/uv/releases/download/$uv_version/$uv_archive"
uv_sha256=ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460
uv_bin_sha256=671793498fe0a545432e2524b6691ffb9eea4540d9fda43ca2f978df2dbf8426
python_version=3.12.14
diffusers_url=https://github.com/huggingface/diffusers.git
diffusers_commit=abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
ffmpeg_archive=ffmpeg-n8.1.2-50-g1a748fe2cd-linux64-gpl-shared-8.1.tar.xz
ffmpeg_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-01-13-13/$ffmpeg_archive"
ffmpeg_sha256=bbb5d5fa85277c47aab85de79101476854b86b136f071ff55d18d6d5764bbbcb
ffmpeg_bin_sha256=2907ab5197982a54a0b5143eb371d993a5780ccd5ebf79b779d7ba3768322eb6
ffprobe_bin_sha256=9c5a4c1cf490b7255d36e885ecac4f3920cbd9acb73fdd7f1f8d3f5629b6ceae

fail() {
  echo "[h3-flash] ERROR: $*" >&2
  exit 2
}

for command in curl git sha256sum tar nvidia-smi; do
  command -v "$command" >/dev/null || fail "required command not found: $command"
done
[[ $(uname -s) == Linux ]] || fail "the current release target is Linux only"
[[ $(uname -m) == x86_64 ]] || fail "the current release target is Linux x86_64 only"

gpu_rows=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits) \
  || fail "nvidia-smi cannot query the GPUs"
gpu_count=$(printf '%s\n' "$gpu_rows" | sed '/^[[:space:]]*$/d' | wc -l)
if [[ $gpu_count -ne 8 ]] || printf '%s\n' "$gpu_rows" | grep -qv 'B200.*, 10\.0'; then
  fail "this release requires exactly 8x NVIDIA B200 (sm100); detected $gpu_count GPU(s): $gpu_rows"
fi
driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)
[[ $driver_cuda == 13.* ]] || fail "an NVIDIA driver reporting CUDA 13.x is required; found ${driver_cuda:-unknown}"

mkdir -p "$tools_root" "$uv_root" "$deps_root" "$uv_cache_root"
temporary=$(mktemp -d "$tools_root/bootstrap.XXXXXX")
trap 'rm -rf "$temporary"' EXIT

if [[ ! -x "$uv_bin" ]] || [[ $($uv_bin --version 2>/dev/null || true) != "uv $uv_version "* ]]; then
  echo "[h3-flash] installing uv $uv_version"
  curl --fail --location --retry 3 "$uv_url" --output "$temporary/$uv_archive"
  printf '%s  %s\n' "$uv_sha256" "$temporary/$uv_archive" | sha256sum --check --status \
    || fail "uv archive SHA-256 mismatch"
  tar -xzf "$temporary/$uv_archive" -C "$temporary"
  install -m 0755 "$temporary/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin"
fi
printf '%s  %s\n' "$uv_bin_sha256" "$uv_bin" | sha256sum --check --status \
  || fail "installed uv binary SHA-256 mismatch"

echo "[h3-flash] installing managed Python $python_version"
"$uv_bin" python install "$python_version" --install-dir "$python_root" --no-bin
python_bin="$python_root/cpython-$python_version-linux-x86_64-gnu/bin/python3.12"
[[ -x $python_bin ]] || fail "uv installed Python but the pinned interpreter was not found: $python_bin"

if [[ ! -x "$venv_root/bin/python" ]]; then
  "$uv_bin" venv "$venv_root" --python "$python_bin" --managed-python
fi
[[ $($venv_root/bin/python -c 'import platform; print(platform.python_version())') == "$python_version" ]] \
  || fail "$venv_root exists but is not Python $python_version"

if [[ -e "$diffusers_root" ]]; then
  [[ -d "$diffusers_root/.git" ]] || fail "$diffusers_root exists but is not a Git checkout"
  actual_commit=$(git -C "$diffusers_root" rev-parse HEAD)
  [[ $actual_commit == "$diffusers_commit" ]] \
    || fail "Diffusers checkout is $actual_commit; expected $diffusers_commit"
  [[ -z $(git -C "$diffusers_root" status --porcelain) ]] \
    || fail "Diffusers checkout has local changes: $diffusers_root"
else
  echo "[h3-flash] fetching pinned Diffusers $diffusers_commit"
  git init --quiet "$diffusers_root"
  git -C "$diffusers_root" remote add origin "$diffusers_url"
  git -C "$diffusers_root" fetch --quiet --depth 1 origin "$diffusers_commit"
  git -C "$diffusers_root" checkout --quiet --detach FETCH_HEAD
fi

if [[ ! -x "$ffmpeg_root/bin/ffmpeg" ]]; then
  [[ ! -e "$ffmpeg_root" ]] || fail "$ffmpeg_root exists but has no ffmpeg executable"
  echo "[h3-flash] installing pinned FFmpeg"
  curl --fail --location --retry 3 "$ffmpeg_url" --output "$temporary/$ffmpeg_archive"
  printf '%s  %s\n' "$ffmpeg_sha256" "$temporary/$ffmpeg_archive" | sha256sum --check --status \
    || fail "FFmpeg archive SHA-256 mismatch"
  tar -xJf "$temporary/$ffmpeg_archive" -C "$temporary"
  ffmpeg_extracted=$(find "$temporary" -mindepth 1 -maxdepth 1 -type d -name 'ffmpeg-*' | head -n1)
  [[ -n $ffmpeg_extracted ]] || fail "unexpected FFmpeg archive layout"
  mv "$ffmpeg_extracted" "$ffmpeg_root"
fi
[[ -x "$ffmpeg_root/bin/ffprobe" ]] || fail "pinned FFprobe executable is missing"
printf '%s  %s\n' "$ffmpeg_bin_sha256" "$ffmpeg_root/bin/ffmpeg" | sha256sum --check --status \
  || fail "installed FFmpeg binary SHA-256 mismatch"
printf '%s  %s\n' "$ffprobe_bin_sha256" "$ffmpeg_root/bin/ffprobe" | sha256sum --check --status \
  || fail "installed FFprobe binary SHA-256 mismatch"

echo "[h3-flash] installing hash-locked CUDA 13.0 runtime"
"$uv_bin" pip install \
  --python "$venv_root/bin/python" \
  --requirements "$runtime_lock" \
  --require-hashes \
  --torch-backend cu130 \
  --link-mode copy \
  --exact \
  --strict
echo "[h3-flash] installing the local H3-Flash package"
"$uv_bin" pip install \
  --python "$venv_root/bin/python" \
  --no-deps \
  --editable "$project_root" \
  --link-mode copy
echo "[h3-flash] validating the installed runtime"
if ! "$project_root/scripts/h3-flash" doctor --deps-root "$deps_root" >/dev/null; then
  "$project_root/scripts/h3-flash" doctor --deps-root "$deps_root" || true
  fail "runtime validation failed"
fi
echo "[h3-flash] runtime validated"
echo "[h3-flash] bootstrap complete"
echo "[h3-flash] runtime root: $h3_flash_runtime_root"
echo "[h3-flash] next: scripts/download_models.sh flash"
