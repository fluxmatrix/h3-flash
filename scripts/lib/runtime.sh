#!/usr/bin/env bash

# Shared filesystem layout. Runtime assets live beside the checkout by default,
# never inside it. Set H3_FLASH_RUNTIME_ROOT to place them on another volume.
h3_flash_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
h3_flash_parent_root=$(dirname "$h3_flash_project_root")
h3_flash_runtime_root=${H3_FLASH_RUNTIME_ROOT:-$h3_flash_parent_root/.h3-flash-runtime}

h3_flash_tools_root="$h3_flash_runtime_root/tools"
h3_flash_python_root="$h3_flash_tools_root/python"
h3_flash_venv_root="$h3_flash_runtime_root/venv"
h3_flash_prompt_venv_root="$h3_flash_runtime_root/prompt-venv"
h3_flash_deps_root="$h3_flash_runtime_root/deps"
h3_flash_diffusers_root="$h3_flash_deps_root/diffusers-minimax-h3"
h3_flash_ffmpeg_root="$h3_flash_runtime_root/ffmpeg"
h3_flash_weights_root=${H3_FLASH_WEIGHTS_ROOT:-$h3_flash_runtime_root/weights}
h3_flash_prompt_weights_root=${H3_FLASH_PROMPT_WEIGHTS_ROOT:-$h3_flash_weights_root/prompt-enhancer/Qwen3.5-4B}
h3_flash_story_weights_root=${H3_FLASH_STORY_WEIGHTS_ROOT:-$h3_flash_weights_root/story-director/Qwen3-8B-FP8}
h3_flash_artifacts_root=${H3_FLASH_ARTIFACTS_ROOT:-$h3_flash_runtime_root/artifacts}
