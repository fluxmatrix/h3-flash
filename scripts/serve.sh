#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/runtime.sh"

mode=FLASH
if [[ $# -gt 0 && ( "${1^^}" == "FLASH" || "${1^^}" == "LOSSLESS" ) ]]; then
  mode=$1
  shift
fi
case "${mode^^}" in
  FLASH)
    profile=flash
    model_root="$h3_flash_weights_root/official-diffusers-turbo4-bf16"
    ;;
  LOSSLESS)
    profile=lossless
    model_root="$h3_flash_weights_root/official-diffusers"
    ;;
  *)
    echo "usage: $0 [FLASH|LOSSLESS] [--host HOST] [--port PORT]" >&2
    exit 2
    ;;
esac

export H3_FLASH_ROOT="$h3_flash_project_root"
export H3_FLASH_FFMPEG_BIN="$h3_flash_ffmpeg_root/bin/ffmpeg"
export PATH="$h3_flash_venv_root/bin:$h3_flash_ffmpeg_root/bin:$PATH"
export LD_LIBRARY_PATH="$h3_flash_ffmpeg_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$h3_flash_project_root/src:$h3_flash_diffusers_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

prompt_pid=
story_pid=
worker_pid=
cleanup() {
  if [[ -n $worker_pid ]] && kill -0 "$worker_pid" 2>/dev/null; then
    kill -INT "$worker_pid" 2>/dev/null || true
  fi
  if [[ -n $prompt_pid ]] && kill -0 "$prompt_pid" 2>/dev/null; then
    kill -TERM "$prompt_pid" 2>/dev/null || true
  fi
  if [[ -n $story_pid ]] && kill -0 "$story_pid" 2>/dev/null; then
    kill -TERM "$story_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

service_args=("$@")
warm_story=0
for service_arg in "${service_args[@]}"; do
  if [[ $service_arg == --warm-story ]]; then
    warm_story=1
    break
  fi
done
if [[ ${H3_FLASH_ENABLE_PROMPT_ENHANCER:-1} == 1 ]]; then
  prompt_port=${H3_FLASH_PROMPT_PORT:-8001}
  prompt_model=${H3_FLASH_PROMPT_MODEL_NAME:-qwen3.5-4b-prompt}
  if curl --fail --silent "http://127.0.0.1:$prompt_port/health" >/dev/null 2>&1; then
    echo "[h3-flash] using prompt enhancer already listening on port $prompt_port"
  else
    "$script_dir/serve_prompt_enhancer.sh" &
    prompt_pid=$!
    echo "[h3-flash] loading Qwen3.5-4B prompt enhancer on GPU ${H3_FLASH_PROMPT_GPU:-0}"
    for attempt in $(seq 1 180); do
      if curl --fail --silent "http://127.0.0.1:$prompt_port/health" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "$prompt_pid" 2>/dev/null; then
        wait "$prompt_pid" || true
        echo "[h3-flash] prompt enhancer failed to start" >&2
        exit 2
      fi
      if [[ $attempt -eq 180 ]]; then
        echo "[h3-flash] prompt enhancer did not become ready in 180 seconds" >&2
        exit 2
      fi
      sleep 1
    done
  fi
  service_args+=(
    --prompt-enhancer-url "http://127.0.0.1:$prompt_port/v1/chat/completions"
    --prompt-enhancer-model "$prompt_model"
  )
  if [[ $warm_story == 1 ]]; then
    if [[ ${H3_FLASH_SEPARATE_STORY_DIRECTOR:-0} == 1 ]]; then
      story_port=${H3_FLASH_STORY_PORT:-8002}
      story_model=${H3_FLASH_STORY_MODEL_NAME:-qwen3-8b-story}
      if curl --fail --silent "http://127.0.0.1:$story_port/health" >/dev/null 2>&1; then
        echo "[h3-flash] using story director already listening on port $story_port"
      else
        "$script_dir/serve_story_director.sh" &
        story_pid=$!
        echo "[h3-flash] loading Qwen3-8B-FP8 story director on GPU ${H3_FLASH_STORY_GPU:-0}"
        for attempt in $(seq 1 180); do
          if curl --fail --silent "http://127.0.0.1:$story_port/health" >/dev/null 2>&1; then
            break
          fi
          if ! kill -0 "$story_pid" 2>/dev/null; then
            wait "$story_pid" || true
            echo "[h3-flash] story director failed to start" >&2
            exit 2
          fi
          if [[ $attempt -eq 180 ]]; then
            echo "[h3-flash] story director did not become ready in 180 seconds" >&2
            exit 2
          fi
          sleep 1
        done
      fi
    else
      story_port=$prompt_port
      story_model=$prompt_model
      echo "[h3-flash] reusing Qwen3.5-4B as the story director"
    fi
    service_args+=(
      --story-director-url "http://127.0.0.1:$story_port/v1/chat/completions"
      --story-director-model "$story_model"
    )
    "$h3_flash_venv_root/bin/python" "$script_dir/warm_story_director.py" \
      --endpoint "http://127.0.0.1:$prompt_port/v1/chat/completions" \
      --model "$prompt_model" \
      --director-endpoint "http://127.0.0.1:$story_port/v1/chat/completions" \
      --director-model "$story_model"
  fi
fi

"$h3_flash_venv_root/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc-per-node 8 \
  --module h3_flash.service \
  --profile "$profile" \
  --model-root "$model_root" \
  --output-root "$h3_flash_runtime_root/artifacts/web" \
  "${service_args[@]}" &
worker_pid=$!
set +e
wait "$worker_pid"
status=$?
set -e
worker_pid=
exit "$status"
