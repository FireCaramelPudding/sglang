#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_MODEL_PATH="/ssd/home/xiaoliangyang/models/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
DEFAULT_TP_SIZE="2"
DEFAULT_EP_SIZE="1"
DEFAULT_PORT="6001"
DEFAULT_HOST="127.0.0.1"
DEFAULT_BIND_HOST="0.0.0.0"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1"
DEFAULT_CONDA_ENV="sglang"

model_name() {
  basename "${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
}

default_topk_ids_dir() {
  printf "%s" "/ssd/home/xiaoliangyang/sglang/benchmark/kernels/fused_moe_triton/topk_ids/$(model_name)_tp${TP_SIZE:-${DEFAULT_TP_SIZE}}"
}

default_tuning_output_dir() {
  printf "%s" "/ssd/home/xiaoliangyang/sglang/benchmark/kernels/fused_moe_triton/generated/$(model_name)_tp${TP_SIZE:-${DEFAULT_TP_SIZE}}"
}

activate_conda_env() {
  local conda_sh="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  local env_name=""

  if [[ ! -f "${conda_sh}" ]]; then
    echo "conda.sh not found at ${conda_sh}" >&2
    return 1
  fi

  if [[ -n "${CONDA_ENV:-}" ]]; then
    env_name="${CONDA_ENV}"
  elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    env_name="${CONDA_DEFAULT_ENV}"
  else
    env_name="${DEFAULT_CONDA_ENV}"
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${env_name}" ]]; then
    conda activate "${env_name}"
  fi
  echo "Using conda env: ${env_name}"
}

wait_for_http_ready() {
  local url="$1"
  local timeout_s="${2:-300}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout_s )); then
      echo "Timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 2
  done
}

triton_version_dir() {
  python - <<'PY'
import triton
print(f"triton_{triton.__version__.replace('.', '_')}")
PY
}

print_step() {
  printf "\n[%s] %s\n" "$(date '+%F %T')" "$1"
}
