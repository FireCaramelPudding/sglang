#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
PORT="${PORT:-${DEFAULT_PORT}}"
HOST="${HOST:-${DEFAULT_HOST}}"
BIND_HOST="${BIND_HOST:-${DEFAULT_BIND_HOST}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
TOPK_IDS_DIR="${TOPK_IDS_DIR:-$(default_topk_ids_dir)}"
TOPK_IDS_MIN_TOKENS="${TOPK_IDS_MIN_TOKENS:-4096}"
TOPK_IDS_MAX_PER_LAYER="${TOPK_IDS_MAX_PER_LAYER:-2}"
CAPTURE_ROUNDS="${CAPTURE_ROUNDS:-2}"
SERVER_LOG="${SERVER_LOG:-${TOPK_IDS_DIR}/capture_server.log}"
START_SERVER="${START_SERVER:-1}"
KEEP_SERVER_RUNNING="${KEEP_SERVER_RUNNING:-0}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-1}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

mkdir -p "${TOPK_IDS_DIR}"
activate_conda_env

export CUDA_VISIBLE_DEVICES
export SGLANG_MOE_TOPK_IDS_DIR="${TOPK_IDS_DIR}"
export SGLANG_MOE_TOPK_IDS_MIN_TOKENS="${TOPK_IDS_MIN_TOKENS}"
export SGLANG_MOE_TOPK_IDS_MAX_PER_LAYER="${TOPK_IDS_MAX_PER_LAYER}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" && "${KEEP_SERVER_RUNNING}" != "1" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

print_step "Capture config"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TP_SIZE=${TP_SIZE}"
echo "PORT=${PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TOPK_IDS_DIR=${TOPK_IDS_DIR}"
echo "CAPTURE_ROUNDS=${CAPTURE_ROUNDS}"
echo "DISABLE_PIECEWISE_CUDA_GRAPH=${DISABLE_PIECEWISE_CUDA_GRAPH}"

if [[ "${START_SERVER}" == "1" ]]; then
  print_step "Starting SGLang server with topk capture enabled"
  read -r -a extra_args <<< "${SERVER_EXTRA_ARGS}"
  server_cmd=(
    python -u -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${BIND_HOST}"
    --tp "${TP_SIZE}"
    --port "${PORT}"
  )
  if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
    server_cmd+=(--skip-server-warmup)
  fi
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
    server_cmd+=(--disable-piecewise-cuda-graph)
  fi
  server_cmd+=("${extra_args[@]}")

  "${server_cmd[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
fi

print_step "Waiting for server readiness"
wait_for_http_ready "http://${HOST}:${PORT}/v1/models" 600

print_step "Driving ${CAPTURE_ROUNDS} capture requests"
for ((round = 1; round <= CAPTURE_ROUNDS; round++)); do
  echo "Capture round ${round}/${CAPTURE_ROUNDS}"
  python "${SGLANG_REPO}/benchmark/kernels/fused_moe_triton/tuning_client.py" \
    --model "${MODEL_PATH}" \
    --ip "${HOST}" \
    --port "${PORT}"
done

print_step "Capture summary"
find "${TOPK_IDS_DIR}" -maxdepth 1 -type f -name 'topk_ids_layer*.pt' | sort | sed -n '1,20p'
captured_count="$(find "${TOPK_IDS_DIR}" -maxdepth 1 -type f -name 'topk_ids_layer*.pt' | wc -l)"
echo "Captured ${captured_count} topk_ids files into ${TOPK_IDS_DIR}"
