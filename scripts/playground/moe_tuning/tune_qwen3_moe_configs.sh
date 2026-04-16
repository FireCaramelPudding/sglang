#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
EP_SIZE="${EP_SIZE:-${DEFAULT_EP_SIZE}}"
TOPK_IDS_DIR="${TOPK_IDS_DIR:-$(default_topk_ids_dir)}"
TUNING_OUTPUT_DIR="${TUNING_OUTPUT_DIR:-$(default_tuning_output_dir)}"
PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH:-}"

activate_conda_env
export PYTHONPATH

if ! find "${TOPK_IDS_DIR}" -maxdepth 1 -type f -name 'topk_ids_layer*.pt' | grep -q .; then
  echo "No captured topk_ids files found in ${TOPK_IDS_DIR}" >&2
  echo "Run capture_qwen3_moe_topk_ids.sh first." >&2
  exit 1
fi

mkdir -p "${TUNING_OUTPUT_DIR}"

print_step "Tuning fused MoE Triton configs"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TP_SIZE=${TP_SIZE}"
echo "EP_SIZE=${EP_SIZE}"
echo "TOPK_IDS_DIR=${TOPK_IDS_DIR}"
echo "TUNING_OUTPUT_DIR=${TUNING_OUTPUT_DIR}"

pushd "${TUNING_OUTPUT_DIR}" >/dev/null
python "${SGLANG_REPO}/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py" \
  --model "${MODEL_PATH}" \
  --tp-size "${TP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --topk-ids-dir "${TOPK_IDS_DIR}" \
  --tune
popd >/dev/null

print_step "Generated config files"
find "${TUNING_OUTPUT_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name 'tuning_result_*.txt' \) | sort
