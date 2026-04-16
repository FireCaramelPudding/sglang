#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

TUNING_OUTPUT_DIR="${TUNING_OUTPUT_DIR:-$(default_tuning_output_dir)}"
CONFIG_ROOT="${CONFIG_ROOT:-${SGLANG_REPO}/python/sglang/srt/layers/moe/fused_moe_triton}"

activate_conda_env

if ! find "${TUNING_OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' | grep -q .; then
  echo "No tuned json files found in ${TUNING_OUTPUT_DIR}" >&2
  echo "Run tune_qwen3_moe_configs.sh first." >&2
  exit 1
fi

TRITON_VERSION_DIR="$(triton_version_dir)"
TARGET_DIR="${CONFIG_ROOT}/configs/${TRITON_VERSION_DIR}"
mkdir -p "${TARGET_DIR}"

print_step "Installing tuned config files"
echo "SOURCE=${TUNING_OUTPUT_DIR}"
echo "TARGET=${TARGET_DIR}"

find "${TUNING_OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' -print0 | while IFS= read -r -d '' json_file; do
  cp -f "${json_file}" "${TARGET_DIR}/"
  echo "Installed $(basename "${json_file}")"
done

print_step "Installed files now present in target dir"
find "${TARGET_DIR}" -maxdepth 1 -type f -name '*.json' | sort | tail -n 20
