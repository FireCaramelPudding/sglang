#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/capture_qwen3_moe_topk_ids.sh"
bash "${SCRIPT_DIR}/tune_qwen3_moe_configs.sh"
bash "${SCRIPT_DIR}/install_qwen3_moe_configs.sh"
