#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-sft-verl}"

remote_run_preflight sft

CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.fsdp_sft_trainer --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name sft_verl trainer.total_epochs=${TOTAL_EPOCHS}"

remote_print_or_submit "${CMD}"
