#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-dpo-trl}"

remote_run_preflight dpo

CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m train_pipeline.train_dpo_trl --config ${REMOTE_ROOT}/train_configs/remote/dpo_trl.yaml --set training.num_train_epochs=${TOTAL_EPOCHS}"

remote_print_or_submit "${CMD}"
