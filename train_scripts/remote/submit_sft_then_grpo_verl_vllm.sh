#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-sft-then-grpo-verl}"
SFT_STAGE_DIR="${REMOTE_OUTPUT_DIR}/stage1_sft/checkpoints"
GRPO_STAGE_DIR="${REMOTE_OUTPUT_DIR}/stage2_grpo/checkpoints"

remote_run_preflight sft_then_grpo

SFT_CMD="${PYTHON} -m verl.trainer.fsdp_sft_trainer --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name sft_verl trainer.default_local_dir=${SFT_STAGE_DIR} trainer.total_epochs=${TOTAL_EPOCHS}"
GRPO_CMD="${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_vllm actor_rollout_ref.model.path=${SFT_STAGE_DIR} trainer.default_local_dir=${GRPO_STAGE_DIR} trainer.total_epochs=${GRPO_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
CMD="cd ${REMOTE_ROOT} && ${SFT_CMD} && ${GRPO_CMD}"

remote_print_or_submit "${CMD}"
