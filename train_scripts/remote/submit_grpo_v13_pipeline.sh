#!/bin/bash -l
# submit_grpo_v13_pipeline.sh
#
# ChainGSM V13 end-to-end pipeline (one sbatch):
#   1) data_preprocess_v13.sh   (raw + supplementary -> grpo_v13_json.parquet)
#   2) V13 verl GRPO training  (uses grpo_v13_json.parquet)
#
# Override:
#   REMOTE_ROOT=...              bash submit_grpo_v13_pipeline.sh
#   DRY_RUN=1                    bash submit_grpo_v13_pipeline.sh  (print only)
#   SKIP_PREPROCESS=1            bash submit_grpo_v13_pipeline.sh  (reuse existing parquet)
#   V13_TOTAL_EPOCHS=3           bash submit_grpo_v13_pipeline.sh  (override epochs)

#SBATCH -J chaingsm-grpo-v13-pipeline
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32GB
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com

set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home/wwq416/snap/wwq/math-chain}"
REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/outputs/train/remote}"
SCRIPT_DIR="${REMOTE_ROOT}/train_scripts/remote"
REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v13_json_verl.py"
CONFIG_NAME="grpo_verl_v13_vllm"
JOB_TAG="v13"
TOTAL_EPOCHS="${V13_TOTAL_EPOCHS:-2}"

if command -v module >/dev/null 2>&1; then
  module load cuda12.4/toolkit
else
  echo "[env] module command not found; assuming CUDA is already available"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"
if command -v conda >/dev/null 2>&1; then
  source activate Reasoning360 || echo "[env] conda activate Reasoning360 failed; using current env"
else
  echo "[env] conda not found; using current Python environment"
fi

mkdir -p ./all_logs

echo "[v13-pipeline] Step 1/2: data preprocess (raw -> grpo_v13_json.parquet)"
echo "[v13-pipeline] Step 2/2: V13 verl GRPO training (epochs=${TOTAL_EPOCHS}, config=${CONFIG_NAME})"

CMD_PRE="bash ${SCRIPT_DIR}/data_preprocess_v13.sh"
CMD_TRAIN="cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name ${CONFIG_NAME} actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-${JOB_TAG}/checkpoints trainer.total_epochs=${TOTAL_EPOCHS}"

echo "[v13-pipeline] Preprocess command:"
echo "  ${CMD_PRE}"
echo "[v13-pipeline] Train command:"
echo "  ${CMD_TRAIN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[v13-pipeline] DRY_RUN=1, skipping actual run"
  exit 0
fi

eval "${CMD_PRE}"
eval "${CMD_TRAIN}"

echo "[v13-pipeline] Done. Ckpts: ${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-${JOB_TAG}/checkpoints"
nvidia-smi
