#!/bin/bash -l
# submit_grpo_v12_pipeline.sh
#
# ChainGSM V12i end-to-end pipeline (one sbatch):
#   1) data_preprocess_v12.sh   (raw + supplementary -> grpo_v12_json.parquet)
#   2) V12i verl GRPO training  (uses grpo_v12_json.parquet)
#
# V12 and V12i share the same parquet (v12 prompt + v12i reward).
# Default end is V12i (4-component LCS reward).
# To run V12 instead:  VERSION=v12 bash submit_grpo_v12_pipeline.sh
#
# Override:
#   REMOTE_ROOT=...              bash submit_grpo_v12_pipeline.sh
#   DRY_RUN=1                    bash submit_grpo_v12_pipeline.sh  (print only)
#   SKIP_PREPROCESS=1            bash submit_grpo_v12_pipeline.sh  (reuse existing parquet)

#SBATCH -J chaingsm-grpo-v12i-pipeline
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32GB
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com

set -euo pipefail

VERSION="${VERSION:-v12i}"
case "${VERSION}" in
  v12|v12i) ;;
  *) echo "[v12-pipeline] unsupported VERSION=${VERSION} (use v12 or v12i)"; exit 2 ;;
esac

REMOTE_ROOT="${REMOTE_ROOT:-/home/wwq416/snap/wwq/math-chain}"
REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/outputs/train/remote}"
SCRIPT_DIR="${REMOTE_ROOT}/train_scripts/remote"

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

echo "[v12-pipeline] VERSION=${VERSION}"
echo "[v12-pipeline] Step 1/2: data preprocess (raw -> grpo_v12_json.parquet)"
echo "[v12-pipeline] Step 2/2: V12i verl GRPO training"

CMD_PRE="bash ${SCRIPT_DIR}/data_preprocess_v12.sh"

if [[ "${VERSION}" == "v12" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12_json_verl.py"
  CONFIG_NAME="grpo_verl_v12_vllm"
  JOB_TAG="v12"
  TOTAL_EPOCHS="${V12_TOTAL_EPOCHS:-2}"
else
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12i_json_verl.py"
  CONFIG_NAME="grpo_verl_v12i_vllm"
  JOB_TAG="v12i"
  TOTAL_EPOCHS="${V12I_TOTAL_EPOCHS:-2}"
fi
export REMOTE_REWARD_PATH

CMD_TRAIN="cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name ${CONFIG_NAME} actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-${JOB_TAG}/checkpoints trainer.total_epochs=${TOTAL_EPOCHS}"

echo "[v12-pipeline] Preprocess command:"
echo "  ${CMD_PRE}"
echo "[v12-pipeline] Train command (${VERSION}, epochs=${TOTAL_EPOCHS}, config=${CONFIG_NAME}):"
echo "  ${CMD_TRAIN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[v12-pipeline] DRY_RUN=1, skipping actual run"
  exit 0
fi

eval "${CMD_PRE}"
eval "${CMD_TRAIN}"

echo "[v12-pipeline] Done. Ckpts: ${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-${JOB_TAG}/checkpoints"
nvidia-smi
