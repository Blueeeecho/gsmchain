#!/bin/bash -l
# data_preprocess_v12.sh
#
# ChainGSM V12 dataset -> parquet preprocessor (SLURM one-shot).
#
# Output: ${REMOTE_DATA_DIR}/grpo_v12_json.parquet
# (V12 and V12i share this file; rerun before any V12/V12i training if you
# edited build_grpo_v12_json.py SYSTEM / USER_TEMPLATE.)
#
# Override:
#   REMOTE_ROOT=/path/to/math-chain   bash data_preprocess_v12.sh
#   REMOTE_DATA_DIR=.../grpo          bash data_preprocess_v12.sh
#   DRY_RUN=1                         bash data_preprocess_v12.sh   (print only)
#   SKIP_PREPROCESS=1                 bash data_preprocess_v12.sh   (assume parquet exists)

#SBATCH -J chaingsm-preprocess-v12
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem 32GB
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com

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

REMOTE_ROOT="${REMOTE_ROOT:-/home/wwq416/snap/wwq/math-chain}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_ROOT}/chaingsm_data/data/final/grpo}"
BUILD_SCRIPT="${REMOTE_ROOT}/chaingsm_data/data/final/sft/build_grpo_v12_json.py"
OUT_PARQUET="${REMOTE_DATA_DIR}/grpo_v12_json.parquet"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

mkdir -p "${REMOTE_DATA_DIR}" ./all_logs

echo "[v12-preprocess] REMOTE_ROOT=${REMOTE_ROOT}"
echo "[v12-preprocess] REMOTE_DATA_DIR=${REMOTE_DATA_DIR}"
echo "[v12-preprocess] BUILD_SCRIPT=${BUILD_SCRIPT}"
echo "[v12-preprocess] OUT_PARQUET=${OUT_PARQUET}"

CMD="cd ${REMOTE_ROOT} && python ${BUILD_SCRIPT}"
echo "[v12-preprocess] Resolved command:"
echo "${CMD}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[v12-preprocess] DRY_RUN=1, skipping build"
  exit 0
fi

if [[ "${SKIP_PREPROCESS}" == "1" ]]; then
  echo "[v12-preprocess] SKIP_PREPROCESS=1, assuming ${OUT_PARQUET} exists"
  exit 0
fi

eval "${CMD}"
echo "[v12-preprocess] Done. Output: ${OUT_PARQUET}"
nvidia-smi
