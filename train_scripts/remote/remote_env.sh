#!/usr/bin/env bash

remote_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_repo_root="$(cd "${remote_env_dir}/../.." && pwd)"

REMOTE_ROOT="${REMOTE_ROOT:-$PWD}"
PYTHON="${PYTHON:-python3}"

REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/outputs/train/remote}"
REMOTE_EVAL_DATA_PATH="${REMOTE_EVAL_DATA_PATH:-${REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl}"
REMOTE_REWARD_PATH="${REMOTE_REWARD_PATH:-${REMOTE_ROOT}/train_pipeline/reward_chaingsm.py}"

REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-Reasoning360}"
REMOTE_CUDA_MODULE="${REMOTE_CUDA_MODULE:-cuda12.4/toolkit}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/export/home/asifali/HF_cache}"
HF_HOME="${HF_HOME:-${REMOTE_HF_HOME}}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REMOTE_HF_HOME}}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REMOTE_OUTPUT_DIR}/slurm_logs}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu-A100}"
SLURM_GPUS="${SLURM_GPUS:-4}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"
SLURM_MEM="${SLURM_MEM:-256GB}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-A100}"
SLURM_QOS="${SLURM_QOS:-a100_qos}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

remote_export_env() {
  export REMOTE_ROOT
  export PYTHON
  export REMOTE_MODEL_PATH
  export REMOTE_DATA_DIR
  export REMOTE_OUTPUT_DIR
  export REMOTE_EVAL_DATA_PATH
  export REMOTE_REWARD_PATH
  export REMOTE_CONDA_ENV
  export REMOTE_CUDA_MODULE
  export HF_HOME
  export HF_DATASETS_CACHE
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"
  export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
}

remote_job_prelude() {
  cat <<EOF
set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module load "${REMOTE_CUDA_MODULE}"
else
  echo "[remote_env] module command not found; assuming CUDA is already available"
fi
if command -v conda >/dev/null 2>&1; then
  source activate "${REMOTE_CONDA_ENV}"
else
  echo "[remote_env] conda command not found; using current Python environment"
fi
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset ROCR_VISIBLE_DEVICES
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
LOCAL_BASE="/var/tmp/\${USER}/\${SLURM_JOB_ID:-manual}"
export RAY_TMPDIR="\${LOCAL_BASE}/ray"
export TMPDIR="\${LOCAL_BASE}/tmp"
export TMP="\${TMPDIR}"
export TEMP="\${TMPDIR}"
mkdir -p "\${RAY_TMPDIR}" "\${TMPDIR}"
chmod 700 "\${LOCAL_BASE}" "\${RAY_TMPDIR}" "\${TMPDIR}" 2>/dev/null || true
export RAY_DISABLE_DASHBOARD=1
unset RAY_ADDRESS RAY_HEAD_IP RAY_PORT
ray stop -f >/dev/null 2>&1 || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
ulimit -n 1048576 2>/dev/null || true
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset CUDA_LAUNCH_BLOCKING
unset CUDA_DEVICE_MAX_CONNECTIONS
EOF
}

remote_sbatch_prefix() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${SLURM_LOG_DIR}"
  fi
  printf 'sbatch'
  printf ' --job-name=%q' "${JOB_NAME}"
  printf ' --partition=%q' "${SLURM_PARTITION}"
  printf ' --gres=%q' "gpu:${SLURM_GPUS}"
  printf ' --cpus-per-task=%q' "${SLURM_CPUS_PER_TASK}"
  printf ' --mem=%q' "${SLURM_MEM}"
  printf ' --account=%q' "${SLURM_ACCOUNT}"
  printf ' --qos=%q' "${SLURM_QOS}"
  printf ' --output=%q' "${SLURM_LOG_DIR}/%j-%x.out"
  printf ' --error=%q' "${SLURM_LOG_DIR}/%j-%x.err"
  printf ' --export=%s' "ALL,REMOTE_ROOT=${REMOTE_ROOT},PYTHON=${PYTHON},REMOTE_MODEL_PATH=${REMOTE_MODEL_PATH},REMOTE_DATA_DIR=${REMOTE_DATA_DIR},REMOTE_OUTPUT_DIR=${REMOTE_OUTPUT_DIR},REMOTE_EVAL_DATA_PATH=${REMOTE_EVAL_DATA_PATH},REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH},REMOTE_CONDA_ENV=${REMOTE_CONDA_ENV},REMOTE_CUDA_MODULE=${REMOTE_CUDA_MODULE},HF_HOME=${HF_HOME},HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
}

remote_print_or_submit() {
  local cmd="$1"
  remote_export_env
  local sbatch_prefix
  sbatch_prefix="$(remote_sbatch_prefix)"
  local wrapped
  wrapped="$(remote_job_prelude)
cd \"${REMOTE_ROOT}\"
${cmd}"
  printf '[remote] Resolved command:\n%s\n' "${cmd}"
  printf '[remote] Submission command:\n%s --wrap=%q\n' "${sbatch_prefix}" "${wrapped}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${sbatch_prefix} --wrap=$(printf '%q' "${wrapped}")"
}

remote_run_preflight() {
  if [[ "${SKIP_PREFLIGHT}" == "1" ]]; then
    echo "[remote] SKIP_PREFLIGHT=1; skipping preflight."
    return 0
  fi
  "${remote_env_dir}/preflight_remote.sh" "$@"
}
