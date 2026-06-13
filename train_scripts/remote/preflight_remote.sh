#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

MODE="${1:-all}"
FAILURES=0

usage() {
  cat <<'EOF'
Usage: preflight_remote.sh [all|sft|dpo|grpo|sft_then_grpo]

Checks the configured remote ChainGSM paths, SLURM commands, environment, and Python imports.
This script is read-only and does not install packages or modify the remote env.
EOF
}

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  usage
  exit 0
fi

pass() {
  printf '[PASS] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  FAILURES=$((FAILURES + 1))
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    pass "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    pass "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_command() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    pass "command available: ${name}"
  else
    fail "command missing: ${name}"
  fi
}

check_cuda_module() {
  if ! command -v module >/dev/null 2>&1; then
    warn "module command unavailable; cannot verify CUDA module ${REMOTE_CUDA_MODULE}"
    return 0
  fi
  if module load "${REMOTE_CUDA_MODULE}" >/tmp/chaingsm_remote_cuda_module.txt 2>&1; then
    pass "CUDA module loadable: ${REMOTE_CUDA_MODULE}"
  else
    fail "CUDA module cannot be loaded: ${REMOTE_CUDA_MODULE}; see /tmp/chaingsm_remote_cuda_module.txt"
  fi
}

check_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    warn "conda command unavailable; cannot verify conda env ${REMOTE_CONDA_ENV}"
    return 0
  fi
  if conda env list | awk '{print $1}' | grep -Fxq "${REMOTE_CONDA_ENV}"; then
    pass "conda env available: ${REMOTE_CONDA_ENV}"
  else
    fail "conda env missing: ${REMOTE_CONDA_ENV}"
  fi
}

check_python_imports() {
  if "${PYTHON}" - <<'PY' >/tmp/chaingsm_remote_imports.txt 2>&1
import importlib

for name in ("torch", "transformers", "datasets", "vllm", "verl"):
    importlib.import_module(name)
print("imports ok")
PY
  then
    pass "Python imports: torch transformers datasets vllm verl"
  else
    fail "Python imports failed with ${PYTHON}; see /tmp/chaingsm_remote_imports.txt"
  fi
}

check_slurm_partition() {
  if ! command -v sinfo >/dev/null 2>&1; then
    warn "sinfo unavailable; cannot verify SLURM partition ${SLURM_PARTITION}"
    return 0
  fi
  if sinfo -h -p "${SLURM_PARTITION}" >/dev/null 2>&1; then
    pass "SLURM partition visible: ${SLURM_PARTITION}"
  else
    fail "SLURM partition not visible: ${SLURM_PARTITION}"
  fi
}

check_gpu_count() {
  if "${PYTHON}" - <<PY >/tmp/chaingsm_remote_gpu.txt 2>&1
import torch
expected = int("${SLURM_GPUS}")
actual = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(actual)
raise SystemExit(0 if actual == 0 or actual >= expected else 1)
PY
  then
    pass "GPU count check is compatible outside or inside allocation"
  else
    fail "GPU count below expected ${SLURM_GPUS}; see /tmp/chaingsm_remote_gpu.txt"
  fi
}

check_common() {
  check_dir "REMOTE_ROOT" "${REMOTE_ROOT}"
  check_dir "REMOTE_DATA_DIR" "${REMOTE_DATA_DIR}"
  check_dir "REMOTE_MODEL_PATH" "${REMOTE_MODEL_PATH}"
  check_file "REMOTE_REWARD_PATH" "${REMOTE_REWARD_PATH}"
  check_file "REMOTE_EVAL_DATA_PATH" "${REMOTE_EVAL_DATA_PATH}"
  check_command sbatch
  check_command sinfo
  check_cuda_module
  check_conda_env
  check_slurm_partition
  check_python_imports
  check_gpu_count
}

check_chain_data() {
  case "$1" in
    sft)
      check_file "SFT parquet" "${REMOTE_DATA_DIR}/verl_sft_train.parquet"
      ;;
    dpo)
      check_file "DPO jsonl" "${REMOTE_DATA_DIR}/dpo_train.jsonl"
      ;;
    grpo)
      check_file "GRPO parquet" "${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
      ;;
    sft_then_grpo)
      check_file "SFT parquet" "${REMOTE_DATA_DIR}/verl_sft_train.parquet"
      check_file "GRPO parquet" "${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
      ;;
    all)
      check_chain_data sft
      check_chain_data dpo
      check_chain_data grpo
      ;;
    *)
      fail "unsupported preflight mode: $1"
      ;;
  esac
}

check_common
check_chain_data "${MODE}"

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "[remote] Preflight failed with ${FAILURES} failure(s)."
  exit 1
fi

echo "[remote] Preflight passed for mode: ${MODE}"
