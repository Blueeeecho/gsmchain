#!/usr/bin/env bash
# submit_grpo_verl_vllm.sh
#
# Unified entry point for submitting ChainGSM GRPO (verl) training jobs to a
# remote SLURM cluster. Supports four prompt / reward variants in one file:
#
#   VERSION  | reward file                                       | data parquet              | config
#   ---------+---------------------------------------------------+---------------------------+------------------------------
#   v12      | reward_chaingsm_v12_json_verl.py                  | grpo_v12_json.parquet     | grpo_verl_v12_vllm.yaml
#   v12i     | reward_chaingsm_v12i_json_verl.py                 | grpo_v12_json.parquet     | grpo_verl_v12i_vllm.yaml
#   v13      | reward_chaingsm_v13_json_verl.py                  | grpo_v13_json.parquet     | grpo_verl_v13_vllm.yaml
#   v14      | reward_chaingsm_v14_reasoning_verl.py             | grpo_v14_reasoning.parquet| grpo_verl_v14_vllm.yaml
#
# v12 / v12i share the same data file (v12 prompt), differing only in reward.
# v14 uses 3 epochs (the others use 2) because its reasoning prompt is longer.
#
# Quick start:
#   1) Edit the USER CONFIG block below (at minimum: VERSION, optionally
#      REMOTE_MODEL_PATH / REMOTE_OUTPUT_DIR / GRPO_EPOCHS).
#   2) Dry-run to inspect the resolved sbatch command without submitting:
#         DRY_RUN=1 SKIP_PREFLIGHT=1 bash submit_grpo_verl_vllm.sh
#   3) Submit for real:
#         bash submit_grpo_verl_vllm.sh
#   4) Override the version from the CLI instead of editing the file:
#         bash submit_grpo_verl_vllm.sh --version v14
#
# What you can override (full list lives in remote_env.sh):
#   - VERSION                 v12 | v12i | v13 | v14     (also: --version CLI flag)
#   - REMOTE_ROOT             repo root on the remote host
#   - REMOTE_MODEL_PATH       base model checkpoint (default: Qwen2.5-0.5B-Instruct)
#   - REMOTE_OUTPUT_DIR       where checkpoints and slurm logs go
#   - REMOTE_DATA_DIR         where the parquet data lives (default: .../data/final/grpo)
#   - GRPO_EPOCHS / TOTAL_EPOCHS   training epochs (default per-version: v12/v12i/v13=2, v14=3)
#   - ROLLOUT_N, TP_SIZE      rollout count and tensor-parallel size (see remote_env.sh)
#   - SLURM_*                 partition / gpus / cpus / mem / account / qos
#   - DRY_RUN=1               print resolved command, do not submit
#   - SKIP_PREFLIGHT=1        skip the pre-flight check (faster iteration)
#
# How data is wired:
#   The verl yaml configs expect the training file to be named
#   `verl_grpo_train.parquet`. Our data files are named per version
#   (`grpo_v12_json.parquet`, `grpo_v13_json.parquet`, `grpo_v14_reasoning.parquet`).
#   This script places a symlink `verl_grpo_train.parquet -> <version source>`
#   under REMOTE_DATA_DIR (or refreshes it if stale). The symlink is the only
#   write this script performs on the remote repo.
#
# Companion docs:
#   - train_scripts/remote/V12_V14_HANDBOOK.md
#   - train_scripts/remote/V12i_HANDBOOK.md
#   - train_configs/remote/grpo_verl_v{12,12i,v13,v14}_vllm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# USER CONFIG: edit values, then run this script.
# All lines below use the ${VAR:-default} pattern so that env-var exports
# (and the --version CLI flag) override these defaults.
# ============================================================================
VERSION="${VERSION:-v12i}"                                                # one of: v12 | v12i | v13 | v14
REMOTE_ROOT="${REMOTE_ROOT:-$PWD}"
REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-${REMOTE_ROOT}/outputs/train/remote}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_ROOT}/chaingsm_data/data/final/grpo}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"                                           # default 2; v14 will be auto-bumped to 3
JOB_NAME="${JOB_NAME:-}"                                                  # auto-derived from VERSION if empty
DRY_RUN="${DRY_RUN:-0}"                                                   # 1 = print only, do not submit
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"                                     # 1 = skip preflight_remote.sh
# ============================================================================

# --- CLI parsing: allow `--version <v>` to override the top-of-file var ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      shift
      if [[ $# -lt 1 ]]; then
        echo "[submit_grpo_verl_vllm] ERROR: --version requires an argument" >&2
        exit 1
      fi
      VERSION="$1"
      shift
      ;;
    --version=*)
      VERSION="${1#--version=}"
      shift
      ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "[submit_grpo_verl_vllm] ERROR: unknown argument: $1" >&2
      echo "[submit_grpo_verl_vllm] Valid: --version <v12|v12i|v13|v14>" >&2
      exit 1
      ;;
  esac
done

# --- Validate VERSION and resolve per-version defaults ---
case "${VERSION}" in
  v12)
    VERSION_REWARD="reward_chaingsm_v12_json_verl.py"
    VERSION_DATA_FILE="grpo_v12_json.parquet"
    VERSION_DEFAULT_EPOCHS=2
    VERSION_DEFAULT_JOB="chaingsm-grpo-verl-v12"
    VERSION_TAG="v12"
    ;;
  v12i)
    VERSION_REWARD="reward_chaingsm_v12i_json_verl.py"
    VERSION_DATA_FILE="grpo_v12_json.parquet"
    VERSION_DEFAULT_EPOCHS=2
    VERSION_DEFAULT_JOB="chaingsm-grpo-verl-v12i"
    VERSION_TAG="v12i"
    ;;
  v13)
    VERSION_REWARD="reward_chaingsm_v13_json_verl.py"
    VERSION_DATA_FILE="grpo_v13_json.parquet"
    VERSION_DEFAULT_EPOCHS=2
    VERSION_DEFAULT_JOB="chaingsm-grpo-verl-v13"
    VERSION_TAG="v13"
    ;;
  v14)
    VERSION_REWARD="reward_chaingsm_v14_reasoning_verl.py"
    VERSION_DATA_FILE="grpo_v14_reasoning.parquet"
    VERSION_DEFAULT_EPOCHS=3
    VERSION_DEFAULT_JOB="chaingsm-grpo-verl-v14"
    VERSION_TAG="v14"
    ;;
  *)
    echo "[submit_grpo_verl_vllm] ERROR: unsupported VERSION='${VERSION}'" >&2
    echo "[submit_grpo_verl_vllm] Valid versions: v12 | v12i | v13 | v14" >&2
    exit 1
    ;;
esac

# --- Apply per-version defaults that the user did not override ---
# Use +u / -u so we can test for "unset" with ${VAR+x} without tripping nounset.
set +u
if [[ -z "${REMOTE_REWARD_PATH+x}" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/${VERSION_REWARD}"
fi
# If the user kept the default GRPO_EPOCHS=2 but picked a version whose default
# is 3 (only v14), bump it so v14 actually trains 3 epochs.
if [[ "${GRPO_EPOCHS}" == "2" && "${VERSION_DEFAULT_EPOCHS}" == "3" ]]; then
  GRPO_EPOCHS=3
  echo "[submit_grpo_verl_vllm] note: ${VERSION} default epochs = 3 (overriding default 2)"
fi
if [[ -z "${TOTAL_EPOCHS+x}" ]]; then
  TOTAL_EPOCHS="${GRPO_EPOCHS}"
fi
if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="${VERSION_DEFAULT_JOB}"
fi
set -u
export REMOTE_REWARD_PATH REMOTE_DATA_DIR GRPO_EPOCHS TOTAL_EPOCHS JOB_NAME

# --- Load shared remote helpers (SLURM flags, env exports, preflight, etc.) ---
source "${SCRIPT_DIR}/remote_env.sh"

# --- Defense check: warn if the resolved reward path does not match VERSION ---
if [[ "${REMOTE_REWARD_PATH}" != *"${VERSION_TAG}"* ]]; then
  echo "[${VERSION_TAG}] WARNING: REMOTE_REWARD_PATH does not contain '${VERSION_TAG}': ${REMOTE_REWARD_PATH}" >&2
  echo "[${VERSION_TAG}] (This is a ${VERSION} submission, expected a reward matching ${VERSION_TAG}.)" >&2
fi
if [[ "${REMOTE_DATA_DIR}" != *"/grpo"* ]]; then
  echo "[${VERSION_TAG}] WARNING: REMOTE_DATA_DIR is not under /grpo/: ${REMOTE_DATA_DIR}" >&2
fi

# --- Symlink the version's source parquet to verl_grpo_train.parquet ---
# (verl yaml configs reference the generic filename; our data files are
# version-suffixed.)
ensure_version_symlink() {
  local source_name="$1"
  local target="${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
  local source="${REMOTE_DATA_DIR}/${source_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[${VERSION_TAG}] DRY_RUN=1, skipping symlink/copy check"
    return 0
  fi
  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[${VERSION_TAG}] NOTE: ${target} already exists (not a symlink); leaving as-is"
    return 0
  fi
  if [[ -L "${target}" ]]; then
    local cur
    cur="$(readlink -f "${target}" 2>/dev/null || true)"
    if [[ "${cur}" == "${source}" ]]; then
      echo "[${VERSION_TAG}] symlink ok: ${target} -> ${source}"
      return 0
    fi
    echo "[${VERSION_TAG}] WARNING: stale symlink ${target} -> ${cur} (expected ${source}); re-creating"
    rm -f "${target}"
  fi
  if [[ ! -f "${source}" ]]; then
    echo "[${VERSION_TAG}] ERROR: source not found: ${source}" >&2
    echo "[${VERSION_TAG}] Put ${source_name} under ${REMOTE_DATA_DIR}/ and rerun" >&2
    return 1
  fi
  ln -s "${source}" "${target}"
  echo "[${VERSION_TAG}] created symlink: ${target} -> ${source}"
}

ensure_version_symlink "${VERSION_DATA_FILE}"

# --- Pre-flight check (optional via SKIP_PREFLIGHT=1) ---
remote_run_preflight grpo

# --- Assemble and submit ---
CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_${VERSION_TAG}_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
