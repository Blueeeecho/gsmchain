# Remote Chain Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the four remote submission chains with the existing A100 SLURM server while preserving local `math_chain_verl` training behavior.

**Architecture:** Add a shared remote shell environment module, a read-only preflight script, and dry-run-aware submit scripts for SFT, DPO, GRPO, and SFT-to-GRPO. Add pytest coverage that validates command construction locally without requiring SLURM, and update stale local docs so they consistently point to `math_chain_verl`.

**Tech Stack:** Bash, SLURM `sbatch`, verl, TRL, pytest, Python standard library.

---

### Task 1: Add Remote Submission Dry-Run Tests

**Files:**
- Create: `tests/test_remote_submission_scripts.py`
- Modify: none
- Test: `tests/test_remote_submission_scripts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_remote_submission_scripts.py` with this content:

```python
from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_DIR = REPO_ROOT / "train_scripts" / "remote"
REMOTE_ROOT = "/export/home/asifali/math-chain"
REMOTE_DATA_DIR = f"{REMOTE_ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946"
REMOTE_OUTPUT_DIR = f"{REMOTE_ROOT}/outputs/train/remote"
REMOTE_MODEL_PATH = "/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct"
REMOTE_EVAL_DATA_PATH = f"{REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl"
REMOTE_REWARD_PATH = f"{REMOTE_ROOT}/train_pipeline/reward_chaingsm.py"


def run_submit_script(script_name: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PREFLIGHT": "1",
            "REMOTE_ROOT": REMOTE_ROOT,
            "REMOTE_MODEL_PATH": REMOTE_MODEL_PATH,
            "REMOTE_DATA_DIR": REMOTE_DATA_DIR,
            "REMOTE_OUTPUT_DIR": REMOTE_OUTPUT_DIR,
            "REMOTE_EVAL_DATA_PATH": REMOTE_EVAL_DATA_PATH,
            "REMOTE_REWARD_PATH": REMOTE_REWARD_PATH,
            "TOTAL_EPOCHS": "1",
            "GRPO_EPOCHS": "1",
            "ROLLOUT_N": "8",
            "TP_SIZE": "2",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(REMOTE_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_common_sbatch_flags(output: str) -> None:
    assert "sbatch" in output
    assert "--partition=gpu-A100" in output
    assert "--gres=gpu:4" in output
    assert "--cpus-per-task=128" in output
    assert "--mem=256GB" in output
    assert "--account=A100" in output
    assert "--qos=a100_qos" in output
    assert "--output=" in output
    assert "--error=" in output
    assert "--export=ALL," in output
    assert "REMOTE_ROOT=/export/home/asifali/math-chain" in output
    assert "REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct" in output


def test_sft_verl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_sft_verl.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.fsdp_sft_trainer" in output
    assert "--config-name sft_verl" in output
    assert "trainer.total_epochs=1" in output


def test_dpo_trl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_dpo_trl.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "train_pipeline.train_dpo_trl" in output
    assert "train_configs/remote/dpo_trl.yaml" in output
    assert "--set training.num_train_epochs=1" in output


def test_grpo_verl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_grpo_verl_vllm.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.main_ppo" in output
    assert "--config-name grpo_verl_vllm" in output
    assert "actor_rollout_ref.rollout.n=8" in output
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in output


def test_sft_then_grpo_dry_run_chains_stage_commands() -> None:
    result = run_submit_script("submit_sft_then_grpo_verl_vllm.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.fsdp_sft_trainer" in output
    assert "verl.trainer.main_ppo" in output
    assert "stage1_sft/checkpoints" in output
    assert "stage2_grpo/checkpoints" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_remote_submission_scripts.py -q
```

Expected:

```text
4 failed
```

The current scripts do not support the expected shared resource flags and dry-run output.

### Task 2: Add Shared Remote Environment Module

**Files:**
- Create: `train_scripts/remote/remote_env.sh`
- Modify: none
- Test: `tests/test_remote_submission_scripts.py`

- [ ] **Step 1: Create the shared shell module**

Create `train_scripts/remote/remote_env.sh` with this content:

```bash
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
  mkdir -p "${SLURM_LOG_DIR}"
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
  printf ' --export=%q' "ALL,REMOTE_ROOT=${REMOTE_ROOT},PYTHON=${PYTHON},REMOTE_MODEL_PATH=${REMOTE_MODEL_PATH},REMOTE_DATA_DIR=${REMOTE_DATA_DIR},REMOTE_OUTPUT_DIR=${REMOTE_OUTPUT_DIR},REMOTE_EVAL_DATA_PATH=${REMOTE_EVAL_DATA_PATH},REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH},REMOTE_CONDA_ENV=${REMOTE_CONDA_ENV},REMOTE_CUDA_MODULE=${REMOTE_CUDA_MODULE},HF_HOME=${HF_HOME},HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
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
```

- [ ] **Step 2: Run shell syntax check**

Run:

```bash
bash -n train_scripts/remote/remote_env.sh
```

Expected:

```text
```

The command should exit with status 0 and no output.

### Task 3: Add Read-Only Remote Preflight Script

**Files:**
- Create: `train_scripts/remote/preflight_remote.sh`
- Modify: none
- Test: local syntax and usage checks

- [ ] **Step 1: Create the preflight script**

Create `train_scripts/remote/preflight_remote.sh` with this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

MODE="${1:-all}"
FAILURES=0

usage() {
  cat <<'EOF'
Usage: preflight_remote.sh [all|sft|dpo|grpo|sft_then_grpo]

Checks the configured remote ChainGSM paths, SLURM commands, and Python imports.
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
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x train_scripts/remote/preflight_remote.sh
```

Expected:

```text
```

- [ ] **Step 3: Verify syntax and help output**

Run:

```bash
bash -n train_scripts/remote/preflight_remote.sh
bash train_scripts/remote/preflight_remote.sh --help
```

Expected help output includes:

```text
Usage: preflight_remote.sh [all|sft|dpo|grpo|sft_then_grpo]
```

### Task 4: Refactor Four Remote Submit Scripts

**Files:**
- Modify: `train_scripts/remote/submit_sft_verl.sh`
- Modify: `train_scripts/remote/submit_dpo_trl.sh`
- Modify: `train_scripts/remote/submit_grpo_verl_vllm.sh`
- Modify: `train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh`
- Test: `tests/test_remote_submission_scripts.py`

- [ ] **Step 1: Replace SFT submit script**

Replace `train_scripts/remote/submit_sft_verl.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-sft-verl}"

remote_run_preflight sft

CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.fsdp_sft_trainer --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name sft_verl trainer.total_epochs=${TOTAL_EPOCHS}"

remote_print_or_submit "${CMD}"
```

- [ ] **Step 2: Replace DPO submit script**

Replace `train_scripts/remote/submit_dpo_trl.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-dpo-trl}"

remote_run_preflight dpo

CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m train_pipeline.train_dpo_trl --config ${REMOTE_ROOT}/train_configs/remote/dpo_trl.yaml --set training.num_train_epochs=${TOTAL_EPOCHS}"

remote_print_or_submit "${CMD}"
```

- [ ] **Step 3: Replace GRPO submit script**

Replace `train_scripts/remote/submit_grpo_verl_vllm.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_env.sh"

JOB_NAME="${JOB_NAME:-chaingsm-grpo-verl-vllm}"

remote_run_preflight grpo

CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
```

- [ ] **Step 4: Replace SFT-to-GRPO submit script**

Replace `train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh` with:

```bash
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
```

- [ ] **Step 5: Verify syntax**

Run:

```bash
bash -n train_scripts/remote/submit_sft_verl.sh
bash -n train_scripts/remote/submit_dpo_trl.sh
bash -n train_scripts/remote/submit_grpo_verl_vllm.sh
bash -n train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

Expected:

```text
```

- [ ] **Step 6: Run remote submission tests**

Run:

```bash
pytest tests/test_remote_submission_scripts.py -q
```

Expected:

```text
4 passed
```

### Task 5: Add Local Environment Consistency Tests

**Files:**
- Create: `tests/test_local_environment_docs.py`
- Modify: none
- Test: `tests/test_local_environment_docs.py`

- [ ] **Step 1: Write local consistency tests**

Create `tests/test_local_environment_docs.py` with this content:

```python
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = "/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"
LOCAL_SCRIPTS = [
    "train_scripts/local/run_preprocess.sh",
    "train_scripts/local/run_sft.sh",
    "train_scripts/local/run_dpo.sh",
    "train_scripts/local/run_grpo.sh",
    "train_scripts/local/run_grpo_verl.sh",
    "train_scripts/local/run_sft_then_grpo.sh",
]
CURRENT_DOCS = [
    "README.md",
    "train_readme.md",
    "baseline_readme.md",
    "code/README.md",
]


def test_local_scripts_default_to_math_chain_verl_python() -> None:
    for script in LOCAL_SCRIPTS:
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert LOCAL_PYTHON in text, script


def test_local_scripts_have_valid_bash_syntax() -> None:
    for script in LOCAL_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / script)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script}\n{result.stderr}"


def test_current_docs_do_not_present_math_noise_as_current_env() -> None:
    for doc in CURRENT_DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        assert "/home/wwq416/miniconda3/envs/math-noise/bin/python" not in text, doc
        assert "conda run --no-capture-output -n math-noise" not in text, doc
```

- [ ] **Step 2: Run tests to verify they fail on stale docs if needed**

Run:

```bash
pytest tests/test_local_environment_docs.py -q
```

Expected before doc cleanup:

```text
1 failed, 2 passed
```

The failing test should identify stale `math-noise` documentation.

### Task 6: Update Local Documentation to `math_chain_verl`

**Files:**
- Modify: `train_readme.md`
- Modify: `baseline_readme.md`
- Modify: `code/README.md`
- Test: `tests/test_local_environment_docs.py`

- [ ] **Step 1: Replace current-environment references**

Update current local environment references:

```text
/home/wwq416/miniconda3/envs/math-noise/bin/python
```

to:

```text
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
```

Update command examples:

```text
conda run --no-capture-output -n math-noise python
```

to:

```text
conda run --no-capture-output -n math_chain_verl python
```

If any section describes old historical results rather than current commands,
rewrite the sentence so it explicitly says `math-noise` was a historical
environment and is not the current recommended environment.

- [ ] **Step 2: Run local consistency tests**

Run:

```bash
pytest tests/test_local_environment_docs.py -q
```

Expected:

```text
3 passed
```

### Task 7: Update Remote Server Documentation

**Files:**
- Modify: `docs/remote-server-settings.md`
- Modify: `README.md`
- Test: shell grep checks

- [ ] **Step 1: Document the new shared remote layer**

In `docs/remote-server-settings.md`, add a section named `Implemented ChainGSM Remote Layer` with:

```markdown
## Implemented ChainGSM Remote Layer

The current repository uses `train_scripts/remote/remote_env.sh` as the shared
remote configuration layer. The four submit scripts source it before building
their chain-specific commands.

Use `DRY_RUN=1 SKIP_PREFLIGHT=1` to inspect commands locally:

```bash
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

Run the read-only remote preflight on the remote server:

```bash
bash train_scripts/remote/preflight_remote.sh all
```
```

In `README.md`, add a concise pointer to `docs/remote-server-settings.md` and
the dry-run commands.

- [ ] **Step 2: Verify docs mention the new scripts**

Run:

```bash
grep -n "remote_env.sh" docs/remote-server-settings.md README.md
grep -n "preflight_remote.sh" docs/remote-server-settings.md README.md
```

Expected:

```text
docs/remote-server-settings.md:<line containing remote_env.sh or preflight_remote.sh>
README.md:<line containing remote_env.sh or preflight_remote.sh>
```

The exact line numbers may differ.

### Task 8: Final Verification

**Files:**
- Modify: none
- Test: all scripts and tests touched in this plan

- [ ] **Step 1: Run shell syntax checks**

Run:

```bash
bash -n train_scripts/remote/remote_env.sh
bash -n train_scripts/remote/preflight_remote.sh
bash -n train_scripts/remote/submit_sft_verl.sh
bash -n train_scripts/remote/submit_dpo_trl.sh
bash -n train_scripts/remote/submit_grpo_verl_vllm.sh
bash -n train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
bash -n train_scripts/local/run_preprocess.sh
bash -n train_scripts/local/run_sft.sh
bash -n train_scripts/local/run_dpo.sh
bash -n train_scripts/local/run_grpo.sh
bash -n train_scripts/local/run_grpo_verl.sh
bash -n train_scripts/local/run_sft_then_grpo.sh
```

Expected:

```text
```

- [ ] **Step 2: Run pytest checks**

Run:

```bash
pytest tests/test_remote_submission_scripts.py tests/test_local_environment_docs.py -q
```

Expected:

```text
7 passed
```

- [ ] **Step 3: Run dry-run smoke checks**

Run:

```bash
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

Expected:

```text
[remote] Resolved command:
[remote] Submission command:
```

Each output should include A100 SLURM flags and the chain-specific Python module.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat
git diff --name-only
```

Expected changed files include:

```text
docs/remote-server-settings.md
README.md
baseline_readme.md
code/README.md
train_readme.md
train_scripts/remote/preflight_remote.sh
train_scripts/remote/remote_env.sh
train_scripts/remote/submit_dpo_trl.sh
train_scripts/remote/submit_grpo_verl_vllm.sh
train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
train_scripts/remote/submit_sft_verl.sh
tests/test_local_environment_docs.py
tests/test_remote_submission_scripts.py
```

- [ ] **Step 5: Commit**

Run:

```bash
git add docs/remote-server-settings.md README.md baseline_readme.md code/README.md train_readme.md train_scripts/remote tests
git commit -m "feat: align local and remote training launch scripts"
```

Expected:

```text
[master <hash>] feat: align local and remote training launch scripts
```
