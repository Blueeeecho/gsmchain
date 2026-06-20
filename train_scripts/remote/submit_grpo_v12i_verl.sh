#!/usr/bin/env bash

#SBATCH -J MTMT-A100 #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:2 #number of gpus needed, default is 1
#SBATCH -c 60  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit
nvidia-smi
source activate Reasoning360

# submit_grpo_v12i_verl.sh
#
# 一键提交 V12i GRPO 训练到远程 SLURM 集群.
# 跟 submit_grpo_verl_vllm.sh 同结构, 但默认配置指向 V12i:
#   - reward  : reward_chaingsm_v12i_json_verl.py (4 项, 跟 V12 公式同构)
#   - data    : grpo_v12_json.parquet (V12 prompt 已内嵌, 3051 行, 5 类别)
#   - config  : train_configs/remote/grpo_verl_v12i_vllm.yaml
#   - epochs  : 2 (跟本地 V12i 一致, 老 V10 config 是 1)
#   - save_freq: 300 (跟本地 V12i 一致, 老 V10 config 是 50)
#
# 用法:
#   # 1) 干跑 (DRY_RUN), 自检提交命令, 不真提交
#   DRY_RUN=1 SKIP_PREFLIGHT=1 bash submit_grpo_v12i_verl.sh
#
#   # 2) 跳过预检, 直接提交
#   SKIP_PREFLIGHT=1 bash submit_grpo_v12i_verl.sh
#
#   # 3) 完整流程 (预检 + 提交)
#   bash submit_grpo_v12i_verl.sh
#
#   # 4) 自定义变量 (覆盖默认值)
#   JOB_NAME=my-v12i-run \
#   REMOTE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
#   REMOTE_ROOT=/path/to/repo \
#   bash submit_grpo_v12i_verl.sh
#
# V12i 训练量: 2 epoch × 762 steps/epoch = 1524 total steps
# V12i 保存: 5 ckpts @ 300/600/900/1200/1500
# V12i 评测: 训练完后用 train_scripts/local/eval_v12_5ckpts.sh 跑
#
# 配套文档: V12i_HANDBOOK.md §5 (远程提交完整流程)
# 配套: train_configs/remote/grpo_verl_v12i_vllm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === V12i 关键覆盖: 在 source remote_env.sh 之前先注入 V12i 默认值 ===
# 实现: 临时关闭 nounset (set +u) 以容忍可能未定义的变量, 然后用普通 if 判断
# "未设则赋值" 模式. 这样后续 source remote_env.sh 时, 内部那些
# "REMOTE_DATA_DIR=${REMOTE_DATA_DIR:-old_default}" 行就不会把我们的值覆盖回去.
set +u
if [[ -z "${REMOTE_REWARD_PATH+x}" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12i_json_verl.py"
fi
if [[ -z "${REMOTE_DATA_DIR+x}" ]]; then
  REMOTE_DATA_DIR="${REMOTE_ROOT}/chaingsm_data/data/final/grpo"
fi
set -u
export REMOTE_REWARD_PATH REMOTE_DATA_DIR
# V12i 训练量: 2 epoch (本地 V12i 是 2, 跟 V10 config 的 1 不一样)
if [[ -z "${GRPO_EPOCHS+x}" ]]; then
  GRPO_EPOCHS=2
fi
if [[ -z "${TOTAL_EPOCHS+x}" ]]; then
  TOTAL_EPOCHS="${GRPO_EPOCHS}"
fi
export GRPO_EPOCHS TOTAL_EPOCHS

# 复用通用远程环境加载器 (SLURM, conda, vLLM, NCCL 等所有 helper 都在 remote_env.sh 里)
source "${SCRIPT_DIR}/remote_env.sh"

# 防御: 万一上面 source 过程有副作用 (不应该, 但保险起见)
if [[ "${REMOTE_REWARD_PATH}" != *"v12i"* ]]; then
  echo "[v12i] WARNING: REMOTE_REWARD_PATH 不含 'v12i': ${REMOTE_REWARD_PATH}" >&2
  echo "[v12i] (这是 V12i 提交, 应该是 reward_chaingsm_v12i_json_verl.py)" >&2
fi
if [[ "${REMOTE_DATA_DIR}" != *"/grpo"* ]]; then
  echo "[v12i] WARNING: REMOTE_DATA_DIR 不在 /grpo/ 下: ${REMOTE_DATA_DIR}" >&2
fi

# 远程 grpo_verl_v12i_vllm.yaml 期望的 parquet 文件名是 verl_grpo_train.parquet
# 我们的 V12i 数据文件叫 grpo_v12_json.parquet.
# 这里在 REMOTE_DATA_DIR 下放一个 symlink/copy 兜底 (不覆盖真源文件, DRY_RUN 时跳过).
ensure_v12i_symlink() {
  local target="${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
  local source="${REMOTE_DATA_DIR}/grpo_v12_json.parquet"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[v12i] DRY_RUN=1, skip symlink/copy check"
    return 0
  fi
  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[v12i] NOTE: ${target} already exists (not a symlink); leaving as-is"
    return 0
  fi
  if [[ -L "${target}" ]]; then
    local cur; cur="$(readlink -f "${target}" 2>/dev/null || true)"
    if [[ "${cur}" == "${source}" ]]; then
      echo "[v12i] symlink ok: ${target} -> ${source}"
      return 0
    fi
    echo "[v12i] WARNING: stale symlink ${target} -> ${cur} (expected ${source}); re-creating"
    rm -f "${target}"
  fi
  if [[ ! -f "${source}" ]]; then
    echo "[v12i] ERROR: source not found: ${source}" >&2
    echo "[v12i] Put grpo_v12_json.parquet under ${REMOTE_DATA_DIR}/ and rerun" >&2
    return 1
  fi
  ln -s "${source}" "${target}"
  echo "[v12i] created symlink: ${target} -> ${source}"
}

ensure_v12i_symlink

# === V12i job 名 ===
JOB_NAME="${JOB_NAME:-chaingsm-grpo-verl-v12i}"

# === 预检 (可选, SKIP_PREFLIGHT=1 跳过) ===
remote_run_preflight grpo

# === 拼装并提交 ===
# 关键差异 (跟 submit_grpo_verl_vllm.sh 对比):
#   --config-name grpo_verl_v12i_vllm   (而非 grpo_verl_vllm)
#   trainer.total_epochs=2              (而非 1)
# 其余参数: 跟 V10 提交器保持一致, 通过 env 变量控制
CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v12i_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
