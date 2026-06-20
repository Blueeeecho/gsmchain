#!/usr/bin/env bash
# submit_grpo_v14_verl.sh
#
# 一键提交 V14 GRPO 训练到远程 SLURM 集群.
# 跟 submit_grpo_v12i_verl.sh 同结构, 但默认配置指向 V14:
#   - reward  : reward_chaingsm_v14_reasoning_verl.py (4 项, 跟 V12 同构)
#   - data    : grpo_v14_reasoning.parquet (V14 prompt 已内嵌, 3051 行, 5 类别)
#   - config  : train_configs/remote/grpo_verl_v14_vllm.yaml
#   - epochs  : 3 (V14 prompt 长, 留更多训练步数)
#   - save_freq: 300
#
# V14 vs V12 / V13 / V12i 关键差异:
# - prompt 4 字段 (target/use_facts/exclude_facts/reasoning[]) - 砍 steps + final_expression
# - reasoning 物理交错 prose / 表达式, 末行 = 最终数值
# - reward 跟 V12i 同构 (4 项), 但 signal source 不同 (V14 用 reasoning[], V12i 用 steps[])
#
# 用法:
#   DRY_RUN=1 SKIP_PREFLIGHT=1 bash submit_grpo_v14_verl.sh
#   SKIP_PREFLIGHT=1 bash submit_grpo_v14_verl.sh
#   bash submit_grpo_v14_verl.sh  # 完整流程 (预检 + 提交)
#
# V14 训练量: 3 epoch × 762 steps/epoch = 2286 total steps
# V14 保存: 每 300 步 (7 ckpts @ 300/600/900/1200/1500/1800/2100)
# V14 评测: 训练完后用 train_pipeline/eval_vllm_chaingsm.py:method=cot_brackets_v14_reasoning
#
# 配套文档: train_scripts/remote/V12_V14_HANDBOOK.md
# 配套: train_configs/remote/grpo_verl_v14_vllm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === V14 关键覆盖: 在 source remote_env.sh 之前先注入 V14 默认值 ===
set +u
if [[ -z "${REMOTE_REWARD_PATH+x}" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v14_reasoning_verl.py"
fi
if [[ -z "${REMOTE_DATA_DIR+x}" ]]; then
  REMOTE_DATA_DIR="${REMOTE_ROOT}/chaingsm_data/data/final/grpo"
fi
# V14 训练量: 3 epoch (V14 prompt 长, 留更多训练步数; V12/V13/V12i 都是 2)
if [[ -z "${GRPO_EPOCHS+x}" ]]; then
  GRPO_EPOCHS=3
fi
if [[ -z "${TOTAL_EPOCHS+x}" ]]; then
  TOTAL_EPOCHS="${GRPO_EPOCHS}"
fi
set -u
export REMOTE_REWARD_PATH REMOTE_DATA_DIR GRPO_EPOCHS TOTAL_EPOCHS

# 复用通用远程环境加载器
source "${SCRIPT_DIR}/remote_env.sh"

# 防御性检查
if [[ "${REMOTE_REWARD_PATH}" != *"v14"* ]]; then
  echo "[v14] WARNING: REMOTE_REWARD_PATH 异常: ${REMOTE_REWARD_PATH}" >&2
  echo "[v14] (这是 V14 提交, 应该是 reward_chaingsm_v14_reasoning_verl.py)" >&2
fi

# 远程 grpo_verl_v14_vllm.yaml 期望 parquet 文件名是 verl_grpo_train.parquet
# 我们的 V14 数据文件叫 grpo_v14_reasoning.parquet.
ensure_v14_symlink() {
  local target="${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
  local source="${REMOTE_DATA_DIR}/grpo_v14_reasoning.parquet"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[v14] DRY_RUN=1, skip symlink/copy check"
    return 0
  fi
  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[v14] NOTE: ${target} already exists (not a symlink); leaving as-is"
    return 0
  fi
  if [[ -L "${target}" ]]; then
    local cur; cur="$(readlink -f "${target}" 2>/dev/null || true)"
    if [[ "${cur}" == "${source}" ]]; then
      echo "[v14] symlink ok: ${target} -> ${source}"
      return 0
    fi
    echo "[v14] WARNING: stale symlink ${target} -> ${cur} (expected ${source}); re-creating"
    rm -f "${target}"
  fi
  if [[ ! -f "${source}" ]]; then
    echo "[v14] ERROR: source not found: ${source}" >&2
    echo "[v14] Put grpo_v14_reasoning.parquet under ${REMOTE_DATA_DIR}/ and rerun" >&2
    return 1
  fi
  ln -s "${source}" "${target}"
  echo "[v14] created symlink: ${target} -> ${source}"
}

ensure_v14_symlink

# === V14 job 名 ===
JOB_NAME="${JOB_NAME:-chaingsm-grpo-verl-v14}"

# === 预检 (可选) ===
remote_run_preflight grpo

# === 拼装并提交 ===
CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v14_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
