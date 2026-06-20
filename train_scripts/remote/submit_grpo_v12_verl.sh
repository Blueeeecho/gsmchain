#!/usr/bin/env bash
# submit_grpo_v12_verl.sh
#
# 一键提交 V12 GRPO 训练到远程 SLURM 集群.
# 跟 submit_grpo_v12i_verl.sh 同结构, 但默认配置指向 V12 (老 4 项 reward):
#   - reward  : reward_chaingsm_v12_json_verl.py (4 项, 软 core 相似度)
#   - data    : grpo_v12_json.parquet (V12 prompt 已内嵌, 3051 行, 5 类别)
#   - config  : train_configs/remote/grpo_verl_v12_vllm.yaml
#   - epochs  : 2
#   - save_freq: 300
#
# V12 是 V12i 的 base, V12i 沿用 V12 prompt + 数据, 只换 reward.
# 跟 V12i 区别: V12 reward 7 项 (含 soft core + invalid_reward),
#               V12i reward 4 项 (LCS + 精确匹配 + per-step distractor).
#
# 用法:
#   DRY_RUN=1 SKIP_PREFLIGHT=1 bash submit_grpo_v12_verl.sh
#   SKIP_PREFLIGHT=1 bash submit_grpo_v12_verl.sh
#   bash submit_grpo_v12_verl.sh  # 完整流程 (预检 + 提交)
#
# V12 训练量: 2 epoch × 762 steps/epoch = 1524 total steps
# V12 保存: 5 ckpts @ 300/600/900/1200/1500
# V12 评测: 训练完后用 train_scripts/local/eval_v12_5ckpts.sh 跑 cot_brackets_v12_json
#
# 配套文档: train_scripts/remote/V12_V14_HANDBOOK.md
# 配套: train_configs/remote/grpo_verl_v12_vllm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === V12 关键覆盖: 在 source remote_env.sh 之前先注入 V12 默认值 ===
set +u
if [[ -z "${REMOTE_REWARD_PATH+x}" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12_json_verl.py"
fi
if [[ -z "${REMOTE_DATA_DIR+x}" ]]; then
  REMOTE_DATA_DIR="${REMOTE_ROOT}/chaingsm_data/data/final/grpo"
fi
# V12 训练量: 2 epoch
if [[ -z "${GRPO_EPOCHS+x}" ]]; then
  GRPO_EPOCHS=2
fi
if [[ -z "${TOTAL_EPOCHS+x}" ]]; then
  TOTAL_EPOCHS="${GRPO_EPOCHS}"
fi
set -u
export REMOTE_REWARD_PATH REMOTE_DATA_DIR GRPO_EPOCHS TOTAL_EPOCHS

# 复用通用远程环境加载器
source "${SCRIPT_DIR}/remote_env.sh"

# 防御性检查
if [[ "${REMOTE_REWARD_PATH}" != *"v12_json_verl"* ]] && [[ "${REMOTE_REWARD_PATH}" != *"v12i_json_verl"* ]]; then
  echo "[v12] WARNING: REMOTE_REWARD_PATH 异常: ${REMOTE_REWARD_PATH}" >&2
  echo "[v12] (这是 V12 提交, 应该是 reward_chaingsm_v12_json_verl.py)" >&2
fi

# 远程 grpo_verl_v12_vllm.yaml 期望 parquet 文件名是 verl_grpo_train.parquet
# 我们的 V12 数据文件叫 grpo_v12_json.parquet (V12 / V12i 共享).
# 这里在 REMOTE_DATA_DIR 下放一个 symlink/copy 兜底.
ensure_v12_symlink() {
  local target="${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
  local source="${REMOTE_DATA_DIR}/grpo_v12_json.parquet"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[v12] DRY_RUN=1, skip symlink/copy check"
    return 0
  fi
  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[v12] NOTE: ${target} already exists (not a symlink); leaving as-is"
    return 0
  fi
  if [[ -L "${target}" ]]; then
    local cur; cur="$(readlink -f "${target}" 2>/dev/null || true)"
    if [[ "${cur}" == "${source}" ]]; then
      echo "[v12] symlink ok: ${target} -> ${source}"
      return 0
    fi
    echo "[v12] WARNING: stale symlink ${target} -> ${cur} (expected ${source}); re-creating"
    rm -f "${target}"
  fi
  if [[ ! -f "${source}" ]]; then
    echo "[v12] ERROR: source not found: ${source}" >&2
    echo "[v12] Put grpo_v12_json.parquet under ${REMOTE_DATA_DIR}/ and rerun" >&2
    return 1
  fi
  ln -s "${source}" "${target}"
  echo "[v12] created symlink: ${target} -> ${source}"
}

ensure_v12_symlink

# === V12 job 名 ===
JOB_NAME="${JOB_NAME:-chaingsm-grpo-verl-v12}"

# === 预检 (可选, SKIP_PREFLIGHT=1 跳过) ===
remote_run_preflight grpo

# === 拼装并提交 ===
CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v12_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
