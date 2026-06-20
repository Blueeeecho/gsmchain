#!/usr/bin/env bash
# submit_grpo_v13_verl.sh
#
# 一键提交 V13 GRPO 训练到远程 SLURM 集群.
# 跟 submit_grpo_v12_verl.sh 同结构, 但默认配置指向 V13:
#   - reward  : reward_chaingsm_v13_json_verl.py (5 项, 硬 per-step 数值匹配)
#   - data    : grpo_v13_json.parquet (V13 prompt 已内嵌, 3051 行, 5 类别)
#   - config  : train_configs/remote/grpo_verl_v13_vllm.yaml
#   - epochs  : 2
#   - save_freq: 300
#
# V13 vs V12 关键差异:
# - prompt 7 字段 (target/use_facts/exclude_facts/steps[3]/final_expression) -> 5 字段 (steps[] 单层)
#   (实际 V13 = 5 字段, 跟 V12 6 字段比少了 1 个; 详见 docs/prompts/v13_long_prompt.md)
# - reward 公式重命名: r_answer->r_final, r_step_value->r_step_val, 新增 r_step_fmt + r_format + r_exclude
# - 砍 r_core (软 trace+final sim, 鼓励"生成相似"而非"算对")
#
# 用法:
#   DRY_RUN=1 SKIP_PREFLIGHT=1 bash submit_grpo_v13_verl.sh
#   SKIP_PREFLIGHT=1 bash submit_grpo_v13_verl.sh
#   bash submit_grpo_v13_verl.sh  # 完整流程 (预检 + 提交)
#
# V13 训练量: 2 epoch × 762 steps/epoch = 1524 total steps
# V13 保存: 5 ckpts @ 300/600/900/1200/1500
# V13 评测: 训练完后用 train_pipeline/eval_vllm_chaingsm.py:method=cot_brackets_v13_json
#
# 配套文档: train_scripts/remote/V12_V14_HANDBOOK.md
# 配套: train_configs/remote/grpo_verl_v13_vllm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === V13 关键覆盖: 在 source remote_env.sh 之前先注入 V13 默认值 ===
set +u
if [[ -z "${REMOTE_REWARD_PATH+x}" ]]; then
  REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v13_json_verl.py"
fi
if [[ -z "${REMOTE_DATA_DIR+x}" ]]; then
  REMOTE_DATA_DIR="${REMOTE_ROOT}/chaingsm_data/data/final/grpo"
fi
# V13 训练量: 2 epoch
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
if [[ "${REMOTE_REWARD_PATH}" != *"v13_json_verl"* ]]; then
  echo "[v13] WARNING: REMOTE_REWARD_PATH 异常: ${REMOTE_REWARD_PATH}" >&2
  echo "[v13] (这是 V13 提交, 应该是 reward_chaingsm_v13_json_verl.py)" >&2
fi

# 远程 grpo_verl_v13_vllm.yaml 期望 parquet 文件名是 verl_grpo_train.parquet
# 我们的 V13 数据文件叫 grpo_v13_json.parquet.
ensure_v13_symlink() {
  local target="${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
  local source="${REMOTE_DATA_DIR}/grpo_v13_json.parquet"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[v13] DRY_RUN=1, skip symlink/copy check"
    return 0
  fi
  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "[v13] NOTE: ${target} already exists (not a symlink); leaving as-is"
    return 0
  fi
  if [[ -L "${target}" ]]; then
    local cur; cur="$(readlink -f "${target}" 2>/dev/null || true)"
    if [[ "${cur}" == "${source}" ]]; then
      echo "[v13] symlink ok: ${target} -> ${source}"
      return 0
    fi
    echo "[v13] WARNING: stale symlink ${target} -> ${cur} (expected ${source}); re-creating"
    rm -f "${target}"
  fi
  if [[ ! -f "${source}" ]]; then
    echo "[v13] ERROR: source not found: ${source}" >&2
    echo "[v13] Put grpo_v13_json.parquet under ${REMOTE_DATA_DIR}/ and rerun" >&2
    return 1
  fi
  ln -s "${source}" "${target}"
  echo "[v13] created symlink: ${target} -> ${source}"
}

ensure_v13_symlink

# === V13 job 名 ===
JOB_NAME="${JOB_NAME:-chaingsm-grpo-verl-v13}"

# === 预检 (可选) ===
remote_run_preflight grpo

# === 拼装并提交 ===
CMD="cd ${REMOTE_ROOT} && ${PYTHON} -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v13_vllm trainer.total_epochs=${TOTAL_EPOCHS} actor_rollout_ref.rollout.n=${ROLLOUT_N} actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"

remote_print_or_submit "${CMD}"
