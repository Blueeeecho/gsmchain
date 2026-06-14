#!/usr/bin/env bash
# LB-PRM (liveness-based PRM) GRPO training for ChainGSM.
#
# This is a THIN WRAPPER over run_grpo_verl.sh that switches:
#   - SFT 2-epoch checkpoint (符合"≤ 2 epoch"约束)
#   - LB-PRM reward (reward_chaingsm_lbprm_verl.py)
#   - LB-PRM reward weights (format=0.2 / answer=0.4 / liveness=0.4 / invalid=-0.5)
#   - DROPS old reward weights (expression_weight / trace_weight / distractor_penalty)
#
# Everything else (rollout N, lr, KL, scheduler, per-epoch eval, etc.) is
# inherited from run_grpo_verl.sh via env vars — no Hydra overrides, because
# run_grpo_verl.sh already uses the correct verl config keys
# (actor_rollout_ref.model.path, data.train_files, etc.) and passing wrong
# keys (model.path, data.path) would fail with "Key not in struct".
#
# Per-epoch eval writes to $OUTPUT_DIR/eval/epoch_summary.jsonl.
# Tail the log:  tail -f $OUTPUT_DIR/logs/grpo_verl_stdout.log
#
# Usage:
#   bash train_scripts/local/run_grpo_lbprm_smoke.sh
#   TOTAL_EPOCHS=4 bash train_scripts/local/run_grpo_lbprm_smoke.sh
#
# See docs/superpowers/specs/2026-06-08-lb-prm-design.md for the design.
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"

# SFT 2-epoch checkpoint (符合"≤ 2 epoch"约束)
MODEL="${MODEL:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best}"

# 新写的 LB-PRM reward
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_lbprm_verl.py}"

# 训练数据 (verl parquet, 7,055 条)
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet}"

# LB-PRM reward weights (see docs/superpowers/specs/2026-06-08-lb-prm-design.md §3.1)
# run_grpo_verl.sh's env-var defaults are tuned for the OLD reward
# (answer_weight=2.5, expression_weight=1.0, trace_weight=1.0, distractor_penalty=0.5).
# We override them to LB-PRM values so the YAML config is correct, and we
# ALSO clear the old reward weights so they don't leak into score_response
# (LB-PRM's score_response absorbs them via **kwargs, but the YAML should
# not carry stale values).
FORMAT_WEIGHT=0.2
ANSWER_WEIGHT=0.4
LIVENESS_WEIGHT=0.4
INVALID_REWARD=-0.5
# zero out the old reward weights so they don't pollute the YAML
EXPRESSION_WEIGHT=0.0
TRACE_WEIGHT=0.0
DISTRACTOR_PENALTY=0.0

# Export so run_grpo_verl.sh inherits and its banner echo + YAML are accurate
export MODEL REWARD_PATH DATA
export FORMAT_WEIGHT ANSWER_WEIGHT LIVENESS_WEIGHT INVALID_REWARD
export EXPRESSION_WEIGHT TRACE_WEIGHT DISTRACTOR_PENALTY

# Pass any extra env vars through (TOTAL_EPOCHS, RUN_NAME, RUN_ID, OUTPUT_DIR, ...)
exec bash "${ROOT}/train_scripts/local/run_grpo_verl.sh" "$@"
