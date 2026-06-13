#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
MODEL="${MODEL:-/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct}"
RUN_NAME="${RUN_NAME:-sft_then_grpo}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/sft_then_grpo/Qwen2.5-0.5B-Instruct/${RUN_NAME}}"
BASE_OUT="${OUTPUT_BASE_DIR}/${RUN_ID}"
SFT_DATA="${SFT_DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_then_grpo_stage1_sft.jsonl}"
GRPO_DATA="${GRPO_DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_then_grpo_stage2_grpo.jsonl}"

SFT_OUT="${BASE_OUT}/stage1_sft"
GRPO_OUT="${BASE_OUT}/stage2_grpo"
SFT_FINAL_OUT="${SFT_OUT}/${RUN_ID}"
GRPO_FINAL_OUT="${GRPO_OUT}/${RUN_ID}"
echo "SFT->GRPO output dir: $BASE_OUT"

ROOT="$ROOT" PYTHON="$PYTHON" MODEL="$MODEL" DATA="$SFT_DATA" OUTPUT_DIR="$SFT_OUT" \
  MAX_SAMPLES="${MAX_SAMPLES:-}" MAX_STEPS="${SFT_MAX_STEPS:-${MAX_STEPS:-}}" \
  "${ROOT}/train_scripts/local/run_sft.sh" ${SFT_EXTRA_ARGS:-}

SFT_MODEL="${SFT_FINAL_OUT}/checkpoints/current"
ROOT="$ROOT" PYTHON="$PYTHON" MODEL="$SFT_MODEL" DATA="$GRPO_DATA" OUTPUT_DIR="$GRPO_OUT" \
  MAX_SAMPLES="${MAX_SAMPLES:-}" MAX_STEPS="${GRPO_MAX_STEPS:-${MAX_STEPS:-}}" \
  SKIP_GRPO_ENV_CHECK="${SKIP_GRPO_ENV_CHECK:-0}" \
  "${ROOT}/train_scripts/local/run_grpo.sh" ${GRPO_EXTRA_ARGS:-}
