#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
CONFIG="${CONFIG:-${ROOT}/train_configs/local/grpo.yaml}"
MODEL="${MODEL:-/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/grpo_train.jsonl}"
RUN_NAME="${RUN_NAME:-grpo}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_ID}"

if [[ ! -f "$DATA" ]]; then
  MAX_SAMPLES="${PREPROCESS_MAX_SAMPLES:-}" "${ROOT}/train_scripts/local/run_preprocess.sh"
fi

ARGS=(--config "$CONFIG" --model "$MODEL" --data "$DATA" --output-dir "$OUTPUT_DIR")
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ "${SKIP_GRPO_ENV_CHECK:-0}" == "1" ]]; then
  ARGS+=(--skip-env-check)
fi

cd "$ROOT"
echo "GRPO output dir: $OUTPUT_DIR"
"$PYTHON" -m train_pipeline.train_grpo_trl "${ARGS[@]}" "$@"
