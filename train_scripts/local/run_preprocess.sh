#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
INPUT="${INPUT:-${ROOT}/chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946}"

ARGS=(--input "$INPUT" --output-dir "$OUTPUT_DIR")
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "${SKIP_PARQUET:-0}" == "1" ]]; then
  ARGS+=(--skip-parquet)
fi

cd "$ROOT"
"$PYTHON" -m train_pipeline.preprocess_chaingsm "${ARGS[@]}" "$@"

