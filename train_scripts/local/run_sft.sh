#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
CONFIG="${CONFIG:-${ROOT}/train_configs/local/sft.yaml}"
MODEL="${MODEL:-/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_train.jsonl}"
RUN_NAME="${RUN_NAME:-sft}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/${RUN_NAME}}"
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

# Ensure vLLM/FlashInfer subprocesses can find ninja and nvcc (CUDA 13.0)
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY

cd "$ROOT"
echo "SFT output dir: $OUTPUT_DIR"
"$PYTHON" -m train_pipeline.train_sft_trl "${ARGS[@]}" "$@"
