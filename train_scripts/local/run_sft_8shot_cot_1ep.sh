#!/usr/bin/env bash
# v8 SFT 1 epoch 8-shot CoT 训练入口
#
# 关联 spec: docs/superpowers/specs/2026-06-14-lbprm-v8-design.md
# 关联 plan: docs/superpowers/plans/2026-06-14-lbprm-v8-plan.md
#
# 关键设计:
#   - 起点: 0.5B base
#   - SFT 数据: sft_train_v2.jsonl (14528 条, free-form CoT)
#   - SFT 1 epoch (908 步 @ batch_size 16)
#   - LR 2e-5
#   - MAX_LENGTH 1024
#   - 关 eval 回调 (避免跟后续 SFT ckpt 评测冲突)
#
# 用法:
#   bash train_scripts/local/run_sft_8shot_cot_1ep.sh 2>&1 | tee /tmp/sft_v8.log

set -uo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"

MODEL="${MODEL:-/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_train_v2.jsonl}"

# 14528 / 16 = 908 步 (1 epoch)
MAX_STEPS="${MAX_STEPS:-908}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

RUN_NAME="${RUN_NAME:-sft_8shot_cot_1ep}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sft/${RUN_NAME}/Qwen2.5-0.5B-Instruct/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_ID}"

[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }

# CUDA env (跟 run_sft.sh 一致)
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY

echo "=== v8 SFT 1 epoch 8-shot CoT ==="
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "MAX_STEPS=$MAX_STEPS BATCH_SIZE=$BATCH_SIZE LR=$LR MAX_LENGTH=$MAX_LENGTH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo ""

cd "$ROOT"

"$PYTHON" -m train_pipeline.train_sft_trl \
  --config train_configs/local/sft.yaml \
  --model "$MODEL" \
  --data "$DATA" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --set training.per_device_train_batch_size=$BATCH_SIZE \
  --set training.gradient_accumulation_steps=$GRAD_ACCUM \
  --set training.learning_rate=$LR \
  --set training.max_length=$MAX_LENGTH \
  --set training.logging_steps=20 \
  --set eval.enabled=false

echo ""
echo "=== SFT 完成. Output: $OUTPUT_DIR ==="
echo "=== Ckpts: $OUTPUT_DIR/checkpoints/{current,best} ==="
