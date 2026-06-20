#!/usr/bin/env bash
# 评测所有 v10 GRPO 训练出的 ckpt
# 用 cot_brackets 方法 (跟训练时 prompt 一致) + gsm8k_test_clean.jsonl
set -euo pipefail

ROOT="/home/wwq416/snap/wwq/math-chain"
PYTHON="/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"

# 默认从哪个目录找 ckpt
CKPT_ROOT="${1:-${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v10_signed}"
# 找最新的 RUN_ID
LATEST_RUN=$(ls -t "$CKPT_ROOT" 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
  echo "No run found in $CKPT_ROOT"
  exit 1
fi
CKPT_DIR="$CKPT_ROOT/$LATEST_RUN/checkpoints"
EVAL_DIR="$CKPT_ROOT/$LATEST_RUN/eval_per_ckpt"
mkdir -p "$EVAL_DIR"

# 评测设置
EVAL_DATA="${EVAL_DATA:-${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl}"
METHOD="cot_brackets"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# 评测目标: 跳过的 step 列表 (跳过 baseline)
SKIP_STEPS="${SKIP_STEPS:-}"

# 环境变量
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY

echo "=========================================="
echo "Evaluating all ckpts in $CKPT_DIR"
echo "Test data: $EVAL_DATA"
echo "Method: $METHOD"
echo "Output: $EVAL_DIR"
echo "=========================================="

# 写 summary 文件
SUMMARY="$EVAL_DIR/eval_summary.jsonl"
> "$SUMMARY"

# 找所有 global_step_*/actor/huggingface
for CKPT_PATH in $(find "$CKPT_DIR" -maxdepth 3 -name "huggingface" -path "*/actor/*" -type d 2>/dev/null | sort); do
  STEP=$(echo "$CKPT_PATH" | sed 's|.*/global_step_||' | sed 's|/actor/huggingface||')
  if [ -n "$SKIP_STEPS" ] && [[ " $SKIP_STEPS " == *" $STEP "* ]]; then
    echo "Skipping step $STEP (in SKIP_STEPS)"
    continue
  fi
  STEP_DIR="$EVAL_DIR/step_${STEP}"
  mkdir -p "$STEP_DIR"
  
  echo ""
  echo "--- Evaluating step $STEP ---"
  echo "Model: $CKPT_PATH"
  echo "Output: $STEP_DIR"
  
  # 清 GPU 缓存 + 杀之前可能残留的 vllm
  pkill -9 -f "wwq416.*vllm" 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  sleep 3
  
  if "$PYTHON" -m train_pipeline.eval_vllm_chaingsm \
    --model-path "$CKPT_PATH" \
    --data-path "$EVAL_DATA" \
    --output-dir "$STEP_DIR" \
    --method "$METHOD" \
    --batch-size "$BATCH_SIZE" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --gpu-memory-utilization-candidate "$GPU_MEM_UTIL" \
    --max-tokens "$MAX_TOKENS" \
    --top-k 1 \
    --dtype auto; then
    
    # 读结果
    if [ -f "$STEP_DIR/eval_result.json" ]; then
      READ=$("$PYTHON" -c "
import json
with open('$STEP_DIR/eval_result.json') as f:
    d = json.load(f)
ov = d['overall_accuracy']
orig = 0.0
for cat in d.get('by_category', []):
    if cat.get('category') == 'original':
        orig = cat.get('accuracy', 0.0)
        break
print(f'{ov:.6f} {orig:.6f}')
")
      OVERALL=$(echo "$READ" | awk '{print $1}')
      ORIGINAL=$(echo "$READ" | awk '{print $2}')
      echo "Step $STEP: overall=$OVERALL original=$ORIGINAL"
      echo "{\"step\":$STEP,\"overall\":$OVERALL,\"original\":$ORIGINAL,\"status\":\"ok\",\"model\":\"$CKPT_PATH\"}" >> "$SUMMARY"
    else
      echo "Step $STEP: eval_result.json not found"
      echo "{\"step\":$STEP,\"status\":\"result_missing\",\"model\":\"$CKPT_PATH\"}" >> "$SUMMARY"
    fi
  else
    echo "Step $STEP: FAILED"
    echo "{\"step\":$STEP,\"status\":\"eval_failed\",\"model\":\"$CKPT_PATH\"}" >> "$SUMMARY"
  fi
done

echo ""
echo "=========================================="
echo "Eval summary:"
cat "$SUMMARY"
echo "=========================================="
