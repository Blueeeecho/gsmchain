#!/usr/bin/env bash
# v11 续训评测: step_400 / 600 / 800 / 1000
# 串行跑, 每个 ckpt 用独立 output dir
# vLLM 进程随 python 主进程退出, 不需要手动 kill

set -uo pipefail

ROOT="/home/wwq416/snap/wwq/math-chain"
PYTHON="/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"

CKPT_ROOT="${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v11_stepvalue/20260618_134820/checkpoints"
EVAL_ROOT="${ROOT}/outputs/v11_eval"
EVAL_DATA="${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl"
METHOD="cot_brackets"
GPU_MEM_UTIL=0.5
BATCH_SIZE=64
MAX_TOKENS=512
MAX_MODEL_LEN=2048

mkdir -p "$EVAL_ROOT"

export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY

SUMMARY="$EVAL_ROOT/eval_summary_v11_4ckpts.jsonl"
> "$SUMMARY"

# 评估之间: 只 stop ray (不动其它 GPU 用户), 然后 sleep 等显存释放
cleanup_between() {
  ray stop --force 2>/dev/null || true
  sleep 8
}

for STEP in 400 600 800 1000; do
  CKPT_PATH="$CKPT_ROOT/global_step_${STEP}/actor/huggingface"
  OUT_DIR="$EVAL_ROOT/step_${STEP}"
  mkdir -p "$OUT_DIR"

  echo ""
  echo "=========================================="
  echo "  [$(date +%H:%M:%S)] Evaluating v11 step $STEP"
  echo "  Model:  $CKPT_PATH"
  echo "  Output: $OUT_DIR"
  echo "=========================================="

  cleanup_between

  T0=$(date +%s)
  # 同步执行 (非后台), python 退出后自动释放显存
  if "$PYTHON" -m train_pipeline.eval_vllm_chaingsm \
      --model-path "$CKPT_PATH" \
      --data-path "$EVAL_DATA" \
      --output-dir "$OUT_DIR" \
      --method "$METHOD" \
      --batch-size "$BATCH_SIZE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --gpu-memory-utilization-candidate "$GPU_MEM_UTIL" \
      --max-tokens "$MAX_TOKENS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --top-k 1 \
      --dtype auto \
      --seed 42 > "$OUT_DIR/eval.log" 2>&1; then
    EXIT_CODE=0
  else
    EXIT_CODE=$?
  fi
  T1=$(date +%s)
  DUR=$((T1 - T0))
  echo "  [$(date +%H:%M:%S)] step $STEP done, exit=$EXIT_CODE, duration=${DUR}s"

  if [[ $EXIT_CODE -eq 0 && -f "$OUT_DIR/eval_result.json" ]]; then
    READ=$("$PYTHON" -c "
import json
with open('$OUT_DIR/eval_result.json') as f: d=json.load(f)
ov=d['overall_accuracy']
orig=0.0
for cat in d.get('by_category', []):
    if cat.get('category')=='original':
        orig=cat.get('accuracy',0.0); break
print(f'{ov:.6f} {orig:.6f}')")
    OVERALL=$(echo "$READ" | awk '{print $1}')
    ORIGINAL=$(echo "$READ" | awk '{print $2}')
    echo "  step $STEP: overall=$OVERALL original=$ORIGINAL"
    echo "{\"step\":$STEP,\"overall\":$OVERALL,\"original\":$ORIGINAL,\"duration_sec\":$DUR,\"status\":\"ok\"}" >> "$SUMMARY"
  else
    echo "  step $STEP FAILED (exit=$EXIT_CODE)"
    echo "{\"step\":$STEP,\"status\":\"failed\",\"exit\":$EXIT_CODE,\"duration_sec\":$DUR}" >> "$SUMMARY"
    tail -20 "$OUT_DIR/eval.log"
  fi
done

echo ""
echo "=========================================="
echo "  v11 4-ckpt eval summary"
echo "=========================================="
cat "$SUMMARY"
