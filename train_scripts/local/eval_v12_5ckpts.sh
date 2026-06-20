# 用 cot_brackets_v12_json method, JSON prompt + JSON 抽取

set -uo pipefail

ROOT="/home/wwq416/snap/wwq/math-chain"
PYTHON="/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"

# Run id 用环境变量传入, 默认用最新
RUN_NAME="${RUN_NAME:-qwen2.5-0.5b-grpo-verl-v12-json}"
RUN_ID="${RUN_ID:-$(ls -td ${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/*/ 2>/dev/null | head -1 | xargs basename)}"
EVAL_ROOT="${ROOT}/outputs/v12_eval"
CKPT_ROOT="${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}"
# v12 实际落盘位置: $RUN_ID/global_step_* (无 checkpoints/ 中间层)
# v12 实际落盘位置: $RUN_ID/global_step_* (无 checkpoints/ 中间层)
EVAL_DATA="${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl"
METHOD="cot_brackets_v12_json"
GPU_MEM_UTIL=0.3
BATCH_SIZE=64
MAX_TOKENS=2048      # 2026-06-19: 1024 会截断 27% 样本, 升 2048 消截断
MAX_MODEL_LEN=4096      # prompt 1700 + response 2048 = 3748, 4096 留余量

mkdir -p "$EVAL_ROOT"

export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export VLLM_USE_FLASHINFER_SAMPLER=0

SUMMARY="$EVAL_ROOT/eval_summary_v12_${RUN_ID}.jsonl"
> "$SUMMARY"

cleanup_between() {
  ray stop --force 2>/dev/null || true
  sleep 8
}

# 默认评 5 ckpts (300/600/900/1200/1500), 可通过 STEPS 覆盖
STEPS="${STEPS:-300 600 900 1200 1500}"

for STEP in $STEPS; do
  CKPT_PATH="$CKPT_ROOT/global_step_${STEP}/actor/huggingface"
  OUT_DIR="$EVAL_ROOT/step_${STEP}"
  mkdir -p "$OUT_DIR"

  if [[ ! -d "$CKPT_PATH" ]]; then
    echo "  WARN: ckpt not found, skip: $CKPT_PATH"
    continue
  fi

  echo ""
  echo "=========================================="
  echo "  [$(date +%H:%M:%S)] Evaluating v12 step $STEP"
  echo "  Model:  $CKPT_PATH"
  echo "  Output: $OUT_DIR"
  echo "=========================================="

  cleanup_between

  T0=$(date +%s)
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
    "$PYTHON" -c "
import json
with open('$OUT_DIR/eval_result.json') as f: d=json.load(f)
print(f'step=$STEP overall={d[\"overall_accuracy\"]:.4f}')
" | tee -a "$SUMMARY"
  fi
done

echo ""
echo "Summary: $SUMMARY"
