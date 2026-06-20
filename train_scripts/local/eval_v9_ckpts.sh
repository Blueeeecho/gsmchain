#!/usr/bin/env bash
# 批量评测 v9 GRPO 5 节点 (global_step_100/200/300/400/500)
set -uo pipefail
ROOT=/home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
EVAL_SCRIPT=$ROOT/train_pipeline/eval_sft_messages_chaingsm.py
TEST=$ROOT/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
TRAIN=$ROOT/chaingsm_data/data/final/sft/all_sft.jsonl
BASE_EVAL=$ROOT/outputs/sft_2ep_all/v9_full/eval
RUN_DIR=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_v9/Qwen2.5-0.5B-Instruct/grpo_v9/20260616_205812

export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME=/home/wwq416/miniconda3/envs/math_chain_verl
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_ROOT=$RUN_DIR
STEPS=(100 200 300 400 500)
echo "=== Eval 5 ckpts: ${STEPS[*]} ==="
echo "RUN_DIR: $RUN_DIR"
echo "TEST: $TEST"
echo ""

for step_num in "${STEPS[@]}"; do
    MODEL=$CKPT_ROOT/global_step_$step_num/actor/huggingface
    if [ ! -d "$MODEL" ]; then
        echo "SKIP step $step_num: no $MODEL"
        continue
    fi
    OUT=$BASE_EVAL/step_$step_num
    mkdir -p $OUT
    if [ -f "$OUT/latest_metrics.json" ]; then
        echo "SKIP step $step_num: already evaluated"
        continue
    fi
    echo "=== EVAL step $step_num ==="
    echo "  model: $MODEL"
    echo "  out:   $OUT"
    $PYTHON $EVAL_SCRIPT \
        --model-path "$MODEL" \
        --data-path "$TEST" \
        --train-data "$TRAIN" \
        --output-dir "$OUT" \
        --max-tokens 512 \
        --batch-size 32 \
        --gpu-memory-utilization 0.85 \
        2>&1 | tee $OUT/eval.log | grep -E "overall|acc|category|samples|Time" | head -30
    echo ""
done

echo "=== 汇总 ==="
for step_num in "${STEPS[@]}"; do
    OUT=$BASE_EVAL/step_$step_num
    if [ -f "$OUT/latest_metrics.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$OUT/latest_metrics.json'))
acc = d.get('accuracy', 0)
cats = d.get('by_category', [])
orig = next((c for c in cats if c.get('category') == 'original'), None)
orig_acc = orig['accuracy'] if orig else 0
print(f'step $step_num: overall={acc:.4f}  original={orig_acc:.4f}')
"
    fi
done
