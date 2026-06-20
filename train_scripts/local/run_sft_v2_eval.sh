#!/usr/bin/env bash
# SFT v2 完整流程: epoch1 + epoch2 + 3 轮评测
# 用法: bash train_scripts/local/run_sft_v2_eval.sh
set -uo pipefail
ROOT=/home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
CONFIG=$ROOT/train_configs/local/sft_v2.yaml
MODEL=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
DATA=$ROOT/chaingsm_data/data/final/sft/all_sft_train_prompt_completion.jsonl
RUN_NAME=sft_v2_abstral
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=$ROOT/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/$RUN_NAME/$RUN_ID
EPOCH1_DIR=$OUTPUT_BASE/epoch1
EPOCH2_DIR=$OUTPUT_BASE/epoch2
EVAL_BASE=$ROOT/outputs/sft_v2_abstral/eval
TEST=$ROOT/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
TRAIN=$ROOT/chaingsm_data/data/final/sft/all_sft.jsonl
EVAL_SCRIPT=$ROOT/train_pipeline/eval_sft_messages_chaingsm.py

export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME=/home/wwq416/miniconda3/envs/math_chain_verl
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p $EPOCH1_DIR $EPOCH2_DIR $EVAL_BASE

# === epoch 1 ===
echo "=== EPOCH 1: $MODEL ==="
$PYTHON -m train_pipeline.train_sft_trl \
  --config $CONFIG \
  --model $MODEL \
  --data $DATA \
  --output-dir $EPOCH1_DIR 2>&1 | tee $EPOCH1_DIR/train.log | grep -E "loss|epoch|grad_norm|Saving" | tail -30

# === epoch 2 (从 epoch1 继续) ===
sed "s|model_name_or_path: .*|model_name_or_path: $EPOCH1_DIR/checkpoints/current|" $CONFIG > $OUTPUT_BASE/sft_v2_epoch2.yaml
echo ""
echo "=== EPOCH 2: $EPOCH1_DIR/checkpoints/current ==="
$PYTHON -m train_pipeline.train_sft_trl \
  --config $OUTPUT_BASE/sft_v2_epoch2.yaml \
  --model $EPOCH1_DIR/checkpoints/current \
  --data $DATA \
  --output-dir $EPOCH2_DIR 2>&1 | tee $EPOCH2_DIR/train.log | grep -E "loss|epoch|grad_norm|Saving" | tail -30

# === 3 轮评测 ===
eval_ckpt() {
    local name=$1
    local model_path=$2
    local out=$EVAL_BASE/$name
    mkdir -p $out
    if [ -f "$out/latest_metrics.json" ]; then
        echo "SKIP $name (already evaluated)"
        return
    fi
    echo ""
    echo "=== EVAL $name: $model_path ==="
    $PYTHON $EVAL_SCRIPT \
        --model-path "$model_path" \
        --data-path "$TEST" \
        --train-data "$TRAIN" \
        --output-dir "$out" \
        --max-tokens 512 \
        --batch-size 32 \
        --gpu-memory-utilization 0.85 2>&1 | tee $out/eval.log | tail -10
}
eval_ckpt "baseline" $MODEL
eval_ckpt "epoch1" $EPOCH1_DIR/checkpoints/current
eval_ckpt "epoch2" $EPOCH2_DIR/checkpoints/current

echo ""
echo "=== 汇总 ==="
for tag in baseline epoch1 epoch2; do
    if [ -f "$EVAL_BASE/$tag/latest_metrics.json" ]; then
        $PYTHON -c "
import json
d = json.load(open('$EVAL_BASE/$tag/latest_metrics.json'))
acc = d.get('accuracy', 0)
cats = d.get('by_category', [])
orig = next((c for c in cats if c.get('category') == 'original'), None)
print(f'$tag: overall={acc:.4f}  original={orig[\"accuracy\"] if orig else 0:.4f}')
"
    fi
done
