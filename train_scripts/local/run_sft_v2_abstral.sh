#!/usr/bin/env bash
# SFT v2 - 对齐 AbstRaL 论文 (单卡适配)
# 2 epoch 串行, 每 epoch 独立 ckpt
set -uo pipefail

ROOT=/home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
CONFIG=$ROOT/train_configs/local/sft_v2.yaml
MODEL=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
DATA=$ROOT/chaingsm_data/data/final/sft/all_sft_train_prompt_completion.jsonl
RUN_NAME=sft_v2_abstral
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE=$ROOT/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/$RUN_NAME/$RUN_ID
LOG_DIR=$OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/$RUN_ID/logs

# epoch 1: 从 base 0.5B 开始
EPOCH1_DIR=$OUTPUT_BASE/epoch1
mkdir -p $EPOCH1_DIR
echo "=== EPOCH 1: from $MODEL ==="
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME=/home/wwq416/miniconda3/envs/math_chain_verl
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -m train_pipeline.train_sft_trl \
  --config $CONFIG \
  --model $MODEL \
  --data $DATA \
  --output-dir $EPOCH1_DIR 2>&1 | tee $EPOCH1_DIR/train.log | tail -30

# epoch 2: 从 epoch1 继续
EPOCH2_DIR=$OUTPUT_BASE/epoch2
mkdir -p $EPOCH2_DIR
# 改 config model.path 指向 epoch1
sed "s|model_name_or_path: .*|model_name_or_path: $EPOCH1_DIR/checkpoints/current|" \
  $CONFIG > $OUTPUT_BASE/sft_v2_epoch2.yaml
echo ""
echo "=== EPOCH 2: from $EPOCH1_DIR ==="
$PYTHON -m train_pipeline.train_sft_trl \
  --config $OUTPUT_BASE/sft_v2_epoch2.yaml \
  --model $EPOCH1_DIR/checkpoints/current \
  --data $DATA \
  --output-dir $EPOCH2_DIR 2>&1 | tee $EPOCH2_DIR/train.log | tail -30

echo ""
echo "=== SFT v2 (abstral-aligned) DONE ==="
echo "epoch1: $EPOCH1_DIR/checkpoints/current"
echo "epoch2: $EPOCH2_DIR/checkpoints/current"
