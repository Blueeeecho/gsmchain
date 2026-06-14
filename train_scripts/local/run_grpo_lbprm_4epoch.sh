#!/usr/bin/env bash
# LB-PRM 4-epoch GRPO training on ChainGSM (7,055 train, Qwen-0.5B SFT 2-ep).
#
# Reuses run_grpo_lbprm_smoke.sh 100% for training logic.
# Differences from the smoke script:
#   1. TOTAL_EPOCHS=4 (smoke script inherits default 20 from run_grpo_verl.sh)
#   2. After training, runs the 8-shot CoT eval on the best checkpoint
#      (code/eval_chaingsm_base_8shot.py, 5,467 test set) so we can compare to the
#      8-shot CoT baseline reported elsewhere in the project.
#   3. Explicit export of MODEL/REWARD_PATH/RUN_NAME/OUTPUT_DIR so the
#      run_grpo_verl.sh banner echo reflects the actual values used.
#
# Per-epoch monitoring (train_json_prompt on 6,575 test set) is built into
# run_grpo_verl.sh and writes to $OUTPUT_DIR/eval/epoch_summary.jsonl.
# Tail it with:
#   tail -f $OUTPUT_DIR/logs/grpo_verl_stdout.log
#   cat   $OUTPUT_DIR/eval/epoch_summary.jsonl
#
# Usage:
#   bash train_scripts/local/run_grpo_lbprm_4epoch.sh
#
# Override:
#   TOTAL_EPOCHS=2 bash run_grpo_lbprm_4epoch.sh
#   SKIP_8SHOT=1   bash run_grpo_lbprm_4epoch.sh   # skip final 8-shot eval
#   MODEL=/path    bash run_grpo_lbprm_4epoch.sh   # use different SFT start
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"

# Defaults — exported so child processes (smoke + run_grpo_verl.sh) inherit them
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
SKIP_8SHOT="${SKIP_8SHOT:-0}"
RUN_NAME="${RUN_NAME:-grpo_verl_lbprm_4ep}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
MODEL="${MODEL:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best}"
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_lbprm_verl.py}"

export TOTAL_EPOCHS RUN_NAME RUN_ID OUTPUT_DIR MODEL REWARD_PATH

echo "============================================"
echo " LB-PRM 4-Epoch GRPO Training"
echo "============================================"
echo "  ROOT:         $ROOT"
echo "  TOTAL_EPOCHS: $TOTAL_EPOCHS"
echo "  SKIP_8SHOT:   $SKIP_8SHOT"
echo "  RUN_NAME:     $RUN_NAME"
echo "  RUN_ID:       $RUN_ID"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"
echo "  MODEL:        $MODEL"
echo "  REWARD:       $REWARD_PATH"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Run 4-epoch GRPO training (per-epoch eval built into the script)
# ------------------------------------------------------------------
echo "[step 1/3] Starting ${TOTAL_EPOCHS}-epoch GRPO training (LB-PRM reward)..."
echo ""

bash "${ROOT}/train_scripts/local/run_grpo_lbprm_smoke.sh"

# ------------------------------------------------------------------
# Step 2: Locate the best checkpoint produced by training
# ------------------------------------------------------------------
BEST_CKPT="${OUTPUT_DIR}/checkpoints/best"
if [[ ! -d "$BEST_CKPT" ]]; then
  echo ""
  echo "=========================================="
  echo " ERROR: Best checkpoint not found at $BEST_CKPT"
  echo "=========================================="
  echo "Training may have failed or skipped eval. Check logs at:"
  echo "  $OUTPUT_DIR/logs/"
  exit 1
fi

echo ""
echo "[step 2/3] Best checkpoint: $BEST_CKPT"
echo ""

# Print per-epoch summary that was already collected during training
EVAL_DIR="${OUTPUT_DIR}/eval"
EPOCH_SUMMARY="${EVAL_DIR}/epoch_summary.jsonl"
if [[ -f "$EPOCH_SUMMARY" ]]; then
  echo "  --- Per-epoch training-prompt eval (6,575 test set, train_json_prompt) ---"
  "$PYTHON" -c "
import json
rows = []
with open('${EPOCH_SUMMARY}') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
ok_rows = [r for r in rows if r.get('eval_status') == 'ok']
for r in ok_rows:
    print(f\"  epoch {r['epoch']:>2}  acc={r.get('overall_accuracy', 0):.4f}  global_step={r.get('global_step', '?')}\")
"
fi

# ------------------------------------------------------------------
# Step 3: Final 8-shot CoT eval on best checkpoint (5,467 test set)
# ------------------------------------------------------------------
EIGHT_SHOT_DIR=""
if [[ "$SKIP_8SHOT" != "1" ]]; then
  echo ""
  echo "[step 3/3] Running 8-shot CoT eval on best checkpoint..."
  echo "  (OFFICIAL baseline comparison: 5,467 test set, 8-shot CoT)"

  EIGHT_SHOT_OUT="${OUTPUT_DIR}/eval_8shot"
  EIGHT_SHOT_RUN_ID="${EIGHT_SHOT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  EIGHT_SHOT_DIR="${EIGHT_SHOT_OUT}/${EIGHT_SHOT_RUN_ID}"

  "$PYTHON" code/eval_chaingsm_base_8shot.py \
    --model "$BEST_CKPT" \
    --data-path "${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl" \
    --output-root "$EIGHT_SHOT_DIR" \
    --batch-size 16 \
    2>&1 | tee "${EIGHT_SHOT_OUT}/eval_stdout.log" || echo "  WARNING: 8-shot eval failed, see ${EIGHT_SHOT_OUT}/eval_stdout.log"

  echo ""
  echo "  --- 8-shot CoT eval result ---"
  if [[ -f "${EIGHT_SHOT_DIR}/summary_overall.json" ]]; then
    cat "${EIGHT_SHOT_DIR}/summary_overall.json"
  fi
  if [[ -f "${EIGHT_SHOT_DIR}/summary_by_category.json" ]]; then
    echo "  --- by category ---"
    cat "${EIGHT_SHOT_DIR}/summary_by_category.json"
  fi
else
  echo ""
  echo "[step 3/3] SKIP_8SHOT=1, skipping 8-shot CoT eval."
fi

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
echo "  Best checkpoint:    $BEST_CKPT"
echo "  Per-epoch summary:  $EVAL_DIR/epoch_summary.jsonl"
echo "  8-shot CoT eval:    $EIGHT_SHOT_DIR"
echo "  Full output:        $OUTPUT_DIR"
echo "============================================"
