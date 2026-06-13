#!/usr/bin/env bash
# verl GRPO training for ChainGSM on local single-GPU (RTX 5090)
#
# Uses the math_chain_verl conda environment (PyTorch 2.11+cu130, vLLM 0.21.0)
# instead of the TRL-based math-noise environment.
#
# Features:
#   - Runs training epoch-by-epoch with vLLM accuracy evaluation after each epoch
#   - Uses verl resume_mode=auto to continue training across epochs (preserves
#     optimizer state, scheduler, dataloader position)
#   - Saves HF-format checkpoints at each epoch boundary (for vLLM eval)
#   - Selects best model by accuracy (same as TRL pipeline)
#
# Usage:
#   conda activate math_chain_verl
#   MODEL=/path/to/sft/checkpoint/best \
#   RUN_NAME=grpo_verl_from_sft \
#   bash train_scripts/local/run_grpo_verl.sh
#
# Override any Hydra parameter by appending key=value after the script.
set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# Model path: default to SFT best checkpoint
MODEL="${MODEL:-/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_20epoch_full_eval/20260528_193544/checkpoints/best}"

# Data path
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet}"

# Reward function path
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_verl.py}"

# Run identification
RUN_NAME="${RUN_NAME:-grpo_verl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_DIR="$OUTPUT_DIR/configs"
METRICS_DIR="$OUTPUT_DIR/metrics"
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$METRICS_DIR"

exec > >(tee -a "$LOG_DIR/grpo_verl_stdout.log") \
  2> >(tee -a "$LOG_DIR/grpo_verl_stderr.log" >&2)

cleanup() {
  ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# Training hyperparameters. The default profile favors stability on 32GB
# Blackwell; set PROFILE=paper to recover the original paper-like defaults.
PROFILE="${PROFILE:-stable}"
if [[ "$PROFILE" == "paper" ]]; then
  DEFAULT_ROLLOUT_N=8
  DEFAULT_ROLLOUT_GPU_MEM_UTIL=0.4
  DEFAULT_TRAIN_BATCH_SIZE=8
  DEFAULT_MAX_RESPONSE_LENGTH=2048
  DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=4
else
  DEFAULT_ROLLOUT_N=4
  DEFAULT_ROLLOUT_GPU_MEM_UTIL=0.3
  DEFAULT_TRAIN_BATCH_SIZE=4
  DEFAULT_MAX_RESPONSE_LENGTH=1024
  DEFAULT_LOG_PROB_MICRO_BATCH_SIZE=1
fi

ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.04}"
ROLLOUT_N="${ROLLOUT_N:-$DEFAULT_ROLLOUT_N}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-$DEFAULT_ROLLOUT_GPU_MEM_UTIL}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$DEFAULT_MAX_RESPONSE_LENGTH}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-$DEFAULT_LOG_PROB_MICRO_BATCH_SIZE}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$LOG_PROB_MICRO_BATCH_SIZE}"

# Optimizer hyperparameters (matching TRL GRPO paper config)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.99]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"

# Reward weights
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.2}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-2.5}"
EXPRESSION_WEIGHT="${EXPRESSION_WEIGHT:-1.0}"
TRACE_WEIGHT="${TRACE_WEIGHT:-1.0}"
DISTRACTOR_PENALTY="${DISTRACTOR_PENALTY:-0.5}"
INVALID_REWARD="${INVALID_REWARD:--0.5}"

# Evaluation settings
EVAL_ENABLED="${EVAL_ENABLED:-1}"
EVAL_DATA_PATH="${EVAL_DATA:-${ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_BASELINE="${EVAL_BASELINE:-1}"  # Run baseline eval before training

# Environment variables for Blackwell compatibility
# VLLM_USE_V1 removed in vLLM 0.21.0 (V1 is the only engine now)
export CUDA_MODULE_LOADING=LAZY
# Ensure vLLM/FlashInfer subprocesses can find ninja and nvcc (CUDA 13.0)
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
# FlashInfer JIT needs CUDA_HOME pointing to nvcc 13.0 for sm_120 (Blackwell)
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
# FlashInfer needs to know GPU arch for sm_120 (Blackwell)
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
# FlashInfer .so needs conda's libstdc++ (GLIBCXX_3.4.32) at runtime
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

# Check data file exists
if [[ ! -f "$DATA" ]]; then
  echo "[run_grpo_verl.sh] verl data not found at $DATA"
  echo "[run_grpo_verl.sh] Running preprocessing to generate parquet..."
  MAX_SAMPLES="${PREPROCESS_MAX_SAMPLES:-}" "${ROOT}/train_scripts/local/run_preprocess.sh"
fi

# Check model path exists
if [[ ! -d "$MODEL" ]]; then
  echo "[run_grpo_verl.sh] ERROR: Model path does not exist: $MODEL" >&2
  exit 1
fi

# Check reward module exists
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "[run_grpo_verl.sh] ERROR: Reward module not found: $REWARD_PATH" >&2
  exit 1
fi

# Calculate steps_per_epoch for save_freq and test_freq.
# verl PPO trainer: total_training_steps = steps_per_epoch * total_epochs
# Its train dataloader uses drop_last=True, so steps_per_epoch is floor(
# num_samples / train_batch_size), not ceil.
NUM_SAMPLES=$("$PYTHON" -c "
import pyarrow.parquet as pq
print(pq.read_metadata('${DATA}').num_rows)
" 2>/dev/null || echo "0")
if [[ "$NUM_SAMPLES" == "0" ]]; then
  echo "[run_grpo_verl.sh] ERROR: Cannot read parquet metadata from $DATA" >&2
  exit 1
fi
STEPS_PER_EPOCH=$(( NUM_SAMPLES / TRAIN_BATCH_SIZE ))
if [[ "$STEPS_PER_EPOCH" -lt 1 ]]; then
  echo "[run_grpo_verl.sh] ERROR: Not enough samples ($NUM_SAMPLES) for train_batch_size=$TRAIN_BATCH_SIZE" >&2
  exit 1
fi
echo "[run_grpo_verl.sh] samples=$NUM_SAMPLES, batch_size=$TRAIN_BATCH_SIZE, steps_per_epoch=$STEPS_PER_EPOCH"

CKPT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/eval"
FULL_TRAINING_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))
SAVE_FREQ="${SAVE_FREQ:-$STEPS_PER_EPOCH}"

# Initialize evaluation tracking variables
BEST_ACCURACY="-1"
BEST_CKPT=""
EVAL_SUMMARY="$EVAL_DIR/epoch_summary.jsonl"
mkdir -p "$CKPT_DIR" "$EVAL_DIR"
> "$EVAL_SUMMARY"

{
  echo "ROOT=$ROOT"
  echo "PYTHON=$PYTHON"
  echo "VERL_HOME=$VERL_HOME"
  echo "MODEL=$MODEL"
  echo "DATA=$DATA"
  echo "REWARD_PATH=$REWARD_PATH"
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "PROFILE=$PROFILE"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "KL_LOSS_COEF=$KL_LOSS_COEF"
  echo "ROLLOUT_N=$ROLLOUT_N"
  echo "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE"
  echo "ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "TOTAL_EPOCHS=$TOTAL_EPOCHS"
  echo "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
  echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
  echo "LOG_PROB_MICRO_BATCH_SIZE=$LOG_PROB_MICRO_BATCH_SIZE"
  echo "REF_LOG_PROB_MICRO_BATCH_SIZE=$REF_LOG_PROB_MICRO_BATCH_SIZE"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "FULL_TRAINING_STEPS=$FULL_TRAINING_STEPS"
  echo "EVAL_ENABLED=$EVAL_ENABLED"
  echo "EVAL_BASELINE=$EVAL_BASELINE"
  echo "EVAL_DATA_PATH=$EVAL_DATA_PATH"
  echo "EVAL_GPU_MEM_UTIL=$EVAL_GPU_MEM_UTIL"
  echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
} > "$CONFIG_DIR/run_env.txt"

float_gt() {
  "$PYTHON" - "$1" "$2" <<'PY'
import sys
left = float(sys.argv[1])
right = float(sys.argv[2])
sys.exit(0 if left > right else 1)
PY
}

cd "$ROOT"
echo "=========================================="
echo "verl GRPO Training (per-epoch eval)"
echo "=========================================="
echo "  Model:         $MODEL"
echo "  Data:          $DATA ($NUM_SAMPLES samples)"
echo "  Reward:        $REWARD_PATH"
echo "  Output:        $OUTPUT_DIR"
echo "  Python:        $PYTHON"
echo "  verl:          $VERL_HOME"
echo "  Profile:       $PROFILE"
echo "  Actor LR:      $ACTOR_LR"
echo "  KL coef:       $KL_LOSS_COEF"
echo "  Rollout n:     $ROLLOUT_N"
echo "  Temperature:    $ROLLOUT_TEMPERATURE"
echo "  Total epochs:  $TOTAL_EPOCHS"
echo "  Steps/epoch:   $STEPS_PER_EPOCH"
echo "  Full steps:    $FULL_TRAINING_STEPS"
echo "  Save freq:     $SAVE_FREQ"
echo "  Max response:  $MAX_RESPONSE_LENGTH"
echo "  Eval enabled:  $EVAL_ENABLED"
echo "  Eval data:     $EVAL_DATA_PATH"
echo "  Logs:          $LOG_DIR"
echo "  Metrics:       $METRICS_DIR/train_metrics.jsonl"
echo "=========================================="

# ── Step 1: Optional baseline evaluation on the initial model ──
if [[ "$EVAL_ENABLED" == "1" && "$EVAL_BASELINE" == "1" ]]; then
  echo ""
  echo "=== Baseline Evaluation (before training) ==="
  BASELINE_EVAL_DIR="$EVAL_DIR/baseline"
  mkdir -p "$BASELINE_EVAL_DIR"
  "$PYTHON" -m train_pipeline.eval_vllm_chaingsm \
    --model-path "$MODEL" \
    --data-path "$EVAL_DATA_PATH" \
    --output-dir "$BASELINE_EVAL_DIR" \
    --method train_json_prompt \
    --batch-size "$EVAL_BATCH_SIZE" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$EVAL_GPU_MEM_UTIL" \
    --gpu-memory-utilization-candidate "$EVAL_GPU_MEM_UTIL" \
    --gpu-memory-utilization-candidate 0.3 \
    --max-tokens 2048 \
    --top-k 1 \
    --dtype auto && {
      BASELINE_ACCURACY=$("$PYTHON" -c "
import json
with open('${BASELINE_EVAL_DIR}/eval_result.json') as f:
    data = json.load(f)
print(f\"{data['overall_accuracy']:.6f}\")
" 2>/dev/null || echo "0.0")
      echo "  Baseline accuracy: $BASELINE_ACCURACY"
      echo "{\"stage\":\"baseline\",\"epoch\":0,\"global_step\":0,\"overall_accuracy\":${BASELINE_ACCURACY},\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"${BASELINE_EVAL_DIR}\",\"current_model_dir\":\"${MODEL}\",\"eval_status\":\"ok\"}" >> "$EVAL_SUMMARY"
      if float_gt "$BASELINE_ACCURACY" "$BEST_ACCURACY"; then
        BEST_ACCURACY="$BASELINE_ACCURACY"
        BEST_CKPT="$MODEL"
      fi
    } || {
      echo "[run_grpo_verl.sh] WARNING: Baseline eval failed, continuing with training"
      echo "{\"stage\":\"baseline\",\"epoch\":0,\"global_step\":0,\"overall_accuracy\":\"\",\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"${BASELINE_EVAL_DIR}\",\"current_model_dir\":\"${MODEL}\",\"eval_status\":\"failed\"}" >> "$EVAL_SUMMARY"
    }
  echo "=== Baseline evaluation complete ==="
  echo ""
fi

# ── Step 2: Run verl GRPO training epoch-by-epoch with per-epoch evaluation ──
for EPOCH_NUM in $(seq 1 $TOTAL_EPOCHS); do
  echo ""
  echo "=========================================="
  echo "  Epoch $EPOCH_NUM / $TOTAL_EPOCHS"
  echo "=========================================="

  # Determine resume settings
  if [[ $EPOCH_NUM -eq 1 ]]; then
    RESUME_MODE="disable"
  else
    RESUME_MODE="auto"
    # Stop Ray from previous epoch to free GPU memory for evaluation
    ray stop --force 2>/dev/null || true
  fi

  # Run verl training for this epoch
  echo "  Starting training (resume_mode=$RESUME_MODE)..."
  if ! "$PYTHON" -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    --config-dir "${VERL_HOME}/verl/trainer/config" \
    \
    `# === Algorithm ===` \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    `# === Data ===` \
    data.train_files="['$DATA']" \
    data.val_files="['$DATA']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    `# === Model ===` \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# === Actor ===` \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.betas="$ADAM_BETAS" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER_TYPE \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    actor_rollout_ref.actor.optim.clip_grad=$MAX_GRAD_NORM \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    `# Save HF-format model for vLLM evaluation` \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']" \
    \
    `# === Rollout ===` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    \
    `# === Reference ===` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# === Reward ===` \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_reward \
    +custom_reward_function.reward_kwargs.format_weight=$FORMAT_WEIGHT \
    +custom_reward_function.reward_kwargs.answer_weight=$ANSWER_WEIGHT \
    +custom_reward_function.reward_kwargs.expression_weight=$EXPRESSION_WEIGHT \
    +custom_reward_function.reward_kwargs.trace_weight=$TRACE_WEIGHT \
    +custom_reward_function.reward_kwargs.distractor_penalty=$DISTRACTOR_PENALTY \
    +custom_reward_function.reward_kwargs.invalid_reward=$INVALID_REWARD \
    \
    `# === Trainer ===` \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name=chaingsm_local_grpo_verl \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=$EPOCH_NUM \
    trainer.total_training_steps=$FULL_TRAINING_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=$RESUME_MODE \
    \
    `# === Ray ===` \
    ray_kwargs.ray_init.num_cpus=16 \
    \
    "$@"; then
    echo "[run_grpo_verl.sh] ERROR: verl training failed during epoch $EPOCH_NUM" >&2
    echo "{\"stage\":\"train_epoch_${EPOCH_NUM}\",\"epoch\":${EPOCH_NUM},\"global_step\":\"\",\"overall_accuracy\":\"\",\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"\",\"current_model_dir\":\"\",\"eval_status\":\"train_failed\"}" >> "$EVAL_SUMMARY"
    exit 1
  fi

  echo "  Epoch $EPOCH_NUM training step complete."

  # Stop Ray to free GPU memory for evaluation
  ray stop --force 2>/dev/null || true

  # Find the latest checkpoint (the one just saved at this epoch boundary)
  # Use basename extraction to avoid path underscores breaking sort field alignment
  LATEST_CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" -printf '%f\n' 2>/dev/null | sort -t_ -k3 -n | tail -1 | xargs -I{} echo "$CKPT_DIR/{}" || true)

  if [[ -z "$LATEST_CKPT" ]]; then
    echo "[run_grpo_verl.sh] ERROR: No checkpoint found after epoch $EPOCH_NUM; cannot resume or evaluate safely" >&2
    echo "{\"stage\":\"epoch_${EPOCH_NUM}\",\"epoch\":${EPOCH_NUM},\"global_step\":\"\",\"overall_accuracy\":\"\",\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"\",\"current_model_dir\":\"\",\"eval_status\":\"checkpoint_missing\"}" >> "$EVAL_SUMMARY"
    exit 1
  fi

  EPOCH_GLOBAL_STEP=$(basename "$LATEST_CKPT" | sed 's/global_step_//')
  CURRENT_EPOCH=$(( EPOCH_GLOBAL_STEP / STEPS_PER_EPOCH ))
  HF_MODEL_DIR="$LATEST_CKPT/actor/huggingface"

  # ── Per-epoch evaluation ──
  if [[ "$EVAL_ENABLED" == "1" && -d "$HF_MODEL_DIR" ]]; then
    EPOCH_EVAL_DIR="$EVAL_DIR/epoch_$(printf '%04d' $CURRENT_EPOCH)"
    echo ""
    echo "  --- Evaluating epoch $CURRENT_EPOCH (global_step=$EPOCH_GLOBAL_STEP) ---"
    echo "  Model: $HF_MODEL_DIR"

    if "$PYTHON" -m train_pipeline.eval_vllm_chaingsm \
      --model-path "$HF_MODEL_DIR" \
      --data-path "$EVAL_DATA_PATH" \
      --output-dir "$EPOCH_EVAL_DIR" \
      --method train_json_prompt \
      --batch-size "$EVAL_BATCH_SIZE" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$EVAL_GPU_MEM_UTIL" \
      --gpu-memory-utilization-candidate "$EVAL_GPU_MEM_UTIL" \
      --gpu-memory-utilization-candidate 0.3 \
      --max-tokens 2048 \
      --top-k 1 \
      --dtype auto; then

      # Read accuracy from eval result
      ACCURACY=$("$PYTHON" -c "
import json
with open('${EPOCH_EVAL_DIR}/eval_result.json') as f:
    data = json.load(f)
print(f\"{data['overall_accuracy']:.6f}\")
" 2>/dev/null || echo "0.0")

      echo "  Accuracy: $ACCURACY"

      # Append to summary
      echo "{\"stage\":\"epoch_${CURRENT_EPOCH}\",\"epoch\":${CURRENT_EPOCH},\"global_step\":${EPOCH_GLOBAL_STEP},\"overall_accuracy\":${ACCURACY},\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"${EPOCH_EVAL_DIR}\",\"current_model_dir\":\"${HF_MODEL_DIR}\",\"eval_status\":\"ok\"}" >> "$EVAL_SUMMARY"

      # Track best model
      if float_gt "$ACCURACY" "$BEST_ACCURACY"; then
        BEST_ACCURACY="$ACCURACY"
        BEST_CKPT="$HF_MODEL_DIR"
        echo "  *** New best model! ***"
      fi
    else
      echo "  Evaluation FAILED"
      echo "{\"stage\":\"epoch_${CURRENT_EPOCH}\",\"epoch\":${CURRENT_EPOCH},\"global_step\":${EPOCH_GLOBAL_STEP},\"overall_accuracy\":\"\",\"best_accuracy\":${BEST_ACCURACY},\"eval_dir\":\"${EPOCH_EVAL_DIR}\",\"current_model_dir\":\"${HF_MODEL_DIR}\",\"eval_status\":\"failed\"}" >> "$EVAL_SUMMARY"
    fi
  else
    if [[ "$EVAL_ENABLED" != "1" ]]; then
      echo "  Evaluation skipped (EVAL_ENABLED=$EVAL_ENABLED)"
    else
      echo "  WARNING: HF model dir not found at $HF_MODEL_DIR, skipping eval"
    fi
  fi

  echo "  Epoch $EPOCH_NUM complete."
done

echo ""
echo "=== All training epochs complete ==="

# ── Step 3: Save best model ──
echo ""
echo "=== Evaluation Summary ==="
echo "  Best accuracy: $BEST_ACCURACY"
echo "  Best checkpoint: $BEST_CKPT"

if [[ -n "$BEST_CKPT" && -d "$BEST_CKPT" ]]; then
  BEST_DIR="$OUTPUT_DIR/checkpoints/best"
  if [[ -d "$BEST_DIR" ]]; then
    rm -rf "$BEST_DIR"
  fi
  cp -r "$BEST_CKPT" "$BEST_DIR"
  echo "  Best model copied to: $BEST_DIR"
fi

# Write latest metrics
"$PYTHON" -c "
import json
best_acc = float('${BEST_ACCURACY}') if '${BEST_ACCURACY}' != '-1' else 0.0
rows = []
with open('${EVAL_SUMMARY}', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]
ok_rows = [row for row in rows if row.get('eval_status') == 'ok']
epoch_ok_rows = [row for row in ok_rows if str(row.get('stage', '')).startswith('epoch_')]
metrics = {
    'best_accuracy': best_acc,
    'best_model_dir': '${BEST_CKPT}',
    'total_eval_rows': len(rows),
    'total_ok_evals': len(ok_rows),
    'total_epoch_checkpoints_evaluated': len(epoch_ok_rows),
    'metrics_jsonl': '${METRICS_DIR}/train_metrics.jsonl',
    'logs_dir': '${LOG_DIR}',
}
with open('${EVAL_DIR}/latest_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
with open('${METRICS_DIR}/latest_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Latest metrics written to ${EVAL_DIR}/latest_metrics.json')
"

echo ""
echo "=== All done ==="
