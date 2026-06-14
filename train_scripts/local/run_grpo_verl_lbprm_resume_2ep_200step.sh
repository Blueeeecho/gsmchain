#!/usr/bin/env bash
# Resume GRPO from existing step 200 checkpoint and run remaining ~1.5 epoch
# to reach step 3526, with per-200-step evaluation on the clean test set.
#
# This is a near-clone of run_grpo_verl_lbprm_base_2ep_200step.sh but:
#   - MODEL points to the existing step 200 HF model (warm start)
#   - First epoch uses resume_mode=auto (so verl continues from step 200)
#   - EVAL_BASELINE=0 (we already have a baseline from the previous run)
#
# Usage: bash train_scripts/local/run_grpo_verl_lbprm_resume_2ep_200step.sh

set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# Reuse the same run dir as the original (has step 200 ckpt)
RUN_NAME="${RUN_NAME:-grpo_verl_lbprm_base_2ep_200step}"
EXISTING_RUN_DIR="$(ls -1dt "$ROOT/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/$RUN_NAME"/*/ 2>/dev/null | head -1)"
if [[ -z "$EXISTING_RUN_DIR" ]]; then
  echo "[resume] ERROR: cannot find an existing run dir under $RUN_NAME/*/" >&2
  exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-${EXISTING_RUN_DIR%/}}"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_DIR="$OUTPUT_DIR/configs"
METRICS_DIR="$OUTPUT_DIR/metrics"
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$METRICS_DIR"

echo "[resume] using existing OUTPUT_DIR=$OUTPUT_DIR"
echo "[resume] existing ckpt: $(cat "$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null)"

# Data
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet}"

# Reward
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_lbprm_v2_verl.py}"

# Warm-start: continue from step 200's HF model
MODEL="${MODEL:-$OUTPUT_DIR/checkpoints/global_step_200/actor/huggingface}"

exec > >(tee -a "$LOG_DIR/grpo_verl_resume_stdout.log") \
  2> >(tee -a "$LOG_DIR/grpo_verl_resume_stderr.log" >&2)

cleanup() {
  ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# Hyperparameters (same as the original run)
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
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$DEFAULT_MAX_RESPONSE_LENGTH}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-$DEFAULT_LOG_PROB_MICRO_BATCH_SIZE}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$LOG_PROB_MICRO_BATCH_SIZE}"

# Optimizer
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.99]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"

# Reward weights (LB-PRM v2)
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.2}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-2.5}"
LIVENESS_WEIGHT="${LIVENESS_WEIGHT:-0.4}"
INVALID_REWARD="${INVALID_REWARD:--0.5}"

# Eval settings
EVAL_ENABLED="${EVAL_ENABLED:-1}"
EVAL_DATA_PATH="${EVAL_DATA:-${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_BASELINE="${EVAL_BASELINE:-0}"   # we already have a baseline

# vLLM env
export CUDA_MODULE_LOADING=LAZY
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

# Check
if [[ ! -f "$DATA" ]]; then
  echo "[resume] ERROR: $DATA not found" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "[resume] ERROR: warm-start model not found: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "[resume] ERROR: reward not found: $REWARD_PATH" >&2
  exit 1
fi

NUM_SAMPLES=$("$PYTHON" -c "
import pyarrow.parquet as pq
print(pq.read_metadata('${DATA}').num_rows)
" 2>/dev/null || echo "0")
STEPS_PER_EPOCH=$(( NUM_SAMPLES / TRAIN_BATCH_SIZE ))
if [[ "$STEPS_PER_EPOCH" -lt 1 ]]; then
  echo "[resume] ERROR: NUM_SAMPLES=$NUM_SAMPLES too small" >&2
  exit 1
fi
echo "[resume] samples=$NUM_SAMPLES, batch_size=$TRAIN_BATCH_SIZE, steps_per_epoch=$STEPS_PER_EPOCH"

CKPT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/eval"
EVAL_SUMMARY="$EVAL_DIR/epoch_summary.jsonl"
mkdir -p "$CKPT_DIR" "$EVAL_DIR"
# don't truncate eval summary: it already has baseline + step_00200 entry

# Save every 200 steps so eval can fire at each.
SAVE_FREQ="${SAVE_FREQ:-200}"
FULL_TRAINING_STEPS="${FULL_TRAINING_STEPS:-3526}"

{
  echo "RESUME_RUN_DIR=$OUTPUT_DIR"
  echo "MODEL=$MODEL"
  echo "DATA=$DATA"
  echo "REWARD_PATH=$REWARD_PATH"
  echo "TOTAL_EPOCHS=$TOTAL_EPOCHS"
  echo "STEPS_PER_EPOCH=$STEPS_PER_EPOCH"
  echo "FULL_TRAINING_STEPS=$FULL_TRAINING_STEPS"
  echo "SAVE_FREQ=$SAVE_FREQ"
} > "$CONFIG_DIR/run_env_resume.txt"

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
echo "verl GRPO Training (RESUME from step 200)"
echo "=========================================="
echo "  Model:         $MODEL"
echo "  Output:        $OUTPUT_DIR"
echo "  Data:          $DATA"
echo "  Steps/epoch:   $STEPS_PER_EPOCH"
echo "  Full steps:    $FULL_TRAINING_STEPS"
echo "  Save freq:     $SAVE_FREQ"
echo "=========================================="

for EPOCH_NUM in $(seq 1 $TOTAL_EPOCHS); do
  echo ""
  echo "=========================================="
  echo "  Epoch $EPOCH_NUM / $TOTAL_EPOCHS (resume from step 200)"
  echo "=========================================="

  # First epoch: auto-resume; later: also auto
  if [[ $EPOCH_NUM -eq 1 ]]; then
    RESUME_MODE="auto"
  else
    RESUME_MODE="auto"
    ray stop --force 2>/dev/null || true
  fi

  echo "  Starting training (resume_mode=$RESUME_MODE)..."
  if ! "$PYTHON" -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    --config-dir "${VERL_HOME}/verl/trainer/config" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    data.train_files="['$DATA']" \
    data.val_files="['$DATA']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
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
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra','hf_model']" \
    \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_reward \
    +custom_reward_function.reward_kwargs.format_weight=$FORMAT_WEIGHT \
    +custom_reward_function.reward_kwargs.answer_weight=$ANSWER_WEIGHT \
    +custom_reward_function.reward_kwargs.liveness_weight=$LIVENESS_WEIGHT \
    +custom_reward_function.reward_kwargs.invalid_reward=$INVALID_REWARD \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name=chaingsm_local_grpo_verl \
    trainer.experiment_name="${RUN_NAME}_resume" \
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
    ray_kwargs.ray_init.num_cpus=16 \
    \
    "$@"; then
    echo "[resume] ERROR: verl training failed during epoch $EPOCH_NUM" >&2
    echo "{\"stage\":\"train_epoch_${EPOCH_NUM}_failed\",\"epoch\":${EPOCH_NUM}}" >> "$EVAL_SUMMARY"
    exit 1
  fi

  echo "  Epoch $EPOCH_NUM training step complete."

  ray stop --force 2>/dev/null || true

  # Discover every new checkpoint saved this epoch that is a multiple of SAVE_FREQ
  # and hasn't been evaluated yet.
  CKPT_STEPS=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "global_step_*" -printf '%f\n' 2>/dev/null \
    | sed 's/global_step_//' \
    | sort -n \
    | awk -v sf="$SAVE_FREQ" '$1 % sf == 0' \
    || true)
  EVALED_STEPS=$(awk -F'"global_step":' '/global_step/ {print $2}' "$EVAL_SUMMARY" 2>/dev/null \
    | awk -F',' '{print $1}' | sort -n | uniq || true)
  PENDING_STEPS=$(comm -23 <(echo "$CKPT_STEPS") <(echo "$EVALED_STEPS") || true)

  if [[ -z "$PENDING_STEPS" ]]; then
    echo "  No new (multiple of $SAVE_FREQ) checkpoints to evaluate this epoch."
  fi

  for CKPT_STEP in $PENDING_STEPS; do
    HF_MODEL_DIR="$CKPT_DIR/global_step_${CKPT_STEP}/actor/huggingface"
    if [[ ! -d "$HF_MODEL_DIR" ]]; then
      echo "  WARN: HF model dir missing for global_step=$CKPT_STEP ($HF_MODEL_DIR); skipping."
      continue
    fi
    CURRENT_EPOCH=$(( CKPT_STEP / STEPS_PER_EPOCH ))
    EPOCH_EVAL_DIR="$EVAL_DIR/step_$(printf '%05d' $CKPT_STEP)"

    echo ""
    echo "  --- Evaluating global_step=$CKPT_STEP (epoch~=$CURRENT_EPOCH) ---"

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

      ACCURACY=$("$PYTHON" -c "
import json
with open('${EPOCH_EVAL_DIR}/eval_result.json') as f:
    data = json.load(f)
print(f\"{data['overall_accuracy']:.6f}\")
" 2>/dev/null || echo "0.0")
      ORIGINAL_ACCURACY=$("$PYTHON" -c "
import json
with open('${EPOCH_EVAL_DIR}/eval_result.json') as f:
    data = json.load(f)
by_cat = data.get('by_category', {})
print(f\"{by_cat.get('original', 0.0):.6f}\")
" 2>/dev/null || echo "0.0")
      echo "  Accuracy: overall=$ACCURACY  original=$ORIGINAL_ACCURACY"

      echo "{\"stage\":\"step_${CKPT_STEP}\",\"epoch\":${CURRENT_EPOCH},\"global_step\":${CKPT_STEP},\"overall_accuracy\":${ACCURACY},\"original_accuracy\":${ORIGINAL_ACCURACY},\"eval_dir\":\"${EPOCH_EVAL_DIR}\",\"current_model_dir\":\"${HF_MODEL_DIR}\",\"eval_status\":\"ok\"}" >> "$EVAL_SUMMARY"
    else
      echo "  Evaluation FAILED at global_step=$CKPT_STEP"
      echo "{\"stage\":\"step_${CKPT_STEP}_failed\",\"epoch\":${CURRENT_EPOCH},\"global_step\":${CKPT_STEP}}" >> "$EVAL_SUMMARY"
    fi
  done

  echo "  Epoch $EPOCH_NUM complete."
done

echo ""
echo "=== All training epochs complete ==="
echo "Summary: $EVAL_SUMMARY"
