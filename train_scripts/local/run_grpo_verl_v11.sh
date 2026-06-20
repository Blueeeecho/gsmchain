#!/usr/bin/env bash
# verl GRPO v11 training: 1000 步 (v10 → v11 奖励优化, 验证 reward 信号强度)
#
# v11 关键变化:
#   - 删 r_format (饱和)
#   - r_calc 替换为 r_step_value (vs gold step values)
#   - r_answer 2.5 → 3.0
#   - r_core 1.2 → 0.5
#   - core_trace_w / core_final_w 0.7/0.3 → 0.8/0.2
#
# 配置:
#   - 1000 步, 起点 SFT epoch3 ckpt-762
#   - 每 200 步保存一次 (共 5 个 ckpt: 200/400/600/800/1000)
#   - 不跑自动 eval, 训练完手动跑
#
# 启动:
#   bash train_scripts/local/run_grpo_verl_v11.sh

set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# 起点: SFT epoch3 ckpt
MODEL="${MODEL:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep_resume/20260617_180315/checkpoints/checkpoint-762}"

# GRPO 数据
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/grpo/all_grpo_cot.parquet}"

# v11 奖励函数
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_v11_verl.py}"

# Run identification
RUN_NAME="${RUN_NAME:-grpo_v11_stepvalue}"
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

# Hyperparameters
TOTAL_GRPO_STEPS="${TOTAL_GRPO_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-200}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.04}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$LOG_PROB_MICRO_BATCH_SIZE}"

# Optimizer
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.99]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"

# Environment variables for Blackwell compatibility
export CUDA_MODULE_LOADING=LAZY
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

# Check model path exists
if [[ ! -d "$MODEL" ]]; then
  echo "[run_grpo_verl_v11.sh] ERROR: Model path does not exist: $MODEL" >&2
  exit 1
fi

# Check reward module exists
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "[run_grpo_verl_v11.sh] ERROR: Reward module not found: $REWARD_PATH" >&2
  exit 1
fi

# steps_per_epoch
NUM_SAMPLES=$("$PYTHON" -c "
import pyarrow.parquet as pq
print(pq.read_metadata('${DATA}').num_rows)
" 2>/dev/null || echo "0")
STEPS_PER_EPOCH=$(( NUM_SAMPLES / TRAIN_BATCH_SIZE ))
echo "[run_grpo_verl_v11.sh] samples=$NUM_SAMPLES, batch=$TRAIN_BATCH_SIZE, steps_per_epoch=$STEPS_PER_EPOCH"

CKPT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "MODEL=$MODEL"
  echo "DATA=$DATA"
  echo "REWARD_PATH=$REWARD_PATH"
  echo "TOTAL_GRPO_STEPS=$TOTAL_GRPO_STEPS"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "KL_LOSS_COEF=$KL_LOSS_COEF"
  echo "ROLLOUT_N=$ROLLOUT_N"
  echo "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE"
  echo "ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
  echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
} > "$CONFIG_DIR/run_env.txt"

cd "$ROOT"
echo "=========================================="
echo "verl GRPO v11 Training (1000 steps, save every 200)"
echo "=========================================="
echo "  Model:        $MODEL"
echo "  Data:         $DATA ($NUM_SAMPLES samples)"
echo "  Reward:       $REWARD_PATH (v11: step_value vs gold)"
echo "  Output:       $OUTPUT_DIR"
echo "  Total steps:  $TOTAL_GRPO_STEPS"
echo "  Save freq:    $SAVE_FREQ (expect ckpts @ 200/400/600/800/1000)"
echo "  Actor LR:     $ACTOR_LR"
echo "  Rollout n:    $ROLLOUT_N"
echo "  Temperature:  $ROLLOUT_TEMPERATURE"
echo "=========================================="

# v11 奖励: total = 3.0*ans + 1.5*step_value + 0.5*core - 0.5*dist
"$PYTHON" -m verl.trainer.main_ppo \
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
  actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']" \
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
  +custom_reward_function.reward_kwargs.answer_weight=3.0 \
  +custom_reward_function.reward_kwargs.step_value_weight=1.5 \
  +custom_reward_function.reward_kwargs.core_weight=0.5 \
  +custom_reward_function.reward_kwargs.core_trace_w=0.8 \
  +custom_reward_function.reward_kwargs.core_final_w=0.2 \
  +custom_reward_function.reward_kwargs.distractor_weight=0.5 \
  +custom_reward_function.reward_kwargs.invalid_reward=-0.5 \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","file"]' \
  trainer.project_name=chaingsm_local_grpo_verl \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=999 \
  trainer.total_training_steps=$TOTAL_GRPO_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=10 \
  \
  ray_kwargs.ray_init.num_cpus=16

echo ""
echo "=== v11 training complete: $TOTAL_GRPO_STEPS steps ==="
echo "  Checkpoints: $CKPT_DIR/global_step_*"
