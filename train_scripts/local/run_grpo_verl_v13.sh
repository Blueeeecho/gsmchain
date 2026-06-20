#!/usr/bin/env bash
# verl GRPO v13 training: 5 字段 JSON + 3 rules + hard per-step 数值 reward
# 起点: Qwen2.5-0.5B-Instruct base (跳过 SFT, V13 重新 GRPO)
# 训练量: 2 epoch × 762 steps/epoch = 1524 total steps (Q4A: 先到 step 300 验证, OK 后继续)
# 保存: 每 300 步 (与 V12 同步)

set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# 起点: base
MODEL_BASE="/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct"

# GRPO 数据 (3051 行)
DATA="${ROOT}/chaingsm_data/data/final/grpo/grpo_v13_json.parquet"

# v12 奖励
REWARD_PATH="${ROOT}/train_pipeline/reward_chaingsm_v13_json_verl.py"

# Run
RUN_NAME="${RUN_NAME:-qwen2.5-0.5b-grpo-verl-v13-json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_DIR="$OUTPUT_DIR/configs"
METRICS_DIR="$OUTPUT_DIR/metrics"
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$METRICS_DIR"

# Hyperparameters
SAVE_FREQ="${SAVE_FREQ:-300}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.04}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"

# v12 关键: max_prompt_length 升 768 -> 1280 (JSON 4 示范 + 12 rules + R12)
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"  # v13 prompt 1328, 留 200 余量
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"

LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"

# Optimizer
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.99]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"

# 显存压缩
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-True}"

# Environment
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

# Reward check
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "ERROR: Reward module not found: $REWARD_PATH" >&2
  exit 1
fi

# Snapshot configs
cp "${ROOT}/train_configs/local/grpo_verl_v12.yaml" "$CONFIG_DIR/grpo_verl_v12.yaml"

# steps_per_epoch
NUM_SAMPLES=$("$PYTHON" -c "import pyarrow.parquet as pq; print(pq.read_metadata('${DATA}').num_rows)")
STEPS_PER_EPOCH=$(( NUM_SAMPLES / TRAIN_BATCH_SIZE ))
echo "[v12] samples=$NUM_SAMPLES, batch=$TRAIN_BATCH_SIZE, steps_per_epoch=$STEPS_PER_EPOCH, total_epochs=2, total_steps=$((STEPS_PER_EPOCH * 2))"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "DATA=$DATA"
  echo "REWARD_PATH=$REWARD_PATH"
  echo "MODEL=$MODEL_BASE"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
  echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
  echo "ROLLOUT_N=$ROLLOUT_N"
  echo "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE"
  echo "ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD"
  echo "ACTOR_OPTIM_OFFLOAD=$ACTOR_OPTIM_OFFLOAD"
} > "$CONFIG_DIR/run_env.txt"

cd "$ROOT"
echo "=========================================="
echo "verl GRPO v12 (JSON output, 2 epoch)"
echo "=========================================="
echo "  Model:         $MODEL_BASE (base, no SFT)"
echo "  Data:          $DATA ($NUM_SAMPLES samples)"
echo "  Reward:        $REWARD_PATH"
echo "  Output:        $OUTPUT_DIR"
echo "  Save freq:     $SAVE_FREQ (expect ckpts @ 300/600/900/1200/1500/1800/2100/2400/2700/3000)"
echo "  Max prompt:    $MAX_PROMPT_LENGTH (v12 JSON prompt + 2x question)"
echo "  Max resp:      $MAX_RESPONSE_LENGTH"
echo "  Total steps:   $((STEPS_PER_EPOCH * 2))"
echo "=========================================="

# v12 奖励: total = 3.0*ans + 1.5*step_value + 0.5*core - 0.5*dist (pred 解析切 JSON)
"$PYTHON" -u -m verl.trainer.main_ppo \
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
  actor_rollout_ref.model.path="$MODEL_BASE" \
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
  actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIM_OFFLOAD \
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
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  reward_model.enable=False \
  \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_reward \
  +custom_reward_function.reward_kwargs.answer_weight=3.0 \
  +custom_reward_function.reward_kwargs.step_value_weight=1.5 \
  +custom_reward_function.reward_kwargs.step_format_weight=0.5 \
  +custom_reward_function.reward_kwargs.format_weight=0.5 \
  +custom_reward_function.reward_kwargs.exclude_weight=0.5 \
  \
  trainer.critic_warmup=0 \
  trainer.project_name=chaingsm_local_grpo_verl \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=2 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.logger="['console','file']" \
  2>&1 | tee "$LOG_DIR/stdout.log"
