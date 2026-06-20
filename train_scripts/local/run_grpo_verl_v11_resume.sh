#!/usr/bin/env bash
# verl GRPO v11 RESUME: 从 step 200 续到 1000
#
# 关键修改 (vs v11 原始脚本):
#   - trainer.resume_mode=auto (默认) — 从 default_local_dir 找最新 global_step_*
#   - trainer.total_training_steps=1000 (绝对目标) — resume 后 global_steps=200, 跑 800 步到 1000
#   - actor.fsdp_config.param_offload=True  (省 ~3-4GB)
#   - actor.fsdp_config.optimizer_offload=True  (省 ~1-2GB)
#   - max_response_length 1024 → 768  (压 KV cache 突发, 崩渍时 max=432 但 batch 极端长度会爆)
#   - OUTPUT_DIR/默认 local dir 都指向已有 run, 便于 resume 找到 ckpt
#   - 不传 model.path (resume 时由 ckpt 决定, 否则会与 init 冲突)
#
# 启动:
#   bash train_scripts/local/run_grpo_verl_v11_resume.sh
#
# 预期:
#   进程从 step 200 续到 1000, save_freq=200 → 后续 400/600/800/1000 各一份 ckpt

set -euo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# v11 续训: 复用已存在的 RUN (resumed from global_step_200)
RUN_NAME="${RUN_NAME:-grpo_v11_stepvalue}"
RUN_ID="${RUN_ID:-20260618_134820}"   # 已有 run id
OUTPUT_DIR="${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}"
LOG_DIR="$OUTPUT_DIR/logs"
CONFIG_DIR="$OUTPUT_DIR/configs"
METRICS_DIR="$OUTPUT_DIR/metrics"
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$METRICS_DIR"

exec > >(tee -a "$LOG_DIR/grpo_verl_stdout_resume.log") \
  2> >(tee -a "$LOG_DIR/grpo_verl_stderr_resume.log" >&2)

cleanup() {
  ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# 起点: SFT epoch3 ckpt (v11 原始配置; resume 时若 ckpt 优先则不读)
MODEL="${MODEL:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep_resume/20260617_180315/checkpoints/checkpoint-762}"

# GRPO 数据
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/grpo/all_grpo_cot.parquet}"

# v11 奖励函数
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_v11_verl.py}"

# Hyperparameters (匹配 v11 原始, 不变)
TOTAL_GRPO_STEPS="${TOTAL_GRPO_STEPS:-1000}"      # 绝对目标
SAVE_FREQ="${SAVE_FREQ:-200}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.04}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"

# === v11 RESUME 关键修改 ===
# 1) 缩短 max_response_length: 1024 -> 768
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"

LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-$LOG_PROB_MICRO_BATCH_SIZE}"

# Optimizer
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.99]}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.1}"

# === 显存压缩 (OOM 修复) ===
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-True}"

# Environment variables for Blackwell compatibility
export CUDA_MODULE_LOADING=LAZY
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

# Check reward module exists
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "[run_grpo_verl_v11_resume.sh] ERROR: Reward module not found: $REWARD_PATH" >&2
  exit 1
fi

# Check existing checkpoint (resume 必要条件)
RESUME_CKPT="$OUTPUT_DIR/checkpoints/global_step_200"
if [[ ! -d "$RESUME_CKPT/actor" ]]; then
  echo "[run_grpo_verl_v11_resume.sh] ERROR: Resume ckpt not found: $RESUME_CKPT" >&2
  exit 1
fi
echo "[run_grpo_verl_v11_resume.sh] Will resume from $RESUME_CKPT"

# steps_per_epoch
NUM_SAMPLES=$("$PYTHON" -c "
import pyarrow.parquet as pq
print(pq.read_metadata('${DATA}').num_rows)
" 2>/dev/null || echo "0")
STEPS_PER_EPOCH=$(( NUM_SAMPLES / TRAIN_BATCH_SIZE ))
echo "[run_grpo_verl_v11_resume.sh] samples=$NUM_SAMPLES, batch=$TRAIN_BATCH_SIZE, steps_per_epoch=$STEPS_PER_EPOCH"

# default_local_dir 指向已有 ckpt 目录, resume_mode=auto 找最新 global_step_*
CKPT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_ID=$RUN_ID"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "RESUMED_FROM=$RESUME_CKPT"
  echo "DATA=$DATA"
  echo "REWARD_PATH=$REWARD_PATH"
  echo "TOTAL_GRPO_STEPS=$TOTAL_GRPO_STEPS (absolute, 200 -> 1000)"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "ACTOR_LR=$ACTOR_LR"
  echo "ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD"
  echo "ACTOR_OPTIM_OFFLOAD=$ACTOR_OPTIM_OFFLOAD"
  echo "MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH (v11-orig=1024, v11-resume=768)"
  echo "ROLLOUT_N=$ROLLOUT_N"
  echo "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE"
  echo "ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
} > "$CONFIG_DIR/run_env_resume.txt"

cd "$ROOT"
echo "=========================================="
echo "verl GRPO v11 RESUME (200 -> 1000 steps)"
echo "=========================================="
echo "  Resuming from: $RESUME_CKPT"
echo "  Data:          $DATA ($NUM_SAMPLES samples)"
echo "  Reward:        $REWARD_PATH (v11)"
echo "  Output:        $OUTPUT_DIR"
echo "  Target:        $TOTAL_GRPO_STEPS (abs), expected +800 steps to reach"
echo "  Save freq:     $SAVE_FREQ (expect ckpts @ 400/600/800/1000)"
echo "  Actor LR:      $ACTOR_LR"
echo "  Max resp len:  $MAX_RESPONSE_LENGTH"
echo "  Param offload: $ACTOR_PARAM_OFFLOAD"
echo "  Optim offload: $ACTOR_OPTIM_OFFLOAD"
echo "=========================================="

# v11 奖励: total = 3.0*ans + 1.5*step_value + 0.5*core - 0.5*dist
# resume_mode=auto: 找 default_local_dir 下的最新 global_step_*
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
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=10 \
  \
  ray_kwargs.ray_init.num_cpus=16

echo ""
echo "=== v11 resume training complete: target=$TOTAL_GRPO_STEPS ==="
echo "  Checkpoints: $CKPT_DIR/global_step_*"
