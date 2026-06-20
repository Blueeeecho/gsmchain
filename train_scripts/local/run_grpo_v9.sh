#!/usr/bin/env bash
# LB-PRM v9 GRPO 训练入口
#
# 关键设计 (相对 v8.2):
#   - 起点: SFT 2 epoch ckpt (本项目刚训完的, acc 20.5%)
#   - 训练 prompt: all_sft.jsonl 协议 (STEP 模板, system+user 自带)
#   - 训练数据: all_grpo_v9.parquet (6102 条, 3051 original + 3051 变体)
#   - REWARD: v9 (4 子项, 0.2/2.5/1.5/-0.5, normalized edit similarity)
#   - MAX_STEPS: 500 (用户授权, 沿用 v8.2 LR/KL/ROLLOUT_N)
#   - SAVE_FREQ: 100 (eval 在训练结束后统一跑, 不在每节点自动跑)
#
# 终止条件: 500 step 自动停
#
# 用法:
#   bash train_scripts/local/run_grpo_v9.sh 2>&1 | tee /tmp/v9_run.log

set -uo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# v9 起点: SFT 2 epoch ckpt
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2ep_all/epoch2/checkpoints/current}"
MODEL="${MODEL:-$SFT_CKPT_DIR}"

# 训练数据
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/sft/all_grpo_v9.parquet}"

# v9 reward
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_v9_verl.py}"

MAX_STEPS="${MAX_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-100}"

RUN_NAME="${RUN_NAME:-grpo_v9}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_v9/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
LOG_DIR="$OUTPUT_DIR/logs"
METRICS_DIR="$OUTPUT_DIR/metrics"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/eval"
mkdir -p "$LOG_DIR" "$METRICS_DIR" "$CKPT_DIR" "$EVAL_DIR"

cleanup() { ray stop --force 2>/dev/null || true; }
trap cleanup EXIT

[[ -d "$MODEL" ]] || { echo "ERROR: SFT ckpt not found: $MODEL" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA" >&2; exit 1; }
[[ -f "$REWARD_PATH" ]] || { echo "ERROR: reward module not found: $REWARD_PATH" >&2; exit 1; }

# 沿用 v8.2 超参
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"

ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"

# v9 reward 权重 (0.2/2.5/1.5/-0.5)
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.2}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-2.5}"
CORE_WEIGHT="${CORE_WEIGHT:-1.5}"
DISTRACTOR_WEIGHT="${DISTRACTOR_WEIGHT:-0.5}"
INVALID_REWARD="${INVALID_REWARD:--0.5}"

export CUDA_MODULE_LOADING=LAZY
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

echo "=== v9 GRPO 训练配置 ==="
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "REWARD=$REWARD_PATH"
echo "MAX_STEPS=$MAX_STEPS SAVE_FREQ=$SAVE_FREQ"
echo "REWARD_WEIGHTS: format=$FORMAT_WEIGHT answer=$ANSWER_WEIGHT core=$CORE_WEIGHT distractor=-$DISTRACTOR_WEIGHT"
echo "ACTOR_LR=$ACTOR_LR KL=$KL_LOSS_COEF ROLLOUT_N=$ROLLOUT_N"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo ""

cd "$ROOT"

"$PYTHON" -m verl.trainer.main_ppo \
  --config-name ppo_trainer \
  --config-dir "$VERL_HOME/verl/trainer/config" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files="['$DATA']" \
  data.val_files="['${ROOT}/chaingsm_data/data/final/sft/_dummy_val.parquet']" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=8 \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=2 \
  \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.optim.betas="[0.9,0.99]" \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.optim.clip_grad=0.1 \
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
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
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
  custom_reward_function.path=$REWARD_PATH \
  custom_reward_function.name=compute_reward \
  +custom_reward_function.reward_kwargs.format_weight=0.2 \
  +custom_reward_function.reward_kwargs.answer_weight=2.5 \
  +custom_reward_function.reward_kwargs.core_weight=1.5 \
  +custom_reward_function.reward_kwargs.distractor_weight=0.5 \
  \
  trainer.project_name=grpo_v9 \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$MAX_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=-1 \
  trainer.critic_warmup=0 \
  trainer.logger="['file']" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.max_actor_ckpt_to_keep=2 \
  "$@"
