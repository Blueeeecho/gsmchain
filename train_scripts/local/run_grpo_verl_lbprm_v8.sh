#!/usr/bin/env bash
# LB-PRM v8 GRPO 训练入口
#
# 关联 spec: docs/superpowers/specs/2026-06-14-lbprm-v8-design.md
# 关联 plan: docs/superpowers/plans/2026-06-14-lbprm-v8-plan.md
#
# 关键设计 (相对 v7):
#   - 起点: SFT 1 epoch 8-shot CoT ckpt (NEW, 不用 sft_2epoch/best, 不用 0.5B base)
#   - 训练 prompt: 8-shot CoT (跟 v7 一致)
#   - REWARD: v8 公式 (4 子项 0.05/0.85/0.05/0.05, 砍 numeric 抢 answer 梯度)
#   - MAX_STEPS: 500 → 1000 (用户授权上限)
#   - SAVE_FREQ: 100 → 200 (5 个 eval 节点 200/400/600/800/1000)
#   - 其他沿用 v7 训练协议 (LR/KL/ROLLOUT_N 等)
#
# 终止条件:
#   1) 1000 step 自动停
#   2) step 200 eval original >= 0.46 → 视为达成
#   3) step 200 eval original < SFT ckpt → 反思 SFT 是否破坏 8-shot
#   4) step 200-400 连续 2 次 original 持平且 < 0.46 → 停
#   5) step 600 eval original >= 0.46 → 强制 stop at 600
#
# 用法:
#   bash train_scripts/local/run_grpo_verl_lbprm_v8.sh 2>&1 | tee /tmp/v8_run.log

set -uo pipefail

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# v8 起点: SFT 1 epoch 8-shot CoT ckpt (默认路径, 用户可覆盖)
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${ROOT}/outputs/sft/sft_8shot_cot_1ep/Qwen2.5-0.5B-Instruct/sft_8shot_cot_1ep/latest}"
MODEL="${MODEL:-$SFT_CKPT_DIR}"

# 8-shot CoT 训练集
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_8shot_cot.parquet}"

# v8 reward
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_lbprm_v8_verl.py}"

# 用户授权 1000 步
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-200}"

RUN_NAME="${RUN_NAME:-grpo_verl_v8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
LOG_DIR="$OUTPUT_DIR/logs"
METRICS_DIR="$OUTPUT_DIR/metrics"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/eval"
mkdir -p "$LOG_DIR" "$METRICS_DIR" "$CKPT_DIR" "$EVAL_DIR"

cleanup() { ray stop --force 2>/dev/null || true; }
trap cleanup EXIT

# 检查 SFT ckpt 存在
if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: SFT ckpt not found: $MODEL" >&2
  echo "请先跑 SFT: bash train_scripts/local/run_sft_8shot_cot_1ep.sh" >&2
  exit 1
fi

ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"

ACTOR_LR="${ACTOR_LR:-5e-7}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"

# v8 reward 权重 (4 子项: format / answer / numeric / step_count)
# answer 0.85 (大幅提, 主线信号, 跟 v7 0.70 区别)
# numeric 0.05 (大幅砍, v7 抢 answer 梯度的元凶)
# step_count 0.05 (新, 鼓励展开算式)
# format 0.05 (降半, base 0.5B 自由推理下基本 ok)
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.05}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-0.85}"
NUMERIC_CORRECTNESS_WEIGHT="${NUMERIC_CORRECTNESS_WEIGHT:-0.05}"
STEP_COUNT_WEIGHT="${STEP_COUNT_WEIGHT:-0.05}"
INVALID_REWARD="${INVALID_REWARD:--0.5}"

EVAL_ENABLED="${EVAL_ENABLED:-1}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${ROOT}/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.3}"
EVAL_BASELINE="${EVAL_BASELINE:-0}"

export CUDA_MODULE_LOADING=LAZY
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
export FLASHINFER_CUDA_ARCH_LIST="12.0f"
export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
export VERL_FILE_LOGGER_PATH="$METRICS_DIR/train_metrics.jsonl"

[[ -f "$DATA" ]] || { echo "ERROR: data not found: $DATA" >&2; exit 1; }
[[ -f "$REWARD_PATH" ]] || { echo "ERROR: reward module not found: $REWARD_PATH" >&2; exit 1; }

echo "=== v8 GRPO 训练配置 ==="
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "REWARD=$REWARD_PATH"
echo "MAX_STEPS=$MAX_STEPS SAVE_FREQ=$SAVE_FREQ"
echo "REWARD_WEIGHTS: format=$FORMAT_WEIGHT answer=$ANSWER_WEIGHT numeric=$NUMERIC_CORRECTNESS_WEIGHT step_count=$STEP_COUNT_WEIGHT"
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
  data.val_files="['$DATA']" \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$TRAIN_BATCH_SIZE \
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
  +custom_reward_function.reward_kwargs.numeric_correctness_weight=$NUMERIC_CORRECTNESS_WEIGHT \
  +custom_reward_function.reward_kwargs.step_count_weight=$STEP_COUNT_WEIGHT \
  +custom_reward_function.reward_kwargs.invalid_reward=$INVALID_REWARD \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","file"]' \
  trainer.project_name=chaingsm_lbprm_v8 \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$MAX_STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable

echo ""
echo "=== Training done, evaluating checkpoints ==="

# 评测用 code/eval_chaingsm_base_8shot.py (跟训练 prompt 完全一致)
for STEP_DIR in $(ls -d $CKPT_DIR/global_step_* 2>/dev/null | sort); do
  STEP_NAME=$(basename $STEP_DIR)
  HF_MODEL_DIR="$STEP_DIR/actor/huggingface"
  if [[ ! -d "$HF_MODEL_DIR" ]]; then
    echo "WARN: no huggingface dir at $HF_MODEL_DIR, skipping $STEP_NAME"
    continue
  fi
  STEP_NUM=${STEP_NAME#global_step_}
  EVAL_STEP_DIR="$EVAL_DIR/step_${STEP_NUM}"
  if [[ -f "$EVAL_STEP_DIR/summary_overall.json" ]]; then
    echo "=== $STEP_NAME already evaluated, skipping ==="
    continue
  fi
  echo ""
  echo "=== Evaluating $STEP_NAME (8-shot CoT) ==="
  mkdir -p "$EVAL_STEP_DIR"
  if cd "$ROOT" && "$PYTHON" code/eval_chaingsm_base_8shot.py \
    --data-path "$EVAL_DATA_PATH" \
    --run-dir "$EVAL_STEP_DIR" \
    --output-root "$EVAL_DIR" \
    --limit 6000 \
    --batch-size $EVAL_BATCH_SIZE \
    --gpu-memory-utilization $EVAL_GPU_MEM_UTIL \
    --model "Qwen2.5-0.5B-Instruct=$HF_MODEL_DIR" > "$LOG_DIR/eval_step_${STEP_NUM}.log" 2>&1; then
    if [[ -f "$EVAL_STEP_DIR/summary_overall.json" ]]; then
      OVERALL_ACC=$(cat "$EVAL_STEP_DIR/summary_overall.json" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)[0]['accuracy'])" 2>/dev/null || echo "?")
      ORIG_ACC=$(cat "$EVAL_STEP_DIR/summary_by_category.json" | "$PYTHON" -c "import json,sys; rows=json.load(sys.stdin); orig=[r for r in rows if r['category']=='original']; print(orig[0]['accuracy'] if orig else 0)" 2>/dev/null || echo "?")
      echo "  eval done: overall=$OVERALL_ACC  original=$ORIG_ACC"
      echo "  summary: $EVAL_STEP_DIR/summary_overall.json"
    else
      echo "  WARN: eval finished without summary_overall.json, see $LOG_DIR/eval_step_${STEP_NUM}.log"
    fi
  else
    echo "  WARN: eval failed for $STEP_NAME, see $LOG_DIR/eval_step_${STEP_NUM}.log"
  fi
done

echo ""
echo "=== All done. Output: $OUTPUT_DIR ==="
