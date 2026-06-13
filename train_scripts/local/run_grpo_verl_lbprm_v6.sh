#!/usr/bin/env bash
# LB-PRM v6 GRPO 训练入口
#
# 关联 spec: docs/superpowers/specs/2026-06-14-lbprm-v6-design.md
# 关联 plan: docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md
#
# 关键设计 (相对 v5 的改动):
#   - 起点: sft_2epoch/best (27.42% / 25.55% Original, JSON schema 协议)
#   - 训练 prompt: 8-shot CoT 完整模板 (同 EIGHT_SHOT_EXAMPLES, 跟评测完全一致)
#   - 训练数据: verl_grpo_train_8shot_cot.parquet (新建, 8-shot CoT prompt 拼接)
#   - REWARD: reward_chaingsm_lbprm_v6_verl.py (新建, 8-shot CoT 协议适配)
#             - format 0.15 (收尾 "The final answer is N.")
#             - answer 0.60 (N 数值匹配)
#             - reasoning_quality 0.25 (0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction)
#   - MAX_PROMPT_LENGTH: 768 → 1280 (8-shot prompt 较长, 验证 max=1029)
#   - MAX_RESPONSE_LENGTH: 1024 → 512 (CoT 推理通常 < 300 token)
#   - MAX_STEPS: 500 → 400 (用户约束: 每类奖励函数 400 步)
#   - SAVE_FREQ: 100 → 80 (5 个 eval 节点: step_80/160/240/320/400)
#   - KL 系数: 跟 v5 一致 0.02
#   - 起点改回 sft_2epoch/best (用户指定)
#
# 终止条件:
#   1) 达到 400 step 自动停
#   2) 200 step 评估后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
#   3) step 80 eval < 0.5B + 8-shot CoT 原生 (43.29% Original) → 立即停, 反思 prompt 模板或 reward
#   4) step 80 eval < v3 best (30.47% overall) → 立即停
#
# 输出:
#   outputs/train/local/grpo_verl_lbprm_v6/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
#
# 用法:
#   RUN_NAME=grpo_verl_v6 \
#   bash train_scripts/local/run_grpo_verl_lbprm_v6.sh 2>&1 | tee /tmp/v6_run.log

set -uo pipefail  # 不开 -e, 因为 baseline eval 失败时希望继续

ROOT="${ROOT:-/home/wwq416/snap/wwq/math-chain}"
PYTHON="${PYTHON:-/home/wwq416/miniconda3/envs/math_chain_verl/bin/python}"
VERL_HOME="${VERL_HOME:-/home/wwq416/snap/wwq/verl_math_chain}"

# v6 起点: sft_2epoch/best (用户指定, 27.42% / 25.55% Original)
MODEL="${MODEL:-${ROOT}/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best}"
# v6 训练集: 8-shot CoT prompt 拼接 (新建)
DATA="${DATA:-${ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_8shot_cot.parquet}"
# v6 reward: 8-shot CoT 协议适配 (新建)
REWARD_PATH="${REWARD_PATH:-${ROOT}/train_pipeline/reward_chaingsm_lbprm_v6_verl.py}"

MAX_STEPS="${MAX_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-80}"

RUN_NAME="${RUN_NAME:-grpo_verl_v6}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_verl_lbprm_v6/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
LOG_DIR="$OUTPUT_DIR/logs"
METRICS_DIR="$OUTPUT_DIR/metrics"
CKPT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/eval"
mkdir -p "$LOG_DIR" "$METRICS_DIR" "$CKPT_DIR" "$EVAL_DIR"

cleanup() { ray stop --force 2>/dev/null || true; }
trap cleanup EXIT

ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
# v6: 8-shot prompt 较长, max=1029, 留余量到 1280
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"
# v6: CoT 推理通常 < 300 token, 留余量到 512
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-1}"

ACTOR_LR="${ACTOR_LR:-5e-7}"
# v6: 跟 v5 一致 0.02 (已验证稳定)
KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"

# v6 reward 权重
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.15}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-0.60}"
REASONING_QUALITY_WEIGHT="${REASONING_QUALITY_WEIGHT:-0.25}"
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
[[ -d "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -f "$REWARD_PATH" ]] || { echo "ERROR: reward module not found: $REWARD_PATH" >&2; exit 1; }

echo "=========================================="
echo "LB-PRM v6 GRPO 训练 (8-shot CoT 协议)"
echo "=========================================="
echo "  RUN_NAME:       $RUN_NAME"
echo "  RUN_ID:         $RUN_ID"
echo "  OUTPUT_DIR:     $OUTPUT_DIR"
echo "  MODEL (起点):   $MODEL"
echo "                   (= sft_2epoch/best, 27.42% / 25.55% Original, JSON schema 协议)"
echo "  DATA:           $DATA (8-shot CoT prompt)"
echo "  REWARD:         $REWARD_PATH (v6 reward, 8-shot CoT 协议适配)"
echo "  MAX_STEPS:      $MAX_STEPS"
echo "  SAVE_FREQ:      $SAVE_FREQ"
echo "  ROLLOUT_N:      $ROLLOUT_N"
echo "  TRAIN_BATCH:    $TRAIN_BATCH_SIZE"
echo "  MAX_PROMPT:     $MAX_PROMPT_LENGTH (8-shot 验证 max=1029)"
echo "  MAX_RESPONSE:   $MAX_RESPONSE_LENGTH (CoT 通常 < 300)"
echo "  FORMAT_WEIGHT:  $FORMAT_WEIGHT"
echo "  ANSWER_WEIGHT:  $ANSWER_WEIGHT"
echo "  REASON_QUALITY: $REASONING_QUALITY_WEIGHT"
echo "  KL_COEF:        $KL_LOSS_COEF"
echo "  ACTOR_LR:       $ACTOR_LR"
echo "=========================================="

# baseline eval 用 8-shot CoT 脚本 (跟训练 prompt 一致)
if [[ "$EVAL_BASELINE" == "1" ]]; then
  echo ""
  echo "=== Baseline eval (sft_2epoch/best with 8-shot CoT) ==="
  BASELINE_DIR="$EVAL_DIR/baseline"
  mkdir -p "$BASELINE_DIR"
  if cd "$ROOT" && PYTHONPATH="$ROOT" "$PYTHON" -m code.eval_chaingsm_base_8shot \
    --data-path "$EVAL_DATA_PATH" \
    --run-dir "$BASELINE_DIR" \
    --output-root "$EVAL_DIR" \
    --limit -1 \
    --batch-size $EVAL_BATCH_SIZE \
    --gpu-memory-utilization $EVAL_GPU_MEM_UTIL \
    --model "Qwen2.5-0.5B-Instruct@$MODEL" > "$LOG_DIR/baseline_eval.log" 2>&1; then
    if [[ -f "$BASELINE_DIR/summary_overall.json" ]]; then
      echo "  baseline eval done: $BASELINE_DIR/summary_overall.json"
    else
      echo "  WARN: baseline eval finished without summary_overall.json, see $LOG_DIR/baseline_eval.log"
    fi
  else
    echo "  WARN: baseline eval failed, see $LOG_DIR/baseline_eval.log"
  fi
fi

echo ""
echo "=== Starting verl GRPO training (max $MAX_STEPS steps) ==="

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
  +custom_reward_function.reward_kwargs.reasoning_quality_weight=$REASONING_QUALITY_WEIGHT \
  +custom_reward_function.reward_kwargs.invalid_reward=$INVALID_REWARD \
  \
  trainer.critic_warmup=0 \
  trainer.logger='["console","file"]' \
  trainer.project_name=chaingsm_lbprm_v6 \
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

# v6 评测用 code/eval_chaingsm_base_8shot.py (跟训练 prompt 完全一致)
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
  echo "=== Evaluating $STEP_NAME (8-shot CoT, code/eval_chaingsm_base_8shot.py) ==="
  mkdir -p "$EVAL_STEP_DIR"
  if cd "$ROOT" && PYTHONPATH="$ROOT" "$PYTHON" -m code.eval_chaingsm_base_8shot \
    --data-path "$EVAL_DATA_PATH" \
    --run-dir "$EVAL_STEP_DIR" \
    --output-root "$EVAL_DIR" \
    --limit -1 \
    --batch-size $EVAL_BATCH_SIZE \
    --gpu-memory-utilization $EVAL_GPU_MEM_UTIL \
    --model "Qwen2.5-0.5B-Instruct@$HF_MODEL_DIR" > "$LOG_DIR/eval_step_${STEP_NUM}.log" 2>&1; then
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
