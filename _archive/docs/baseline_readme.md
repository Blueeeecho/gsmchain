# ChainGSM 四种训练流程 Baseline 说明

本文档用于统一记录当前项目中的四种训练 baseline：

1. SFT
2. DPO
3. GRPO
4. 先 SFT 再 GRPO

当前本地 baseline 默认使用 TRL 后端，远程 GRPO/SFT->GRPO 方向保留 verl + vLLM 配置。本文重点说明本地可直接运行的四条流程，以及它们共同使用的数据、提示词、评测、日志和模型保存策略。

## 1. 公共设置

### 默认环境

```text
Python 环境：
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python

默认模型：
/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct

训练数据根目录：
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/

测试数据：
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
```

### 输出目录规则

所有本地训练脚本都会强制使用时间戳子目录，避免重复运行时覆盖结果。

```text
outputs/train/local/<method>/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
```

其中：

```text
RUN_NAME = 实验名称
RUN_ID   = 默认启动时间戳，例如 20260528_190200
```

即使显式指定 `OUTPUT_DIR=/path/to/base_dir`，实际输出也会是：

```text
/path/to/base_dir/<RUN_ID>/
```

每个 run 目录包含：

```text
checkpoints/
  current/   # 当前训练模型
  best/      # 测试集 overall accuracy 最高模型
logs/
  *_stdout.log
  *_stderr.log
metrics/
  train_result.json
configs/
  resolved_config.yaml
eval/
  baseline/
    predictions.csv
    summary_overall.csv
    summary_by_category.csv
  epoch_0001/
    predictions.csv
    summary_overall.csv
    summary_by_category.csv
  epoch_summary.csv
  latest_metrics.json
```

### 训练前和每轮训练后评测

所有本地训练入口默认都开启：

```yaml
eval:
  enabled: true
  baseline_before_train: true
```

因此训练开始前会先运行一轮 baseline 测试。每个 epoch 结束后也会运行一轮测试。

评测使用 vLLM 加速，默认确定性解码：

```yaml
method: train_json_prompt
max_tokens: 2048
temperature: 0.0
top_k: 1
top_p: 1.0
batch_size: 64
tensor_parallel_size: 1
gpu_memory_utilization_candidates: [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
```

说明：vLLM 在 `temperature=0.0` 时会走 greedy 路径，内部打印的 `top_k` 可能被规范化成 `0`，但实际行为仍是确定性解码。

评测输出：

```text
summary_overall.csv       # 整体准确率
summary_by_category.csv   # original + 四类变体准确率
predictions.csv           # 每条样本输出、预测答案、正确性
```

测试集包含：

```text
original
independent_decoy
attribute_mismatch
path_competition
target_scope_misalignment
```

### 显存鲁棒性

评测前会读取当前 GPU 空闲显存，并跳过明显会失败的 `gpu_memory_utilization` 候选值。如果 vLLM 加载失败，会自动降档重试。所有候选值都失败时，该轮评测会被标记为 failed，训练不会直接中断。

### 公共提示词

训练和评测默认使用同一套 JSON 输出提示词。

System prompt：

```text
You are a careful mathematical reasoning assistant. Select only the computation chain that answers the question, ignore distractor chains, and return JSON only.
```

User prompt 模板：

```text
Solve the following grade-school math problem. Return exactly one JSON object with this schema:
{
  "target": "short target quantity",
  "selected_steps": [
    {"variable": "name", "description": "short explanation", "expression": "arithmetic expression", "value": "computed value"}
  ],
  "final_expression": "arithmetic expression",
  "answer": "final answer"
}

Problem:
{question}
```

模型应输出 JSON，例如：

```json
{
  "target": "earnings",
  "selected_steps": [
    {
      "variable": "minute_rate",
      "description": "Compute step 1 for the correct chain.",
      "expression": "12 / 60",
      "value": "0.2"
    },
    {
      "variable": "earnings",
      "description": "Compute step 2 for the correct chain.",
      "expression": "12 / 60 * 50",
      "value": "10"
    }
  ],
  "final_expression": "12/60*50",
  "answer": "10"
}
```

变量名会保留，但 reward 不做变量名语义 exact match；`variable` 主要用于格式完整性和中间变量意识。

## 2. SFT Baseline

### 目标

SFT 用于教模型稳定输出目标 JSON 格式，并学习从带干扰的题目中选择正确核心计算链。

### 数据

```text
sft_train.jsonl
```

每条样本主要字段：

```json
{
  "id": "sample_id",
  "prompt": "完整 user prompt",
  "response": "gold JSON response",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

训练入口会将 `prompt/response` 转为当前 TRL 需要的：

```json
{
  "prompt": "...",
  "completion": "..."
}
```

并使用 `completion_only_loss=true`，只在模型应输出的 JSON completion 上计算 loss。

### 默认训练设置

配置文件：

```text
train_configs/local/sft.yaml
```

关键参数：

```yaml
training:
  max_length: 3072
  completion_only_loss: true
  packing: false
  num_train_epochs: 1
  max_steps: -1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-5
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  logging_strategy: steps
  logging_first_step: true
  logging_steps: 10
  disable_tqdm: false
  save_strategy: "no"
  bf16: true
  gradient_checkpointing: true
```

`save_strategy: "no"` 是刻意设置的：Trainer 不自动堆积 checkpoint，由我们的回调只维护 `checkpoints/current` 和 `checkpoints/best`。

### 运行命令

20 epoch 完整训练：

```bash
RUN_NAME=sft_20epoch_full_eval \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh \
  --set training.num_train_epochs=20
```

小样本调试：

```bash
RUN_NAME=sft_debug \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh \
  --max-samples 8 \
  --max-steps 1 \
  --set eval.limit=20
```

### 适用场景

SFT 是最稳定的第一阶段 baseline。它主要解决：

```text
1. 模型是否会按 JSON schema 输出
2. 模型是否能显式列出 selected_steps
3. 模型是否能把最终答案放到 answer 字段
```

## 3. DPO Baseline

### 目标

DPO 用于偏好学习。它让模型更偏向正确核心链输出，远离干扰链输出。

### 数据

```text
dpo_train.jsonl
```

每条样本主要字段：

```json
{
  "id": "sample_id",
  "prompt": "完整 user prompt",
  "chosen": "gold JSON response",
  "rejected": "distractor JSON response"
}
```

其中：

```text
chosen   = 由 gold_trace 构造的正确 JSON
rejected = 由 distractor_trace 构造的干扰 JSON
```

即使 distractor 的数值偶尔与 gold 接近或相同，DPO 仍通过 trace 和 final_expression 学习偏好正确链路。

### 默认训练设置

配置文件：

```text
train_configs/local/dpo.yaml
```

关键参数：

```yaml
training:
  max_length: 3072
  max_prompt_length: 768
  max_completion_length: 2048
  beta: 0.1
  loss_type: sigmoid
  num_train_epochs: 1
  max_steps: -1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 5.0e-6
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  logging_steps: 10
  disable_tqdm: false
  save_strategy: "no"
  bf16: true
  gradient_checkpointing: true
```

### 运行命令

```bash
RUN_NAME=dpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh \
  --set training.num_train_epochs=3
```

小样本调试：

```bash
RUN_NAME=dpo_debug \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh \
  --max-samples 8 \
  --max-steps 1 \
  --set eval.limit=20
```

### 适用场景

DPO 适合回答：

```text
给定同一个题目，模型能否更偏好核心链而不是干扰链？
```

它不直接通过 reward 在线采样，而是使用预构造的 chosen/rejected 偏好对。

## 4. GRPO Baseline

### 目标

GRPO 用于在线强化学习。模型会对同一 prompt 生成多条候选输出，再通过规则 reward 评估并更新策略。

### 数据

```text
grpo_train.jsonl
```

每条样本主要字段：

```json
{
  "id": "sample_id",
  "prompt": "完整 user prompt",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reward_reference": {
    "gold_answer": "...",
    "gold_expression": "...",
    "gold_trace": [],
    "distractor_expression": "...",
    "distractor_trace": [],
    "category": "...",
    "difficulty_tags": {}
  }
}
```

### Reward 设置

Reward 实现在：

```text
train_pipeline/reward_chaingsm.py
```

默认权重：

```yaml
reward:
  format_weight: 0.2
  answer_weight: 0.4
  expression_weight: 0.2
  trace_weight: 0.2
  distractor_penalty: 0.3
  invalid_reward: -0.5
```

含义：

```text
format_weight       JSON 格式和字段完整性
answer_weight       answer 是否等于 gold answer
expression_weight   final_expression 是否与 gold expression 数值等价
trace_weight        selected_steps 是否覆盖 gold trace
distractor_penalty  命中 distractor trace/expression 时惩罚
invalid_reward      非 JSON 或不可解析输出的失败分
```

### 默认训练设置

配置文件：

```text
train_configs/local/grpo.yaml
```

关键参数：

```yaml
training:
  max_prompt_length: 768
  max_completion_length: 2048
  num_generations: 4
  temperature: 0.7
  top_p: 1.0
  num_train_epochs: 1
  max_steps: -1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-6
  logging_steps: 1
  disable_tqdm: false
  save_strategy: "no"
  bf16: true
  gradient_checkpointing: true
```

注意：这里的 `training.temperature=0.7` 是 GRPO 训练采样温度，用于产生多样候选；评测仍然使用 `temperature=0.0, top_k=1, top_p=1.0`。

### 环境注意

当前本地环境曾检测到：

```text
TRL 0.26.1
vLLM 0.13.0
```

TRL GRPO 对 vLLM 版本较敏感。脚本会先做环境检查，如不兼容会给出明确报错。远程更推荐使用 verl + vLLM 跑 GRPO。

### 运行命令

```bash
RUN_NAME=grpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh \
  --set training.num_train_epochs=1
```

如果只是检查数据和 reward，可先小样本：

```bash
RUN_NAME=grpo_debug \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh \
  --max-samples 8 \
  --max-steps 1 \
  --set eval.limit=20
```

如需临时跳过本地 GRPO 环境检查：

```bash
SKIP_GRPO_ENV_CHECK=1 bash train_scripts/local/run_grpo.sh
```

## 5. SFT -> GRPO Baseline

### 目标

这是当前最推荐的主流程：

```text
先用 SFT 学会稳定输出 JSON 和基础链路选择
再用 GRPO 根据 reward 强化“选对核心链、避开干扰链”
```

### 数据

第一阶段：

```text
sft_then_grpo_stage1_sft.jsonl
```

第二阶段：

```text
sft_then_grpo_stage2_grpo.jsonl
```

这两个文件默认分别复制自 `sft_train.jsonl` 和 `grpo_train.jsonl`，单独命名是为了后续可以对两阶段做不同筛选。

### 流程

脚本：

```text
train_scripts/local/run_sft_then_grpo.sh
```

执行顺序：

```text
1. 运行 SFT
2. 保存第一阶段模型到 stage1_sft/<RUN_ID>/checkpoints/current
3. 第二阶段 GRPO 使用该 SFT 模型作为初始模型
4. GRPO 每轮继续执行 baseline/eval/current/best 保存策略
```

### 运行命令

```bash
RUN_NAME=sft_then_grpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh \
  --set eval.limit=100
```

给两个阶段分别传参数：

```bash
RUN_NAME=sft_then_grpo_debug \
SFT_EXTRA_ARGS="--set training.num_train_epochs=1 --set eval.limit=20" \
GRPO_EXTRA_ARGS="--set training.num_train_epochs=1 --set eval.limit=20" \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

小样本整体调试：

```bash
RUN_NAME=sft_then_grpo_debug \
MAX_SAMPLES=8 \
SFT_MAX_STEPS=1 \
GRPO_MAX_STEPS=1 \
SFT_EXTRA_ARGS="--set eval.limit=20" \
GRPO_EXTRA_ARGS="--set eval.limit=20" \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

### 适用场景

SFT -> GRPO 适合做正式 baseline，因为它把两个问题拆开：

```text
SFT 解决格式和基础行为
GRPO 解决 reward 下的链路偏好和抗干扰能力
```

## 6. 数据预处理说明

预处理脚本：

```text
train_pipeline/preprocess_chaingsm.py
```

运行：

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

输入：

```text
chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl
```

输出：

```text
chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/
```

主要输出文件：

```text
source_augmented_with_traces.jsonl
sft_train.jsonl
dpo_train.jsonl
grpo_train.jsonl
sft_then_grpo_stage1_sft.jsonl
sft_then_grpo_stage2_grpo.jsonl
trace_build_errors.jsonl
preprocess_stats.json
verl_sft_train.parquet
verl_grpo_train.parquet
```

当前全量预处理统计：

```text
input_count: 7419
kept_count: 7055
error_count: 364
```

表达式无法安全解析的样本会写入 `trace_build_errors.jsonl`，默认不进入训练。

## 7. 四种流程对比

| 流程 | 数据 | 主要学习目标 | 是否使用 reward | 推荐用途 |
|---|---|---|---|---|
| SFT | prompt/completion | 学格式、学正确链路输出 | 否 | 最基础 baseline |
| DPO | prompt/chosen/rejected | 偏好正确链而非干扰链 | 否，使用偏好 loss | 偏好学习 baseline |
| GRPO | prompt + reward_reference | 在线采样并按规则奖励优化 | 是 | 强化学习 baseline |
| SFT -> GRPO | SFT 数据 + GRPO 数据 | 先学格式，再强化抗干扰 | 第二阶段使用 | 推荐正式流程 |

## 8. 常用命令汇总

SFT 20 epoch：

```bash
RUN_NAME=sft_20epoch_full_eval \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh \
  --set training.num_train_epochs=20
```

DPO：

```bash
RUN_NAME=dpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh \
  --set training.num_train_epochs=3
```

GRPO：

```bash
RUN_NAME=grpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh \
  --set training.num_train_epochs=1
```

SFT -> GRPO：

```bash
RUN_NAME=sft_then_grpo_baseline \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

快速调试任意流程时建议加：

```bash
--max-samples 8 --max-steps 1 --set eval.limit=20
```

