# ChainGSM 训练流程说明

本文档说明当前项目中的四类训练流程：

1. SFT
2. DPO
3. GRPO
4. 先 SFT 再 GRPO

本地用于快速调试，统一走 TRL；远程优先继承旧项目的 verl + vLLM + SLURM 训练链路。

## 1. 目录结构

新增训练相关目录如下：

```text
train_pipeline/
  preprocess_chaingsm.py     # ChainGSM 数据预处理
  train_sft_trl.py           # 本地 TRL SFT
  train_dpo_trl.py           # 本地/远程 TRL DPO
  train_grpo_trl.py          # 本地 TRL GRPO
  reward_chaingsm.py         # GRPO 规则奖励
  config_utils.py            # YAML 配置与命令行覆盖工具

train_configs/
  local/
    sft.yaml
    dpo.yaml
    grpo.yaml
    sft_then_grpo.yaml
  remote/
    sft_verl.yaml
    dpo_trl.yaml
    grpo_verl_vllm.yaml
    sft_then_grpo_verl_vllm.yaml

train_scripts/
  local/
    run_preprocess.sh
    run_sft.sh
    run_dpo.sh
    run_grpo.sh
    run_sft_then_grpo.sh
  remote/
    submit_sft_verl.sh
    submit_dpo_trl.sh
    submit_grpo_verl_vllm.sh
    submit_sft_then_grpo_verl_vllm.sh
```

## 2. 数据预处理

默认输入文件：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl
```

默认输出目录：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/
```

运行：

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

快速小样本检查：

```bash
MAX_SAMPLES=32 bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

生成文件：

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

预处理只解析安全算术表达式。无法解析的样本不会硬修，会写入 `trace_build_errors.jsonl`，默认不进入训练。

## 3. 本地训练

本地默认 Python：

```text
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
```

本地默认模型：

```text
/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
```

### SFT

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

### DPO

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh
```

### GRPO

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh
```

注意：当前本地环境中 `TRL 0.26.1` 与 `vLLM 0.13.0` 不兼容。TRL GRPO 当前支持的 vLLM 版本为：

```text
0.10.2, 0.11.0, 0.11.1, 0.11.2
```

如果版本不兼容，`run_grpo.sh` 会先报清晰错误，不会静默进入训练。确认环境后可运行：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh
```

### SFT -> GRPO

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

该脚本先训练 SFT，并把第二阶段 GRPO 的初始模型指向：

```text
stage1_sft/checkpoints/final
```

## 4. 命令行覆盖

脚本支持通过环境变量覆盖常用路径：

```bash
MODEL=/path/to/model \
DATA=/path/to/sft_train.jsonl \
OUTPUT_DIR=/path/to/output \
MAX_SAMPLES=128 \
MAX_STEPS=10 \
bash train_scripts/local/run_sft.sh
```

训练 Python 入口也支持 YAML 点路径覆盖：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.train_sft_trl \
  --config train_configs/local/sft.yaml \
  --set training.learning_rate=1e-5 \
  --set training.per_device_train_batch_size=2
```

## 5. 奖励设置

奖励函数位于：

```text
train_pipeline/reward_chaingsm.py
```

主要超参数在 `train_configs/local/grpo.yaml` 和 `train_configs/remote/grpo_verl_vllm.yaml` 中：

```yaml
reward:
  format_weight: 0.2
  answer_weight: 0.4
  expression_weight: 0.2
  trace_weight: 0.2
  distractor_penalty: 0.3
  invalid_reward: -0.5
```

奖励不对变量名做语义 exact match。`variable` 字段只作为 JSON 格式完整性的一部分，避免因为变量命名差异误伤模型。

## 6. 每个 Epoch 后的 vLLM 测试

本地 TRL 的 SFT、DPO、GRPO 入口都已经接入同一个 epoch-end 回调；SFT -> GRPO 会分别在两个阶段启用该机制。每个训练阶段会：

1. 训练前先运行一次 baseline 测试
2. 每个 epoch 后保存当前模型到 `checkpoints/current/`
3. 使用 vLLM 在测试集上跑一轮评测
4. 在终端和日志中输出整体准确率和各类别准确率
5. 如果当前整体准确率更高，复制为 `checkpoints/best/`
6. 不保留其他 epoch checkpoint，控制磁盘占用

默认测试集：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
```

默认评测提示词为 `train_json_prompt`，也就是和 SFT 训练相同的 JSON schema 提示。评分时优先读取模型输出 JSON 中的 `answer` 字段；如果 JSON 解析失败，再回退到普通数字抽取。

每轮结果按子目录保存：

```text
outputs/train/local/<method>/<model>/<run>/eval/
  epoch_summary.jsonl
  latest_metrics.json
  epoch_0001/
    predictions.jsonl
    summary_overall.jsonl
    summary_by_category.jsonl
  epoch_0002/
    predictions.jsonl
    summary_overall.jsonl
    summary_by_category.jsonl
```

`summary_overall.jsonl` 包含整体准确率；`summary_by_category.jsonl` 包含 `original` 和四类变体的准确率。

可在配置中调整测试参数：

```yaml
eval:
  enabled: true
  baseline_before_train: true
  method: train_json_prompt
  data_path: /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
  limit: null
  batch_size: 64
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  gpu_memory_utilization_candidates: [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
  max_tokens: 2048
  top_k: 1
  top_p: 1.0
```

当前默认最大输出为 2048。对应设置包括：评测 vLLM 的 `eval.max_tokens=2048`，DPO/GRPO 的 `training.max_completion_length=2048`，远程 verl GRPO 的 `data.max_response_length=2048`。SFT 的 `training.max_length` 是 prompt + completion 的总长度，因此默认设为 3072。

评测解码使用确定性设置：`temperature=0.0`、`top_k=1`、`top_p=1.0`。当前 vLLM 在 `temperature=0.0` 时会走 greedy 路径，内部打印的 `SamplingParams.top_k` 可能被规范化为 `0`，但实际行为仍是确定性解码。

如果服务器或本地 GPU 被其他任务占用，vLLM 加载失败时会自动按
`gpu_memory_utilization_candidates` 降档重试。评测启动前还会读取当前 GPU 空闲显存，自动跳过明显超过可用显存的候选值。所有候选值都失败时，本轮 eval 会写入
`eval_failed.json` 和 `epoch_summary.csv` 的失败记录，但不会直接中断训练。

SFT -> GRPO 如果要给两个阶段分别传额外参数，可以使用：

```bash
SFT_EXTRA_ARGS="--set eval.limit=100" \
GRPO_EXTRA_ARGS="--set eval.limit=100" \
bash train_scripts/local/run_sft_then_grpo.sh
```

如果只想手动评测某个 checkpoint：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.eval_vllm_chaingsm \
  --model-path /path/to/checkpoint \
  --output-dir /path/to/eval_dir
```

## 7. 远程训练

远程脚本只提供提交结构和格式检查，当前本地不实际提交远程任务。远程默认使用 SLURM，并尽量继承旧项目的 verl + vLLM 方式。

远程常用环境变量：

```bash
REMOTE_ROOT=/path/to/math-chain
REMOTE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
REMOTE_DATA_DIR=/path/to/rl_preprocessed/gsm8k_train_balanced_one_variant_14946
REMOTE_OUTPUT_DIR=/path/to/outputs/train/remote
PYTHON=python3
```

### 远程 SFT，verl

```bash
bash train_scripts/remote/submit_sft_verl.sh
```

### 远程 GRPO，verl + vLLM

```bash
bash train_scripts/remote/submit_grpo_verl_vllm.sh
```

可覆盖 rollout 设置：

```bash
ROLLOUT_N=8 TP_SIZE=2 bash train_scripts/remote/submit_grpo_verl_vllm.sh
```

### 远程 DPO，TRL

旧项目没有成熟的离线 DPO 主入口。当前远程 DPO 使用标准 TRL DPO：

```bash
bash train_scripts/remote/submit_dpo_trl.sh
```

这不是 SPIN/online DPO。如果后续要完全 verl 化离线 DPO，需要新增单独 recipe。

### 远程 SFT -> GRPO

```bash
bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

该脚本在同一个 SLURM job 中先运行 verl SFT，再把 GRPO 的 `actor_rollout_ref.model.path` 指向第一阶段输出。

## 8. 输出目录

本地默认输出：

```text
outputs/train/local/{sft,dpo,grpo,sft_then_grpo}/Qwen2.5-0.5B-Instruct/<run_name>/<run_id>/
```

`RUN_NAME` 用于标记实验语义，`RUN_ID` 默认是启动时间戳。即使复用同一个 `RUN_NAME`，每次训练也会写入新的时间戳子目录，避免覆盖旧结果。例如：

```bash
RUN_NAME=sft_20epoch_full_eval bash train_scripts/local/run_sft.sh
```

会输出到类似：

```text
outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_20epoch_full_eval/20260528_190102/
```

如果显式设置 `OUTPUT_DIR=/path/to/run_base`，脚本仍然会追加时间戳子目录：

```text
/path/to/run_base/<run_id>/
```

也就是说 `OUTPUT_DIR` 表示输出根目录，不表示最终运行目录。只有显式设置 `RUN_ID` 时，才会使用你指定的子目录名。

每个本地运行目录包含：

```text
checkpoints/
  current/
  best/
logs/
metrics/
configs/resolved_config.yaml
generated_samples/
eval/
```

远程默认输出：

```text
outputs/train/remote/
```

## 9. 当前全量预处理结果

当前默认数据已完成一次全量预处理：

```text
input_count: 7419
kept_count: 7055
error_count: 364
```

失败样本集中在表达式无法安全解析、等号链式表达式、文本式赋值、特殊操作符等情况。
