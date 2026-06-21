# ChainGSM Math-Chain Training

ChainGSM Math-Chain 是一个面向小学数学推理鲁棒性的训练与评测项目。项目围绕带干扰链的 GSM8K 风格数据，构建统一的数据预处理、SFT、DPO、GRPO 强化学习训练、vLLM 评测与实验输出监控流程。

当前版本的本地训练统一使用 `math_chain_verl` 环境。SFT / DPO / TRL-GRPO 使用 TRL 训练入口；推荐的本地 GRPO 加速路径使用 verl + vLLM rollout。远程服务器支持 SLURM + verl / TRL 提交流程。

## 当前状态总览

当前项目主线已经收敛到：

- 本地环境：`/home/wwq416/miniconda3/envs/math_chain_verl/bin/python`
- 本地训练：TRL SFT、TRL DPO、TRL GRPO、verl GRPO
- 本地评测：`train_pipeline.eval_vllm_chaingsm`
- 远程启动层：`train_scripts/remote/remote_env.sh`
- 远程只读预检：`train_scripts/remote/preflight_remote.sh`
- 当前远程状态：由于暂时不能登录远程服务器，真实 preflight、SLURM 提交、远程环境 import、checkpoint/log 验证仍等待远程验证

本地已经可以验证的内容：

- 所有本地和远程 shell 脚本可以通过 `bash -n` 语法检查。
- 四条远程链路可以通过 `DRY_RUN=1 SKIP_PREFLIGHT=1` 在本地检查命令构造。
- 当前推荐文档不再把 `math-noise` 作为运行环境。
- `__pycache__/` 和 `*.pyc` 这类 Python 字节码缓存应删除并保持不进入版本管理。

暂时不要删除：

- `Noise_math_data-main/`：远程四条链路验证完成前，它仍是远程服务器配置和 verl 用法的参考来源。
- `outputs/`、`code/results/`：这些目录可能包含训练和评测历史。
- `chaingsm_data/data/final/rl_preprocessed/`：当前训练链路依赖这些预处理产物。

当前中文维护记录：

```text
docs/superpowers/plans/2026-06-02-local-only-maintenance.md
```


## 最近一次训练结果 (v8.2 收尾, 2026-06-14)

主线 `original >= 0.46` 目标在 v3 → v7 → v8.2 三次迭代中均未达成：

- **baseline** (0.5B base + 8-shot CoT 原生): original 0.4329 (571/1319)
- **v7 best** (step_100, 3 子项 reward): original 0.4428 (+0.99pp) — 失败，numeric 抢梯度
- **v8.2 best** (step_200, 5 机制 reward + 修数据 schema): original 0.4003 (-3.26pp) — 失败
- **v8.2 best by overall** (step_800): overall 0.3170 (+7.32pp) — 4 类变体齐涨 +10-14pp
- **核心 trade-off 发现**: 0.5B 容量下抗干扰能力跟无干扰推理能力是零和博弈

完整结果：`docs/superpowers/reports/2026-06-14-lbprm-v8-2-report.md`
评测摘要：`outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/grpo_verl_v82_base/20260614_170033/eval/latest_metrics.json`

下一次迭代 v9 候选方向（详见 v8.2 报告 §7）：

- 候选 A: 修 target_recognition number-only bug + 加 8-shot 模板相似度子项
- 候选 B: SFT + 8-shot 协议对齐 reward
- 候选 C: 承认 0.5B 上限，转 1.5B
- 候选 D: 集成 v7 + v8.2 step_200 + SFT ckpt


## 目录

- [当前状态总览](#当前状态总览)
- [项目目标](#项目目标)
- [代码结构](#代码结构)
- [运行环境](#运行环境)
- [数据与预处理](#数据与预处理)
- [本地训练流程](#本地训练流程)
- [verl GRPO 推荐流程](#verl-grpo-推荐流程)
- [评测与指标](#评测与指标)
- [输出目录与监控](#输出目录与监控)
- [本地可验证流程](#本地可验证流程)
- [远程服务器运行](#远程服务器运行)
- [维护与清理边界](#维护与清理边界)
- [常用调试命令](#常用调试命令)
- [关键文件说明](#关键文件说明)
- [排障指南](#排障指南)
- [推荐工作流](#推荐工作流)

## 项目目标

本项目的核心目标是训练模型在存在干扰计算链时仍然选择正确推理链，并输出结构化 JSON 解答。

模型训练目标：

- 识别题目真正要求的目标量。
- 忽略与目标无关或竞争的 distractor chain。
- 选择正确计算步骤。
- 生成符合 schema 的 JSON。
- 输出可解析的最终答案、计算表达式与推理步骤。

默认训练模型：

```text
/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
```

默认项目根目录：

```text
/home/wwq416/snap/wwq/math-chain
```

## 代码结构

```text
math-chain/
  README.md
  train_readme.md
  baseline_readme.md

  chaingsm_data/
    data/
      final/
        train_balanced_one_variant/
        rl_preprocessed/
        gsm8k_test_full/
    README.md

  train_pipeline/
    preprocess_chaingsm.py       # 训练数据预处理
    train_sft_trl.py             # TRL SFT 训练入口
    train_dpo_trl.py             # TRL DPO 训练入口
    train_grpo_trl.py            # TRL GRPO 训练入口
    reward_chaingsm.py           # TRL / remote verl 奖励函数
    reward_chaingsm_verl.py      # 本地 verl 奖励函数
    eval_vllm_chaingsm.py        # vLLM 评测入口
    eval_callback.py             # TRL epoch-end vLLM 评测回调
    eval_constants.py            # 轻量评测常量
    config_utils.py              # YAML 与输出目录工具

  train_configs/
    local/
      sft.yaml
      dpo.yaml
      grpo.yaml
      grpo_verl.yaml
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
      run_grpo_verl.sh
      run_sft_then_grpo.sh
    remote/
      remote_env.sh
      preflight_remote.sh
      submit_sft_verl.sh
      submit_dpo_trl.sh
      submit_grpo_verl_vllm.sh          <-- 4 行平铺，编辑文件切换 v12/v12i/v13/v14
      submit_sft_then_grpo_verl_vllm.sh

  tests/
    test_local_environment_docs.py
    test_remote_submission_scripts.py

  code/
    eval_chaingsm.py
    eval_abstral_baselines.py
    results/

  outputs/
    train/
      local/
      remote/
```

## 运行环境

### 本地推荐环境

本地默认 Python：

```text
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
```

激活方式：

```bash
conda activate math_chain_verl
```

该环境用于：

- 数据预处理
- TRL SFT
- TRL DPO
- TRL GRPO
- verl GRPO
- vLLM 训练中评测

关键组件：

```text
PyTorch 2.11.0+cu130
vLLM 0.21.0
verl 0.8.0.dev0
TRL 1.5.1
Ray
datasets
transformers
pyarrow
```

本地 verl 源码路径：

```text
/home/wwq416/snap/wwq/verl_math_chain
```

本地 Blackwell / RTX 5090 相关环境变量由 `train_scripts/local/run_grpo_verl.sh` 自动设置：

```bash
CUDA_MODULE_LOADING=LAZY
CUDA_HOME=/home/wwq416/miniconda3/envs/math_chain_verl
FLASHINFER_CUDA_ARCH_LIST=12.0f
LD_LIBRARY_PATH=/home/wwq416/miniconda3/envs/math_chain_verl/lib:$LD_LIBRARY_PATH
```

## 数据与预处理

### 默认输入数据

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl
```

### 默认预处理输出目录

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/
```

### 运行预处理

```bash
conda activate math_chain_verl

bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

小样本检查：

```bash
MAX_SAMPLES=32 \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

跳过 Parquet 输出：

```bash
SKIP_PARQUET=1 \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

### 预处理产物

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

数据用途：

| 文件 | 用途 |
|---|---|
| `sft_train.jsonl` | TRL SFT |
| `dpo_train.jsonl` | TRL DPO |
| `grpo_train.jsonl` | TRL GRPO |
| `sft_then_grpo_stage1_sft.jsonl` | 两阶段训练第 1 阶段 SFT |
| `sft_then_grpo_stage2_grpo.jsonl` | 两阶段训练第 2 阶段 TRL GRPO |
| `verl_grpo_train.parquet` | verl GRPO |
| `verl_sft_train.parquet` | verl SFT / remote SFT |

预处理会解析安全算术表达式并构造 gold trace / distractor trace。无法解析的样本会写入 `trace_build_errors.jsonl`，默认不进入训练。

## 本地训练流程

### 通用环境变量

所有本地脚本都支持以下常见覆盖：

```bash
ROOT=/home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
MODEL=/path/to/model
DATA=/path/to/train_data
OUTPUT_DIR=/path/to/output_base_or_run
RUN_NAME=my_run_name
RUN_ID=20260529_custom
MAX_SAMPLES=128
MAX_STEPS=10
```

也可以给 Python 入口传递 YAML 点路径覆盖：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.train_sft_trl \
  --config /home/wwq416/snap/wwq/math-chain/train_configs/local/sft.yaml \
  --set training.learning_rate=1e-5 \
  --set training.num_train_epochs=2
```

### SFT

默认配置：

```text
train_configs/local/sft.yaml
```

运行：

```bash
conda activate math_chain_verl

RUN_NAME=sft_train_json \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 RUN_NAME=sft_smoke \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

默认输出：

```text
outputs/train/local/sft/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
```

### DPO

默认配置：

```text
train_configs/local/dpo.yaml
```

运行：

```bash
conda activate math_chain_verl

RUN_NAME=dpo_train_json \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 RUN_NAME=dpo_smoke \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_dpo.sh
```

默认输出：

```text
outputs/train/local/dpo/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
```

### TRL GRPO

默认配置：

```text
train_configs/local/grpo.yaml
```

运行：

```bash
conda activate math_chain_verl

RUN_NAME=grpo_trl_train_json \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=8 MAX_STEPS=1 RUN_NAME=grpo_trl_smoke \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo.sh
```

默认输出：

```text
outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
```

### SFT -> TRL GRPO

该脚本先运行 SFT，再把第二阶段 GRPO 的初始模型指向第一阶段输出的 `checkpoints/current`。

运行：

```bash
conda activate math_chain_verl

RUN_NAME=sft_then_grpo_train_json \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

分阶段控制步数：

```bash
SFT_MAX_STEPS=20 \
GRPO_MAX_STEPS=20 \
RUN_NAME=sft_then_grpo_smoke \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft_then_grpo.sh
```

默认输出：

```text
outputs/train/local/sft_then_grpo/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
  stage1_sft/
  stage2_grpo/
```

## verl GRPO 推荐流程

本地推荐使用 `train_scripts/local/run_grpo_verl.sh` 进行 GRPO 强化学习训练。该入口使用 verl 的 FSDP + vLLM rollout，并在每个 epoch 后调用项目的 vLLM 评测脚本。

### 默认输入

默认初始模型：

```text
/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_20epoch_full_eval/20260528_193544/checkpoints/best
```

默认训练数据：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet
```

默认奖励函数：

```text
/home/wwq416/snap/wwq/math-chain/train_pipeline/reward_chaingsm_verl.py
```

### 稳定默认 profile

`run_grpo_verl.sh` 当前默认使用 `PROFILE=stable`，适合本地 RTX 5090 32GB 稳定训练与监控：

```text
TRAIN_BATCH_SIZE=4
ROLLOUT_N=4
ROLLOUT_GPU_MEM_UTIL=0.3
MAX_RESPONSE_LENGTH=1024
LOG_PROB_MICRO_BATCH_SIZE=1
REF_LOG_PROB_MICRO_BATCH_SIZE=1
EVAL_GPU_MEM_UTIL=0.3
```

运行：

```bash
conda activate math_chain_verl

TOTAL_EPOCHS=8 \
RUN_NAME=grpo_verl_from_sft_8epoch \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

### paper profile

如需更接近论文式参数，可以使用：

```bash
PROFILE=paper \
TOTAL_EPOCHS=8 \
RUN_NAME=grpo_verl_from_sft_8epoch_paper \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

`PROFILE=paper` 默认：

```text
TRAIN_BATCH_SIZE=8
ROLLOUT_N=8
ROLLOUT_GPU_MEM_UTIL=0.4
MAX_RESPONSE_LENGTH=2048
LOG_PROB_MICRO_BATCH_SIZE=4
```

如果本地显存不足，优先降低：

```bash
ROLLOUT_GPU_MEM_UTIL=0.3
ROLLOUT_N=4
TRAIN_BATCH_SIZE=4
MAX_RESPONSE_LENGTH=1024
```

### 快速 smoke test

用于验证数据、reward、verl、checkpoint、日志能否连通：

```bash
conda activate math_chain_verl

EVAL_BASELINE=0 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=20 \
TRAIN_BATCH_SIZE=4 \
ROLLOUT_N=2 \
ROLLOUT_GPU_MEM_UTIL=0.3 \
MAX_RESPONSE_LENGTH=1024 \
RUN_NAME=grpo_verl_smoke \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

### 常用覆盖项

```bash
MODEL=/path/to/sft/checkpoint/best
DATA=/path/to/verl_grpo_train.parquet
REWARD_PATH=/path/to/reward_chaingsm_verl.py
OUTPUT_DIR=/path/to/output/run
RUN_NAME=my_verl_run
RUN_ID=manual_id
TOTAL_EPOCHS=8
SAVE_FREQ=100
EVAL_BASELINE=0
EVAL_ENABLED=1
EVAL_BATCH_SIZE=32
```

Hydra 参数也可追加在脚本后：

```bash
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  +custom_reward_function.reward_kwargs.answer_weight=2.5
```

## 评测与指标

### 默认测试集

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
```

### vLLM 评测入口

```bash
conda activate math_chain_verl

/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.eval_vllm_chaingsm \
  --model-path /path/to/model_or_checkpoint \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl \
  --output-dir /path/to/eval_output \
  --method train_json_prompt \
  --batch-size 64 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --gpu-memory-utilization-candidate 0.3 \
  --gpu-memory-utilization-candidate 0.25 \
  --max-tokens 2048 \
  --top-k 1 \
  --dtype auto
```

评测方法：

| method | 说明 |
|---|---|
| `train_json_prompt` | 使用训练时 JSON schema prompt，默认推荐 |
| `direct` | 直接问答 |
| `zero_shot_cot` | zero-shot chain-of-thought 风格 prompt |

评测输出：

```text
predictions.jsonl
summary_overall.jsonl
summary_by_category.jsonl
eval_result.json
eval_errors.json          # 仅失败时出现
```

字段说明：

| 文件 | 关键字段 |
|---|---|
| `predictions.jsonl` | `id`, `base_id`, `category`, `question`, `gold_answer`, `raw_output`, `pred_answer`, `correct` |
| `summary_overall.jsonl` | `correct`, `total`, `accuracy` |
| `summary_by_category.jsonl` | `category`, `correct`, `total`, `accuracy` |
| `eval_result.json` | `overall_accuracy`, `by_category`, `prediction_count`, `gpu_memory_utilization` |

### 训练中评测

TRL SFT / DPO / GRPO 入口通过 `EpochVLLMEvalCallback` 进行：

1. 训练前 baseline 评测。
2. 每个 epoch 结束保存 `checkpoints/current`。
3. 子进程启动 vLLM 评测。
4. 如果准确率更高，复制为 `checkpoints/best`。
5. 将评测摘要写入 `eval/epoch_summary.jsonl` 和 `eval/latest_metrics.json`。

verl GRPO 入口通过 shell 脚本管理：

1. 可选 baseline 评测。
2. 按 epoch 运行 verl。
3. 每到 checkpoint 保存 HF-format actor。
4. 调用 `eval_vllm_chaingsm.py` 评测 checkpoint。
5. 按 `overall_accuracy` 维护 `checkpoints/best`。

## 输出目录与监控

### TRL 训练输出结构

SFT / DPO / TRL-GRPO 输出结构：

```text
outputs/train/local/<method>/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
  checkpoints/
    current/
    best/
  configs/
    resolved_config.yaml
  logs/
    <method>_stdout.log
    <method>_stderr.log
  metrics/
    train_result.json
  eval/
    baseline/
      predictions.jsonl
      summary_overall.jsonl
      summary_by_category.jsonl
      eval_result.json
    epoch_0001/
    epoch_0002/
    epoch_summary.jsonl
    latest_metrics.json
  generated_samples/
```

### verl GRPO 输出结构

```text
outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
  checkpoints/
    global_step_*/
      actor/
        huggingface/
    best/
  configs/
    run_env.txt
  logs/
    grpo_verl_stdout.log
    grpo_verl_stderr.log
  metrics/
    train_metrics.jsonl
    latest_metrics.json
  eval/
    baseline/
      predictions.jsonl
      summary_overall.jsonl
      summary_by_category.jsonl
      eval_result.json
    epoch_0001/
    epoch_0002/
    epoch_summary.jsonl
    latest_metrics.json
```

### 监控训练日志

查看主日志：

```bash
tail -f /path/to/run/logs/grpo_verl_stdout.log
```

查看错误日志：

```bash
tail -f /path/to/run/logs/grpo_verl_stderr.log
```

查看 verl 每步 metrics：

```bash
tail -f /path/to/run/metrics/train_metrics.jsonl
```

查看评测摘要：

```bash
tail -f /path/to/run/eval/epoch_summary.jsonl
```

查看最新 best：

```bash
cat /path/to/run/eval/latest_metrics.json
```

### verl 每步关键指标

verl 控制台和 `metrics/train_metrics.jsonl` 会记录每一步训练指标。关键字段包括：

```text
training/global_step
training/epoch
critic/score/mean
critic/score/max
critic/score/min
critic/rewards/mean
critic/rewards/max
critic/rewards/min
actor/...
response_length/mean
response_length/clip_ratio
prompt_length/mean
throughput/...
timing_s/...
```

项目自定义 reward 组件会额外记录：

```text
reward/accuracy
reward_components/accuracy/mean
reward_components/format/mean
reward_components/answer/mean
reward_components/expression/mean
reward_components/trace_overlap/mean
reward_components/distractor_overlap/mean
```

其中：

- `reward/accuracy` 是当前训练 batch 中 answer 正确比例，属于训练 reward 视角的 step-level 近似准确率。
- `eval/*/overall_accuracy` 是在 held-out 测试集上的 vLLM 评测准确率。
- 两者用途不同：前者用于观察 RL reward 学习动态，后者用于模型质量选择与对比。

## 本地可验证流程

当前不能登录远程服务器时，先完成以下本地检查。这些检查只能证明脚本语法和命令构造正确，不能替代远程环境验证。

### Shell 脚本语法检查

```bash
bash -n train_scripts/remote/remote_env.sh
bash -n train_scripts/remote/preflight_remote.sh
bash -n train_scripts/remote/submit_sft_verl.sh
bash -n train_scripts/remote/submit_dpo_trl.sh
bash -n train_scripts/remote/submit_grpo_verl_vllm.sh
bash -n train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
bash -n train_scripts/local/run_preprocess.sh
bash -n train_scripts/local/run_sft.sh
bash -n train_scripts/local/run_dpo.sh
bash -n train_scripts/local/run_grpo.sh
bash -n train_scripts/local/run_grpo_verl.sh
bash -n train_scripts/local/run_sft_then_grpo.sh
```

### 远程提交命令 dry-run

```bash
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

dry-run 输出应包含：

```text
[remote] Resolved command:
[remote] Submission command:
--partition=gpu-A100
--gres=gpu:4
--cpus-per-task=128
--mem=256GB
--account=A100
--qos=a100_qos
```

四条链路还应分别包含：

| 链路 | 关键入口 |
|---|---|
| verl SFT | `verl.trainer.fsdp_sft_trainer` |
| TRL DPO | `train_pipeline.train_dpo_trl` |
| verl GRPO | `verl.trainer.main_ppo` |
| verl SFT -> verl GRPO | 同时包含 `fsdp_sft_trainer` 和 `main_ppo` |

### 本地测试断言

当前环境可能没有安装 `pytest`。如果已安装，可运行：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest \
  tests/test_remote_submission_scripts.py \
  tests/test_local_environment_docs.py \
  -q
```

如果没有 `pytest`，可以直接执行测试函数：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python - <<'PY'
import importlib.util
from pathlib import Path

for path in [Path("tests/test_remote_submission_scripts.py"), Path("tests/test_local_environment_docs.py")]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_"):
            getattr(module, name)()
            print(f"PASS {path}:{name}")
PY
```

### 缓存清理检查

```bash
find . \
  -path './.git' -prune -o \
  -path './.worktrees' -prune -o \
  -path './Noise_math_data-main' -prune -o \
  \( -type d -name '__pycache__' -o -type f -name '*.pyc' \) \
  -print
```

如果输出非空，可以删除这些缓存文件。不要在这个阶段删除数据、训练输出、评测结果或 `Noise_math_data-main/`。

## 远程服务器运行

远程脚本位于：

```text
train_scripts/remote/
```

远程配置位于：

```text
train_configs/remote/
```

远程服务器细节见：

```text
docs/remote-server-settings.md
```

当前提交脚本共用：

```text
train_scripts/remote/remote_env.sh
```

只读预检脚本为：

```text
train_scripts/remote/preflight_remote.sh
```

在无法登录远程服务器时，只能用 dry-run 检查 SLURM 命令构造：

```bash
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

在远程服务器上提交前，可运行只读预检：

```bash
bash train_scripts/remote/preflight_remote.sh all
```

只有远程 preflight 通过后，才提交真实训练任务。运行前通常需要在远程机器设置：

```bash
export REMOTE_ROOT=/export/home/asifali/math-chain
export REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct
export REMOTE_DATA_DIR=${REMOTE_ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946
export REMOTE_OUTPUT_DIR=${REMOTE_ROOT}/outputs/train/remote
export REMOTE_EVAL_DATA_PATH=${REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
export REMOTE_REWARD_PATH=${REMOTE_ROOT}/train_pipeline/reward_chaingsm.py
```

远程脚本默认通过 SLURM `sbatch --wrap` 提交，并使用参考 A100 配置：

```text
partition: gpu-A100
gres: gpu:4
cpus-per-task: 128
mem: 256GB
account: A100
qos: a100_qos
conda env: Reasoning360
CUDA module: cuda12.4/toolkit
```

这些值来自 `Noise_math_data-main/` 参考项目。远程验证完成前不要删除该目录。

### 远程验证恢复后的顺序

远程登录恢复后，按顺序执行：

```bash
bash train_scripts/remote/preflight_remote.sh all
```

preflight 通过后，先跑小规模 smoke job，再考虑完整训练。smoke job 需要确认：

- SLURM 日志能写入 `${REMOTE_OUTPUT_DIR}/slurm_logs/`
- 模型路径、数据路径、reward 路径存在
- `Reasoning360` 能导入 `torch`、`transformers`、`datasets`、`vllm`、`verl`
- SFT / DPO / GRPO / SFT->GRPO 四条链路能启动并生成预期输出

### 远程 verl GRPO

```bash
cd "$REMOTE_ROOT"

TOTAL_EPOCHS=8 \
ROLLOUT_N=8 \
TP_SIZE=2 \
JOB_NAME=chaingsm-grpo-verl-vllm \
bash train_scripts/remote/submit_grpo_verl_vllm.sh
```

远程 GRPO 配置：

```text
train_configs/remote/grpo_verl_vllm.yaml
```

远程 SLURM 日志：

```text
${REMOTE_OUTPUT_DIR}/slurm_logs/
```

### 远程 SFT -> GRPO

```bash
cd "$REMOTE_ROOT"

TOTAL_EPOCHS=1 \
GRPO_EPOCHS=8 \
ROLLOUT_N=8 \
TP_SIZE=2 \
JOB_NAME=chaingsm-sft-then-grpo-verl \
bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

## 维护与清理边界

当前仓库里既有源码，也有数据、评测结果、训练输出和远程参考项目。清理时必须区分“确定无用缓存”和“可能承载实验历史或远程验证价值的内容”。

### 可以直接清理

| 内容 | 原因 |
|---|---|
| `__pycache__/` | Python 自动生成缓存，无维护价值 |
| `*.pyc` | Python 字节码缓存，无维护价值 |
| `.pytest_cache/` | 本地测试缓存 |
| 临时 dry-run 输出 | 可重新生成 |

清理命令：

```bash
find . \
  -path './.git' -prune -o \
  -path './.worktrees' -prune -o \
  -path './Noise_math_data-main' -prune -o \
  \( -type d -name '__pycache__' -o -type f -name '*.pyc' \) \
  -print -exec rm -rf {} +
```

### 暂时保留

| 内容 | 当前处理 | 原因 |
|---|---|---|
| `Noise_math_data-main/` | 保留 | 远程 A100/SLURM 链路验证前仍是参考来源 |
| `outputs/` | 保留 | 可能包含训练历史、checkpoint、metrics |
| `code/results/` | 保留 | 可能包含历史评测结果和对比数据 |
| `chaingsm_data/data/final/rl_preprocessed/` | 保留 | 当前训练链路依赖的预处理产物 |
| `plan_1.md` | 暂保留 | 需要确认是否仍有历史规划价值 |

### 删除 `Noise_math_data-main/` 的前置条件

只有同时满足以下条件后，才考虑删除 `Noise_math_data-main/`：

1. 远程服务器可以登录。
2. `bash train_scripts/remote/preflight_remote.sh all` 在远程通过。
3. 四条远程链路 smoke job 均能提交并生成日志。
4. verl SFT、TRL DPO、verl GRPO、SFT->GRPO 的输出目录和 checkpoint 正常。
5. `docs/remote-server-settings.md` 已记录所有远程差异和最终配置。

删除前建议先单独提交远程验证结果，再做清理提交。

## 常用调试命令

### 检查数据文件是否存在

```bash
ls -lh /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/
```

### 查看 Parquet 样本数

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python - <<'PY'
import pyarrow.parquet as pq
path = "/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet"
print(pq.read_metadata(path).num_rows)
PY
```

### 单独评测一个 checkpoint

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.eval_vllm_chaingsm \
  --model-path /path/to/checkpoints/best \
  --output-dir /tmp/chaingsm_eval \
  --method train_json_prompt \
  --gpu-memory-utilization 0.3 \
  --gpu-memory-utilization-candidate 0.3 \
  --gpu-memory-utilization-candidate 0.25 \
  --batch-size 32
```

### 查看某次 verl 训练的最新状态

```bash
RUN_DIR=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>

tail -n 50 "$RUN_DIR/logs/grpo_verl_stdout.log"
tail -n 20 "$RUN_DIR/eval/epoch_summary.jsonl"
cat "$RUN_DIR/eval/latest_metrics.json"
```

### 查看每步 reward 与 accuracy

```bash
RUN_DIR=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>

tail -n 5 "$RUN_DIR/metrics/train_metrics.jsonl"
```

### 清理残留 Ray

```bash
ray stop --force
```

`run_grpo_verl.sh` 已经内置 `trap cleanup EXIT`，正常失败或中断时会自动尝试清理 Ray。

## 关键文件说明

### `train_pipeline/preprocess_chaingsm.py`

负责将 ChainGSM 原始 JSONL 转为训练数据：

- SFT prompt / completion
- DPO chosen / rejected
- GRPO prompt / reward reference
- verl Parquet 数据
- gold trace / distractor trace

### `train_pipeline/reward_chaingsm.py`

通用奖励函数。可用于 TRL GRPO 和远程 verl。核心 reward 组件：

```text
format
answer
expression
trace_overlap
distractor_overlap
```

### `train_pipeline/reward_chaingsm_verl.py`

本地 verl 奖励函数入口。返回字段包括：

```text
score
accuracy
format
answer
expression
trace_overlap
distractor_overlap
```

这些字段会被本地 patched verl trainer 汇总到每步日志中。

### `train_pipeline/eval_vllm_chaingsm.py`

vLLM 评测入口。特点：

- JSONL 实时写入 predictions。
- 支持 GPU memory utilization candidate 重试。
- 生成 `eval_result.json` 供脚本读取。
- 优先解析模型 JSON 输出中的 `answer` 字段。
- JSON 解析失败时回退到数字抽取。

### `train_pipeline/eval_callback.py`

TRL 训练回调。每个 epoch 后：

- 保存当前模型。
- 子进程运行 vLLM 评测，避免 vLLM 在父进程中残留显存。
- 更新 `checkpoints/best`。

### `train_scripts/local/run_grpo_verl.sh`

本地推荐 GRPO 入口。负责：

- 设置 Blackwell / vLLM / FlashInfer 环境变量。
- 建立标准输出目录。
- tee stdout / stderr 到日志文件。
- 写入运行环境快照。
- 调用 verl 训练。
- 记录每步 metrics。
- 按 epoch 评测并维护 best checkpoint。

## 排障指南

### 1. 没有看到旧的 `logs/` 或 `metrics/`

当前 `run_grpo_verl.sh` 会在新输出路径下生成：

```text
outputs/train/local/grpo_verl/...
```

不要在 `outputs/train/local/grpo/...` 下找 verl 训练结果。`grpo/` 是 TRL GRPO 的路径，`grpo_verl/` 是 verl GRPO 的路径。

### 2. baseline eval 有结果，但没有 epoch eval

通常说明训练尚未保存 checkpoint 或训练在第一个 checkpoint 前失败。

检查：

```bash
tail -n 100 /path/to/run/logs/grpo_verl_stderr.log
find /path/to/run/checkpoints -maxdepth 2 -type d
```

如果想更早得到 checkpoint：

```bash
SAVE_FREQ=20 EVAL_BASELINE=0 TOTAL_EPOCHS=1 RUN_NAME=grpo_verl_debug \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

### 3. vLLM rollout OOM

优先降低以下参数：

```bash
ROLLOUT_GPU_MEM_UTIL=0.25
ROLLOUT_N=2
TRAIN_BATCH_SIZE=2
MAX_RESPONSE_LENGTH=768
LOG_PROB_MICRO_BATCH_SIZE=1
REF_LOG_PROB_MICRO_BATCH_SIZE=1
```

示例：

```bash
ROLLOUT_GPU_MEM_UTIL=0.25 \
ROLLOUT_N=2 \
TRAIN_BATCH_SIZE=2 \
MAX_RESPONSE_LENGTH=768 \
RUN_NAME=grpo_verl_low_mem \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

### 4. vLLM 评测 OOM

降低评测显存与 batch：

```bash
EVAL_GPU_MEM_UTIL=0.25
EVAL_BATCH_SIZE=16
```

### 5. 训练 step 日志过多

verl console logger 会每步打印大量指标，这是当前监控设计的一部分。如果只想保留文件指标，可以在脚本中将：

```bash
trainer.logger='["console","file"]'
```

改为：

```bash
trainer.logger='["file"]'
```

### 6. 找不到 best checkpoint

检查：

```bash
cat /path/to/run/eval/latest_metrics.json
tail -n 20 /path/to/run/eval/epoch_summary.jsonl
```

如果 `total_epoch_checkpoints_evaluated` 为 0，说明还没有成功完成 epoch checkpoint 评测。可先用 `SAVE_FREQ=20` 做 smoke test。

## 推荐工作流

当前不能登录远程服务器时，先执行：

```bash
bash -n train_scripts/remote/remote_env.sh
bash -n train_scripts/remote/preflight_remote.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

远程登录恢复前，不删除 `Noise_math_data-main/`，不把 dry-run 当作远程训练可用的最终证明。

首次运行或数据变化后：

```bash
conda activate math_chain_verl
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_preprocess.sh
```

训练 SFT：

```bash
RUN_NAME=sft_train_json \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

使用 SFT best checkpoint 做 verl GRPO：

```bash
MODEL=/path/to/sft/checkpoints/best \
TOTAL_EPOCHS=8 \
RUN_NAME=grpo_verl_from_sft_8epoch \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

监控：

```bash
RUN_DIR=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_verl_from_sft_8epoch/<RUN_ID>

tail -f "$RUN_DIR/logs/grpo_verl_stdout.log"
tail -f "$RUN_DIR/metrics/train_metrics.jsonl"
tail -f "$RUN_DIR/eval/epoch_summary.jsonl"
```

最终模型：

```text
<RUN_DIR>/checkpoints/best/
```

最终评测摘要：

```text
<RUN_DIR>/eval/latest_metrics.json
```
