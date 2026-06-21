# ChainGSM V12-V14 远程提交器总览

> **作者**: 训练组
> **日期**: 2026-06-20
> **覆盖版本**: V12 / V12i / V13 / V14
> **配套**: 4 个 submit_grpo_v{12,12i,13,14}_verl.sh + 4 个 grpo_verl_v{12,12i,13,14}_vllm.yaml
> **本文档**: 跟 4 个提交器一起放 train_scripts/remote/, 合作者一眼看到全部

---

## 0. 快速回答: 4 个版本一览

| 提交器 | JOB_NAME | 数据 | Reward | 训练量 | 评测 method |
|---|---|---|---|---|---|
| `submit_grpo_v12_verl.sh` | chaingsm-grpo-verl-v12 | grpo_v12_json.parquet | reward_chaingsm_v12_json_verl.py (4 项软 core) | 2 epoch | cot_brackets_v12_json |
| `submit_grpo_v12i_verl.sh` | chaingsm-grpo-verl-v12i | grpo_v12_json.parquet | reward_chaingsm_v12i_json_verl.py (4 项 LCS) | 2 epoch | cot_brackets_v12_json |
| `submit_grpo_v13_verl.sh` | chaingsm-grpo-verl-v13 | grpo_v13_json.parquet | reward_chaingsm_v13_json_verl.py (5 项硬匹配) | 2 epoch | cot_brackets_v13_json |
| `submit_grpo_v14_verl.sh` | chaingsm-grpo-verl-v14 | grpo_v14_reasoning.parquet | reward_chaingsm_v14_reasoning_verl.py (4 项重做) | 3 epoch | cot_brackets_v14_reasoning |

**关键**: **V12 和 V12i 共用同一份训练数据** `grpo_v12_json.parquet` (V12 prompt 已内嵌, V12i 沿用 V12 prompt, 只换 reward); V13 / V14 各自独立数据.

---

## 1. 一键提交 (以 V12 为例)

```bash
# 0) 上传仓库
rsync -avz --exclude='outputs/' --exclude='.git/' \
    ./math-chain/ user@remote:~/math-chain/
cd ~/math-chain

# 1) 干跑自检 (不真提交)
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12_verl.sh

#    期望看到 (关键检查):
#      REMOTE_REWARD_PATH=.../reward_chaingsm_v12_json_verl.py   ← V12
#      REMOTE_DATA_DIR=.../chaingsm_data/data/final/grpo          ← V12
#      --config-name grpo_verl_v12_vllm                          ← V12 config
#      trainer.total_epochs=2                                     ← V12 训练量
#      JOB_NAME=chaingsm-grpo-verl-v12                           ← V12 任务名
#      [v12] DRY_RUN=1, skip symlink/copy check

# 2) 跳过预检, 正式提交
SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12_verl.sh

# 3) 完整流程 (带预检)
bash train_scripts/remote/submit_grpo_v12_verl.sh
```

**其他版本替换提交器名即可** (`v12` -> `v12i` / `v13` / `v14`).

---

## 2. 4 个版本的设计差异

### 2.1 Prompt 字段差异

| 版本 | 字段数 | 字段 | 评测 method |
|---|---|---|---|
| V12 | 6 | target / use_facts / exclude_facts / steps[{explanation,expression,value}] / final_expression / answer | cot_brackets_v12_json |
| V12i | 6 | 同 V12 (V12i 沿用 V12 prompt) | cot_brackets_v12_json |
| V13 | 5 | target / use_facts / exclude_facts / steps[] / final_expression (steps 元素不同) | cot_brackets_v13_json |
| V14 | 4 | target / use_facts / exclude_facts / reasoning[] (prose/表达式物理交错) | cot_brackets_v14_reasoning |

### 2.2 Reward 公式差异

| 版本 | 公式 | 关键差异 |
|---|---|---|
| V12 | 3.0·r_answer + 1.5·r_step_value + 0.5·r_core (软 trace+final sim) - 0.5·r_distractor | 软 core 鼓励"看起来像" |
| V12i | 3.0·r_answer + 1.5·r_step_value_lcs + 0.5·r_step_value_exact - 0.5·r_distractor_per_step | LCS + 0.5% 容差, 治浮点失败, per-step distractor |
| V13 | 3.0·r_final + 1.5·r_step_val + 0.5·r_step_fmt + 0.5·r_format + 0.5·r_exclude | 5 项, 硬匹配为主, 砍 core 软相似度 |
| V14 | 3.0·r_answer + 1.5·r_step_value + 0.5·r_path_alignment - 0.5·r_distractor_leak | 4 项, 跟 V12 公式同构, 信号源重做 (reasoning[]) |

### 2.3 训练量差异

| 版本 | 训练量 | ckpts | save_freq |
|---|---|---|---|
| V12 | 2 epoch × 762 = 1524 steps | 5 (300/600/900/1200/1500) | 300 |
| V12i | 2 epoch × 762 = 1524 steps | 5 (300/600/900/1200/1500) | 300 |
| V13 | 2 epoch × 762 = 1524 steps | 5 (300/600/900/1200/1500) | 300 |
| V14 | 3 epoch × 762 = 2286 steps | 7 (300/600/900/1200/1500/1800/2100) | 300 |

---

## 3. 数据准备: One-Stop Pipeline (推荐) / symlink (legacy)

### 3.1 现状 (2026-06-21)

合作者**不再需要手动 push 数据 parquet**. 仓库自带 3 个 build 脚本, 在 SLURM 任务里**自动**从 raw + supplementary jsonl 重新生成当前版本的 parquet, 然后立即用同一份 parquet 启动训练. 改 prompt 后重提一次任务即可, 不用动任何训练入口.

资源配 (与 `data_preprocess_parsed_v6a_ALL.sh` 案例一致): 1 GPU / 8 CPU / 32GB / `gpu-all` partition, 日志写到 `./all_logs/%j-%x-slurm.{out,err}`.

### 3.2 3 个 build 脚本 (改 prompt 就改这里)

| 版本 | build 脚本 (顶部 `SYSTEM` / `USER_TEMPLATE` / `V14_*` 常量就是 prompt) | 输出的 parquet |
|---|---|---|
| V12 / V12i | `chaingsm_data/data/final/sft/build_grpo_v12_json.py` | `chaingsm_data/data/final/grpo/grpo_v12_json.parquet` |
| V13 | `chaingsm_data/data/final/sft/build_grpo_v13_json.py` | `chaingsm_data/data/final/grpo/grpo_v13_json.parquet` |
| V14 | `chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py` | `chaingsm_data/data/final/grpo/grpo_v14_reasoning.parquet` |

### 3.3 3 个 pipeline 提交器 (preprocess + 训练 一条 sbatch)

```bash
# 默认 V12i (V12 prompt + V12i 4 项 LCS reward)
sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# 想跑 V12 (V12 prompt + V12 7 项 reward)
VERSION=v12 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# V13 / V14
sbatch train_scripts/remote/submit_grpo_v13_pipeline.sh
sbatch train_scripts/remote/submit_grpo_v14_pipeline.sh
```

每个 pipeline 内部:
1. `bash data_preprocess_vXX.sh` 跑对应 build 脚本生成 `grpo_vXX_*.parquet` (1 GPU, 几秒到几分钟)
2. 同一任务内立即 `python -m verl.trainer.main_ppo --config-name grpo_verl_vXX_vllm ...` 启动训练 (1 GPU, 接力)

环境变量:

| 变量 | 默认 | 用途 |
|---|---|---|
| `DRY_RUN=1` | 0 | 只打印要执行的命令, 不真跑 (本地自检用) |
| `SKIP_PREPROCESS=1` | 0 | 跳过 build, 复用已存在的 parquet |
| `VERSION=v12\|v12i` | `v12i` | 仅 V12 pipeline 有效 |
| `V12_TOTAL_EPOCHS` / `V12I_TOTAL_EPOCHS` | 2 / 2 | 覆盖 V12 训练轮数 |
| `V13_TOTAL_EPOCHS` | 2 | 覆盖 V13 训练轮数 |
| `V14_TOTAL_EPOCHS` | 3 | 覆盖 V14 训练轮数 |
| `REMOTE_ROOT` | `/home/wwq416/snap/wwq/math-chain` | 仓库根 |
| `REMOTE_MODEL_PATH` | `/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct` | 起点模型 |

### 3.4 Legacy: 手动 push 数据 + symlink (不推荐, 保留兼容)

如果合作者**已经**手动维护了一份 parquet (例如 V12 长期稳定版), 可以跳过预处理:

```bash
SKIP_PREPROCESS=1 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
```

老提交器 `submit_grpo_v{12,12i,13,14}_verl.sh` 也保留可用 (它们走 `submit_grpo_verl_vllm.sh`, 需要 parquet 已经在 `chaingsm_data/data/final/grpo/` 下; 主脚本自己放 `ln -sf grpo_vXX_*.parquet verl_grpo_train.parquet`).

> 注: V12 和 V12i 共用 `grpo_v12_json.parquet`; V13 / V14 各自独立 parquet. 跑哪个版本就预处理哪个.

---

## 4. 自定义变量 (4 个版本通用)

```bash
# 改模型路径
REMOTE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
    bash submit_grpo_v12_verl.sh

# 改 GPU 数 / rollout 采样数
SLURM_GPUS=2 ROLLOUT_N=4 TP_SIZE=1 \
    bash submit_grpo_v12_verl.sh

# 改训练量
GRPO_EPOCHS=3 \
    bash submit_grpo_v12_verl.sh

# 改 job 名
JOB_NAME=my-v12-experiment \
    bash submit_grpo_v12_verl.sh
```

---

## 5. 4 个提交器 vs 老 submit_grpo_verl_vllm.sh 对比

| 项 | submit_grpo_verl_vllm.sh (老 V10) | submit_grpo_v12/12i/13/14_verl.sh (新) |
|---|---|---|
| config | grpo_verl_vllm.yaml | **grpo_verl_v{12,12i,13,14}_vllm.yaml** |
| reward | reward_chaingsm.py (V10 风格, 5 项) | **reward_chaingsm_v{12,12i,13,14}_*.py (4-5 项)** |
| data | rl_preprocessed/gsm8k_train_..._14946 | **grpo_v{12_json,13_json,14_reasoning}.parquet** |
| epochs | 1 | **2 (V12/V12i/V13) / 3 (V14)** |
| save_freq | 50 | **300** |
| symlink | 手动 | **自动** |

老脚本**保留**在包里 (`submit_grpo_verl_vllm.sh`), 跑 V10 老实验仍可用.

---

## 6. 4 个版本演进故事 (给合作者看历史)

```
V10 (5 项老 reward, soft sim 为主)
  ↓ numeric 子项抢梯度, 砍掉
V11 (5 项, 修 data schema)
  ↓ best step_200 0.4003 (-3.26pp), 失败
V12 (4 项, soft core)
  ↓ 0.5B 容量下抗干扰 vs 无干扰零和
V12i (4 项, LCS + per-step distractor, V12 reward 改良)
  ↓ V12 prompt 沿用, 改 reward signal
V13 (5 项, 硬 per-step 数值匹配, 砍 core 软相似度)
  ↓ 砍 r_answer 治 R11 违反 4.3% 失分
V14 (4 项, 跟 V12 同构, 信号源用 reasoning[])
  ↓ 砍 7 字段到 4 字段, prose/表达式物理交错
```

详细设计 / 失败 / 成功 报告都在 `docs/superpowers/` 下, 关键看:

- `docs/superpowers/specs/2026-06-17-grpo-v10-signed-design.md` (V10 设计源头)
- `docs/superpowers/reports/2026-06-18-grpo-v8-v11-systematic-report.md` (V8-V11 系统报告)
- `docs/superpowers/reports/2026-06-18-grpo-v11-final-report.md` (V11 最终)
- `docs/superpowers/reports/2026-06-18-paper-innovation-points.md` (创新点归纳)
- `docs/analysis/v12_failure_analysis.md` (V12 失败分析, V12i 改进依据)

---

## 7. 评测 (4 个版本, 各自 method)

```bash
# 评测入口统一用 train_pipeline.eval_vllm_chaingsm
# 通过 --method 参数切换:

# V12 / V12i
python -m train_pipeline.eval_vllm_chaingsm \
    --method cot_brackets_v12_json \
    ...

# V13
python -m train_pipeline.eval_vllm_chaingsm \
    --method cot_brackets_v13_json \
    ...

# V14
python -m train_pipeline.eval_vllm_chaingsm \
    --method cot_brackets_v14_reasoning \
    ...
```

评测 jsonl: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (已 push).
所有 4 个 method 都用同一份评测 jsonl (评测时按 method 拼不同 prompt).

---

## 8. 远程预检 (4 个版本通用)

```bash
bash train_scripts/remote/preflight_remote.sh grpo
```

检查 CUDA module / conda env / SLURM partition / 数据路径 / Python imports. `SKIP_PREFLIGHT=1` 可跳过.

---

## 9. 依赖清单 (4 个版本通用)

- **Python 3.12** (`math_chain_verl` env)
- **torch** + **CUDA 12.x**
- **transformers** (Qwen2.5-0.5B-Instruct)
- **vllm** (rollout, 与 verl 兼容)
- **verl** (`verl_math_chain` patch 版)
- **ray**
- **Levenshtein** (`pip install python-Levenshtein`) — V12 / V12i 必需
- **datasets**, **pyarrow** (parquet 读写)

---

## 10. 文件清单

```
train_scripts/remote/
├── remote_env.sh                    <-- 通用远程环境加载器 (所有 helper)
├── preflight_remote.sh              <-- 远程预检
├── submit_grpo_verl_vllm.sh         <-- 老 V10 提交器 (保留)
├── submit_grpo_v12_verl.sh          <-- ★ V12 提交器 (本次新增)
├── submit_grpo_v12i_verl.sh         <-- V12i 提交器 (上次新增)
├── submit_grpo_v13_verl.sh          <-- ★ V13 提交器 (本次新增)
├── submit_grpo_v14_verl.sh          <-- ★ V14 提交器 (本次新增)
├── V12i_HANDBOOK.md                 <-- V12i 单独说明 (上次新增)
├── V12_V14_HANDBOOK.md              <-- ★ V12-V14 总览 (本次新增)
├── submit_sft_verl.sh
├── submit_sft_then_grpo_verl_vllm.sh
└── submit_dpo_trl.sh

train_configs/remote/
├── grpo_verl_vllm.yaml              <-- 老 V10 远程 config
├── grpo_verl_v12_vllm.yaml          <-- ★ V12 远程 config (本次新增)
├── grpo_verl_v12i_vllm.yaml         <-- V12i 远程 config (上次新增)
├── grpo_verl_v13_vllm.yaml          <-- ★ V13 远程 config (本次新增)
├── grpo_verl_v14_vllm.yaml          <-- ★ V14 远程 config (本次新增)
├── sft_verl.yaml
├── sft_then_grpo_verl_vllm.yaml
└── dpo_trl.yaml
```

---

## 11. 联系方式与版本

- **本资源包 commit 基础**: master 分支, 截至 2026-06-20
- **V12 版本**: 4 项 reward (soft core), baseline 0.4003
- **V12i 版本**: 4 项 reward (LCS + 精确匹配 + per-step distractor), 沿用 V12 prompt
- **V13 版本**: 5 项 reward (硬匹配), 沿用 V13 prompt
- **V14 版本**: 4 项 reward (跟 V12 同构, 信号源 reasoning[]), V14 prompt

合作者跑任何一个版本, **DRY_RUN 自检通过** 就能 `bash` 正式提交. 任何问题先看对应版本的 spec/report (`docs/superpowers/` 下).
