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

## 3. 数据 symlink 自动处理

`grpo_verl_v{12,12i,13,14}_vllm.yaml` 都期望 parquet 文件名是 `verl_grpo_train.parquet`, 但实际数据文件叫 `grpo_v{12_json,13_json,14_reasoning}.parquet`.

4 个提交器**各自自动**在 `${REMOTE_DATA_DIR}/` 下放一个 symlink (`verl_grpo_train.parquet -> grpo_v{XX}.parquet`), 合作者无需手动操作.

**前提条件**: 对应数据文件必须**已经存在于** `${REMOTE_DATA_DIR}/` 下. 本资源包**已** push:

- ✅ `chaingsm_data/data/final/grpo/grpo_v12_json.parquet` (2.6 MB, V12 / V12i 共用)
- ❌ `chaingsm_data/data/final/grpo/grpo_v13_json.parquet` (2.1 MB, **没 push**)
- ❌ `chaingsm_data/data/final/grpo/grpo_v14_reasoning.parquet` (3.3 MB, **没 push**)

**V13 / V14 训练数据需要从作者单独拿** (3 个 parquet 共 5.4 MB, 跟 V12i 数据同一目录, 容易补). 拿下来放进 `chaingsm_data/data/final/grpo/` 即可.

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
