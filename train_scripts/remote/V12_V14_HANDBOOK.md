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
| `submit_grpo_v12_pipeline.sh` (V12) | chaingsm-grpo-v12-pipeline | grpo_v12_json.parquet | reward_chaingsm_v12_json_verl.py (4 项软 core) | 2 epoch | cot_brackets_v12_json |
| `submit_grpo_v12_pipeline.sh` (V12i, **默认**) | chaingsm-grpo-v12i-pipeline | grpo_v12_json.parquet | reward_chaingsm_v12i_json_verl.py (4 项 LCS) | 2 epoch | cot_brackets_v12_json |
| `submit_grpo_v13_pipeline.sh` | chaingsm-grpo-v13-pipeline | grpo_v13_json.parquet | reward_chaingsm_v13_json_verl.py (5 项硬匹配) | 2 epoch | cot_brackets_v13_json |
| `submit_grpo_v14_pipeline.sh` | chaingsm-grpo-v14-pipeline | grpo_v14_reasoning.parquet | reward_chaingsm_v14_reasoning_verl.py (4 项重做) | 3 epoch | cot_brackets_v14_reasoning |

**关键**: **V12 和 V12i 共用同一份训练数据** `grpo_v12_json.parquet` (V12 prompt 已内嵌, V12i 沿用 V12 prompt, 只换 reward); V13 / V14 各自独立数据.

---

## 1. 一键提交 (以 V12 为例)

```bash
# 0) 上传仓库
rsync -avz --exclude='outputs/' --exclude='.git/' \
    ./math-chain/ user@remote:~/math-chain/
cd ~/math-chain

# 1) 干跑自检 (不真提交, 打印 preprocess + train 两条命令)
DRY_RUN=1 SKIP_PREPROCESS=1 bash train_scripts/remote/submit_grpo_v12_pipeline.sh

#    期望看到 (关键检查):
#      preprocess: bash .../data_preprocess_v12.sh (生成 grpo_v12_json.parquet)
#      train: ... --config-name grpo_verl_v12i_vllm ... (默认 V12i, 2 epochs)

# 2) 正式提交 (一条 sbatch, 从 raw 到 ckpt)
sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# 3) 想跑 V12 而不是 V12i
VERSION=v12 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# 4) 其他版本
sbatch train_scripts/remote/submit_grpo_v13_pipeline.sh
sbatch train_scripts/remote/submit_grpo_v14_pipeline.sh

# 5) 复用已有 parquet, 跳过 preprocess
SKIP_PREPROCESS=1 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
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

合作者**不再需要手动 push 数据 parquet**. 仓库自带 1 个 unified 训练 jsonl + 3 个 build 脚本, 在 SLURM 任务里**自动**从该 jsonl 重新生成当前版本的 parquet, 然后立即用同一份 parquet 启动训练. 改 prompt 后重提一次任务即可, 不用动任何训练入口.

**训练数据规模** (×2 训练, 2026-06-21): unified jsonl 7070 行 (变体 4019 + 原题 3051, 每条 base_id 出现 2 次, 跟评测集 5467=3051+3051 对称) → V12/V13/V14 build 脚本内部 gold_trace 校验后保留 **6102 行** (变体 3051 + 原题 3051). 5 类别分布: original 3051 / attribute_mismatch 910 / independent_decoy 837 / path_competition 814 / target_scope_misalignment 490. 968 个 supp-only 因为 gold_trace 不是 dict 列表被 build 脚本过滤掉 (这是历史代码的过滤, 跟改 prompt 无关).  **原题行** category='original', distractor_chain=None, distractor_trace=None — 模型在原题上学习 '正常解题', 在变体上学习 '排除分心'.

资源配 (与 `data_preprocess_parsed_v6a_ALL.sh` 案例一致): 1 GPU / 8 CPU / 32GB / `gpu-all` partition, 日志写到 `./all_logs/%j-%x-slurm.{out,err}`.

### 3.2 3 个 build 脚本 (改 prompt 就改这里)

**Prompt 嵌在每个 build 脚本顶部的三引号常量里** (跟案例 `our_pre_process_zebrapuzzle_to_guru_parsed_v6a_LTLT.py` 的 `SOLUTION_PROMPT_1_SHOT_SYS` 同模式). 改完保存, 不需要查任何外部 md.

| 版本 | build 脚本 (顶部 `SYSTEM` / `USER_TEMPLATE` / `V14_*` 常量就是 prompt) | 输出的 parquet |
|---|---|---|
| V12 / V12i | `chaingsm_data/data/final/sft/build_grpo_v12_json.py` (`SYSTEM` L24, `USER_TEMPLATE` L75) | `chaingsm_data/data/final/grpo/grpo_v12_json.parquet` |
| V13 | `chaingsm_data/data/final/sft/build_grpo_v13_json.py` (`SYSTEM` L36, `USER_TEMPLATE` L72) | `chaingsm_data/data/final/grpo/grpo_v13_json.parquet` |
| V14 | `chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py` (`V14_SYSTEM_PROMPT` L32, `V14_USER_TEMPLATE` L77) | `chaingsm_data/data/final/grpo/grpo_v14_reasoning.parquet` |

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

### 3.4 复用已有 parquet

合作者如果**已经**手动维护了一份 parquet (例如 V12 长期稳定版), 可以跳过预处理:

```bash
SKIP_PREPROCESS=1 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
```

> 注: V12 和 V12i 共用 `grpo_v12_json.parquet`; V13 / V14 各自独立 parquet. 跑哪个版本就预处理哪个.

### 3.5 已删除的 legacy 提交器 (2026-06-21)

- `submit_grpo_verl_vllm.sh` (4 选 1 注释版, 实际不接 `--version`)
- `submit_grpo_v{12,12i,13,14}_verl.sh` (4 个 wrapper, 全部 `exec` 到上面那个老接口, 实际不切档)

合作者跑 V12-V14 **只走 pipeline** (`submit_grpo_v{12,13,14}_pipeline.sh`).

### 3.6 评测 prompt 自动同步 (2026-06-21)

build 脚本每次跑会同步把 `SYSTEM` 写到 `${REMOTE_DATA_DIR}/_system_prompt.txt` (跟 parquet 同目录, 随 prompt 改动自动更新).

`train_pipeline/eval_vllm_chaingsm.py` 加了 `method=parquet_prompt` 模式, 评测时直接读 `_system_prompt.txt` 当 system prompt —— **跟训练完全一致**, 避免 train-test prompt mismatch.

`train_scripts/local/eval_v12_5ckpts.sh` 默认就是这个新模式. USER 段按 parquet 目录名 (v12/v13/v14) 自动选对应模板 + 答案抽取器. 不需要任何额外参数.

旧 method (历史报告对比用) 仍可指定: `METHOD=cot_brackets_v12_json bash train_scripts/local/eval_v12_5ckpts.sh`.

---

## 4. 自定义变量 (4 个版本通用)

```bash
# 改模型路径
REMOTE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
    sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# 改训练量 (V12i 默认 2, V14 默认 3)
V12I_TOTAL_EPOCHS=3 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
V14_TOTAL_EPOCHS=2 sbatch train_scripts/remote/submit_grpo_v14_pipeline.sh
```

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
├── submit_grpo_v{12,13,14}_pipeline.sh         <-- 老 V10 提交器 (保留)
├── submit_grpo_v12_pipeline.sh          <-- ★ V12 提交器 (本次新增)
├── submit_grpo_v12_pipeline.sh         <-- V12i 提交器 (上次新增)
├── submit_grpo_v13_pipeline.sh          <-- ★ V13 提交器 (本次新增)
├── submit_grpo_v14_pipeline.sh          <-- ★ V14 提交器 (本次新增)
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
