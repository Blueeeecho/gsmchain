# math-chain V12i 资源包交付说明 (v2: 含数据 + V12i 专用提交器)

> **作者**: 训练组
> **日期**: 2026-06-20
> **当前状态**: V12i GRPO 训练正在作者本机跑 (PID 3806358), 已跑约 7 小时
> **V12i 一句话**: V12 prompt (v4) + 改良 4 项 reward (LCS + 0.5% 容差 + 精确匹配额外加分 + per-step distractor 检查)
> **本资源包目标**: 让合作者在本机或远程 SLURM 集群跑 V12i GRPO 训练, 沿用 `submit_grpo_v12i_verl.sh` 一键提交, 无需手改 remote_env.sh

---

## 0. 快速回答: 合作者跑 V12i 需要做什么

只需 **3 步** (本机) 或 **3 步** (远程, 比 v1 还少 1 步):

### 本机 (Linux + 单卡/多卡 GPU)

```bash
tar -xzf math-chain-v12i.tar.gz && cd math-chain-v12i
ROOT=$(pwd)
sed -i "s|/home/wwq416/snap/wwq/math-chain|$ROOT|g" \
    train_configs/local/grpo_verl_v12i.yaml \
    train_scripts/local/run_grpo_verl_v12i.sh
bash train_scripts/local/run_grpo_verl_v12i.sh
```

### 远程 (SLURM 集群)

```bash
# 1) 上传仓库
rsync -avz ./math-chain-v12i/ user@remote:~/math-chain/
cd ~/math-chain

# 2) 干跑自检
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12i_verl.sh
#    期望: --export 行里 REMOTE_REWARD_PATH=...v12i, REMOTE_DATA_DIR=.../grpo

# 3) 正式提交
bash train_scripts/remote/submit_grpo_v12i_verl.sh
```

**`submit_grpo_v12i_verl.sh` 是 V12i 专用提交器**, 默认 reward / data / config / epoch 全是 V12i, **无需手改 remote_env.sh**. 详见 §5.

---

## 1. 目录结构 (解包后)

```
math-chain-v12i/
├── V12i_HANDBOOK.md                              <-- 你正在读
├── train_pipeline/
│   ├── reward_chaingsm_v12i_json_verl.py         <-- V12i 奖励函数 (主)
│   ├── eval_vllm_chaingsm.py                    <-- 评测入口 (含 cot_brackets_v12_json method)
│   ├── config_utils.py / eval_constants.py      <-- 评测/训练通用配置加载
│   ├── eval_sft_messages_chaingsm.py            <-- SFT 消息化评测
│   ├── eval_callback.py                         <-- 训练中回调评测
│   ├── preprocess_chaingsm.py                   <-- 主预处理 (生成 chain/distractor)
│   └── preprocess_chaingsm_8shot_cot.py         <-- 8-shot CoT 预处理
├── code/
│   └── gsm_answer_extractor.py                  <-- V12i 奖励 import 依赖
├── chaingsm_data/                                <-- ★ v2 新增: 训练 + 评测数据
│   ├── data/
│   │   ├── final/grpo/grpo_v12_json.parquet     <-- ★ V12i 训练 parquet (2.6 MB, 3051 行)
│   │   └── gsmchain/
│   │       ├── gsm8k_test_clean.jsonl           <-- ★ V12i 评测 jsonl (8.6 MB)
│   │       ├── cleaning_stats.json
│   │       └── README.md
├── train_configs/
│   ├── local/grpo_verl_v12i.yaml                <-- V12i 本地配置 (主)
│   └── remote/
│       ├── grpo_verl_vllm.yaml                  <-- 老 V10 远程 GRPO 配置
│       ├── grpo_verl_v12i_vllm.yaml             <-- ★ v2 新增: V12i 远程 GRPO 配置
│       ├── sft_verl.yaml
│       ├── sft_then_grpo_verl_vllm.yaml
│       └── dpo_trl.yaml
├── train_scripts/
│   ├── local/
│   │   ├── run_grpo_verl_v12i.sh                <-- V12i 本地启动 (主)
│   │   ├── eval_v12_5ckpts.sh                   <-- 评测 5 ckpts 样例
│   │   └── aggregate_v10_results.py             <-- 评测结果汇总
│   └── remote/                                  <-- 远程提交链
│       ├── remote_env.sh                        <-- 通用远程环境加载器 (所有 helper)
│       ├── preflight_remote.sh                  <-- 远程预检
│       ├── submit_grpo_verl_vllm.sh             <-- 老 V10 提交器 (保留, V12i 不走)
│       ├── submit_grpo_v12i_verl.sh             <-- ★ v2 新增: V12i 专用一键提交器
│       ├── submit_sft_verl.sh
│       ├── submit_sft_then_grpo_verl_vllm.sh
│       └── submit_dpo_trl.sh
├── docs/
│   ├── prompts/v12_long_prompt.md               <-- V12 prompt (V12i 沿用)
│   ├── analysis/v12_failure_analysis.md         <-- V12 失败分析 (V12i 改进依据)
│   └── superpowers/
│       ├── specs/2026-06-17-grpo-v10-signed-design.md         <-- V12i 演进源头
│       ├── specs/2026-06-17-sft-abstral-params-cot-protocol-design.md
│       ├── reports/2026-06-18-grpo-v11-final-report.md
│       ├── reports/2026-06-18-grpo-v8-v11-systematic-report.md
│       ├── reports/2026-06-18-error-case-deep-analysis.md
│       ├── reports/2026-06-18-paper-innovation-points.md
│       └── reports/2026-06-17-v9-2000-report.md
├── _ref/                                         <-- V12i 演进前序 (供对比, 不参与运行)
│   ├── reward_chaingsm.py
│   ├── reward_chaingsm_v10_verl.py
│   ├── v13_long_prompt.md
│   └── v14_long_prompt.md
└── tests/
    ├── test_gsm_answer_extractor.py
    ├── test_official_gsm_eval.py
    └── test_remote_submission_scripts.py
```

---

## 2. V12i 设计核心

**Prompt**: V12 (v4) 6 字段 JSON + 12 rules. V12i **沿用** V12 prompt, 不改. (实际 prompt 已内嵌在 `grpo_v12_json.parquet` 的 `prompt` 字段, 训练时直接读)

**Reward (4 项, 跟 V12 公式同构, 信号源重做)**:

```
R = 3.0 * r_answer
  + 1.5 * r_step_value_lcs          (NEW: LCS + 0.5% 容差, 治浮点失败)
  + 0.5 * r_step_value_exact         (NEW: 严格相等, 抓重复算)
  - 0.5 * r_distractor_per_step      (改良: 每行 + final 都跟 dist 比)
```

**vs V12 的差异**:
- V12 的 r_step_value 是软相似度 → V12i 改成 LCS 顺序匹配 + 0.5% 浮点容差
- V12 的 r_core 被砍 (鼓励"看起来像"反而让模型走偏) → 改成 r_step_value_exact
- V12 的 r_distractor 只看末态 → V12i 改成 per-step 检查

**起点**: Qwen2.5-0.5B-Instruct base (跳过 SFT)
**数据**: `grpo_v12_json.parquet` (3051 行, 5 类别, V12 同源) — **本资源包已包含**
**训练量**: 2 epoch × 762 steps = 1524 total steps, 每 300 步存 ckpt (5 个 ckpts)
**预期**: overall 22.9% → 26-30% (V12 baseline +3-7 pp)

详细演进源头: `docs/superpowers/specs/2026-06-17-grpo-v10-signed-design.md`.

---

## 3. 数据准备 (本资源包已包含, 无需自备)

**本资源包已经包含**:
- `chaingsm_data/data/final/grpo/grpo_v12_json.parquet` (2.6 MB, 3051 行) — **训练用**
- `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (8.6 MB) — **评测用**
- `chaingsm_data/data/gsmchain/cleaning_stats.json` + `README.md` — 评测数据元信息

合作者**解包即用**, 无需自己生成.

**如果合作者要自己重新生成** (验证数据管线, 可选):

```bash
python train_pipeline/preprocess_chaingsm_8shot_cot.py \
    --output-path chaingsm_data/data/final/grpo/grpo_v12_json.parquet
```

**输入数据要求** (自行生成时):
- GSM8K raw `train.jsonl` (OpenAI 公开, 7473 条)
- ChainGSM 干扰链构造逻辑 (在 `preprocess_chaingsm_8shot_cot.py` 内, 5 类别) — 这部分**就是本项目的核心创新**

---

## 4. 合作者本机需要修改的硬编码路径

| 出现在 | 含义 | 本机示例 |
|---|---|---|
| `train_configs/local/grpo_verl_v12i.yaml` | 数据 / 奖励 / 模型 / 输出 | 见下 sed |
| `train_scripts/local/run_grpo_verl_v12i.sh` | ROOT / PYTHON / VERL_HOME / MODEL_BASE | 同上 |

**一键替换** (Linux/macOS bash):

```bash
ROOT=/path/to/your/math-chain
sed -i "s|/home/wwq416/snap/wwq/math-chain|$ROOT|g" \
    train_configs/local/grpo_verl_v12i.yaml \
    train_scripts/local/run_grpo_verl_v12i.sh
sed -i "s|/home/wwq416/miniconda3/envs/math_chain_verl/bin/python|/path/to/your/python|g" \
    train_scripts/local/run_grpo_verl_v12i.sh
sed -i "s|/home/wwq416/snap/wwq/verl_math_chain|/path/to/your/verl_math_chain|g" \
    train_scripts/local/run_grpo_verl_v12i.sh
```

**远程**几乎不需要改路径: `submit_grpo_v12i_verl.sh` 默认值用 `${REMOTE_ROOT}` 拼路径, 合作者只需要 `export REMOTE_ROOT=/path/to/repo` 即可.

---

## 5. 远程提交: V12i 一键提交器

本资源包**新增了** `train_scripts/remote/submit_grpo_v12i_verl.sh` —— 为 V12i 量身定做, 默认就指向 V12i 的 reward / data / config / epoch. 合作者**不再需要手改 remote_env.sh**, 一行命令就能提交.

### 5.1 一键提交 (推荐流程)

```bash
# 1) 上传仓库到远程
rsync -avz --exclude='outputs/' --exclude='.git/' \
    ./math-chain-v12i/ user@remote:~/math-chain/
cd ~/math-chain

# 2) 干跑自检 (不真提交, 只打印 sbatch 命令)
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12i_verl.sh

#    期望看到 (关键检查):
#      REMOTE_REWARD_PATH=.../reward_chaingsm_v12i_json_verl.py   ← V12i
#      REMOTE_DATA_DIR=.../chaingsm_data/data/final/grpo          ← V12i
#      --config-name grpo_verl_v12i_vllm                          ← V12i config
#      trainer.total_epochs=2                                     ← V12i 训练量
#      JOB_NAME=chaingsm-grpo-verl-v12i                           ← V12i 任务名
#      [v12i] DRY_RUN=1, skip symlink/copy check

# 3) 跳过预检, 正式提交
SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12i_verl.sh

#    或完整流程 (带预检):
bash train_scripts/remote/submit_grpo_v12i_verl.sh
```

### 5.1 One-Stop Pipeline (2026-06-21 新增, 推荐)

V12i 推荐改用 `submit_grpo_v12_pipeline.sh` —— **preprocess + 训练 一条 sbatch**:

```bash
# 默认 V12i (V12 prompt + V12i 4 项 LCS reward)
sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# V12 prompt 但 V12 7 项 reward
VERSION=v12 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
```

合作者**不需要**手动 push `grpo_v12_json.parquet`. pipeline 内部先跑 `data_preprocess_v12.sh` (生成 V12 parquet), 再立即用同一份 parquet 启动 V12i 训练. 改 prompt 就改 `chaingsm_data/data/final/sft/build_grpo_v12_json.py` 顶部 `SYSTEM` / `USER_TEMPLATE` 然后重提.

环境变量: `DRY_RUN=1` (自检), `SKIP_PREPROCESS=1` (复用已有 parquet), `REMOTE_ROOT`, `REMOTE_MODEL_PATH`, `V12I_TOTAL_EPOCHS=2`. 详见 `train_scripts/remote/data_preprocess_README.md`.

### 5.2 数据 symlink 自动处理

`grpo_verl_v12i_vllm.yaml` 期望 parquet 文件名是 `verl_grpo_train.parquet`, V12i 数据叫 `grpo_v12_json.parquet`. `submit_grpo_v12i_verl.sh` **自动**在 `${REMOTE_DATA_DIR}/` 下放一个 symlink (`verl_grpo_train.parquet -> grpo_v12_json.parquet`), 合作者无需手动操作.

**前提条件**: V12i 数据 (`grpo_v12_json.parquet`) 必须**已经存在于** `${REMOTE_DATA_DIR}/` 下. 本资源包**已经**把数据打包进去, 解包后 `chaingsm_data/data/final/grpo/` 下就会有这个文件.

### 5.3 自定义变量 (可选)

```bash
# 改模型路径
REMOTE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
    bash submit_grpo_v12i_verl.sh

# 改 GPU 数 / rollout 采样数
SLURM_GPUS=2 ROLLOUT_N=4 TP_SIZE=1 \
    bash submit_grpo_v12i_verl.sh

# 改训练量
GRPO_EPOCHS=3 \
    bash submit_grpo_v12i_verl.sh

# 改 job 名
JOB_NAME=my-v12i-experiment \
    bash submit_grpo_v12i_verl.sh
```

### 5.4 跟老 submit_grpo_verl_vllm.sh 的对比

| 项 | submit_grpo_verl_vllm.sh (老) | submit_grpo_v12i_verl.sh (新) |
|---|---|---|
| config | grpo_verl_vllm.yaml | **grpo_verl_v12i_vllm.yaml** |
| reward | reward_chaingsm.py (V10 风格, 5 项) | **reward_chaingsm_v12i_json_verl.py (4 项)** |
| data | rl_preprocessed/gsm8k_train_..._14946 | **grpo_v12_json.parquet** |
| epochs | 1 | **2** |
| save_freq | 50 | **300** |
| JOB_NAME | chaingsm-grpo-verl-vllm | **chaingsm-grpo-verl-v12i** |
| symlink | 手动 | **自动** |

老脚本**保留**在包里 (`submit_grpo_verl_vllm.sh`), 仍然可以用, 但跑 V12i 必须用新脚本.

### 5.5 远程预检 (可选)

```bash
bash train_scripts/remote/preflight_remote.sh grpo
```

检查 CUDA module / conda env / SLURM partition / 数据路径 / Python imports. `SKIP_PREFLIGHT=1` 可跳过.

---

## 6. 评测 V12i 训练产物

V12i 训练每 300 步存一个 ckpt (5 个 ckpts @ 300/600/900/1200/1500). 评测入口已就绪 (`eval_vllm_chaingsm.py:571`):

```bash
STEPS="300 600 900 1200 1500" \
RUN_NAME=qwen2.5-0.5b-grpo-verl-v12i-reward \
RUN_ID=<your_run_id> \
bash train_scripts/local/eval_v12_5ckpts.sh
```

评测数据: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` — **本资源包已包含**.

评测指标:
- overall_accuracy
- 各 category 准确率 (original / path_competition / attribute_mismatch / numerical_irrelevance / scope_isolation)
- V12i 重点: distractor 类提升幅度 (per-step distractor reward 应该把这类拉起来)

汇总: `python train_scripts/local/aggregate_v10_results.py <eval_root>`

---

## 7. 验证 (跑测试)

```bash
# GSM answer extractor 单元测试
python tests/test_gsm_answer_extractor.py

# 远程提交脚本语法 (DRY_RUN)
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_v12i_verl.sh

# bash 语法
bash -n train_scripts/local/run_grpo_verl_v12i.sh
bash -n train_scripts/remote/submit_grpo_v12i_verl.sh
```

V12i 奖励函数本身不带 `__main__` 测试 (依赖 `Levenshtein` 第三方库), 但奖励逻辑可以手动 sanity check: 跑 1 个 sample, 看 metrics dict 的 4 项是否合理.

---

## 8. 依赖清单

- **Python 3.12** (本机 `math_chain_verl` env)
- **torch** + **CUDA 12.x**
- **transformers** (Qwen2.5-0.5B-Instruct)
- **vllm** (rollout, 与 verl 兼容版本)
- **verl** (本机 patch 过的 `verl_math_chain`, 含 vLLM rollout)
- **ray**
- **Levenshtein** (`pip install python-Levenshtein`) — V12i reward 必需
- **datasets**, **pyarrow** (parquet 读写)

合作者 `pip install python-Levenshtein` 后, 其他应该都在 `math_chain_verl` env 里就绪.

---

## 9. 已知问题与合作边界

- **不包含**: GSM8K raw jsonl, Qwen2.5 模型权重, verl_math_chain 源码, ray cluster 配置
- **不包含**: 任何训练 ckpt / 输出 (体积太大)
- **包含** V12i 训练 + 评测数据 (合作者无需自备):
  - `chaingsm_data/data/final/grpo/grpo_v12_json.parquet` (2.6 MB, 3051 行, 5 类别)
  - `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (8.6 MB)
- **V12i 训练可能未完成**: 作者本机 V12i 训练截至 2026-06-20 仍在跑, 实际 best ckpt 待定
- **远程默认指向 V12i**: v2 资源包新增 `submit_grpo_v12i_verl.sh` 已经把 2 个变量默认指 V12i, 合作者无需手改

---

## 10. 文件清单 + 行数参考

```
train_pipeline/reward_chaingsm_v12i_json_verl.py       ~375 行
train_pipeline/eval_vllm_chaingsm.py                   ~1100 行
train_pipeline/preprocess_chaingsm_8shot_cot.py        ~250 行
docs/prompts/v12_long_prompt.md                        ~480 行
train_configs/local/grpo_verl_v12i.yaml                ~95 行
train_configs/remote/grpo_verl_v12i_vllm.yaml          ~106 行
train_scripts/local/run_grpo_verl_v12i.sh              ~165 行
train_scripts/remote/remote_env.sh                     ~155 行
train_scripts/remote/submit_grpo_v12i_verl.sh          ~125 行
chaingsm_data/data/final/grpo/grpo_v12_json.parquet    2.6 MB
chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl     8.6 MB
```

---

## 11. 联系方式与版本

- **本资源包 commit 基础**: master 分支, 截至 2026-06-20 14:47
- **V12i 版本号**: V12i-reward (V12 prompt v4 + 4 项 reward 改良)
- **V12i 预期**: overall 22.9% → 26-30%, V12 baseline 上 +3-7 pp
- **资源包版本**: v2 (v1 = 仅代码, v2 = 代码 + 数据 + V12i 专用提交器)

合作者有任何问题, 先看 `docs/superpowers/specs/2026-06-17-grpo-v10-signed-design.md` (V12i 演进的源头) + `docs/analysis/v12_failure_analysis.md` (V12 为什么失败, V12i 为什么这样改). 路径问题先看 §4. 远程跑问题看 §5.
