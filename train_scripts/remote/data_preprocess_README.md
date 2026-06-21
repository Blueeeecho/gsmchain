# ChainGSM Remote — 数据预处理 + 训练 (One-Stop Pipeline)

> **目的**: 让合作者**只改 prompt** 就能重跑训练, 不用手动准备 parquet / 不用 push 数据.
>
> **prompt 写在 build 脚本顶部** —— 跟案例 `our_pre_process_zebrapuzzle_to_guru_parsed_v6a_LTLT.py` 的
> `SOLUTION_PROMPT_1_SHOT_SYS` 同一种风格, **打开 build 脚本 → 滚到顶部 → 改三引号里的内容 → 保存 → 重提**,
> **不需要查任何外部 md 文档**.

## 文件清单

| 脚本 | 作用 | **prompt 在 build 脚本顶部的常量** (改这里) |
|---|---|---|---|
| `data_preprocess_v12.sh` | raw + supplementary → `grpo_v12_json.parquet` | `chaingsm_data/data/final/sft/build_grpo_v12_json.py` → 顶部 `SYSTEM` (L24) + `USER_TEMPLATE` (L75) |
| `data_preprocess_v13.sh` | raw + supplementary → `grpo_v13_json.parquet` | `chaingsm_data/data/final/sft/build_grpo_v13_json.py` → 顶部 `SYSTEM` (L36) + `USER_TEMPLATE` (L72) |
| `data_preprocess_v14.sh` | raw + supplementary → `grpo_v14_reasoning.parquet` | `chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py` → 顶部 `V14_SYSTEM_PROMPT` (L32) + `V14_USER_TEMPLATE` (L77) |
| `submit_grpo_v12_pipeline.sh` | **V12 / V12i 一键** preprocess + 训练 (默认 V12i) | — |
| `submit_grpo_v13_pipeline.sh` | **V13 一键** preprocess + 训练 | — |
| `submit_grpo_v14_pipeline.sh` | **V14 一键** preprocess + 训练 | — |

## 为什么 prompt 直接写在 build 脚本里 (跟案例同模式)

案例 `data_preprocess_parsed_v6a_ALL.sh` 把 `python ./data_preprocess/logic/v6a/our_pre_process_zebrapuzzle_to_guru_parsed_v6a_LTLT.py`
作为黑盒调用, prompt 全部嵌在该 py 顶部的 `SOLUTION_PROMPT_1_SHOT_SYS` 等常量里 — 改 prompt 不需要查任何外部 md.

我们这里 3 个 build 脚本 (`build_grpo_v12_json.py` / `build_grpo_v13_json.py` / `build_grpo_v14_reasoning.py`)
是**同一种模式**: `SYSTEM` / `USER_TEMPLATE` / `V14_SYSTEM_PROMPT` / `V14_USER_TEMPLATE`
三引号常量写在文件**前 80 行**, 改完保存即可. 不存在 `prompt → 外部 .md → py` 的三跳链.

`docs/prompts/v{12,13,14}_long_prompt.md` 是**历史快照** (跟 `eval_vllm_chaingsm.py` 评测侧同步用的),
**不是**改 prompt 的入口. 改 prompt 永远只动 build 脚本顶部常量.

## 合作者最简流程 (3 步)

1. **改 prompt** (跟案例同模式 — prompt 常量嵌在 build 脚本顶部):
   ```bash
   vim chaingsm_data/data/final/sft/build_grpo_v12_json.py    # V12/V12i: 改 SYSTEM (L24) + USER_TEMPLATE (L75)
   vim chaingsm_data/data/final/sft/build_grpo_v13_json.py    # V13:     改 SYSTEM (L36) + USER_TEMPLATE (L72)
   vim chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py  # V14: 改 V14_SYSTEM_PROMPT (L32) + V14_USER_TEMPLATE (L77)
   ```
   只改三引号 `"""..."""` 里的内容, 文件其他逻辑 (SRC / SUP / DST 路径, row schema) 不要动.
2. **重提任务** (单条 sbatch, 从 raw 到 ckpt 一气呵成):
   ```bash
   # 默认 V12i
   sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

   # 想跑 V12 而不是 V12i
   VERSION=v12 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

   # V13 / V14
   sbatch train_scripts/remote/submit_grpo_v13_pipeline.sh
   sbatch train_scripts/remote/submit_grpo_v14_pipeline.sh
   ```
3. **不用 push 任何数据** — 任务会先跑预处理生成 parquet, 再立即用同一份 parquet 启动训练.

## 环境变量 (可选)

| 变量 | 默认 | 说明 |
|---|---|---|
| `REMOTE_ROOT` | `/home/wwq416/snap/wwq/math-chain` | 仓库根目录 |
| `REMOTE_MODEL_PATH` | `/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct` | 起点模型 |
| `REMOTE_OUTPUT_DIR` | `${REMOTE_ROOT}/outputs/train/remote` | 训练输出 |
| `REMOTE_DATA_DIR` | `${REMOTE_ROOT}/chaingsm_data/data/final/grpo` | parquet 输出 |
| `DRY_RUN=1` | 0 | 只打印要执行的命令, 不真跑 |
| `SKIP_PREPROCESS=1` | 0 | 跳过 build 步骤 (parquet 已有, 不重生成) |
| `VERSION=v12\|v12i` | `v12i` | 仅 V12 pipeline 有效 |
| `V12I_TOTAL_EPOCHS` / `V12_TOTAL_EPOCHS` | 2 / 2 | 覆盖训练轮数 |
| `V13_TOTAL_EPOCHS` | 2 | 覆盖训练轮数 |
| `V14_TOTAL_EPOCHS` | 3 | 覆盖训练轮数 |

## 自检 (干跑)

```bash
DRY_RUN=1 SKIP_PREPROCESS=1 bash train_scripts/remote/submit_grpo_v12_pipeline.sh
DRY_RUN=1 SKIP_PREPROCESS=1 bash train_scripts/remote/submit_grpo_v13_pipeline.sh
DRY_RUN=1 SKIP_PREPROCESS=1 bash train_scripts/remote/submit_grpo_v14_pipeline.sh
```
期望看到打印 preprocess + train 两条命令, 不真跑任何东西.

## 不动的东西

- `submit_sft_verl.sh` / `submit_dpo_trl.sh` / `submit_sft_then_grpo_verl_vllm.sh`: 3 个 SFT/DPO 链提交器 (跟 V12 pipeline 独立, 走 `source remote_env.sh` + `preflight` 派).
- `preflight_remote.sh` / `remote_env.sh`: 共享环境层与预检, 不动.
- `V12_V14_HANDBOOK.md` / `V12i_HANDBOOK.md`: 4 版本总览, 仍在.

## 已删除的 legacy 提交器 (2026-06-21)

- `submit_grpo_verl_vllm.sh` (4 选 1 注释版, 实际不接 `--version`)
- `submit_grpo_v{12,12i,13,14}_verl.sh` (4 个 wrapper, 全部 `exec` 到上面那个老接口, 实际不切档)

合作者跑 V12-V14 **只走 pipeline** (`submit_grpo_v{12,13,14}_pipeline.sh`).

## 资源 / 队列 (与案例一致)

- 1 GPU / 8 CPU / 32GB / `gpu-all` partition
- 日志: `./all_logs/%j-%x-slurm.{out,err}`
- CUDA module: `cuda12.4/toolkit`
- Conda env: `Reasoning360`
- HF cache: `/export/home/asifali/HF_cache`
