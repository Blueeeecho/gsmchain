# ChainGSM Remote — 数据预处理 + 训练 (One-Stop Pipeline)

> **目的**: 让合作者**只改 prompt** 就能重跑训练, 不用手动准备 parquet / 不用 push 数据.

## 文件清单

| 脚本 | 作用 | 配套 build 脚本 (改 prompt 就改这里) |
|---|---|---|
| `data_preprocess_v12.sh` | raw + supplementary → `grpo_v12_json.parquet` | `chaingsm_data/data/final/sft/build_grpo_v12_json.py` (顶部 `SYSTEM` / `USER_TEMPLATE`) |
| `data_preprocess_v13.sh` | raw + supplementary → `grpo_v13_json.parquet` | `chaingsm_data/data/final/sft/build_grpo_v13_json.py` (顶部 `SYSTEM` / `USER_TEMPLATE`) |
| `data_preprocess_v14.sh` | raw + supplementary → `grpo_v14_reasoning.parquet` | `chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py` (顶部 `V14_SYSTEM_PROMPT` / `V14_USER_TEMPLATE`) |
| `submit_grpo_v12_pipeline.sh` | **V12 / V12i 一键** preprocess + 训练 (默认 V12i) | — |
| `submit_grpo_v13_pipeline.sh` | **V13 一键** preprocess + 训练 | — |
| `submit_grpo_v14_pipeline.sh` | **V14 一键** preprocess + 训练 | — |

## 合作者最简流程 (3 步)

1. **改 prompt**: 打开对应版本的 `build_grpo_vXX_*.py`, 编辑顶部 prompt 常量 (`SYSTEM` / `USER_TEMPLATE` / `V14_*`).
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

- `submit_grpo_v{12,12i,13,14}_verl.sh` (4 个 wrapper) 保留, 只想直接训练的合作者仍可调用.
- `submit_grpo_verl_vllm.sh` (4 选 1 注释版) 保留, 但**不被 pipeline 调用** (它目前不接 `--version`, 是个老接口).
- `preflight_remote.sh` / `remote_env.sh` 不动.

## 资源 / 队列 (与案例一致)

- 1 GPU / 8 CPU / 32GB / `gpu-all` partition
- 日志: `./all_logs/%j-%x-slurm.{out,err}`
- CUDA module: `cuda12.4/toolkit`
- Conda env: `Reasoning360`
- HF cache: `/export/home/asifali/HF_cache`
