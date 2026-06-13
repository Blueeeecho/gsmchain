# 评测输出重构记录

**日期**: 2026-05-28

## 变更目标

1. 评测过程文件实时写入（而非全部推理完成后一次性写入）
2. 移除输出文件中冗余的 `model_name`、`model_path`、`method` 字段（已在输出路径中区分）
3. 输出格式从 CSV 改为 JSONL，便于逐行查看

---

## 修改文件清单

### 1. `train_pipeline/eval_vllm_chaingsm.py`

| 改动位置 | 改动内容 |
|---|---|
| 顶部 import | 移除 `import csv`（不再需要） |
| `write_csv()` 函数 (原 L181-186) | 替换为 `write_jsonl()` 和 `append_jsonl()` 两个函数 |
| `summarize()` 函数签名 (原 L245) | 移除 `model_name` 和 `method` 参数 |
| `summarize()` 函数体 | 返回的 `category_rows` 和 `overall_rows` 中移除 `model_name`、`method` 字段 |
| `evaluate_with_vllm()` 函数体 | **核心改动**：详见下方 |

#### `evaluate_with_vllm()` 核心改动细节

- **新增** `predictions_path` 变量，指向 `predictions.jsonl`（原为 `predictions.csv`）
- **新增** 评测开始前清空 `predictions.jsonl` 文件（`predictions_path.write_text("")`）
- **新增** 每次 candidate 重试时也清空文件
- **改动** prediction row 字典中移除 `model_name`、`model_path`、`method` 三个字段
- **改动** 每生成一条 prediction 后立即调用 `append_jsonl(predictions_path, row)` 实时写入
- **改动** `summarize()` 调用不再传 `model_path.name` 和 `method`
- **改动** 输出文件名：`predictions.csv` → `predictions.jsonl`，`summary_by_category.csv` → `summary_by_category.jsonl`，`summary_overall.csv` → `summary_overall.jsonl`
- **改动** `write_jsonl()` 替代 `write_csv()`，不再需要 fieldnames 参数

### 2. `train_pipeline/eval_callback.py`

| 改动位置 | 改动内容 |
|---|---|
| 顶部 import | 移除 `import csv`；新增从 `eval_vllm_chaingsm` 导入 `append_jsonl` |
| `_append_csv()` 函数 (原 L31-38) | 替换为 `_append_jsonl()` 函数，不再需要 fieldnames 参数 |
| `_append_summary()` 方法 (L89-95) | 文件名从 `epoch_summary.csv` → `epoch_summary.jsonl`；调用从 `_append_csv()` → `_append_jsonl()`；不再传 fieldnames |

### 3. `train_readme.md`

| 改动位置 | 改动内容 |
|---|---|
| L227-241 输出目录结构示例 | 文件扩展名 `.csv` → `.jsonl` |

---

## 输出文件变化对比

### 修改前

```
eval/
  epoch_summary.csv          (CSV, 含 model_name/method 列)
  baseline/
    predictions.csv           (CSV, 含 model_name/model_path/method 列)
    summary_overall.csv       (CSV, 含 model_name/method 列)
    summary_by_category.csv   (CSV, 含 model_name/method 列)
  epoch_0001/
    predictions.csv
    summary_overall.csv
    summary_by_category.csv
```

### 修改后

```
eval/
  epoch_summary.jsonl        (JSONL, 不含冗余字段)
  baseline/
    predictions.jsonl         (JSONL, 实时写入, 不含冗余字段)
    summary_overall.jsonl     (JSONL, 不含冗余字段)
    summary_by_category.jsonl (JSONL, 不含冗余字段)
  epoch_0001/
    predictions.jsonl
    summary_overall.jsonl
    summary_by_category.jsonl
```

### 字段变化

**predictions 行字段**:
- 移除: `model_name`, `model_path`, `method`
- 保留: `id`, `base_id`, `category`, `question`, `gold_answer`, `raw_output`, `pred_answer`, `correct`

**summary_by_category 行字段**:
- 移除: `model_name`, `method`
- 保留: `category`, `correct`, `total`, `accuracy`

**summary_overall 行字段**:
- 移除: `model_name`, `method`
- 保留: `correct`, `total`, `accuracy`

**epoch_summary 行字段**:
- 无变化（原本就不含 model_name/method）

---

## 第二次变更：修复评测后训练 OOM

**日期**: 2026-05-28

### 问题

baseline 评测完成后，训练第一步即 OOM：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 15.75 MiB is free.
Including non-PyTorch memory, this process has 23.75 GiB memory in use.
```

### 根因

vLLM 在进程内运行评测后，即使调用 `cleanup_vllm()` (gc.collect + torch.cuda.empty_cache + destroy_model_parallel)，
CUDA 显存仍无法完全释放（vLLM 已知问题）。评测完成后恢复训练模型到 GPU，残留显存导致前向传播 OOM。

### 解决方案

将 vLLM 评测改为**子进程**运行。子进程退出后，OS 自动回收全部 GPU 显存，从根源解决内存泄漏问题。

### 修改文件清单

#### 1. `train_pipeline/eval_vllm_chaingsm.py`

| 改动位置 | 改动内容 |
|---|---|
| `evaluate_with_vllm()` 末尾 (L384-396) | 新增写入 `eval_result.json`，将返回值 dict 序列化到文件，供子进程模式传递结果 |
| `main()` CLI argparse (L422-423) | 新增 `--trust-remote-code` / `--no-trust-remote-code` 参数 |
| `main()` 调用 (L441) | 传入 `trust_remote_code=args.trust_remote_code` |

#### 2. `train_pipeline/eval_callback.py`

| 改动位置 | 改动内容 |
|---|---|
| 顶部 import | 新增 `import subprocess`, `import sys`；移除 `from eval_vllm_chaingsm import append_jsonl, evaluate_with_vllm`，改为只导入 `DEFAULT_TEST_DATA` |
| `_run_eval()` 方法 (L71-113) | **完全重写**：从直接调用 `evaluate_with_vllm()` 改为通过 `subprocess.run()` 启动子进程执行 `python -m train_pipeline.eval_vllm_chaingsm`，子进程退出后从 `eval_dir/eval_result.json` 读取结果 |

### 子进程模式工作流程

```
训练进程 (parent)                          评测子进程 (child)
─────────────────                          ─────────────────
1. offload 训练模型到 CPU
2. 保存模型到 checkpoints/current/
3. subprocess.run(
     python -m train_pipeline.eval_vllm_chaingsm
     --model-path ...
     --output-dir ...
   )
                                           4. 加载 vLLM
                                           5. 逐条推理 + 实时写入 predictions.jsonl
                                           6. 写入 summary_*.jsonl
                                           7. 写入 eval_result.json
                                           8. 进程退出 → OS 回收全部 GPU 显存
9. 读取 eval_result.json
10. 恢复训练模型到 GPU
11. 继续训练
```

### 保留的设计

- `_offload_model()` / `_restore_model()` 仍然保留：子进程与父进程共享同一 GPU，offload 训练模型可为 vLLM 腾出显存
- 所有评测参数（batch_size, gpu_memory_utilization_candidates 等）通过 CLI 参数原样传递
- 子进程的 stdout/stderr 直接输出到终端，用户可实时看到评测进度

---

## 第三次变更：修复 vLLM 0.13.0 与 TRL 0.26.1 不兼容问题

**日期**: 2026-05-29

### 问题

运行 GRPO 训练时报错：
```
ImportError: cannot import name 'GuidedDecodingParams' from 'vllm.sampling_params'
```

### 根因

TRL 0.26.1 的 `grpo_trainer.py` (L101) 硬编码导入：
```python
from vllm.sampling_params import GuidedDecodingParams
```

但在 vLLM >= 0.12.0 中，`GuidedDecodingParams` 被重命名为 `StructuredOutputsParams`。该类被用于 `guided_decoding_regex` 功能（L1349）。

**版本关系**：

| 组件 | 版本 | 兼容性 |
|------|------|--------|
| TRL | 0.26.1 | 期望 vLLM 0.10.2~0.11.2 |
| vLLM | 0.13.0 (math-noise 环境) | `GuidedDecodingParams` → `StructuredOutputsParams` |

### 解决方案

在导入 TRL 之前，通过 monkey-patch 将 `StructuredOutputsParams` 别名为 `GuidedDecodingParams`。
两者构造函数兼容：`GuidedDecodingParams(regex=...)` 与 `StructuredOutputsParams(regex=...)` 等价（`regex` 参数在两者中都存在）。

### 修改文件清单

#### `train_pipeline/train_grpo_trl.py`

| 改动位置 | 改动内容 |
|---|---|
| 新增 `VLLM_PATCHED_VERSIONS` 常量 | 列出支持 monkey-patch 的 vLLM 版本 (0.12.0~0.21.0) |
| 新增 `patch_vllm_for_trl()` 函数 | 检测 vLLM 版本，若缺少 `GuidedDecodingParams` 则将 `StructuredOutputsParams` 别名为其 |
| 修改 `check_grpo_environment()` | 增加 `VLLM_PATCHED_VERSIONS` 分支：若版本在补丁列表中，自动调用 `patch_vllm_for_trl()` 而非报错 |
| 修改 `main()` 中环境检查逻辑 | `--skip-env-check` 时仍调用 `patch_vllm_for_trl()` 确保补丁生效；否则走 `check_grpo_environment()`（内含补丁调用） |

### 行为变化

| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| vLLM 0.11.x | 正常运行 | 正常运行（无变化） |
| vLLM 0.13.0 + `--skip-env-check` | `ImportError` 崩溃 | 自动补丁 → 正常运行 |
| vLLM 0.13.0 不带 skip | `RuntimeError` 崩溃 | 自动补丁 → 正常运行 |
| vLLM 不在两个列表中 | `RuntimeError` | `RuntimeError`（提示更新补丁列表） |

### 注意事项

- TRL 运行时仍会输出 `UserWarning: TRL currently supports vLLM versions: 0.10.2, ...` 警告，这是无害的
- `ResourceTracker.__del__` 的 `AttributeError` 是 multiprocess 库的已知清理问题，不影响训练
- 如果未来升级 TRL 至支持 vLLM 0.12+ 的版本，可移除此补丁

---

## 第四次变更：添加 verl 框架 GRPO 训练支持

**日期**: 2026-05-29

### 背景

原有训练管线基于 TRL (math-noise conda 环境)，存在以下问题：
1. TRL 0.26.1 与 vLLM 0.13.0 不兼容（需 monkey-patch）
2. math-noise 环境的 vLLM 0.13.0 不支持 RTX 5090 (Blackwell sm_120)
3. TRL 的 GRPOTrainer 在单卡场景下性能受限（rollout 和训练无法并行）

新建 `math_chain_verl` conda 环境（PyTorch 2.11+cu130, vLLM 0.21.0, verl 0.8.0.dev0），
完整支持 Blackwell 架构，verl 的 FSDP + vLLM rollout 架构更加高效。

### 新增文件清单

#### 1. `train_pipeline/reward_chaingsm_verl.py`

独立的 verl 奖励函数模块，自包含（不依赖 train_pipeline 内部导入）。

|| 内容 | 说明 |
||---|---|
|| `score_response()` | 与 `reward_chaingsm.py` 完全相同的评分逻辑 |
|| `compute_reward()` | verl 兼容入口，签名 `compute_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)` → `{"score": float, "metrics": dict}` |

verl 通过 `custom_reward_function.path` + `name` 加载此文件，`reward_kwargs` 自动注入 `**kwargs`。

#### 2. `train_configs/local/grpo_verl.yaml`

verl GRPO 本地单卡训练的完整 Hydra 配置，对应论文参数：

|| 参数 | 值 | 对应 TRL 参数 |
||---|---|---|
|| `actor.optim.lr` | 5e-7 | `training.learning_rate` |
|| `actor.kl_loss_coef` | 0.04 | `training.beta` |
|| `actor.clip_ratio` | 0.2 | (TRL 内置) |
|| `actor.loss_agg_mode` | token-mean | (默认) |
|| `rollout.n` | 8 | `training.num_generations` |
|| `rollout.temperature` | 0.9 | `training.temperature` |
|| `rollout.top_p` | 1.0 | `training.top_p` |
|| `rollout.top_k` | 50 | `training.top_k` |
|| `rollout.tensor_model_parallel_size` | 1 | (单卡) |
|| `rollout.gpu_memory_utilization` | 0.5 | (单卡需留显存给训练) |
|| `algorithm.adv_estimator` | grpo | (GRPO 算法) |
|| `trainer.total_epochs` | 20 | `training.num_train_epochs` |
|| `trainer.n_gpus_per_node` | 1 | (单卡) |
|| `custom_reward_function.reward_kwargs` | 论文权重 | `reward.*` |

#### 3. `train_scripts/local/run_grpo_verl.sh`

verl GRPO 本地训练启动脚本，功能：
- 自动设置 `VLLM_USE_V1=1` 和 `CUDA_MODULE_LOADING=LAZY`（Blackwell 优化）
- 默认使用 SFT best checkpoint 作为初始模型
- 自动检查数据文件是否存在
- 支持通过环境变量和命令行参数覆盖配置

### 迁移对照

|| 项目 | TRL (math-noise) | verl (math_chain_verl) |
||---|---|---|
|| Python | 3.12 (math-noise) | 3.12 (math_chain_verl) |
|| PyTorch | 2.x+cu128 | 2.11.0+cu130 |
|| vLLM | 0.13.0 (需 patch) | 0.21.0 (原生支持) |
|| 训练框架 | TRL GRPOTrainer | verl RayPPOTrainer |
|| 奖励函数 | `reward_chaingsm.py` | `reward_chaingsm_verl.py` |
|| 数据格式 | JSONL (`grpo_train.jsonl`) | Parquet (`verl_grpo_train.parquet`) |
|| Blackwell 支持 | 不支持 | 完整支持 |
|| 分布式 | 不支持 | FSDP (Ray) |

### 运行命令

```bash
conda activate math_chain_verl

# 使用论文参数 + 自定义奖励权重
MODEL=/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_20epoch_full_eval/20260528_193544/checkpoints/best \
RUN_NAME=grpo_verl_from_sft_paper_params \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

覆盖特定参数示例：
```bash
bash run_grpo_verl.sh \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  custom_reward_function.reward_kwargs.answer_weight=2.5
```

### 注意事项

1. verl 使用 `main_ppo.py` (已标记废弃，未来替换为 `main_ppo_sync.py`)，但当前仍可用
2. 单卡训练时 `rollout.gpu_memory_utilization` 设为 0.5（留一半显存给 Actor/Ref 训练）
3. `rollout.enforce_eager=True` 避免 CUDA graph 在 Blackwell 上的兼容性问题
4. 当前未实现 epoch-end vLLM 评测回调（verl 有内置 `test_freq` 但逻辑不同）
5. 数据文件 `verl_grpo_train.parquet` 已由 `preprocess_chaingsm.py` 生成（7055 条）
