# 无法登录远程时的本地维护计划

> **For agentic workers:** 当前阶段必须继续使用 superpowers 流程推进。远程 A100/SLURM 验证暂时不可执行，所有结论必须区分“本地已验证”和“等待远程验证”。

**目标：** 在无法登录远程服务器的情况下，先完成本地可验证的脚本、文档、清理边界和后续流程整理。

**当前项目主线：** ChainGSM Math-Chain 项目已经迁移到 `math_chain_verl` 本地环境，并新增远程 A100/SLURM 启动层。远程链路设计目标仍然是支持 verl SFT、TRL DPO、verl GRPO、verl SFT 后接 verl GRPO 四条训练路径。

**本地可处理范围：**

- 保持 `train_scripts/remote/remote_env.sh` 作为远程共享配置入口。
- 保持 `train_scripts/remote/preflight_remote.sh` 作为只读远程预检入口。
- 通过 `DRY_RUN=1 SKIP_PREFLIGHT=1` 在本地检查四条远程提交命令的构造。
- 通过 `bash -n` 检查本地和远程 shell 脚本语法。
- 通过测试断言检查当前文档不再把 `math-noise` 作为推荐环境。
- 删除或保持删除 Python 字节码缓存：`__pycache__/`、`*.pyc`。
- 不删除 `Noise_math_data-main/`，直到远程四条链路完成验证。

**暂时阻塞范围：**

- 不能确认远程 `REMOTE_ROOT` 实际路径。
- 不能确认远程 `gpu-A100`、`A100`、`a100_qos` 是否完全匹配。
- 不能确认远程 `Reasoning360` 环境是否能 import `torch`、`transformers`、`datasets`、`vllm`、`verl`。
- 不能确认远程数据、模型、reward 文件路径是否存在。
- 不能提交 smoke job，也不能检查 SLURM 日志和 checkpoint。

## 当前本地执行计划

### 任务 1：本地脚本和命令构造验证

- [x] 运行所有本地训练脚本的 shell 语法检查。
- [x] 运行所有远程提交脚本的 shell 语法检查。
- [x] 运行四条远程链路 dry-run。
- [x] 记录 dry-run 输出中必须出现的关键项：
  - `--partition=gpu-A100`
  - `--gres=gpu:4`
  - `--cpus-per-task=128`
  - `--mem=256GB`
  - `--account=A100`
  - `--qos=a100_qos`
  - `verl.trainer.fsdp_sft_trainer`
  - `train_pipeline.train_dpo_trl`
  - `verl.trainer.main_ppo`

### 任务 2：当前环境文档一致性

- [x] 确认当前推荐本地环境为：

```text
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
```

- [x] 确认当前文档不再把下面路径作为推荐环境：

```text
/home/wwq416/miniconda3/envs/math-noise/bin/python
```

- [x] 允许 `math-noise` 只在历史兼容说明或旧环境说明中出现。

### 任务 3：清理边界

- [x] 删除 Python 字节码缓存。
- [x] 保留 `Noise_math_data-main/`，因为它仍是远程链路验证前的参考项目。
- [x] 保留训练数据和评测结果，不在本阶段删除。
- [x] 对大型未跟踪目录只做记录，不做破坏性清理。

### 任务 4：远程恢复后的下一步

远程登录恢复后，按顺序执行：

```bash
bash train_scripts/remote/preflight_remote.sh all
```

如果 preflight 通过，再分别做四条链路的小规模 smoke job：

```bash
bash train_scripts/remote/submit_sft_verl.sh
bash train_scripts/remote/submit_dpo_trl.sh
bash train_scripts/remote/submit_grpo_verl_vllm.sh
bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

远程 smoke job 验证完成后，才进入 `Noise_math_data-main/` 删除决策。

## 清理决策表

| 内容 | 当前处理 | 原因 |
|---|---|---|
| `__pycache__/`、`*.pyc` | 删除 | Python 自动生成，已有 `.gitignore`，无维护价值 |
| `Noise_math_data-main/` | 保留 | 远程服务器验证前仍是参考来源 |
| `outputs/` | 保留 | 可能包含训练历史和指标 |
| `code/results/` | 保留 | 可能包含评测历史和对比结果 |
| `chaingsm_data/data/final/rl_preprocessed/` | 保留 | 当前训练链路依赖的预处理产物 |
| `plan_1.md` | 暂保留 | 需要确认是否仍有历史规划价值 |

## 完成标准

本地阶段完成时必须满足：

- shell 语法检查通过。
- dry-run 能构造四条远程提交命令。
- 当前文档推荐环境一致指向 `math_chain_verl`。
- 本地缓存清理完成。
- 明确记录哪些事项因无法登录远程而阻塞。

## 2026-06-02 本地执行记录

本轮在无法登录远程服务器的前提下完成了以下本地验证：

- `bash -n` 检查通过：
  - `train_scripts/remote/remote_env.sh`
  - `train_scripts/remote/preflight_remote.sh`
  - `train_scripts/remote/submit_sft_verl.sh`
  - `train_scripts/remote/submit_dpo_trl.sh`
  - `train_scripts/remote/submit_grpo_verl_vllm.sh`
  - `train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh`
  - `train_scripts/local/run_preprocess.sh`
  - `train_scripts/local/run_sft.sh`
  - `train_scripts/local/run_dpo.sh`
  - `train_scripts/local/run_grpo.sh`
  - `train_scripts/local/run_grpo_verl.sh`
  - `train_scripts/local/run_sft_then_grpo.sh`
- 四条远程链路 dry-run 均能输出 `[remote] Resolved command:` 和 `[remote] Submission command:`。
- dry-run 输出包含 A100 SLURM 资源参数和对应训练入口。
- 当前环境没有可用的 `pytest` 命令，因此使用 `/home/wwq416/miniconda3/envs/math_chain_verl/bin/python` 直接导入并执行测试函数。
- 直接执行的 7 个测试函数全部通过：
  - `test_dpo_trl_dry_run_uses_remote_env_and_resources`
  - `test_grpo_verl_dry_run_uses_remote_env_and_resources`
  - `test_sft_then_grpo_dry_run_chains_stage_commands`
  - `test_sft_verl_dry_run_uses_remote_env_and_resources`
  - `test_current_docs_do_not_present_math_noise_as_current_env`
  - `test_local_scripts_default_to_math_chain_verl_python`
  - `test_local_scripts_have_valid_bash_syntax`
- 缓存扫描命令未发现剩余的 `__pycache__/` 或 `*.pyc` 文件。

仍然不能在本地确认：

- 远程服务器是否存在 `REMOTE_ROOT`、`REMOTE_MODEL_PATH`、`REMOTE_DATA_DIR`。
- 远程 SLURM partition/account/qos 是否和脚本默认值完全一致。
- 远程 `Reasoning360` 环境能否导入所有依赖。
- 四条训练链路能否实际提交并生成日志、checkpoint、metrics。
