# GRPO / verl 输出与代码改动审查报告

日期：2026-05-29

审查范围：

- `train_readme.md`
- `codebuddy/eval_output_refactor.md`
- `/home/wwq416/snap/math_chain_verl_environment_setup.md`
- `train_pipeline/eval_vllm_chaingsm.py`
- `train_pipeline/eval_callback.py`
- `train_pipeline/train_grpo_trl.py`
- `train_pipeline/reward_chaingsm.py`
- `train_pipeline/reward_chaingsm_verl.py`
- `train_scripts/local/run_grpo.sh`
- `train_scripts/local/run_grpo_verl.sh`
- `train_configs/local/grpo_verl.yaml`
- `train_scripts/remote/submit_grpo_verl_vllm.sh`
- `train_configs/remote/grpo_verl_vllm.yaml`

## 1. 总体结论

当前修正方向总体是合理的：将评测输出从 CSV 改为 JSONL、逐条写入 `predictions.jsonl`、并将 vLLM 评测放入子进程，都是对长评测和显存释放问题更稳的处理；为 RTX 5090 / Blackwell 新增 `math_chain_verl` + verl GRPO 路线，也比继续强行在旧 TRL + vLLM 组合上打补丁更合适。

但当前实现还没有完全达到“替代旧训练输出结构”的成熟度。你这次运行：

```bash
conda activate math_chain_verl && \
TOTAL_EPOCHS=8 \
RUN_NAME=grpo_verl_from_sft_8epoch \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh
```

实际不会写到旧路径：

```text
/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/grpo_from_sft_paper_params/...
```

而是写到新路径：

```text
/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_verl_from_sft_8epoch/<RUN_ID>/
```

此外，新 `run_grpo_verl.sh` 目前没有像 TRL 入口那样调用 `prepare_run_dir()` / `setup_run_logging()`，因此不会自动生成旧结构中的：

```text
configs/resolved_config.yaml
logs/grpo_stdout.log
logs/grpo_stderr.log
metrics/
generated_samples/
```

这不是单纯的路径误会，还有一个真实训练稳定性问题：我检查了本机输出和 Ray 临时日志，你的 `grpo_verl_from_sft_8epoch` 运行已经完成了 baseline 评测，但训练进入第一个 epoch 后在约第 41 个 step 发生 vLLM rollout worker CUDA OOM。由于脚本设置 `save_freq=steps_per_epoch=882`，第一个 epoch checkpoint 尚未保存，所以输出目录里只有 baseline eval，没有 `checkpoints/global_step_*` 和 epoch eval。

## 2. 已确认的现象与证据

### 2.1 旧 GRPO 输出路径

旧 TRL 脚本 `train_scripts/local/run_grpo.sh` 将输出目录设置为：

```bash
OUTPUT_BASE_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_ID}"
```

对应旧路径：

```text
outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/<run_name>/<run_id>/
```

并且 Python 入口 `train_grpo_trl.py` 会执行：

```python
run_dir = prepare_run_dir(config)
setup_run_logging(run_dir, "grpo")
```

因此会创建：

```text
checkpoints/
configs/resolved_config.yaml
generated_samples/
logs/grpo_stdout.log
logs/grpo_stderr.log
metrics/
```

你提到的旧目录 `20260529_104541` 实际只有：

```text
configs/resolved_config.yaml
logs/grpo_stdout.log
logs/grpo_stderr.log
```

其 `grpo_stderr.log` 显示旧 TRL 运行失败在：

```text
ImportError: cannot import name 'GuidedDecodingParams' from 'vllm.sampling_params'
```

这正是文档里后续用 monkey-patch 处理的 TRL 0.26.1 + vLLM 0.13.0 不兼容问题。

### 2.2 新 verl 输出路径

新脚本 `train_scripts/local/run_grpo_verl.sh` 第 36-39 行定义：

```bash
RUN_NAME="${RUN_NAME:-grpo_verl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/${RUN_NAME}/${RUN_ID}}"
```

所以当前命令的实际输出根目录是：

```text
outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_verl_from_sft_8epoch/
```

我看到已有 3 次运行：

```text
20260529_143346/
20260529_144232/
20260529_145016/
```

这些目录内均只有：

```text
eval/baseline/eval_result.json
eval/baseline/predictions.jsonl
eval/baseline/summary_by_category.jsonl
eval/baseline/summary_overall.jsonl
eval/epoch_summary.jsonl
```

`epoch_summary.jsonl` 行数为 0，说明还没有完成任何 epoch 后评测。

### 2.3 baseline 评测已成功

baseline `eval_result.json` 显示初始 SFT best checkpoint 评测成功：

```json
{
  "overall_accuracy": 0.26783269961977185,
  "prediction_count": 6575,
  "gpu_memory_utilization": 0.4,
  "attempt_errors": []
}
```

分类准确率大致在 0.259-0.283 之间。也就是说，评测输出 JSONL 路线本身是跑通的。

### 2.4 训练在第一个 epoch 内 OOM，未到 checkpoint

`run_grpo_verl.sh` 第 113-118 行按样本数和 batch size 计算：

```bash
STEPS_PER_EPOCH=$(( (NUM_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
FULL_TRAINING_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS))
```

当前数据 7055 条、`TRAIN_BATCH_SIZE=8`，所以：

```text
steps_per_epoch = 882
TOTAL_EPOCHS=8
full_steps = 7056
```

脚本第 272-275 行设置：

```bash
trainer.total_epochs=$EPOCH_NUM
trainer.total_training_steps=$FULL_TRAINING_STEPS
trainer.save_freq=$STEPS_PER_EPOCH
trainer.test_freq=-1
```

也就是只有到 step 882 才会保存第一个 epoch checkpoint。Ray 日志显示训练实际跑到了约第 41 个 step，随后 vLLM rollout 进程 OOM：

```text
Training Progress: 41/7056
CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:139
RuntimeError: Worker failed with error 'CUDA Error: out of memory ...'
```

因此没有 `checkpoints/global_step_882/actor/huggingface`，也就没有 epoch eval、best checkpoint 和 latest metrics。

## 3. 代码改动合理性分析

### 3.1 `eval_vllm_chaingsm.py`: JSONL 与实时写入合理

合理点：

- `write_jsonl()` / `append_jsonl()` 替代 CSV，便于中途中断时查看已有结果。
- 每条 prediction 生成后立即写入 `predictions.jsonl`，比全部推理完成后再写安全。
- 输出字段去掉 `model_name`、`model_path`、`method`，在当前目录结构下是合理的，减少冗余。
- 写入 `eval_result.json`，方便父进程或 shell 脚本读取评测结果。
- GPU memory candidate 重试逻辑保留，且有 `eval_errors.json`。

需要改进点：

- `predictions_path.write_text("")` 会在每次 candidate 重试时清空当前 `predictions.jsonl`。这符合“只保留成功尝试”的语义，但如果第一次尝试已经生成了部分结果后失败，部分结果会被清掉。建议额外写 `attempt_<gpu_mem>/predictions.jsonl` 或在 `eval_errors.json` 里记录失败发生时已有行数。
- `eval_callback.py` 只想导入 `DEFAULT_TEST_DATA`，但它从 `train_pipeline.eval_vllm_chaingsm` 顶层导入，而 `eval_vllm_chaingsm.py` 顶层会立即 import `vllm`。这会让 SFT/DPO/GRPO 的训练父进程也 import vLLM，削弱“vLLM 只在子进程里出现”的隔离性。建议将 `DEFAULT_TEST_DATA` 移到轻量模块，例如 `train_pipeline/eval_constants.py`，或在 `eval_callback.py` 里直接定义常量，避免父进程导入 `vllm`。

### 3.2 `eval_callback.py`: 子进程评测方向正确

合理点：

- 子进程评测可以绕开 vLLM 进程内显存释放不彻底的问题，这是比 `gc.collect()` / `torch.cuda.empty_cache()` 更可靠的路线。
- `_offload_model()` 在评测前把训练模型移到 CPU，为 vLLM 留显存，这对 32GB 单卡尤其重要。
- `epoch_summary.jsonl` 和 `latest_metrics.json` 结构清晰。

需要改进点：

- 如上所述，顶层导入 `eval_vllm_chaingsm` 会间接 import vLLM。
- `_restore_model()` 捕获异常后静默忽略，若恢复失败，训练可能继续以异常状态运行。建议至少打印 warning。
- `subprocess.run(cmd)` 没有显式传入当前环境快照或日志文件；在本地可用，但如果远程环境变量较复杂，建议把关键 env 打印进日志。

### 3.3 `reward_chaingsm_verl.py`: 独立 reward 模块合理，但可减少双份逻辑

合理点：

- verl 通过 `custom_reward_function.path` 加载文件，独立自包含可以减少导入路径问题。
- `compute_reward()` 签名兼容 verl，返回 `{"score": float, "metrics": dict}` 是正确方向。

风险：

- `reward_chaingsm.py` 与 `reward_chaingsm_verl.py` 现在复制了同一套评分逻辑，后续很容易出现本地 TRL 与 verl reward 漂移。

建议：

- 如果远程加载路径稳定，保留 `reward_chaingsm.py` 为唯一实现，让 `reward_chaingsm_verl.py` 只做薄 wrapper。
- 如果确实需要自包含文件，则加一个最小一致性测试，比较同一批 completion/reference 在两个模块下分数完全一致。

### 3.4 `train_grpo_trl.py`: monkey-patch 思路合理，但旧失败说明 patch 时机曾经不够早

现在 `patch_vllm_for_trl()` 在动态获取 `GRPOTrainer` 前调用，逻辑上能解决 TRL 0.26.x 导入 `GuidedDecodingParams` 的问题。旧目录 `20260529_104541` 的失败日志来自 patch 完成之前或未使用当前版本脚本的运行。

当前版本对 TRL >= 1.0 与 vLLM >= 0.22 的判断较宽松，但 `VLLM_PATCHED_VERSIONS` 写死到 0.21.0。更稳的判断方式应基于能力而不是版本号：

```python
if not hasattr(sp, "GuidedDecodingParams") and hasattr(sp, "StructuredOutputsParams"):
    sp.GuidedDecodingParams = sp.StructuredOutputsParams
```

也就是说，版本列表可以只用于提示，不应成为主要兼容条件。

## 4. `run_grpo_verl.sh` 当前主要问题

### 4.1 输出结构未对齐旧 TRL 结构

旧 TRL 输出结构由 Python 入口统一创建；新 verl 脚本只创建了：

```bash
mkdir -p "$EVAL_DIR"
> "$EVAL_SUMMARY"
```

因此不会自动出现：

```text
configs/
logs/
metrics/
generated_samples/
```

建议在 shell 层补齐统一 run dir 初始化：

```text
$OUTPUT_DIR/
  configs/
    resolved_config.yaml 或 resolved_hydra_args.txt
  logs/
    grpo_verl_stdout.log
    grpo_verl_stderr.log
  metrics/
    train_result.json 或 latest_metrics.json
  eval/
  checkpoints/
```

### 4.2 没有日志 tee，失败后只能去 `/tmp/ray`

当前脚本直接把 baseline eval 和 verl training 输出到终端。训练失败后，项目输出目录没有 `logs/`，只能到 `/tmp/ray/session_.../logs/` 找错误，这会让复现实验很痛苦。

建议在确定 `OUTPUT_DIR` 后立即：

```bash
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/configs" "$OUTPUT_DIR/metrics"
exec > >(tee -a "$OUTPUT_DIR/logs/grpo_verl_stdout.log") \
     2> >(tee -a "$OUTPUT_DIR/logs/grpo_verl_stderr.log" >&2)
```

### 4.3 单卡 32GB 默认参数偏激进，已出现 rollout OOM

当前默认：

```bash
TRAIN_BATCH_SIZE=8
ROLLOUT_N=8
max_response_length=2048
ROLLOUT_GPU_MEM_UTIL=0.5
actor/ref log_prob_micro_batch_size_per_gpu=4
actor param_offload=False
optimizer_offload=False
```

对 RTX 5090 32GB 来说，这组参数追求吞吐，但当前证据显示不稳。OOM 发生在 vLLM rollout worker，而不是 baseline eval。建议把本地默认改成“稳态默认”，再允许通过环境变量调大：

```bash
TRAIN_BATCH_SIZE=4
ROLLOUT_N=4
ROLLOUT_GPU_MEM_UTIL=0.30 或 0.35
MAX_RESPONSE_LENGTH=1024  # smoke / debug 默认
LOG_PROB_MICRO_BATCH_SIZE=1 或 2
REF_LOG_PROB_MICRO_BATCH_SIZE=1 或 2
```

如果要保留论文参数作为 full run，可提供：

```bash
PROFILE=paper
```

或单独脚本 `run_grpo_verl_paper.sh`，而不是让最常用入口默认 OOM。

### 4.4 checkpoint 频率导致早期失败没有任何模型产物

`save_freq=882` 表示必须完整跑完第一个 epoch 才有 checkpoint。当前失败发生在第 41 step，所以没有 `checkpoints/global_step_*` 是预期结果。

建议增加调试参数：

```bash
SAVE_FREQ="${SAVE_FREQ:-$STEPS_PER_EPOCH}"
```

这样 smoke test 可以运行：

```bash
EVAL_BASELINE=0 SAVE_FREQ=20 TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 ...
```

并尽早看到 checkpoint 是否可加载。

### 4.5 `bc` 依赖可能导致 best model 永远不更新

第 336 行：

```bash
if (( $(echo "$ACCURACY > $BEST_ACCURACY" | bc -l 2>/dev/null || echo "0") )); then
```

如果系统没有 `bc`，比较表达式会回退为 0，`BEST_CKPT` 永远不会更新。建议改成 Python 比较，既然脚本已经依赖 `$PYTHON`：

```bash
if "$PYTHON" -c "import sys; sys.exit(0 if float('$ACCURACY') > float('$BEST_ACCURACY') else 1)"; then
    ...
fi
```

### 4.6 建议改用数组组织 Hydra 参数

当前脚本在长命令里使用反引号包裹注释：

```bash
`# === Algorithm ===` \
```

这在 bash 中通常会被当作空的 command substitution，不一定马上失败，但可读性和可维护性都不好。建议改成：

```bash
HYDRA_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  ...
)
"$PYTHON" -m verl.trainer.main_ppo \
  --config-name ppo_trainer \
  --config-dir "${VERL_HOME}/verl/trainer/config" \
  "${HYDRA_ARGS[@]}" \
  "$@"
```

### 4.7 缺少失败清理 trap

脚本只在 epoch 前后手动 `ray stop --force`。如果中途 OOM 或用户中断，Ray 可能残留。建议加：

```bash
cleanup() { ray stop --force 2>/dev/null || true; }
trap cleanup EXIT
```

## 5. 更适合的本地版本建议

建议把本地 verl GRPO 分成两个 profile：

### 5.1 `stable` 默认 profile

用于确认训练、checkpoint、eval、resume 全链路：

```bash
TRAIN_BATCH_SIZE=4
ROLLOUT_N=4
ROLLOUT_GPU_MEM_UTIL=0.30
EVAL_GPU_MEM_UTIL=0.30
MAX_RESPONSE_LENGTH=1024
LOG_PROB_MICRO_BATCH_SIZE=1
SAVE_FREQ=50
EVAL_BASELINE=1
```

目标：不追求速度，先保证 50-100 step 内能稳定保存 HF checkpoint，并能被 `eval_vllm_chaingsm.py` 加载。

### 5.2 `paper` full profile

用于稳定后跑正式实验：

```bash
TRAIN_BATCH_SIZE=8
ROLLOUT_N=8
ROLLOUT_GPU_MEM_UTIL=0.35 或 0.40
MAX_RESPONSE_LENGTH=2048
SAVE_FREQ=$STEPS_PER_EPOCH
TOTAL_EPOCHS=8
```

注意：当前 `ROLLOUT_GPU_MEM_UTIL=0.5` 已有 OOM 证据，不建议作为 32GB 单卡默认值。

### 5.3 输出目录建议

保留算法区分，但补齐旧结构：

```text
outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/<RUN_NAME>/<RUN_ID>/
  configs/
    resolved_args.txt
    environment.txt
  logs/
    grpo_verl_stdout.log
    grpo_verl_stderr.log
  checkpoints/
    global_step_0050/
    ...
    best/
  eval/
    baseline/
    epoch_0001/
    epoch_summary.jsonl
    latest_metrics.json
  metrics/
    latest_metrics.json
```

如果你希望和旧 `grpo/` 路径完全兼容，也可以设置：

```bash
OUTPUT_DIR=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo/Qwen2.5-0.5B-Instruct/grpo_from_sft_paper_params/<RUN_ID>
```

但从实验管理角度，我更建议保留 `grpo_verl/`，避免 TRL GRPO 与 verl GRPO 混在一起。

## 6. 对远程服务器运行的影响

### 6.1 本地 `run_grpo_verl.sh` 的硬编码不会直接影响远程

本地脚本硬编码了：

```text
/home/wwq416/miniconda3/envs/math_chain_verl
/home/wwq416/snap/wwq/verl_math_chain
RTX 5090 / Blackwell 相关环境变量
```

远程脚本 `train_scripts/remote/submit_grpo_verl_vllm.sh` 不调用本地脚本，所以这些硬编码本身不会影响远程 SLURM 任务。

### 6.2 共享 Python 文件可能影响远程

如果远程也同步了 `train_pipeline/eval_callback.py`，要注意它顶层导入 `eval_vllm_chaingsm.py`，从而顶层 import vLLM。若远程某个 TRL SFT/DPO 环境没有 vLLM，单纯 import callback 就可能失败。建议尽快移除这个顶层依赖。

`reward_chaingsm.py` 已包含 `compute_reward()`，远程 `custom_reward_function.path=${REMOTE_REWARD_PATH}` 默认指向它，目前是可用的。

### 6.3 远程 config 中 `ray_init` 键可能不匹配当前 verl 主线

当前本地 verl 主线 `ppo_trainer.yaml` 使用：

```yaml
ray_kwargs:
  ray_init:
```

本地脚本也覆盖：

```bash
ray_kwargs.ray_init.num_cpus=16
```

但远程 `train_configs/remote/grpo_verl_vllm.yaml` 末尾写的是：

```yaml
ray_init:
  num_cpus: 32
```

如果远程使用同一版 verl，这个键大概率不会被 `main_ppo.py` 使用。建议改为：

```yaml
ray_kwargs:
  ray_init:
    num_cpus: 32
```

这不是导致本地问题的原因，但会影响远程 CPU 配置是否生效。

### 6.4 远程参数与本地参数已有明显分叉

远程 GRPO 仍使用：

```yaml
kl_loss_coef: 0.001
temperature: 0.7
answer_weight: 0.4
expression_weight: 0.2
trace_weight: 0.2
distractor_penalty: 0.3
```

本地 verl 脚本使用：

```bash
KL_LOSS_COEF=0.04
ROLLOUT_TEMPERATURE=0.9
ANSWER_WEIGHT=2.5
EXPRESSION_WEIGHT=1.0
TRACE_WEIGHT=1.0
DISTRACTOR_PENALTY=0.5
```

这意味着本地和远程不是同一套实验参数。若目标是先本地验证、再远程复现实验，应把这些参数统一到一个 profile 或文档中，否则远程结果不能直接对齐本地结果。

## 7. 建议优先级

P0：

- 降低本地 `run_grpo_verl.sh` 默认显存压力：`ROLLOUT_GPU_MEM_UTIL=0.30/0.35`、`TRAIN_BATCH_SIZE=4`、`ROLLOUT_N=4` 或先用 debug profile。
- 给 `run_grpo_verl.sh` 增加 `logs/` tee，否则下一次失败仍然只能翻 `/tmp/ray`。
- 增加 `trap cleanup EXIT`，保证 OOM/中断后 Ray 被清理。

P1：

- 补齐 `configs/`、`metrics/` 输出，让 verl 路径拥有和旧 TRL 路径类似的可审计结构。
- 将 `DEFAULT_TEST_DATA` 从 `eval_vllm_chaingsm.py` 拆出，避免训练父进程顶层 import vLLM。
- 将 `SAVE_FREQ`、`MAX_RESPONSE_LENGTH`、`LOG_PROB_MICRO_BATCH_SIZE` 参数化。
- 用 Python 替代 `bc` 做 best accuracy 比较。

P2：

- 把 long Hydra command 改成 bash array。
- 为 `reward_chaingsm.py` 与 `reward_chaingsm_verl.py` 增加一致性测试。
- 远程 config 修正 `ray_init` 为 `ray_kwargs.ray_init`。
- 统一本地/远程 reward 权重、KL、temperature 等实验参数。

## 8. 推荐下一次验证命令

先做一个稳态 smoke test，不跑完整 baseline：

```bash
conda activate math_chain_verl

EVAL_BASELINE=0 \
TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=4 \
ROLLOUT_N=2 \
ROLLOUT_GPU_MEM_UTIL=0.30 \
EVAL_GPU_MEM_UTIL=0.30 \
RUN_NAME=grpo_verl_smoke_stable \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_grpo_verl.sh \
  data.max_response_length=1024 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.save_freq=20
```

但注意：当前脚本内部会在后面固定传 `trainer.save_freq=$STEPS_PER_EPOCH`，命令行追加的 `trainer.save_freq=20` 是否能覆盖，取决于 Hydra 对重复 key 的处理顺序。更稳的做法是脚本里显式加入：

```bash
SAVE_FREQ="${SAVE_FREQ:-$STEPS_PER_EPOCH}"
trainer.save_freq=$SAVE_FREQ
```

然后用：

```bash
SAVE_FREQ=20 ...
```

如果 smoke test 可以稳定产生：

```text
checkpoints/global_step_20/actor/huggingface/
```

再逐步提高 `ROLLOUT_N`、`TRAIN_BATCH_SIZE` 和 `max_response_length`。

## 9. 最终判断

当前修正不是错误方向，核心设计是对的；问题在于本地 verl 训练脚本还缺少旧训练入口已有的“实验目录治理”和“失败可诊断性”，并且默认显存配置对 RTX 5090 32GB 偏激进，已经实际触发 vLLM rollout OOM。

对远程运行的直接破坏风险不高，因为本地 `run_grpo_verl.sh` 不会被远程脚本调用；但共享代码和远程配置仍有两个需要尽快处理的点：`eval_callback.py` 顶层间接 import vLLM，以及远程 `ray_init` 键可能不匹配当前 verl 配置结构。
