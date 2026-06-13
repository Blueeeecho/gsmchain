# Remote Server Settings and Reuse Guide

This document captures the remote A100/SLURM environment inferred from the
reference project under `Noise_math_data-main/` and compares it with the current
ChainGSM remote scripts. The reference copy is intentionally kept in this
repository until the current remote launch path fully matches the server setup.

## Current Status

The current ChainGSM repository now has a remote launch layer aligned with the
reference A100/SLURM server assumptions. It still needs to be validated on the
remote server before `Noise_math_data-main/` can be removed as a reference.

## 当前无法登录远程服务器时的处理原则

当前阶段不能登录远程服务器，因此只能完成本地可验证事项。远程相关结论必须
明确标记为“等待远程验证”，不能因为本地 dry-run 通过就删除参考项目或确认
远程训练链路可用。

本地可以继续执行：

- `bash -n` 检查本地和远程 shell 脚本语法。
- `DRY_RUN=1 SKIP_PREFLIGHT=1` 检查四条远程提交命令构造。
- 检查文档是否统一指向 `math_chain_verl` 本地环境。
- 清理 `__pycache__/` 和 `*.pyc` 等确定无用的缓存文件。
- 用中文记录当前执行状态和远程阻塞项。

本地不能替代执行：

- `train_scripts/remote/preflight_remote.sh all` 的真实远程检查。
- SLURM `sbatch` 提交。
- 远程 CUDA module、conda env、GPU、模型路径、数据路径验证。
- `Noise_math_data-main/` 删除决策。

当前中文本地维护记录见：

```text
docs/superpowers/plans/2026-06-02-local-only-maintenance.md
```

Implemented:

- ChainGSM has remote launch scripts under `train_scripts/remote/`.
- All four remote chains submit through SLURM `sbatch`:
  - verl SFT
  - TRL DPO
  - verl GRPO with vLLM rollout
  - verl SFT followed by verl GRPO
- Remote GRPO/SFT use verl entry points:
  - `python -m verl.trainer.fsdp_sft_trainer`
  - `python -m verl.trainer.main_ppo`
- Remote configs and scripts use environment variables for paths:
  - `REMOTE_ROOT`
  - `REMOTE_MODEL_PATH`
  - `REMOTE_DATA_DIR`
  - `REMOTE_OUTPUT_DIR`
  - `REMOTE_REWARD_PATH`
  - `REMOTE_EVAL_DATA_PATH`
- Remote scripts default to the reference A100 resources:
  `gpu-A100`, `--gres gpu:4`, `--cpus-per-task=128`, `--mem=256GB`,
  `--account=A100`, and `--qos=a100_qos`.
- The shared remote layer loads the reference CUDA module and activates the
  reference conda environment inside the submitted job when those commands are
  available.
- The shared remote layer sets the Ray, cache, NCCL, vLLM, transformers, and
  temporary-directory environment used by the reference scripts.
- The read-only preflight checks model, data, reward file, conda env, CUDA
  module, SLURM command/partition, Python imports, and GPU compatibility.

Still requiring remote validation:

- Confirm the remote checkout path used for `REMOTE_ROOT`.
- Confirm the remote server exposes `gpu-A100`, `A100`, and `a100_qos` exactly
  as configured.
- Confirm `Reasoning360` can import the package versions required by this
  ChainGSM code.
- Confirm the ChainGSM preprocessed data exists on the remote server.
- Submit smoke jobs for the four chains and inspect logs/checkpoints.

## Implemented ChainGSM Remote Layer

The current repository uses `train_scripts/remote/remote_env.sh` as the shared
remote configuration layer. The four submit scripts source it before building
their chain-specific commands.

Use `DRY_RUN=1 SKIP_PREFLIGHT=1` to inspect commands locally:

```bash
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_verl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_dpo_trl.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_grpo_verl_vllm.sh
DRY_RUN=1 SKIP_PREFLIGHT=1 bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

Run the read-only remote preflight on the remote server:

```bash
bash train_scripts/remote/preflight_remote.sh all
```

## Reference Server Identity

The reference scripts indicate a remote user and project layout:

```text
Remote user home: /export/home/asifali
Reference project: /export/home/asifali/Noise_math_data
Shared log root: /export/home/asifali/Noise_math_data/all_logs
HF cache root: /export/home/asifali/HF_cache
Default 0.5B model: /export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct
Default 1.5B fallback model: /export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct
```

For ChainGSM, use the same pattern but point `REMOTE_ROOT` to the remote
checkout of this repository:

```bash
export REMOTE_ROOT=/export/home/asifali/math-chain
export REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct
export REMOTE_DATA_DIR=${REMOTE_ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946
export REMOTE_OUTPUT_DIR=${REMOTE_ROOT}/outputs/train/remote
export REMOTE_EVAL_DATA_PATH=${REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
export REMOTE_REWARD_PATH=${REMOTE_ROOT}/train_pipeline/reward_chaingsm.py
export SLURM_LOG_DIR=${REMOTE_OUTPUT_DIR}/slurm_logs
```

## SLURM Resource Baseline

The reference A100 job scripts use:

```bash
#SBATCH -J MTMT-A100
#SBATCH -p gpu-A100
#SBATCH --gres gpu:4
#SBATCH -c 128
#SBATCH --mem 256GB
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
```

The launcher-level submission uses:

```bash
sbatch \
  --output="${SLURM_LOG_DIR}/%j-%x.out" \
  --error="${SLURM_LOG_DIR}/%j-%x.err" \
  --export="ALL,KEY=value,..." \
  path/to/run_case_or_train_script.sh arg1 arg2 arg3
```

For ChainGSM, the current remote scripts use `sbatch --wrap="$CMD"`. That can
work, but it should be expanded to include the same resource flags when running
on the reference server:

```bash
sbatch \
  --job-name="$JOB_NAME" \
  --partition=gpu-A100 \
  --gres=gpu:4 \
  --cpus-per-task=128 \
  --mem=256GB \
  --account=A100 \
  --qos=a100_qos \
  --output="${SLURM_LOG_DIR}/%j-%x.out" \
  --error="${SLURM_LOG_DIR}/%j-%x.err" \
  --export="ALL,REMOTE_ROOT=${REMOTE_ROOT},REMOTE_MODEL_PATH=${REMOTE_MODEL_PATH},REMOTE_DATA_DIR=${REMOTE_DATA_DIR},REMOTE_OUTPUT_DIR=${REMOTE_OUTPUT_DIR},REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH},REMOTE_EVAL_DATA_PATH=${REMOTE_EVAL_DATA_PATH}" \
  --wrap="$CMD"
```

## Environment Baseline

The reference training script does the following at job start:

```bash
module load cuda12.4/toolkit
nvidia-smi
source activate Reasoning360
```

Required environment variables from the reference:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/export/home/asifali/HF_cache
export HF_DATASETS_CACHE=/export/home/asifali/HF_cache
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
```

Ray and temp directory handling:

```bash
LOCAL_BASE=/var/tmp/$USER/${SLURM_JOB_ID}
export RAY_TMPDIR=$LOCAL_BASE/ray
export TMPDIR=$LOCAL_BASE/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
chmod 700 "$LOCAL_BASE" "$RAY_TMPDIR" "$TMPDIR"
export RAY_DISABLE_DASHBOARD=1
unset RAY_ADDRESS RAY_HEAD_IP RAY_PORT
ray stop -f || true
pkill -9 raylet gcs_server plasma_store dashboard 2>/dev/null || true
ulimit -n 1048576 2>/dev/null || true
```

NCCL settings from the reference:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NVLS_ENABLE=0
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset CUDA_LAUNCH_BLOCKING
unset CUDA_DEVICE_MAX_CONNECTIONS
```

W&B should be configured through environment variables. Do not hard-code API
keys in scripts:

```bash
export WANDB_API_KEY=<set outside the repository>
export WANDB_PROJECT=chaingsm_remote_grpo
export DISABLE_WANDB=0
```

For reproducible non-W&B runs:

```bash
export DISABLE_WANDB=1
```

## Package and Runtime Expectations

The reference server environment is named:

```text
Reasoning360
```

The reference scripts assume the environment can run:

```bash
python3 -m verl.trainer.main_ppo
python3 -m verl.trainer.fsdp_sft_trainer
python3 -c "import torch, transformers, datasets, vllm, verl"
```

The current local ChainGSM environment is documented as `math_chain_verl`, but
the remote reference uses `Reasoning360`. Before submitting ChainGSM remotely,
verify these packages inside the remote environment:

```bash
module load cuda12.4/toolkit
source activate Reasoning360
python3 - <<'PY'
import torch
import transformers
import datasets
import vllm
import verl

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("vllm", vllm.__version__)
print("verl", getattr(verl, "__version__", "unknown"))
PY
```

Also verify SLURM visibility:

```bash
sinfo -p gpu-A100
sacctmgr show assoc user=$USER
```

## ChainGSM Remote Data

Remote training expects preprocessed ChainGSM data:

```text
${REMOTE_DATA_DIR}/sft_train.jsonl
${REMOTE_DATA_DIR}/dpo_train.jsonl
${REMOTE_DATA_DIR}/grpo_train.jsonl
${REMOTE_DATA_DIR}/verl_sft_train.parquet
${REMOTE_DATA_DIR}/verl_grpo_train.parquet
```

Remote evaluation expects:

```text
${REMOTE_EVAL_DATA_PATH}
```

Default project value:

```text
${REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl
```

Preflight check:

```bash
test -f "${REMOTE_DATA_DIR}/verl_sft_train.parquet"
test -f "${REMOTE_DATA_DIR}/verl_grpo_train.parquet"
test -f "${REMOTE_DATA_DIR}/dpo_train.jsonl"
test -f "${REMOTE_EVAL_DATA_PATH}"
```

If data is missing, run preprocessing before remote training:

```bash
cd "${REMOTE_ROOT}"
python3 -m train_pipeline.preprocess_chaingsm \
  --input-path chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl \
  --output-dir "${REMOTE_DATA_DIR}"
```

## ChainGSM Remote Submission Commands

### verl SFT

```bash
cd "${REMOTE_ROOT}"
REMOTE_ROOT="${REMOTE_ROOT}" \
PYTHON=python3 \
REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct \
REMOTE_DATA_DIR="${REMOTE_DATA_DIR}" \
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR}" \
TOTAL_EPOCHS=1 \
JOB_NAME=chaingsm-sft-verl \
bash train_scripts/remote/submit_sft_verl.sh
```

This submits:

```bash
python3 -m verl.trainer.fsdp_sft_trainer \
  --config-dir ${REMOTE_ROOT}/train_configs/remote \
  --config-name sft_verl \
  trainer.total_epochs=${TOTAL_EPOCHS}
```

### verl GRPO with vLLM Rollout

```bash
cd "${REMOTE_ROOT}"
REMOTE_ROOT="${REMOTE_ROOT}" \
PYTHON=python3 \
REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct \
REMOTE_DATA_DIR="${REMOTE_DATA_DIR}" \
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR}" \
REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm.py" \
TOTAL_EPOCHS=1 \
ROLLOUT_N=8 \
TP_SIZE=2 \
JOB_NAME=chaingsm-grpo-verl-vllm \
bash train_scripts/remote/submit_grpo_verl_vllm.sh
```

This submits:

```bash
python3 -m verl.trainer.main_ppo \
  --config-dir ${REMOTE_ROOT}/train_configs/remote \
  --config-name grpo_verl_vllm \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
```

### TRL DPO

```bash
cd "${REMOTE_ROOT}"
REMOTE_ROOT="${REMOTE_ROOT}" \
PYTHON=python3 \
REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct \
REMOTE_DATA_DIR="${REMOTE_DATA_DIR}" \
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR}" \
REMOTE_EVAL_DATA_PATH="${REMOTE_EVAL_DATA_PATH}" \
TOTAL_EPOCHS=1 \
JOB_NAME=chaingsm-dpo-trl \
bash train_scripts/remote/submit_dpo_trl.sh
```

### SFT then GRPO

```bash
cd "${REMOTE_ROOT}"
REMOTE_ROOT="${REMOTE_ROOT}" \
PYTHON=python3 \
REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct \
REMOTE_DATA_DIR="${REMOTE_DATA_DIR}" \
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR}/sft_then_grpo" \
REMOTE_REWARD_PATH="${REMOTE_ROOT}/train_pipeline/reward_chaingsm.py" \
TOTAL_EPOCHS=1 \
GRPO_EPOCHS=1 \
ROLLOUT_N=8 \
TP_SIZE=2 \
JOB_NAME=chaingsm-sft-then-grpo-verl \
bash train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

## Current Remote Config Values

`train_configs/remote/sft_verl.yaml`:

- `data.train_batch_size: 256`
- `data.micro_batch_size_per_gpu: 4`
- `data.max_length: 1024`
- `model.strategy: fsdp2`
- `model.lora_rank: 0`
- `optim.lr: 2.0e-5`
- `trainer.n_gpus_per_node: 4`
- `trainer.logger: [console, wandb]`

`train_configs/remote/grpo_verl_vllm.yaml`:

- `data.train_batch_size: 256`
- `data.val_batch_size: 128`
- `data.max_prompt_length: 768`
- `data.max_response_length: 2048`
- `actor_rollout_ref.rollout.name: vllm`
- `actor_rollout_ref.rollout.tensor_model_parallel_size: 2`
- `actor_rollout_ref.rollout.gpu_memory_utilization: 0.6`
- `actor_rollout_ref.rollout.n: 8`
- `actor_rollout_ref.rollout.temperature: 0.7`
- `actor_rollout_ref.actor.ppo_mini_batch_size: 64`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8`
- `actor_rollout_ref.actor.kl_loss_coef: 0.001`
- `custom_reward_function.name: compute_reward`
- `algorithm.adv_estimator: grpo`
- `trainer.n_gpus_per_node: 4`
- `trainer.logger: [console, wandb]`
- `ray_kwargs.ray_init.num_cpus: 32`

## Reference Noise Math Differences

The reference project uses extra options that ChainGSM does not currently need:

- Case-specific prompt and reward modes: `case_1` through `case_6`.
- `PROMPT_VERSION`, often `case_4` for `case_5` and `case_6`.
- Step-level rubric reward settings such as `step_acc_weight`,
  `require_source_grounding`, and `bad_on_idle_chain`.
- Shared evaluation dataset names such as `global_training_summary.csv`.
- Data preparation through
  `/export/home/asifali/Noise_math_data/scripts_qwen_1_5B/train/prepare_data.sh`.
- Evaluation parquet generation through
  `examples/noise_math/scripts/prepare_test_eval_data.py`.

These are useful as operational references but should not be copied directly
into ChainGSM unless the ChainGSM reward/data format needs those semantics.

## Recommended Next Alignment Work

Before deleting `Noise_math_data-main/`, complete these tasks:

1. Update `train_scripts/remote/*.sh` to accept explicit SLURM resource flags
   and default them to the reference A100 values.
2. Add remote environment setup variables:
   `REMOTE_CONDA_ENV`, `REMOTE_CUDA_MODULE`, `HF_HOME`, `HF_DATASETS_CACHE`,
   `RAY_TMPDIR`, `TMPDIR`, and NCCL settings.
3. Add a preflight script that verifies model, data, reward file, Python imports,
   SLURM partition, GPU count, and CUDA visibility.
4. Replace placeholder defaults such as `/path/to/Qwen2.5-0.5B-Instruct` with
   remote-server-aware defaults or require the user to pass them explicitly.
5. Run one remote smoke job for each required path:
   SFT, GRPO, DPO, and SFT-then-GRPO.
6. Confirm logs, checkpoints, metrics, and W&B runs land in the expected
   locations.
7. Only then remove `Noise_math_data-main/` from this repository.
