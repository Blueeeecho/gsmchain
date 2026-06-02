# Remote Chain Alignment Design

## Overview

This design aligns the current ChainGSM project with the existing remote A100
SLURM server without changing the remote environment. The goal is to make four
remote training paths submit cleanly from this repository:

- verl SFT
- TRL DPO
- verl GRPO with vLLM rollout
- verl SFT followed by verl GRPO

The reference directory `Noise_math_data-main/` stays in the repository until
these paths are validated against the remote server assumptions.

The design also protects the local training path. Local ChainGSM training has
moved to the existing `math_chain_verl` conda environment at
`/home/wwq416/miniconda3/envs/math_chain_verl`; remote alignment must not
reintroduce `math-noise` defaults or weaken the local verl/vLLM setup.

## Goals

- Centralize remote server defaults so the four submission scripts share the
  same paths, SLURM resources, cache locations, and runtime environment.
- Add a read-only preflight check that reports whether the current remote server
  can run each chain.
- Keep the remote environment unchanged. The project may adapt to the server,
  but it must not install packages, alter conda environments, change CUDA
  modules, or modify SLURM configuration.
- Make each submission command inspectable before it is submitted.
- Document any mismatch that cannot be fixed from this repository.
- Keep local scripts and documentation aligned with the current
  `math_chain_verl` environment.

## Non-Goals

- Do not delete `Noise_math_data-main/` in this phase.
- Do not change the remote `Reasoning360` environment.
- Do not require a new CUDA module or package install.
- Do not run full training as the initial validation target.
- Do not rewrite the local training pipeline.
- Do not change the local `math_chain_verl` environment or install packages in
  it as part of this alignment work.

## Remote Server Baseline

The reference scripts indicate this remote shape:

- Home root: `/export/home/asifali`
- Reference project: `/export/home/asifali/Noise_math_data`
- Shared log root: `/export/home/asifali/Noise_math_data/all_logs`
- Hugging Face cache: `/export/home/asifali/HF_cache`
- Default 0.5B model:
  `/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct`
- CUDA module: `cuda12.4/toolkit`
- Conda environment: `Reasoning360`
- SLURM partition: `gpu-A100`
- SLURM account: `A100`
- SLURM qos: `a100_qos`
- Resource baseline: 4 A100 GPUs, 128 CPUs, 256GB memory

These values become configurable defaults in the ChainGSM remote layer.

## Local Environment Baseline

Local training and evaluation should consistently use:

```text
Local conda env: /home/wwq416/miniconda3/envs/math_chain_verl
Local Python: /home/wwq416/miniconda3/envs/math_chain_verl/bin/python
Local verl source: /home/wwq416/snap/wwq/verl_math_chain
Local project root: /home/wwq416/snap/wwq/math-chain
```

Local scripts already expected to use this environment:

```text
train_scripts/local/run_preprocess.sh
train_scripts/local/run_sft.sh
train_scripts/local/run_dpo.sh
train_scripts/local/run_grpo.sh
train_scripts/local/run_grpo_verl.sh
train_scripts/local/run_sft_then_grpo.sh
```

The implementation plan should verify and, where needed, update local docs and
scripts so they do not point users back to `math-noise`. Known stale references
exist in older training/evaluation docs and should be reconciled with the root
README.

## Architecture

### Shared Remote Environment

Create a shared shell module:

```text
train_scripts/remote/remote_env.sh
```

It will define defaults and helper functions for:

- `REMOTE_ROOT`
- `REMOTE_MODEL_PATH`
- `REMOTE_DATA_DIR`
- `REMOTE_OUTPUT_DIR`
- `REMOTE_EVAL_DATA_PATH`
- `REMOTE_REWARD_PATH`
- `REMOTE_CONDA_ENV`
- `REMOTE_CUDA_MODULE`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `SLURM_LOG_DIR`
- SLURM partition/account/qos/GPU/CPU/memory flags
- Ray temp directories
- NCCL settings
- vLLM and transformers environment variables

It will not submit jobs by itself. It only prepares common variables and helper
functions used by the four submit scripts and the preflight script.

### Preflight Script

Create:

```text
train_scripts/remote/preflight_remote.sh
```

The preflight script will be read-only. It checks:

- Required files and directories exist:
  - `REMOTE_ROOT`
  - `REMOTE_MODEL_PATH`
  - `REMOTE_DATA_DIR`
  - `REMOTE_REWARD_PATH`
  - `REMOTE_EVAL_DATA_PATH`
- Chain-specific data exists:
  - SFT: `verl_sft_train.parquet`
  - DPO: `dpo_train.jsonl`
  - GRPO: `verl_grpo_train.parquet`
  - SFT to GRPO: both SFT and GRPO parquet files
- SLURM commands are available:
  - `sbatch`
  - `sinfo`
- The configured partition appears in `sinfo`.
- The configured Python can import the expected packages:
  - `torch`
  - `transformers`
  - `datasets`
  - `vllm`
  - `verl`
- GPU visibility is compatible with the expected GPU count when running inside
  an allocated job or interactive allocation.

It will print PASS/WARN/FAIL rows. FAIL rows block submission by default.

### Submit Scripts

Modify the four existing scripts:

```text
train_scripts/remote/submit_sft_verl.sh
train_scripts/remote/submit_dpo_trl.sh
train_scripts/remote/submit_grpo_verl_vllm.sh
train_scripts/remote/submit_sft_then_grpo_verl_vllm.sh
```

Each script will:

- Source `remote_env.sh`.
- Build a chain-specific command.
- Run preflight unless `SKIP_PREFLIGHT=1`.
- Print the resolved command and exported environment.
- Submit with the shared SLURM resource flags.
- Support `DRY_RUN=1` to print without calling `sbatch`.

### Chain Commands

SFT command:

```bash
python3 -m verl.trainer.fsdp_sft_trainer \
  --config-dir "${REMOTE_ROOT}/train_configs/remote" \
  --config-name sft_verl \
  trainer.total_epochs="${TOTAL_EPOCHS}"
```

DPO command:

```bash
python3 -m train_pipeline.train_dpo_trl \
  --config "${REMOTE_ROOT}/train_configs/remote/dpo_trl.yaml" \
  --set training.num_train_epochs="${TOTAL_EPOCHS}"
```

GRPO command:

```bash
python3 -m verl.trainer.main_ppo \
  --config-dir "${REMOTE_ROOT}/train_configs/remote" \
  --config-name grpo_verl_vllm \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}"
```

SFT to GRPO command:

```bash
python3 -m verl.trainer.fsdp_sft_trainer \
  --config-dir "${REMOTE_ROOT}/train_configs/remote" \
  --config-name sft_verl \
  trainer.default_local_dir="${REMOTE_OUTPUT_DIR}/stage1_sft/checkpoints" \
  trainer.total_epochs="${TOTAL_EPOCHS}" && \
python3 -m verl.trainer.main_ppo \
  --config-dir "${REMOTE_ROOT}/train_configs/remote" \
  --config-name grpo_verl_vllm \
  actor_rollout_ref.model.path="${REMOTE_OUTPUT_DIR}/stage1_sft/checkpoints" \
  trainer.default_local_dir="${REMOTE_OUTPUT_DIR}/stage2_grpo/checkpoints" \
  trainer.total_epochs="${GRPO_EPOCHS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}"
```

The chained path uses the stage 1 SFT checkpoint as the stage 2 GRPO model path.

### Local Script Alignment

Local alignment will stay smaller than remote alignment. It will verify that
local scripts:

- Default `PYTHON` to
  `/home/wwq416/miniconda3/envs/math_chain_verl/bin/python`.
- Preserve local CUDA/vLLM helper variables already required for the RTX 5090
  path:
  - `CUDA_MODULE_LOADING=LAZY`
  - `CUDA_HOME=/home/wwq416/miniconda3/envs/math_chain_verl`
  - `FLASHINFER_CUDA_ARCH_LIST=12.0f`
  - `LD_LIBRARY_PATH=/home/wwq416/miniconda3/envs/math_chain_verl/lib`
- Keep local verl GRPO using
  `/home/wwq416/snap/wwq/verl_math_chain`.
- Keep local output paths under `outputs/train/local/<method>/<model>/<run_name>/<run_id>`.
- Do not inherit remote A100 SLURM settings.

Local documentation should describe `math_chain_verl` as the single current
local environment. Any `math-noise` references should either be removed or
clearly marked as historical if still needed for old result reproduction.

## Error Handling

Because the remote environment is fixed, the scripts must not attempt to repair
remote problems. They should report clear messages instead.

Examples:

- Missing conda environment: report `Reasoning360` is unavailable.
- Missing CUDA module: report `cuda12.4/toolkit` cannot be loaded.
- Missing model path: report the exact `REMOTE_MODEL_PATH`.
- Missing data: report which preprocessing artifact is absent.
- Missing package import: report the module that failed.
- Missing SLURM partition/account/qos: report the configured value and command
  used to check it.

Submission fails on hard preflight failures unless the user explicitly sets
`SKIP_PREFLIGHT=1`.

## Testing Strategy

Local verification:

- Shell syntax check each remote script with `bash -n`.
- Shell syntax check local training scripts with `bash -n`.
- Run `DRY_RUN=1 SKIP_PREFLIGHT=1` for each submit script to verify command
  construction without contacting SLURM.
- Run `preflight_remote.sh --help` or equivalent usage output locally.
- Verify local scripts resolve `PYTHON` to `math_chain_verl` by default without
  launching full training.

Remote verification:

- Run `preflight_remote.sh` on the remote server.
- Run dry-run commands on the remote server.
- Submit one smoke job for each chain with very small epoch/step settings when
  supported by the target trainer.
- Confirm each submitted job writes logs under `SLURM_LOG_DIR`.
- Confirm successful chains produce expected output directories, metrics, and
  checkpoints.

## Success Criteria

- Four submit scripts share one remote configuration layer.
- Four submit scripts support dry-run and preflight.
- Four dry-runs produce commands with the expected A100 SLURM resources.
- Local scripts remain configured for `math_chain_verl`, not `math-noise`.
- Local docs no longer give contradictory current-environment instructions.
- Preflight reports all non-fixable remote mismatches clearly.
- At least the GRPO verl + vLLM chain can be smoke-submitted from the current
  repository once the remote server has the required data/model paths.
- `Noise_math_data-main/` remains available until the four-chain remote path is
  validated.

## Open Constraints

- If the remote `Reasoning360` environment lacks a required package, this
  project will report the issue but will not install anything.
- If SLURM resource names differ from the reference values, the scripts will
  need environment-variable overrides.
- If remote ChainGSM data is not synchronized, preprocessing or data transfer
  must happen outside this alignment script unless explicitly added in a later
  phase.
- If W&B is required, `WANDB_API_KEY` must be provided through the environment;
  it must not be committed to the repository.
