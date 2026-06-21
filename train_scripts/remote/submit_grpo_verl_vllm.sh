#!/bin/bash -l
# submit_grpo_verl_vllm.sh
#
# One-stop SLURM submit for ChainGSM GRPO (verl) training. Lists all four
# prompt / reward variants in one file. The currently-active variant is
# uncommented; the other three are commented out. To run a different variant,
# swap which python line is uncommented (see the comment block above each line).
#
# Variants (uncomment exactly ONE python invocation below to choose):
#
#   variant | --config-name          | training parquet               | epochs | sbatch --job-name
#   --------+------------------------+--------------------------------+--------+----------------------
#   v12     | grpo_verl_v12_vllm     | grpo_v12_json.parquet          |   2    | chaingsm-grpo-verl-v12
#   v12i    | grpo_verl_v12i_vllm    | grpo_v12_json.parquet          |   2    | chaingsm-grpo-verl-v12i
#   v13     | grpo_verl_v13_vllm     | grpo_v13_json.parquet          |   2    | chaingsm-grpo-verl-v13
#   v14     | grpo_verl_v14_vllm     | grpo_v14_reasoning.parquet     |   3    | chaingsm-grpo-verl-v14
#
# v12 and v12i share the same data file (v12 prompt); v14 uses 3 epochs because
# its reasoning prompt is longer than the others.
#
# Before submitting, edit the two paths in section 3 below if your repo /
# model cache live somewhere other than the defaults:
#   - REMOTE_ROOT     (the math-chain repo on the cluster)
#   - REMOTE_MODEL_PATH (the base model checkpoint)
#
# To use this file:
#   1. Edit section 3 paths if needed.
#   2. Comment / uncomment the python line in section 5 for the variant you want.
#   3. Run:  bash submit_grpo_verl_vllm.sh
#
# Companion docs: train_scripts/remote/V12_V14_HANDBOOK.md
#                 train_scripts/remote/V12i_HANDBOOK.md

# ============================================================================
# 1. SLURM resource request
# ============================================================================
#SBATCH -J chaingsm-grpo-verl                # job name (overridden by --job-name if you set one)
#SBATCH -p gpu-all                           # queue / partition
#SBATCH --gres gpu:1                         # GPUs per node
#SBATCH -c 8                                 # CPUs per task
#SBATCH --mem 32GB                           # RAM
#SBATCH --output=./all_logs/%j-%x-slurm.out  # stdout log
#SBATCH --error=./all_logs/%j-%x-slurm.err   # stderr log
#SBATCH --mail-user=asif6827@gmail.com

# ============================================================================
# 2. Environment: CUDA module + HF cache + conda env
# ============================================================================
module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

source activate Reasoning360

# ============================================================================
# 3. Paths — EDIT THESE TWO LINES IF YOUR SETUP DIFFERS
# ============================================================================
REMOTE_ROOT="/home/wwq416/snap/wwq/math-chain"
REMOTE_MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct"
REMOTE_DATA_DIR="${REMOTE_ROOT}/chaingsm_data/data/final/grpo"
REMOTE_OUTPUT_DIR="${REMOTE_ROOT}/outputs/train/remote"

# Reward function file (one per variant — picked below in section 5)
REMOTE_REWARD_PATH_V12="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12_json_verl.py"
REMOTE_REWARD_PATH_V12I="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v12i_json_verl.py"
REMOTE_REWARD_PATH_V13="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v13_json_verl.py"
REMOTE_REWARD_PATH_V14="${REMOTE_ROOT}/train_pipeline/reward_chaingsm_v14_reasoning_verl.py"

# ============================================================================
# 4. Symlink the variant's data parquet to the generic name verl expects.
# Run ONLY the one matching the variant you uncommented in section 5.
# (The verl yaml configs read from verl_grpo_train.parquet.)
# ============================================================================

# V12 / V12i share grpo_v12_json.parquet:
ln -sf ${REMOTE_DATA_DIR}/grpo_v12_json.parquet ${REMOTE_DATA_DIR}/verl_grpo_train.parquet

# V13:
#ln -sf ${REMOTE_DATA_DIR}/grpo_v13_json.parquet ${REMOTE_DATA_DIR}/verl_grpo_train.parquet

# V14:
#ln -sf ${REMOTE_DATA_DIR}/grpo_v14_reasoning.parquet ${REMOTE_DATA_DIR}/verl_grpo_train.parquet

# ============================================================================
# 5. Training command — UNCOMMENT EXACTLY ONE LINE
# Each line is a complete, runnable training invocation for that variant.
# ============================================================================

# --- V12i: 4-component reward, shared V12 prompt, 2 epochs (default / current focus) ---
cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH_V12I} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v12i_vllm actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-v12i/checkpoints trainer.total_epochs=2

# --- V12: 7-component reward (with soft core + invalid_reward), V12 prompt, 2 epochs ---
#cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH_V12} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v12_vllm actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-v12/checkpoints trainer.total_epochs=2

# --- V13: 5-component reward (hard per-step numeric), 5-field prompt, 2 epochs ---
#cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH_V13} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v13_vllm actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-v13/checkpoints trainer.total_epochs=2

# --- V14: 4-component reward, reasoning prompt (long), 3 epochs ---
#cd ${REMOTE_ROOT} && export REMOTE_REWARD_PATH=${REMOTE_REWARD_PATH_V14} && python -m verl.trainer.main_ppo --config-dir ${REMOTE_ROOT}/train_configs/remote --config-name grpo_verl_v14_vllm actor_rollout_ref.model.path=${REMOTE_MODEL_PATH} actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.default_local_dir=${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-v14/checkpoints trainer.total_epochs=3


# ============================================================================
# Notes / testing variants on a different host
# ============================================================================
# REMOTE_ROOT=  REMOTE_MODEL_PATH=  bash submit_grpo_verl_vllm.sh
# (override the two paths from section 3 without editing the file)

nvidia-smi
