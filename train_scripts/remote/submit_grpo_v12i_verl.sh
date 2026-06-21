#!/usr/bin/env bash

#SBATCH -J MTMT-A100 #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:2 #number of gpus needed, default is 1
#SBATCH -c 60  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit
nvidia-smi
source activate Reasoning360

# submit_grpo_v12i_verl.sh
#
# Thin back-compat wrapper: forwards to submit_grpo_verl_vllm.sh with
# VERSION=v12i pre-set. All real logic lives in submit_grpo_verl_vllm.sh.
#
# Prefer calling submit_grpo_verl_vllm.sh directly with --version v12i (or
# editing its USER CONFIG block) — this wrapper exists so existing runbooks
# and docs that reference the per-version filenames keep working.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/submit_grpo_verl_vllm.sh" --version "v12i" "$@"
