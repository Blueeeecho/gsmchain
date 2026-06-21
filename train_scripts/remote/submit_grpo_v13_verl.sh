#!/usr/bin/env bash
# submit_grpo_v13_verl.sh
#
# Thin back-compat wrapper: forwards to submit_grpo_verl_vllm.sh with
# VERSION=v13 pre-set. All real logic lives in submit_grpo_verl_vllm.sh.
#
# Prefer calling submit_grpo_verl_vllm.sh directly with --version v13 (or
# editing its USER CONFIG block) — this wrapper exists so existing runbooks
# and docs that reference the per-version filenames keep working.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/submit_grpo_verl_vllm.sh" --version "v13" "$@"
