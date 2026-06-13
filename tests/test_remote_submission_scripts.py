from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_DIR = REPO_ROOT / "train_scripts" / "remote"
REMOTE_ROOT = "/export/home/asifali/math-chain"
REMOTE_DATA_DIR = f"{REMOTE_ROOT}/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946"
REMOTE_OUTPUT_DIR = f"{REMOTE_ROOT}/outputs/train/remote"
REMOTE_MODEL_PATH = "/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct"
REMOTE_EVAL_DATA_PATH = f"{REMOTE_ROOT}/chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl"
REMOTE_REWARD_PATH = f"{REMOTE_ROOT}/train_pipeline/reward_chaingsm.py"


def run_submit_script(script_name: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PREFLIGHT": "1",
            "REMOTE_ROOT": REMOTE_ROOT,
            "REMOTE_MODEL_PATH": REMOTE_MODEL_PATH,
            "REMOTE_DATA_DIR": REMOTE_DATA_DIR,
            "REMOTE_OUTPUT_DIR": REMOTE_OUTPUT_DIR,
            "REMOTE_EVAL_DATA_PATH": REMOTE_EVAL_DATA_PATH,
            "REMOTE_REWARD_PATH": REMOTE_REWARD_PATH,
            "TOTAL_EPOCHS": "1",
            "GRPO_EPOCHS": "1",
            "ROLLOUT_N": "8",
            "TP_SIZE": "2",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(REMOTE_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_common_sbatch_flags(output: str) -> None:
    assert "sbatch" in output
    assert "--partition=gpu-A100" in output
    assert "--gres=gpu:4" in output
    assert "--cpus-per-task=128" in output
    assert "--mem=256GB" in output
    assert "--account=A100" in output
    assert "--qos=a100_qos" in output
    assert "--output=" in output
    assert "--error=" in output
    assert "--export=ALL," in output
    assert "REMOTE_ROOT=/export/home/asifali/math-chain" in output
    assert "REMOTE_MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct" in output


def test_sft_verl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_sft_verl.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.fsdp_sft_trainer" in output
    assert "--config-name sft_verl" in output
    assert "trainer.total_epochs=1" in output


def test_dpo_trl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_dpo_trl.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "train_pipeline.train_dpo_trl" in output
    assert "train_configs/remote/dpo_trl.yaml" in output
    assert "--set training.num_train_epochs=1" in output


def test_grpo_verl_dry_run_uses_remote_env_and_resources() -> None:
    result = run_submit_script("submit_grpo_verl_vllm.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.main_ppo" in output
    assert "--config-name grpo_verl_vllm" in output
    assert "actor_rollout_ref.rollout.n=8" in output
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in output


def test_sft_then_grpo_dry_run_chains_stage_commands() -> None:
    result = run_submit_script("submit_sft_then_grpo_verl_vllm.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.fsdp_sft_trainer" in output
    assert "verl.trainer.main_ppo" in output
    assert "stage1_sft/checkpoints" in output
    assert "stage2_grpo/checkpoints" in output
