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


def test_grpo_verl_vllm_lists_all_four_variants() -> None:
    # New-style flat script: bash -n check + assert each variant is referenced.
    # The 4 python invocations are mutually exclusive (3 are commented out at
    # any time), so we only assert their config-name strings exist as
    # commented or uncommented lines.
    script = REMOTE_DIR / "submit_grpo_verl_vllm.sh"
    assert script.is_file(), script
    # Parse-check: bash -n returns 0 for valid syntax.
    parse = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=REPO_ROOT, env=os.environ.copy(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert parse.returncode == 0, parse.stderr
    body = script.read_text()
    # All four config-names must appear in the file (one per python line).
    for cfg in ("grpo_verl_v12_vllm", "grpo_verl_v12i_vllm",
                "grpo_verl_v13_vllm", "grpo_verl_v14_vllm"):
        assert f"--config-name {cfg}" in body, f"missing {cfg} in script"
    # v12i is the uncommented default; assert it is the active line.
    active = [ln for ln in body.splitlines()
              if ln.startswith("cd ") and "grpo_verl_v12i_vllm" in ln]
    assert active, "expected v12i to be the uncommented default"
    # The other three variants should be present but commented out.
    for cfg in ("grpo_verl_v12_vllm", "grpo_verl_v13_vllm", "grpo_verl_v14_vllm"):
        commented = [ln for ln in body.splitlines()
                     if ln.startswith("#") and f"--config-name {cfg}" in ln]
        assert commented, f"expected {cfg} to be commented out"
    # Symlink block must include the v12/v12i shared parquet and the v13/v14
    # specific parquets.
    for src in ("grpo_v12_json.parquet", "grpo_v13_json.parquet",
                "grpo_v14_reasoning.parquet"):
        assert src in body, f"missing symlink source {src}"


def test_sft_then_grpo_dry_run_chains_stage_commands() -> None:
    result = run_submit_script("submit_sft_then_grpo_verl_vllm.sh")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert_common_sbatch_flags(output)
    assert "verl.trainer.fsdp_sft_trainer" in output
    assert "verl.trainer.main_ppo" in output
    assert "stage1_sft/checkpoints" in output
    assert "stage2_grpo/checkpoints" in output
