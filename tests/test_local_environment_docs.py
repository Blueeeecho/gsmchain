from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTHON = "/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"
LOCAL_SCRIPTS = [
    "train_scripts/local/run_preprocess.sh",
    "train_scripts/local/run_sft.sh",
    "train_scripts/local/run_dpo.sh",
    "train_scripts/local/run_grpo.sh",
    "train_scripts/local/run_grpo_verl.sh",
    "train_scripts/local/run_sft_then_grpo.sh",
]
CURRENT_DOCS = [
    "README.md",
    "train_readme.md",
]  # baseline_readme.md 与 code/README.md 已移到 _archive/,不再是 "current docs"



def test_local_scripts_default_to_math_chain_verl_python() -> None:
    for script in LOCAL_SCRIPTS:
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert LOCAL_PYTHON in text, script


def test_local_scripts_have_valid_bash_syntax() -> None:
    for script in LOCAL_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / script)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script}\n{result.stderr}"


def test_current_docs_do_not_present_math_noise_as_current_env() -> None:
    for doc in CURRENT_DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        assert "/home/wwq416/miniconda3/envs/math-noise/bin/python" not in text, doc
        assert "conda run --no-capture-output -n math-noise" not in text, doc
