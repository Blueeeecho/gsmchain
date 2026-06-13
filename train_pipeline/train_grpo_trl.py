from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Sequence

from datasets import Dataset
from transformers import AutoTokenizer

from train_pipeline.config_utils import (
    accepted_kwargs,
    apply_common_cli_overrides,
    config_arg_parser,
    load_config_with_overrides,
    prepare_run_dir,
    resolve_path,
    setup_run_logging,
)
from train_pipeline.eval_callback import EpochVLLMEvalCallback
from train_pipeline.reward_chaingsm import score_response


SUPPORTED_TRL_VLLM = {"0.10.2", "0.11.0", "0.11.1", "0.11.2"}

# vLLM >= 0.12.0 renamed GuidedDecodingParams to StructuredOutputsParams.
# TRL >= 1.0 no longer imports GuidedDecodingParams, so this patch is
# only needed when running with TRL 0.26.x + vLLM >= 0.12.0 (the old
# math-noise environment).  It is harmless under TRL >= 1.0.
VLLM_PATCHED_VERSIONS = {
    "0.12.0", "0.13.0", "0.14.0", "0.14.1",
    "0.15.0", "0.15.1", "0.16.0", "0.17.0", "0.17.1",
    "0.18.0", "0.18.1", "0.19.0", "0.19.1",
    "0.20.0", "0.20.1", "0.20.2", "0.21.0",
}


def patch_vllm_for_trl() -> None:
    """Patch vLLM compatibility for TRL 0.26.x.

    TRL 0.26.x imports ``GuidedDecodingParams`` from ``vllm.sampling_params``,
    but vLLM >= 0.12.0 renamed this class to ``StructuredOutputsParams``.
    This function aliases the new name back to the old one so TRL can import
    successfully.  The constructor signature is a superset (``regex`` param
    exists in both), so the patch is forward-compatible.
    """
    try:
        import vllm.sampling_params as sp
    except ImportError:
        return  # vLLM not installed; TRL will fall back to non-vLLM mode
    if not hasattr(sp, "GuidedDecodingParams") and hasattr(sp, "StructuredOutputsParams"):
        sp.GuidedDecodingParams = sp.StructuredOutputsParams
        print("[patch] Aliased vllm.sampling_params.GuidedDecodingParams "
              "-> StructuredOutputsParams (vLLM >= 0.12.0 compat)", flush=True)


def check_grpo_environment() -> None:
    try:
        import importlib.metadata as md

        trl_version = md.version("trl")
        vllm_version = md.version("vllm")
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect TRL/vLLM versions: {exc}") from exc
    if vllm_version in SUPPORTED_TRL_VLLM:
        return  # natively supported (TRL 0.26.x + old vLLM)
    if vllm_version in VLLM_PATCHED_VERSIONS:
        patch_vllm_for_trl()
        return  # patched for TRL 0.26.x compat; harmless under TRL >= 1.0
    # TRL >= 1.0 does not import GuidedDecodingParams, so newer vLLM
    # versions should work without patching.
    major_vllm = tuple(int(x) for x in vllm_version.split(".")[:2])
    if major_vllm >= (0, 22):
        print(f"[info] vLLM {vllm_version} + TRL {trl_version}: no patch needed "
              "(TRL >= 1.0 does not import GuidedDecodingParams)")
        return
    raise RuntimeError(
        "Local TRL GRPO environment is not compatible: "
        f"trl={trl_version}, vllm={vllm_version}. "
        "TRL 0.26.x expects vLLM in {0.10.2, 0.11.0, 0.11.1, 0.11.2}; "
        "newer versions (0.12.0–0.21.0) require a compat patch. "
        "Please install a supported vLLM version or update the patch list."
    )


def load_rows(path: str, max_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append({"prompt": row["prompt"], "reward_reference": row["reward_reference"]})
                if max_samples and len(rows) >= max_samples:
                    break
    return rows


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def make_reward_fn(reward_config: dict):
    def _reward(completions, reward_reference=None, **kwargs):
        refs = reward_reference or [{} for _ in completions]
        rewards = []
        for completion, ref in zip(completions, refs):
            reward, _ = score_response(completion_to_text(completion), ref or {}, **reward_config)
            rewards.append(reward)
        return rewards

    return _reward


def main() -> None:
    parser = config_arg_parser("Train ChainGSM GRPO with TRL.")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-env-check", action="store_true")
    args = parser.parse_args()

    config = apply_common_cli_overrides(load_config_with_overrides(args.config, args.overrides), args)
    run_dir = prepare_run_dir(config)
    setup_run_logging(run_dir, "grpo")

    if args.skip_env_check:
        patch_vllm_for_trl()  # always patch when skipping env check
    else:
        check_grpo_environment()  # patches inside if needed

    trl_mod = importlib.import_module("trl")
    GRPOConfig = getattr(trl_mod, "GRPOConfig")
    GRPOTrainer = getattr(trl_mod, "GRPOTrainer")

    model_path = resolve_path(config["model"]["model_name_or_path"])
    train_file = resolve_path(config["data"]["train_file"])
    rows = load_rows(train_file, config["data"].get("max_samples"))
    dataset = Dataset.from_list(rows)
    print(f"GRPO run dir: {run_dir}", flush=True)
    print(f"GRPO model: {model_path}", flush=True)
    print(f"GRPO train file: {train_file}", flush=True)
    print(f"GRPO train examples: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config["model"].get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_cfg = dict(config["training"])
    training_cfg["output_dir"] = str(run_dir / "checkpoints")
    training_cfg.setdefault("logging_dir", str(run_dir / "logs"))
    grpo_args = GRPOConfig(**accepted_kwargs(GRPOConfig.__init__, training_cfg))
    eval_callback = EpochVLLMEvalCallback(run_dir, config.get("eval", {}), tokenizer)

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=make_reward_fn(config.get("reward", {})),
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )
    trainer.train()
    current_dir = run_dir / "checkpoints" / "current"
    best_dir = run_dir / "checkpoints" / "best"
    if not current_dir.exists():
        trainer.save_model(str(current_dir))
        tokenizer.save_pretrained(str(current_dir))
    if not best_dir.exists():
        shutil.copytree(current_dir, best_dir)
    with open(run_dir / "metrics" / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
    print(f"GRPO finished. Output: {run_dir}")


if __name__ == "__main__":
    main()
