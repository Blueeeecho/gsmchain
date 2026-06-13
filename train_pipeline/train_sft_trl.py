from __future__ import annotations

import json
import shutil

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

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


def load_rows(path: str, max_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # TRL >= 0.26 treats datasets with a "prompt" column as prompt-completion
                # datasets and expects the target column to be named "completion".
                rows.append(
                    {
                        "id": row.get("id"),
                        "prompt": row["prompt"],
                        "completion": row.get("completion") or row.get("response", ""),
                    }
                )
                if max_samples and len(rows) >= max_samples:
                    break
    return rows


def main() -> None:
    parser = config_arg_parser("Train ChainGSM SFT with TRL.")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    config = apply_common_cli_overrides(load_config_with_overrides(args.config, args.overrides), args)
    run_dir = prepare_run_dir(config)
    setup_run_logging(run_dir, "sft")
    model_path = resolve_path(config["model"]["model_name_or_path"])
    train_file = resolve_path(config["data"]["train_file"])
    rows = load_rows(train_file, config["data"].get("max_samples"))
    print(f"SFT run dir: {run_dir}", flush=True)
    print(f"SFT model: {model_path}", flush=True)
    print(f"SFT train file: {train_file}", flush=True)
    print(f"SFT train examples: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=config["model"].get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(rows)
    training_cfg = dict(config["training"])
    training_cfg["output_dir"] = str(run_dir / "checkpoints")
    training_cfg.setdefault("logging_dir", str(run_dir / "logs"))
    sft_args = SFTConfig(**accepted_kwargs(SFTConfig.__init__, training_cfg))
    eval_callback = EpochVLLMEvalCallback(run_dir, config.get("eval", {}), tokenizer)

    peft_config = None
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg.get("target_modules", "all-linear"),
        )

    trainer = SFTTrainer(
        model=model_path,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
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
    print(f"SFT finished. Output: {run_dir}")


if __name__ == "__main__":
    main()
