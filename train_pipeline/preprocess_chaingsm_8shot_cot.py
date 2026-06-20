"""Preprocess ChainGSM JSONL for 8-shot CoT GRPO training.

与 preprocess_chaingsm.py 区别:
  - 训练 prompt 改成 8-shot CoT 完整模板 (8 examples + system)
  - 跟评测 prompt 完全一致 (EIGHT_SHOT_EXAMPLES 同源)
  - 输出 verl_grpo_train_8shot_cot.parquet
  - 训练 prompt tokenize 后, MAX_PROMPT_LENGTH 1024
  - 不需要 selected_steps / final_expression / JSON schema

用法:
  /home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
  train_pipeline/preprocess_chaingsm_8shot_cot.py \
  --input chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl \
  --output-dir chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

# 复用 code/eval_chaingsm_base_8shot.py 的 8-shot examples
from eval_chaingsm_base_8shot import EIGHT_SHOT_EXAMPLES, EIGHT_SHOT_SYSTEM_PROMPT  # noqa: E402


def build_8shot_messages(question: str) -> list[dict[str, str]]:
    """复用评测 prompt 模板, 保证训练-评测 prompt 100% 对齐"""
    messages = [{"role": "system", "content": EIGHT_SHOT_SYSTEM_PROMPT}]
    for ex_q, ex_a in EIGHT_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Q: {ex_q}\nA:"})
        messages.append({"role": "assistant", "content": ex_a})
    messages.append({"role": "user", "content": f"Q: {question}\nA: Let's think step by step."})
    return messages


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """用 tokenizer 的 chat_template 渲染成字符串 (跟评测完全一致)"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def to_verl_grpo_8shot_row(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    """输出 verl GRPO row: prompt 字段 = messages 列表 (verl 会用 tokenizer apply_chat_template)"""
    question = record.get("question_distracted") or record.get("question_original") or ""
    messages = build_8shot_messages(question.strip())
    return {
        "data_source": "chaingsm_8shot_cot",
        "prompt": messages,  # verl 期望 messages 列表
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "gold_answer": str(record.get("answer", "")),
                "gold_expression": str(record.get("gold_expression", "")),
                "core_chain": record.get("core_chain") or [],
                "distractor_chain": record.get("distractor_chain") or [],
                "question": (record.get("question_distracted") or record.get("question_original") or "").strip(),
                "category": str(record.get("category", "")),
            },
        },
        "extra_info": {
            "split": split,
            "index": index,
            "id": record.get("id", f"row_{index}"),
            "category": record.get("category", ""),
            "base_id": record.get("base_id", ""),
        },
    }


def read_jsonl(path: str | os.PathLike[str], max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if pd is None:
        raise RuntimeError("pandas is required to write parquet files")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ChainGSM JSONL for 8-shot CoT GRPO")
    parser.add_argument(
        "--input",
        default=ROOT / "chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokenizer", default="/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct",
                        help="Tokenizer 路径, 用于验证 prompt 长度 (默认 Qwen2.5-0.5B-Instruct)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始数据
    raw_rows = read_jsonl(args.input, args.max_samples)
    print(f"Loaded {len(raw_rows)} rows from {args.input}")

    # 转成 verl GRPO row
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(raw_rows):
        try:
            rows.append(to_verl_grpo_8shot_row(record, "train", idx))
        except Exception as exc:
            print(f"WARN: skip row {idx}: {exc}")

    # 写 parquet
    out_path = output_dir / "verl_grpo_train_8shot_cot.parquet"
    write_parquet(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # 用 tokenizer 验证 prompt 长度分布
    if args.tokenizer and os.path.isdir(args.tokenizer):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        prompt_lens = []
        for row in rows[:100]:  # 抽检前 100 条
            text = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            tokens = tok(text, return_tensors=None)["input_ids"]
            prompt_lens.append(len(tokens))
        if prompt_lens:
            avg = sum(prompt_lens) / len(prompt_lens)
            print(f"Prompt length (前 100 条): min={min(prompt_lens)} max={max(prompt_lens)} avg={avg:.0f}")
            if max(prompt_lens) > 1024:
                print(f"WARN: max prompt length {max(prompt_lens)} > 1024, 需要调 MAX_PROMPT_LENGTH")
            else:
                print(f"OK: max prompt length {max(prompt_lens)} <= 1024 (MAX_PROMPT_LENGTH=1024 够用)")

    # 写 stats
    stats = {
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "input_count": len(raw_rows),
        "output_count": len(rows),
        "output_parquet": str(out_path),
        "protocol": "8shot_cot",
    }
    with open(output_dir / "preprocess_8shot_cot_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
