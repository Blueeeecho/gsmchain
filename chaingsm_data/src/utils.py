"""Utility functions for JSONL, answer parsing, and deterministic selection."""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset


FINAL_ANSWER_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d+)?|\d+\s*/\s*[-+]?\d+)$"
)


def parse_final_answer(answer_text):
    """Extract the final GSM8K answer after #### as a normalized string."""
    if not isinstance(answer_text, str) or "####" not in answer_text:
        return None

    candidate = answer_text.rsplit("####", 1)[-1].strip().splitlines()[0].strip()
    candidate = candidate.replace(",", "")
    candidate = candidate.replace("$", "").strip()
    candidate = re.sub(r"\s*/\s*", "/", candidate)

    if FINAL_ANSWER_RE.match(candidate):
        return candidate
    return None


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path, record, lock):
    ensure_parent(path)
    with lock:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def existing_ids(path):
    return {record.get("id") for record in read_jsonl(path) if record.get("id")}


def sanitize_name(value):
    value = Path(str(value)).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "run"


def _normalize_sample(row, ordinal, base_id_prefix):
    question = row.get("question")
    answer = row.get("answer")
    if not question or not answer:
        return None

    final_answer = row.get("final_answer") or parse_final_answer(answer)
    if final_answer is None:
        return None

    return {
        "base_id": row.get("base_id") or f"{base_id_prefix}_{ordinal:06d}",
        "source_index": row.get("source_index", ordinal - 1),
        "question": question,
        "answer": answer,
        "final_answer": final_answer,
        **({"category": row["category"]} if row.get("category") else {}),
        **({"failed_id": row["failed_id"]} if row.get("failed_id") else {}),
    }


def _load_source_records(input_path, input_format):
    input_path = Path(input_path)
    resolved_format = input_format
    if resolved_format == "auto":
        if input_path.suffix == ".jsonl":
            resolved_format = "jsonl"
        elif input_path.suffix == ".parquet":
            resolved_format = "parquet"
        else:
            raise ValueError(f"Cannot infer input format from {input_path}")

    if resolved_format == "jsonl":
        return read_jsonl(input_path)
    if resolved_format == "parquet":
        return list(load_dataset("parquet", data_files=str(input_path), split="train"))
    raise ValueError(f"Unsupported input format: {input_format}")


def load_or_select_samples(
    input_path,
    selected_path,
    count=None,
    force=False,
    input_format="auto",
    base_id_prefix="chaingsm_train",
):
    """Reuse selected samples unless forced, otherwise take the first count valid rows."""
    selected_path = Path(selected_path)
    if selected_path.exists() and not force:
        return read_jsonl(selected_path)

    source_records = _load_source_records(input_path, input_format)
    selected = []
    for row_index, row in enumerate(source_records):
        sample = _normalize_sample(row, len(selected) + 1, base_id_prefix)
        if sample is None:
            continue
        if "source_index" not in row:
            sample["source_index"] = row_index
        selected.append(sample)
        if count is not None and len(selected) >= count:
            break

    if count is not None and len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} valid samples, expected {count}.")

    write_jsonl(selected_path, selected)
    return selected


def sort_records(records, category_order):
    order = {category: index for index, category in enumerate(category_order)}
    return sorted(
        records,
        key=lambda r: (r.get("base_id", ""), order.get(r.get("category", ""), 999)),
    )
