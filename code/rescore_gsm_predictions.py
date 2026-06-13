#!/usr/bin/env python3
"""Re-score GSM prediction JSONL files with the shared answer extractor."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from gsm_answer_extractor import extract_answer, is_correct


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    for row in rows:
        row_correct = int(bool(row["correct"]))
        correct += row_correct
        stats = grouped[str(row.get("category", "unknown"))]
        stats["correct"] += row_correct
        stats["total"] += 1

    total = len(rows)
    by_category = []
    for category, stats in sorted(grouped.items()):
        category_total = stats["total"]
        by_category.append(
            {
                "category": category,
                "correct": stats["correct"],
                "total": category_total,
                "accuracy": stats["correct"] / category_total if category_total else 0.0,
            }
        )
    return {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "accuracy_percent": 100.0 * correct / total if total else 0.0,
        },
        "by_category": by_category,
    }


def rescore_prediction_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    for row in rows:
        pred_answer = extract_answer(row.get("raw_output"))
        row["pred_answer"] = pred_answer
        row["correct"] = is_correct(pred_answer, str(row["gold_answer"]))
    write_jsonl_atomic(path, rows)
    summary = summarize_rows(rows)
    write_json(path.parent / "summary.json", summary)
    return summary


def rescore_run(run_dir: Path) -> None:
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for predictions_path in sorted((run_dir / "model_outputs").glob("*/predictions.jsonl")):
        summary = rescore_prediction_file(predictions_path)
        rows = read_jsonl(predictions_path)
        if not rows:
            continue
        first = rows[0]
        identity = {
            "model_name": first.get("model_name", predictions_path.parent.name),
            "model_path": first.get("model_path", ""),
            "prompt_profile": first.get("prompt_profile", first.get("method", "")),
        }
        overall_rows.append({**identity, **summary["overall"]})
        for category in summary["by_category"]:
            category_rows.append({**identity, **category})

    write_json(run_dir / "summary_overall.json", overall_rows)
    write_json(run_dir / "summary_by_category.json", category_rows)
    write_csv(run_dir / "summary_overall.csv", overall_rows)
    write_csv(run_dir / "summary_by_category.csv", category_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rescore_run(args.run_dir)
    print(f"Rescored results written to {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
