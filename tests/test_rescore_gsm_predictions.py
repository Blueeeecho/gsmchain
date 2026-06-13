from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from rescore_gsm_predictions import rescore_prediction_file, rescore_run  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rescore_prediction_file_preserves_raw_output_and_updates_score(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    raw_output = "So, Terry spends $75.00 on yogurt over 30 days."
    write_jsonl(
        predictions,
        [
            {
                "id": "one",
                "category": "original",
                "gold_answer": "75",
                "raw_output": raw_output,
                "pred_answer": "30",
                "correct": False,
            }
        ],
    )

    summary = rescore_prediction_file(predictions)

    row = json.loads(predictions.read_text(encoding="utf-8"))
    assert row["raw_output"] == raw_output
    assert row["pred_answer"] == "75.00"
    assert row["correct"] is True
    assert summary["overall"]["correct"] == 1


def test_rescore_run_rebuilds_model_and_combined_summaries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "model_outputs" / "model-a" / "predictions.jsonl",
        [
            {
                "id": "one",
                "model_name": "Model A",
                "model_path": "/models/a",
                "prompt_profile": "profile-a",
                "category": "original",
                "gold_answer": "4",
                "raw_output": "If they have 24 slices, they last 24 / 6 = 4 days.",
                "pred_answer": "6",
                "correct": False,
            }
        ],
    )

    rescore_run(run_dir)

    model_summary = json.loads(
        (run_dir / "model_outputs" / "model-a" / "summary.json").read_text(encoding="utf-8")
    )
    combined = json.loads((run_dir / "summary_overall.json").read_text(encoding="utf-8"))
    assert model_summary["overall"]["accuracy_percent"] == 100.0
    assert combined[0]["model_name"] == "Model A"
    assert combined[0]["correct"] == 1
