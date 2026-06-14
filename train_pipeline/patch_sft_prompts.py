"""Patch sft_train.jsonl: change only 3 places, keep JSON protocol intact.

Input:
  sft_train.jsonl (7,055 rows, variants only, original 4-class set)
  source_augmented_with_traces.jsonl (for extracting natural-language step descriptions from solution_original)

Output:
  sft_train_neutral.jsonl (7,055 rows, only system/description patched)

Changes per advisor feedback (2026-06-08):
  1. system: remove "ignore distractor chains" and "return JSON only"
     -> "You are an expert grade-school math solver. Identify the quantity being
        asked, solve it carefully step by step, and follow the requested output
        format."
  2. user: keep JSON schema, but prepend "First identify the target quantity ...
     Do not include calculations not needed for the target ... Do not add text
     outside the JSON object."
  3. response: keep JSON intact; replace "Compute step X for the correct chain."
     with natural-language descriptions extracted from solution_original sentences
     (mathematical meaning, no mention of "correct chain").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "chaingsm_data" / "data" / "final" / "rl_preprocessed" / "gsm8k_train_balanced_one_variant_14946"
SOURCE_SFT = DATA_DIR / "sft_train.jsonl"
SOURCE_AUG = DATA_DIR / "source_augmented_with_traces.jsonl"
OUTPUT_FILE = DATA_DIR / "sft_train_neutral.jsonl"

# ---------- Neutral prompts ----------

NEUTRAL_SYSTEM = (
    "You are an expert grade-school math solver. Identify the quantity being asked, "
    "solve it carefully step by step, and follow the requested output format."
)

NEUTRAL_USER_PREAMBLE = """Solve the following grade-school math problem.
First identify the target quantity asked by the question. Then write the calculation steps needed to compute that target.
Return one valid JSON object with this schema:
{
  "target": "short description of the quantity being asked",
  "selected_steps": [
    {
      "variable": "short variable name",
      "description": "mathematical meaning of this step",
      "expression": "arithmetic expression",
      "value": "computed value"
    }
  ],
  "final_expression": "arithmetic expression that computes the answer",
  "answer": "final numeric answer"
}
Requirements:
- The JSON must be parseable.
- Use meaningful step descriptions.
- Do not include calculations that are not needed for the target quantity.
- Do not add text outside the JSON object.
Problem:
"""


# ---------- Description extraction from solution_original ----------

def _split_solution_lines(solution: str) -> list[str]:
    out = []
    for raw in solution.split("\n"):
        line = raw.strip()
        if not line or line.startswith("####"):
            continue
        out.append(line)
    return out


def _extract_calc(line: str) -> tuple[str, str, str] | None:
    """Return (description, expression, value) extracted from one solution line.

    Handles:
      "... text ... expr = <<expr=val>>val"
      "... text ... expr = $val"
      "... text ... expr = val"
    The description is the natural-language part BEFORE the final equation.
    """
    # GSM8K calculator pattern
    m = re.search(r"^(.*?)=\s*<<([^>=]+)=([^>]+)>>\s*\$?([\-\d\.,]+)", line)
    if m:
        desc = m.group(1).strip()
        expr = m.group(2).strip()
        val = m.group(4).strip().rstrip(".")
        return desc, expr, val
    # Plain "= val" at end of line
    m = re.search(r"^(.*?)=\s*\$?([\-\d\.,]+)\s*\.?\s*$", line)
    if m:
        head = m.group(1).strip()
        val = m.group(2).strip().rstrip(".")
        # Find the expression right before "= val" in the head
        expr_m = re.search(r"([\d\.\s\+\-\*\/\(\)]+?)\s*$", head)
        expr = expr_m.group(1).strip() if expr_m else ""
        if expr:
            return head[: head.rfind(expr)].strip().rstrip(",").rstrip(), expr, val
        return head, "", val
    return None


def _build_desc_lookup(solution: str) -> dict[tuple[str, str], str]:
    """Map (expr_normalized, val_normalized) -> natural-language description from solution."""
    lookup: dict[tuple[str, str], str] = {}
    for line in _split_solution_lines(solution):
        e = _extract_calc(line)
        if e is None:
            continue
        desc, expr, val = e
        if not desc:
            continue
        key = (expr.replace(" ", ""), val)
        # First description wins (longest usually most informative)
        if key not in lookup or len(lookup[key]) < len(desc):
            lookup[key] = desc
    return lookup


def _normalize_num(s: str) -> str:
    s = s.strip().rstrip(".")
    # treat "10" and "10.0" as same key
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _derive_step_descriptions(gold_trace: list[dict], source_row: dict) -> list[dict]:
    """For each gold_trace step, return a dict with semantic description.

    Falls back to a minimal expression-based description if no natural-language match.
    """
    solution = source_row.get("solution_original", "")
    lookup = _build_desc_lookup(solution)

    result = []
    used = set()
    for step in gold_trace:
        expr_norm = step["expression"].replace(" ", "")
        val_norm = _normalize_num(str(step["value"]))
        desc = None
        # Try a few key variants
        for k_val in {val_norm, val_norm.lstrip("0") or "0"}:
            key = (expr_norm, k_val)
            if key in lookup and lookup[key] not in used:
                desc = lookup[key]
                used.add(desc)
                break
        if desc is None:
            # Fallback: use the variable name as a weak hint
            var = step["variable"].replace("_", " ")
            desc = f"Compute {var}."
        result.append({
            "variable": step["variable"],
            "description": desc,
            "expression": step["expression"],
            "value": str(step["value"]),
        })
    return result


# ---------- Main patching ----------

def _patch_user_prompt(old_user: str) -> str:
    """Replace the old user prompt with neutral preamble; preserve the question text."""
    # The old user ends with "Problem:\n<question>\n" — keep only the question
    m = re.search(r"Problem:\s*\n(.+?)\s*$", old_user, re.DOTALL)
    question = m.group(1).strip() if m else ""
    return NEUTRAL_USER_PREAMBLE + question


def _patch_assistant_response(old_response_str: str, source_row: dict) -> str:
    """Replace 'Compute step X for the correct chain.' descriptions with semantic ones."""
    obj = json.loads(old_response_str)
    if "selected_steps" in obj and obj["selected_steps"]:
        obj["selected_steps"] = _derive_step_descriptions(obj["selected_steps"], source_row)
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    # Load source_augmented_with_traces for description lookup
    source_by_id: dict[str, dict] = {}
    with SOURCE_AUG.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            source_by_id[d["id"]] = d

    n_total = 0
    n_written = 0
    n_missing_source = 0
    cat_counter: dict[str, int] = {}
    desc_lens: list[int] = []

    with SOURCE_SFT.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            d = json.loads(line)
            cat_counter[d.get("category", "unknown")] = cat_counter.get(d.get("category", "unknown"), 0) + 1
            src = source_by_id.get(d["id"])
            if src is None:
                n_missing_source += 1
                continue

            new_messages = [
                {"role": "system", "content": NEUTRAL_SYSTEM},
                {"role": "user", "content": _patch_user_prompt(d["messages"][1]["content"])},
                {"role": "assistant", "content": _patch_assistant_response(d["messages"][2]["content"], src)},
            ]
            # Track description length
            try:
                asst_obj = json.loads(new_messages[2]["content"])
                for s in asst_obj.get("selected_steps", []):
                    desc_lens.append(len(s.get("description", "")))
            except Exception:
                pass

            out = {
                "id": d["id"],
                "base_id": d.get("base_id"),
                "category": d.get("category"),
                "messages": new_messages,
                "prompt": new_messages[1]["content"],
                "response": new_messages[2]["content"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Input rows: {n_total}")
    print(f"Missing source rows: {n_missing_source}")
    print(f"Written: {n_written}")
    print(f"Category distribution: {cat_counter}")
    if desc_lens:
        import statistics
        print(f"Step description length: avg={statistics.mean(desc_lens):.1f} chars, "
              f"median={statistics.median(desc_lens):.0f}, "
              f"min={min(desc_lens)}, max={max(desc_lens)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
