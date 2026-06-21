"""Build a single unified training jsonl by merging raw + supplementary.

Output: chaingsm_data/data/final/sft/gsm8k_train_unified_6102.jsonl

设计 (2026-06-21):
- 跟评测集 gsm8k_test_clean.jsonl (5467 = 3051 original + 3051 distracted) 对称
- 每条变体配一条原题 (×2 训练): 变体用 question_distracted 字段存 distracted 题,
  原题用 question_distracted 字段存 original 题, category=original 区分
- supp 中 4019 行 (含 968 supp-only 无原题) -> 实际生成 变体 3051 + 原题 3051 = 6102 行
  (968 supp-only 无 raw_row, 变体行直接跳过, 跟原题行 3051 完美配对)

字段 (跟之前 4019 jsonl 一致, 保持命名不变):
- id, base_id, source_index, category, variant_type
- question_distracted (这是 build 脚本读的字段, 变体行=distracted 题, 原题行=original 题)
- question_original (raw 字段, 仅作参考)
- answer, solution_original
- core_chain, distractor_chain (原题行: distractor_chain=None / distractor_trace=None)
- gold_expression, distractor_expression (原题行: distractor_expression="")
- difficulty_tags, metadata
- gold_trace, distractor_trace (原题行: distractor_trace=None)
- supp_prompt, supp_messages

行 id 区分:
- 变体行: "{raw_id}_vdistracted"
- 原题行: "{raw_id}_voriginal"
- 互不冲突, build 脚本过滤后保留 变体 3051 + 原题 3051 = 6102
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/wwq416/snap/wwq/math-chain")
RAW = ROOT / "chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946_clean.jsonl"
SUPP = ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/grpo_train_clean_v2.jsonl"
DST = ROOT / "chaingsm_data/data/final/sft/gsm8k_train_unified_6102.jsonl"

ID_SUFFIX_TO_CATEGORY = {
    "independent_decoy": "independent_decoy",
    "attribute_mismatch": "attribute_mismatch",
    "path_competition": "path_competition",
    "numerical_irrelevance": "numerical_irrelevance",
    "scope_isolation": "scope_isolation",
    "target_scope_misalignment": "target_scope_misalignment",
}


def extract_question_distracted_from_supp_prompt(prompt: str) -> str:
    m = re.search(r"Problem:\s*\n(.*?)\s*$", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def category_from_id(sid: str) -> str | None:
    parts = sid.split("_")
    if len(parts) < 4:
        return None
    suffix = "_".join(parts[3:])
    return ID_SUFFIX_TO_CATEGORY.get(suffix, suffix)


def make_row(sid: str, raw_row: dict | None, supp_row: dict, variant_type: str) -> dict:
    """variant_type: 'distracted' 或 'original'

    关键: 保持 question_distracted 字段名不变.
    - 变体行: question_distracted = raw.question_distracted (或从 supp.prompt 提)
    - 原题行: question_distracted = raw.question_original (保持字段名, 值换成原题)
    """
    rr = supp_row.get("reward_reference", {})
    raw_question_distracted = (raw_row or {}).get("question_distracted")
    raw_question_original = (raw_row or {}).get("question_original")

    # supp-only 变体 (无 raw_row): 不生成变体行, 训练时只保留 3051 变体 + 3051 原题 = 6102
    if variant_type == "distracted" and raw_row is None:
        return None

    if variant_type == "original":
        # 原题行: question_distracted 字段存原始题 (没分心), category=original, 无分心链
        if not raw_question_original:
            # raw 没有, supp-only: 不生成原题行 (没法拿原题文本)
            return None
        question_distracted_value = raw_question_original
        category = "original"
        distractor_chain = None
        distractor_expression = ""
        distractor_trace = None
    else:
        # 变体行: question_distracted 字段存 distracted 题
        question_distracted_value = raw_question_distracted
        if not question_distracted_value:
            question_distracted_value = extract_question_distracted_from_supp_prompt(supp_row.get("prompt", ""))
        category = (raw_row or {}).get("category") or category_from_id(sid)
        distractor_chain = (raw_row or {}).get("distractor_chain")
        distractor_expression = (raw_row or {}).get("distractor_expression") or rr.get("distractor_expression") or ""
        distractor_trace = rr.get("distractor_trace")

    return {
        "id": f"{sid}_v{variant_type}",
        "raw_id": sid,
        "base_id": (raw_row or {}).get("base_id"),
        "source_index": (raw_row or {}).get("source_index"),
        "category": category,
        "variant_type": variant_type,
        "question_distracted": question_distracted_value,
        "question_original": raw_question_original,
        "answer": (raw_row or {}).get("answer") or rr.get("gold_answer"),
        "solution_original": (raw_row or {}).get("solution_original"),
        "core_chain": (raw_row or {}).get("core_chain"),
        "distractor_chain": distractor_chain,
        "gold_expression": (raw_row or {}).get("gold_expression") or rr.get("gold_expression"),
        "distractor_expression": distractor_expression,
        "difficulty_tags": (raw_row or {}).get("difficulty_tags"),
        "metadata": (raw_row or {}).get("metadata"),
        "gold_trace": rr.get("gold_trace"),
        "distractor_trace": distractor_trace,
        "supp_prompt": supp_row.get("prompt"),
        "supp_messages": supp_row.get("messages"),
    }


def main() -> None:
    raw_by_id: dict[str, dict] = {}
    with RAW.open() as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            raw_by_id[d["id"]] = d
    print(f"loaded raw: {len(raw_by_id)} rows", flush=True)

    supp_by_id: dict[str, dict] = {}
    with SUPP.open() as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            supp_by_id[d["id"]] = d
    print(f"loaded supp: {len(supp_by_id)} rows", flush=True)

    DST.parent.mkdir(parents=True, exist_ok=True)
    n_distracted = n_original = n_supp_only_skip = 0
    with DST.open("w") as fout:
        # 遍历 supp 完整集 (4019), 每行同时生成 变体 + 原题 (如果 raw 有原题)
        for sid, supp_row in supp_by_id.items():
            raw_row = raw_by_id.get(sid)

            # 变体行
            row_d = make_row(sid, raw_row, supp_row, "distracted")
            if row_d is None:
                n_supp_only_skip += 1
            else:
                fout.write(json.dumps(row_d, ensure_ascii=False) + "\n")
                n_distracted += 1

            # 原题行 (必须有 raw.question_original 才有, supp-only 没原题就跳过)
            row_o = make_row(sid, raw_row, supp_row, "original")
            if row_o is None:
                n_supp_only_skip += 1
            else:
                fout.write(json.dumps(row_o, ensure_ascii=False) + "\n")
                n_original += 1

    print(f"wrote {n_distracted + n_original} rows to {DST}", flush=True)
    print(f"  变体 (distracted): {n_distracted}", flush=True)
    print(f"  原题 (original):   {n_original}", flush=True)
    print(f"  supp-only 跳过 (无原题): {n_supp_only_skip}", flush=True)


if __name__ == "__main__":
    main()
