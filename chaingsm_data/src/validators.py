"""Automatic structural validation for generated ChainGSM records."""

from __future__ import annotations

from .schemas import CATEGORIES, DIFFICULTY_KEYS, REQUIRED_FINAL_FIELDS

FORBIDDEN_HINTS = ["irrelevant", "unrelated", "not used", "does not affect"]


def _chain_errors(chain, field_name, require_non_empty):
    errors = []
    if not isinstance(chain, list):
        return [f"{field_name} must be a list"]
    if require_non_empty and not chain:
        errors.append(f"{field_name} must be non-empty")
    for index, triple in enumerate(chain):
        if not isinstance(triple, list) or len(triple) != 3:
            errors.append(f"{field_name}[{index}] must be a list of length 3")
        elif any(not isinstance(item, str) or not item.strip() for item in triple):
            errors.append(f"{field_name}[{index}] must contain three non-empty strings")
    return errors


def validate_generated_payload(
    payload,
    expected_category,
    final_answer,
    original_question=None,
):
    """Return a list of validation errors for a raw LLM generated payload."""
    errors = []

    if expected_category not in CATEGORIES:
        errors.append(f"unknown category: {expected_category}")

    if not isinstance(payload, dict):
        return ["payload must be a dict"]

    question_distracted = payload.get("question_distracted")
    if not isinstance(question_distracted, str) or not question_distracted.strip():
        errors.append("question_distracted must be a non-empty string")

    if payload.get("answer") != final_answer:
        errors.append("answer must equal final_answer")

    errors.extend(_chain_errors(payload.get("core_chain"), "core_chain", False))
    errors.extend(_chain_errors(payload.get("distractor_chain"), "distractor_chain", True))

    if not isinstance(payload.get("gold_expression"), str) or not payload["gold_expression"].strip():
        errors.append("gold_expression must be non-empty")

    if (
        not isinstance(payload.get("distractor_expression"), str)
        or not payload["distractor_expression"].strip()
    ):
        errors.append("distractor_expression must be non-empty")

    difficulty_tags = payload.get("difficulty_tags")
    if not isinstance(difficulty_tags, dict):
        errors.append("difficulty_tags must be a dict")
    else:
        for key in DIFFICULTY_KEYS:
            if key not in difficulty_tags:
                errors.append(f"difficulty_tags missing {key}")

    if original_question is not None and question_distracted == original_question:
        errors.append("generated question must differ from original question")

    if isinstance(question_distracted, str) and "####" in question_distracted:
        errors.append("generated question must not contain ####")

    text_forbidden_check = [question_distracted or ""]
    for field_name in ["core_chain", "distractor_chain"]:
        chain = payload.get(field_name)
        if isinstance(chain, list):
            for triple in chain:
                if isinstance(triple, list):
                    text_forbidden_check.extend(str(item) for item in triple)
    lowered = "\n".join(text_forbidden_check).lower()
    for phrase in FORBIDDEN_HINTS:
        if phrase in lowered:
            errors.append(f"generated output must not explicitly say '{phrase}'")

    return errors


def validate_final_record(record):
    """Return a list of validation errors for the final JSONL record shape."""
    if not isinstance(record, dict):
        return ["record must be a dict"]

    errors = []
    for field in REQUIRED_FINAL_FIELDS:
        if field not in record:
            errors.append(f"missing field: {field}")

    if "category" in record and record["category"] not in ["original", *CATEGORIES]:
        errors.append(f"unknown category: {record['category']}")

    if "difficulty_tags" in record and not isinstance(record["difficulty_tags"], dict):
        errors.append("difficulty_tags must be a dict")

    if "core_chain" in record:
        errors.extend(_chain_errors(record["core_chain"], "core_chain", False))
    if "distractor_chain" in record:
        require_non_empty = record.get("category") != "original"
        errors.extend(_chain_errors(record["distractor_chain"], "distractor_chain", require_non_empty))

    return errors
