"""Shared constants and lightweight schema helpers."""

CATEGORIES = [
    "independent_decoy",
    "attribute_mismatch",
    "path_competition",
    "target_scope_misalignment",
]

CATEGORY_ORDER = [
    "original",
    "independent_decoy",
    "attribute_mismatch",
    "path_competition",
    "target_scope_misalignment",
]

DIFFICULTY_KEYS = [
    "entity_overlap",
    "operation_similarity",
    "answer_proximity",
    "computational_complexity",
]

ORIGINAL_DIFFICULTY_TAGS = {
    "entity_overlap": "none",
    "operation_similarity": "none",
    "answer_proximity": "none",
    "computational_complexity": "none",
}

REQUIRED_FINAL_FIELDS = [
    "id",
    "base_id",
    "source_index",
    "category",
    "question_original",
    "question_distracted",
    "answer",
    "solution_original",
    "core_chain",
    "distractor_chain",
    "gold_expression",
    "distractor_expression",
    "difficulty_tags",
    "metadata",
]


def make_original_record(sample, seed):
    """Build a final-record-shaped original sample."""
    return {
        "id": f"{sample['base_id']}_original",
        "base_id": sample["base_id"],
        "source_index": sample["source_index"],
        "category": "original",
        "question_original": sample["question"],
        "question_distracted": sample["question"],
        "answer": sample["final_answer"],
        "solution_original": sample["answer"],
        "core_chain": [],
        "distractor_chain": [],
        "gold_expression": "",
        "distractor_expression": "",
        "difficulty_tags": dict(ORIGINAL_DIFFICULTY_TAGS),
        "metadata": {
            "generator_model": None,
            "seed": seed,
            "variant_type": "original",
        },
    }
