"""Prompt builders for generation and validation."""

from __future__ import annotations

import json


def build_generation_prompt(
    category,
    question,
    solution_original,
    final_answer,
):
    system = (
        "You are a careful dataset generator for grade-school math word problems. "
        "Your job is to add one coherent arithmetic distractor chain to a given GSM8K "
        "problem without changing the original correct answer. The distractor must be "
        "natural, internally computable, and misleading, but it must not contradict the "
        "original problem or change the target quantity. You must also provide structured "
        "chain annotations."
    )

    user = f"""Given the original GSM8K problem, create one distracted version according to the specified category.

Original question:
{question}

Original solution:
{solution_original}

Original final answer:
{final_answer}

Category:
{category}

Category definitions:
- independent_decoy: add a separate arithmetic chain using mostly unrelated entities/items.
- attribute_mismatch: add a chain that may share entities but uses a different attribute or unit.
- path_competition: add a chain that branches from an original entity/quantity but leads to a non-target quantity.
- target_scope_misalignment: add a chain after the target, or in a different temporal/hypothetical/story scope, while the original target remains unchanged.

Rules:
1. Preserve the original problem and its correct answer.
2. The final answer must remain exactly: {final_answer}.
3. Add 1 to 4 natural sentences as the distractor chain.
4. The added distractor chain must be arithmetically coherent and computable.
5. Do not introduce contradictions.
6. Do not make the problem ambiguous.
7. Do not directly tell the solver that the added chain is irrelevant.
8. The generated question should still be a natural grade-school math word problem.
9. Keep the original question's target quantity unchanged.
10. The generated question should include the original problem content plus the added distractor chain.
11. Return JSON only.
12. All required fields must be present.
13. core_chain and distractor_chain must be lists of triples: [source_quantity, target_quantity, operation].
14. Use readable symbolic names such as "Li_age", "Zhang_age", "Jung_age", "Ming_badges".
15. operation should be a compact string such as "*2", "+3", "-5", "/2", "+out0", "aggregate_sum", or a short natural operation label if needed.
16. gold_expression should be a compact expression that computes the original final answer.
17. distractor_expression should be a compact expression that computes the plausible distractor answer.
18. Return one complete JSON object only.
19. Keep string values concise.
20. If the original problem has few named entities, branch from a known quantity such as an item count, rate, subtotal, time amount, or intermediate computed value.
21. Do not add facts that change the original computation conditions, such as changed rates, extra required materials, discounts, boosts, or new required quantities.
22. The generated question's final sentence should ask the original target quantity.

Return exactly this JSON schema:

{{
  "question_distracted": "string",
  "answer": "{final_answer}",
  "core_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "distractor_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "gold_expression": "string",
  "distractor_expression": "string",
  "difficulty_tags": {{
    "entity_overlap": "low | medium | high | unknown",
    "operation_similarity": "low | medium | high | unknown",
    "answer_proximity": "far | near | same | unknown",
    "computational_complexity": "simple | multi_step | aggregate | unknown"
  }}
}}

Important:
- The JSON must not contain markdown.
- The JSON must not contain comments.
- The JSON must be valid and parseable.
- Do not omit any field."""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_validation_prompt(original_question, generated_record):
    system = (
        "You are a strict validator for generated math word problems. Check whether "
        "the generated problem preserves the original answer and whether the added "
        "distractor matches the requested category."
    )

    generated_record_json = json.dumps(generated_record, ensure_ascii=False, indent=2)
    user = f"""Original question:
{original_question}

Generated record:
{generated_record_json}

Please check:
1. Does the generated question preserve the original core problem?
2. Does the correct final answer remain the same?
3. Is the added distractor chain coherent and computable?
4. Does the distractor match the expected category?
5. Is there any contradiction or ambiguity?
6. Are core_chain and distractor_chain reasonable list-of-triples annotations?

Return JSON only:

{{
  "pass": true,
  "answer_unchanged": true,
  "category_correct": true,
  "distractor_coherent": true,
  "chains_reasonable": true,
  "has_contradiction": false,
  "has_ambiguity": false,
  "reason": "short explanation"
}}"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
