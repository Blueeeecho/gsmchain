# ChainGSM Cross-Model Baseline Evaluation Design

## Goal

Evaluate five unmodified base Instruct checkpoints on the full ChainGSM test set
with the already validated 8-shot CoT protocol, while preserving the prompt
requirements of each model family.

Models:

- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-Math-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Llama-3.2-1B-Instruct

No SFT, DPO, or GRPO checkpoint is included in this baseline run.

## Evaluation Contract

All models use:

- Full ChainGSM test set: 6,575 examples.
- The same eight GSM8K CoT demonstrations.
- Greedy decoding with temperature 0, top-p 1.0, and at most 512 new tokens.
- The same numeric answer normalization and correctness comparison.
- Per-example predictions, overall accuracy, and accuracy by ChainGSM category.

The semantic task and demonstrations are shared. Model-specific serialization is
not forced to be identical because doing so would disadvantage Llama.

## Qwen Prompt Path

Qwen models use the validated single chat turn:

1. A system message describing step-by-step mathematical problem solving.
2. One user message containing eight question-answer demonstrations followed by
   the target question.
3. The tokenizer's native chat template with a generation prompt.

The rendered string is passed to vLLM. This matches the Qwen baseline path that
reproduced the official GSM8K result.

## Llama Prompt Path

Llama uses the validated lm-eval-compatible multi-turn representation:

1. Eight pairs of user demonstration and assistant answer messages.
2. One final user message containing the target problem.
3. The tokenizer's native chat template with a generation prompt.
4. Token IDs are passed directly to vLLM.

The evaluator must assert that the resulting token IDs contain exactly one BOS
token. Passing the rendered Llama prompt back through vLLM string tokenization is
not allowed because it previously produced duplicate BOS tokens.

Llama generation stops at the model end-of-turn markers, a generated next-user
header, `Q:`, or `</s>`.

## Outputs

One timestamped run directory contains:

- `run_config.json`: model paths, prompt profile, dataset, and generation options.
- `model_outputs/<model>/<profile>/predictions.jsonl`: all example outputs.
- `summary_overall.json` and CSV.
- `summary_by_category.json` and CSV.
- `errors.jsonl` files for model loading, generation, or scoring failures.

The summary records the model-specific prompt profile so results cannot be
mistaken for a single shared serialization.

## Validation

Automated tests cover:

- Qwen uses a rendered chat string.
- Llama uses multi-turn messages and token IDs.
- Llama input contains one BOS token.
- The expected model family selects the expected prompt profile.
- Stop sequences are applied only where required.
- Existing extraction and category summaries remain unchanged.

A smoke test runs a small ChainGSM subset for Qwen and Llama before the full
five-model evaluation.

## Success Criteria

The run is complete only when:

- Each model has exactly 6,575 prediction records.
- No model-level generation error is present.
- Overall and all five category accuracies are available for every model.
- The saved run configuration identifies the selected prompt profile.
- Llama's saved tokenization diagnostics confirm one BOS token.

