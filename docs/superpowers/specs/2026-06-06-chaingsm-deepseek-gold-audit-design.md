# ChainGSM DeepSeek Gold Audit Design

## Goal

Use `deepseek-v4-flash` as an independent strong-model judge to audit every
ChainGSM test group. Determine whether low variant accuracy is caused by model
robustness, incorrect retained answers, ambiguity, or malformed variants.

The audit covers all 1,319 `base_id` groups and all 6,575 currently available
records. It does not regenerate missing variants.

## Input And Grouping

Input:

`chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl`

Records are grouped by `base_id`. Each request contains:

- the Original question;
- every available generated variant in that group;
- the dataset answer retained for the group;
- stable record IDs and category names.

Grouping Original and variants in one request lets the judge compare whether a
variant preserves the intended mathematical target while still solving each
question independently.

## API Configuration

- Base URL: `https://api.deepseek.com`
- Model: `deepseek-v4-flash`
- Thinking: explicitly disabled with
  `{"thinking": {"type": "disabled"}}`
- Temperature: `0`
- Response format: JSON object
- API key source: `DEEPSEEK_API_KEY` only

No API key is stored in source files, reports, commands printed by the script,
or result records.

## Judge Output

For each record, the judge returns:

- `predicted_answer`: independently calculated final answer;
- `dataset_answer_correct`: whether the supplied dataset answer matches;
- `problem_valid`: whether the presented question is mathematically solvable;
- `answer_unique`: whether the question determines one answer;
- `preserves_original_target`: whether a variant still asks for the same
  mathematical target as the Original;
- `issue_type`: one of `none`, `wrong_label`, `ambiguous`,
  `contradictory`, `insufficient_information`, `target_changed`,
  `malformed`, or `judge_uncertain`;
- `confidence`: a number from 0 to 1;
- `reason`: a concise arithmetic or semantic justification.

The judge must not accept the supplied answer by default. It solves first, then
compares. Original records are audited as calibration data rather than assumed
correct.

## Local Verification

The audit runner normalizes numeric answers before comparison, including
commas, currency symbols, percentages, fractions, and equivalent decimal
forms. The final label-consistency field records both:

- the judge's explicit decision;
- the local normalized comparison between `predicted_answer` and the dataset
  answer.

Disagreement between those two signals is marked for manual review instead of
being silently resolved.

## Execution And Recovery

The runner uses a bounded thread pool, defaults to four workers, retries
transient failures with exponential backoff, and stops early on authentication
or balance errors.

Each successful group is appended immediately to a JSONL checkpoint. On
restart, completed `base_id` values are skipped. Failed requests are written to
a separate JSONL file with the error and attempt count. Writes are protected by
a lock.

The runner supports:

- a small pilot selected by group count;
- full execution;
- resume into an existing run directory;
- configurable workers, retries, and token limit;
- deterministic ordering in the finalized output.

## Outputs

Default run directory:

`chaingsm_data/reports/deepseek_gold_audit/<timestamp>/`

Artifacts:

- `audit_groups.jsonl`: checkpointed raw group judgments;
- `audit_records.jsonl`: flattened, sorted per-record judgments;
- `failed_groups.jsonl`: unresolved API or parsing failures;
- `summary_overall.json`: global validity and answer-consistency statistics;
- `summary_by_category.json`: Original and variant-category breakdowns;
- `flagged_records.jsonl`: invalid, ambiguous, changed-target, wrong-label, or
  low-confidence records;
- `report.md`: concise findings and recommended dataset actions;
- `run_config.json`: non-secret execution configuration and resume command.

The summary separates model-solving disagreement from confirmed dataset
problems. A variant is not declared wrong solely because the strong model gives
a different answer when confidence is low or the local comparison disagrees.

## Validation

Before a full run:

1. Unit-test grouping, answer normalization, response-schema validation,
   checkpoint resume, and summary aggregation.
2. Run a pilot on representative complete and incomplete groups.
3. Inspect all pilot flags and confirm JSON stability with thinking disabled.
4. Run the full 1,319-group audit.
5. Report coverage, unresolved failures, and category-level invalidity rates.

## Security Note

The API key pasted into the conversation should be revoked and replaced. The
runner will refuse to start when `DEEPSEEK_API_KEY` is absent, so credentials
cannot silently fall back to a key embedded in the repository.
