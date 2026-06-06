# ChainGSM DeepSeek Gold Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a resumable `deepseek-v4-flash` audit of all 1,319 ChainGSM test groups with thinking disabled.

**Architecture:** A focused Python CLI loads and groups the existing JSONL dataset, builds one structured judge request per `base_id`, validates responses, checkpoints group results, and materializes per-record summaries. Pure grouping, normalization, validation, and aggregation functions are tested without API access; the API runner reuses the repository's OpenAI-compatible client.

**Tech Stack:** Python 3.12, OpenAI Python client, JSON/JSONL, `concurrent.futures`, pytest.

---

### Task 1: Pure Audit Data Model And Prompt

**Files:**
- Create: `chaingsm_data/audit_deepseek_gold.py`
- Create: `tests/test_deepseek_gold_audit.py`

- [ ] **Step 1: Write failing grouping and prompt tests**

Add tests proving records are grouped in source order, Original appears first,
missing variants are accepted, and the prompt requires independent solving plus
all schema fields.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest tests/test_deepseek_gold_audit.py -v
```

Expected: import failure because `audit_deepseek_gold.py` does not exist.

- [ ] **Step 3: Implement grouping and prompt construction**

Implement `read_jsonl`, `group_records`, and `build_audit_messages`. Include
record IDs, categories, presented questions, and the retained dataset answer.
Require the agreed per-record JSON fields and concise reasoning.

- [ ] **Step 4: Verify the tests pass**

Run the focused pytest command and expect all Task 1 tests to pass.

### Task 2: Response Validation And Answer Normalization

**Files:**
- Modify: `chaingsm_data/audit_deepseek_gold.py`
- Modify: `tests/test_deepseek_gold_audit.py`

- [ ] **Step 1: Write failing schema and numeric-equivalence tests**

Cover comma/currency normalization, integer-decimal equivalence, fractions,
percentages, invalid answers, missing record judgments, duplicate IDs, unknown
issue types, and judge/local-comparison disagreement.

- [ ] **Step 2: Verify the new tests fail**

Run focused pytest and confirm failures name the missing validation functions.

- [ ] **Step 3: Implement minimal validation**

Implement `normalize_numeric_answer`, `answers_equivalent`,
`validate_group_judgment`, and `flatten_group_judgment`. Preserve both the
judge decision and local answer comparison and flag disagreements.

- [ ] **Step 4: Verify all focused tests pass**

Run focused pytest and expect zero failures.

### Task 3: Resumable Concurrent API Runner

**Files:**
- Modify: `chaingsm_data/audit_deepseek_gold.py`
- Modify: `tests/test_deepseek_gold_audit.py`

- [ ] **Step 1: Write failing resume and API-configuration tests**

Test that completed groups are skipped, duplicate checkpoint rows are resolved
to the latest result, API key absence is fatal, and the client is constructed
with model `deepseek-v4-flash` and thinking `disabled`.

- [ ] **Step 2: Verify the new tests fail**

Run focused pytest and inspect the expected missing runner behavior.

- [ ] **Step 3: Implement CLI and runner**

Add arguments for input, run directory, group limit, workers, retries,
max tokens, and force. Use four workers by default, append successes and
failures under a lock, stop on fatal API errors, and emit a secret-free
`run_config.json`.

- [ ] **Step 4: Verify all focused tests pass**

Run focused pytest and expect zero failures.

### Task 4: Summary Artifacts

**Files:**
- Modify: `chaingsm_data/audit_deepseek_gold.py`
- Modify: `tests/test_deepseek_gold_audit.py`

- [ ] **Step 1: Write failing aggregation tests**

Test global/category counts, invalidity rates, confidence flags, unresolved
failures, and deterministic sorted output.

- [ ] **Step 2: Verify the tests fail**

Run focused pytest and confirm aggregation behavior is absent.

- [ ] **Step 3: Implement materialization and Markdown reporting**

Create `audit_records.jsonl`, `summary_overall.json`,
`summary_by_category.json`, `flagged_records.jsonl`, and `report.md` from the
checkpoint, including coverage and unresolved failure counts.

- [ ] **Step 4: Verify all focused tests pass**

Run focused pytest and expect zero failures.

### Task 5: Pilot And Full Audit

**Files:**
- Runtime outputs only:
  `chaingsm_data/reports/deepseek_gold_audit/<timestamp>/`

- [ ] **Step 1: Run the full local test suite**

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run a three-group API pilot**

Run the CLI with `--limit-groups 3`, verify thinking is disabled in config,
inspect JSON schema and flags, and confirm all three groups complete.

- [ ] **Step 3: Run the full audit**

Run without a group limit against all 1,319 groups. Resume the same run
directory if transient failures occur.

- [ ] **Step 4: Verify final coverage and report results**

Require 1,319 completed groups, 6,575 flattened records, zero unresolved fatal
errors, and readable overall/category summaries. If the API key is unavailable
or rejected, report the exact blocker without embedding the secret.
