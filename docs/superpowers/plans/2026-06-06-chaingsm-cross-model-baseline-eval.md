# ChainGSM Cross-Model Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate five base Instruct models on full ChainGSM with validated Qwen and Llama 8-shot prompt paths.

**Architecture:** Extend the existing ChainGSM evaluator with explicit prompt profiles. Qwen receives a native chat-template string, while Llama receives lm-eval-style multi-turn token IDs with one BOS; scoring and reporting remain shared.

**Tech Stack:** Python, pytest, Transformers tokenizers, vLLM, JSONL/CSV reports.

---

### Task 1: Add Prompt-Profile Unit Tests

**Files:**
- Create: `tests/test_chaingsm_eval.py`
- Modify: `code/eval_chaingsm.py`

- [ ] **Step 1: Write failing tests**

Test model-family routing, Qwen string rendering, Llama multi-turn token IDs,
single-BOS validation, and profile-specific stop sequences using fake tokenizers.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_chaingsm_eval.py
```

Expected: failures because prompt profiles and model-input routing do not exist.

- [ ] **Step 3: Implement minimal prompt-profile functions**

Add:

```python
def select_prompt_profile(model_name: str) -> str: ...
def build_qwen_eight_shot_messages(question: str) -> list[dict[str, str]]: ...
def build_llama_eight_shot_messages(question: str) -> list[dict[str, str]]: ...
def build_model_input(tokenizer, question: str, profile: str) -> str | list[int]: ...
def stop_sequences_for_profile(profile: str) -> list[str]: ...
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_chaingsm_eval.py
```

Expected: all prompt-profile tests pass.

### Task 2: Integrate Profiles With vLLM Evaluation

**Files:**
- Modify: `code/eval_chaingsm.py`
- Test: `tests/test_chaingsm_eval.py`

- [ ] **Step 1: Write failing integration-oriented unit tests**

Verify that generated model inputs preserve list-of-token-IDs batches for Llama
and that run summaries include the prompt profile.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_chaingsm_eval.py
```

- [ ] **Step 3: Update evaluation and reporting**

Route each model to its profile, construct profile-specific `SamplingParams`,
save prompt diagnostics, and include `prompt_profile` in prediction and summary
records.

- [ ] **Step 4: Run focused and existing tests**

Run:

```bash
pytest -q tests/test_chaingsm_eval.py tests/test_official_gsm_eval.py
```

Expected: all tests pass.

### Task 3: Run Two-Model Smoke Test

**Files:**
- No source changes expected.
- Output: `code/results/chaingsm_base_8shot/<timestamp>/`

- [ ] **Step 1: Run 10-example Qwen smoke test**

Use Qwen2.5-0.5B-Instruct, full generation settings, and a low-memory vLLM
configuration.

- [ ] **Step 2: Run 10-example Llama smoke test**

Use Llama-3.2-1B-Instruct and confirm saved diagnostics report one BOS.

- [ ] **Step 3: Validate smoke outputs**

Check prediction counts, absence of generation errors, summary files, and prompt
profile labels.

### Task 4: Run Full Five-Model Evaluation

**Files:**
- Output: `code/results/chaingsm_base_8shot/<timestamp>/`

- [ ] **Step 1: Run Qwen2.5-0.5B-Instruct**
- [ ] **Step 2: Run Qwen2.5-1.5B-Instruct**
- [ ] **Step 3: Run Qwen2.5-Math-1.5B-Instruct**
- [ ] **Step 4: Run Qwen2.5-3B-Instruct**
- [ ] **Step 5: Run Llama-3.2-1B-Instruct**
- [ ] **Step 6: Verify each model has 6,575 predictions**
- [ ] **Step 7: Regenerate combined summaries**

### Task 5: Report Results

**Files:**
- Create: `docs/chaingsm_base_8shot_results_2026-06-06.md`

- [ ] **Step 1: Record exact run configuration**
- [ ] **Step 2: Add overall accuracy table**
- [ ] **Step 3: Add five-category accuracy table**
- [ ] **Step 4: Record errors and tokenization diagnostics**
- [ ] **Step 5: State how this baseline will anchor later SFT and RL comparisons**

