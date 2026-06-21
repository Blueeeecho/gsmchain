# ChainGSM Remote — Collaborator Run Guide

> **Goal**: remote collaborators can **edit a prompt → push → one `sbatch`** to finish a training run, with no manual data prep and no cluster-path editing.
>
> **Core principle**: prompts are embedded as top-of-file string constants (`SYSTEM` / `USER_TEMPLATE`) inside the build scripts. Edit them there, save, resubmit. **No external `.md` is the source of truth.**

## 0. What's in the repo (ready to use right after `clone`)

| File | Size | Purpose |
|---|---|---|
| `chaingsm_data/data/final/sft/gsm8k_train_unified_6102.jsonl` | 25 MB, 6102 rows | The single training data source (3051 variants + 3051 originals) |
| `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` | 8.9 MB, 5467 questions | Eval set, 5 categories |
| `chaingsm_data/data/final/sft/build_grpo_v{12,13,14}_*.py` | — | Build scripts; prompt constants live at the top |
| `train_pipeline/reward_chaingsm_v{12,12i,13,14}_*_verl.py` | — | Reward functions |
| `train_pipeline/eval_vllm_chaingsm.py` | — | Eval entry point (auto-syncs with training prompt) |
| `train_configs/remote/grpo_verl_v{12,12i,13,14}_vllm.yaml` | — | Remote training configs |
| `train_scripts/remote/submit_grpo_v{12,13,14}_pipeline.sh` | — | One-shot submitter (preprocess + train) |
| `train_scripts/remote/data_preprocess_v{12,13,14}.sh` | — | Preprocess only (rarely needed standalone) |
| `train_scripts/remote/preflight_remote.sh` / `remote_env.sh` | — | Preflight + shared env |

**Not in the repo** (regenerated locally on demand):
- Training parquets (`grpo_v12_json.parquet` etc., 3 files) — built by build scripts on the fly
- `_system_prompt.txt` — written by build scripts
- Any checkpoint or model weight

## 1. End-to-end workflow (5 steps)

### Step 1: Clone the repo

```bash
git clone https://github.com/Blueeeecho/gsmchain.git
cd gsmchain
```

### Step 2: Dry-run self-check (highly recommended, no GPU)

```bash
DRY_RUN=1 SKIP_PREPROCESS=1 \
    bash train_scripts/remote/submit_grpo_v12_pipeline.sh
```

**You should see two command lines printed** (nothing actually runs):
```
[v12-pipeline] Preprocess command:
  bash .../data_preprocess_v12.sh
[v12-pipeline] Train command (v12i, epochs=2, config=grpo_verl_v12i_vllm):
  ... --config-name grpo_verl_v12i_vllm ...
```

If you see these, environment / paths / configs are wired up correctly. **Do not `sbatch` if anything reports FAIL.**

### Step 3: (Optional) Edit a prompt

**Every prompt is embedded as a top-of-file triple-quoted constant in the build script.** Open the file, scroll up, edit the constant. No external doc lookup needed:

| Version | File | Lines to edit |
|---|---|---|
| V12 / V12i | `chaingsm_data/data/final/sft/build_grpo_v12_json.py` | `SYSTEM` (L24) + `USER_TEMPLATE` (L75) |
| V13 | `chaingsm_data/data/final/sft/build_grpo_v13_json.py` | `SYSTEM` (L36) + `USER_TEMPLATE` (L72) |
| V14 | `chaingsm_data/data/final/sft/build_grpo_v14_reasoning.py` | `V14_SYSTEM_PROMPT` (L32) + `V14_USER_TEMPLATE` (L77) |

**Edit only the content inside `"""..."""`. Do not touch any other logic** (paths, row schema, etc.).

Commit and push:
```bash
git add chaingsm_data/data/final/sft/build_grpo_v*.py
git commit -m "v12: tighten prompt wording"
git push
```

**Important**: editing a prompt does **not** require re-running `build_unified_train_jsonl.py` (that script only merges raw + supp; it is prompt-agnostic). Just resubmit the pipeline — the build step inside the pipeline will regenerate the parquet with the new prompt.

### Step 4: Submit training (one `sbatch`, raw → ckpt)

```bash
# Default: V12i (4-component LCS reward, recommended)
sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# Run V12 instead (4-component soft-core reward)
VERSION=v12 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh

# V13 (5-component hard-match reward)
sbatch train_scripts/remote/submit_grpo_v13_pipeline.sh

# V14 (4-component reworked reward, default 3 epochs)
sbatch train_scripts/remote/submit_grpo_v14_pipeline.sh
```

**This single `sbatch` runs two steps sequentially:**

1. **Step 1/2 — preprocess**: `data_preprocess_vXX.sh` invokes `build_grpo_vXX_*.py`, producing `chaingsm_data/data/final/grpo/grpo_vXX_*.parquet` (6102 rows) and writing `SYSTEM` to the same directory as `_system_prompt.txt`.
2. **Step 2/2 — train**: launches verl GRPO training (config: `grpo_verl_vXX_vllm`, base model `Qwen2.5-0.5B-Instruct`, 2–3 epochs).

Output checkpoint path: `${REMOTE_OUTPUT_DIR}/chaingsm-grpo-verl-${JOB_TAG}/checkpoints/global_step_*/actor/huggingface/`

**Reuse an existing parquet** (e.g. when sweeping reward hyperparameters):
```bash
SKIP_PREPROCESS=1 sbatch train_scripts/remote/submit_grpo_v12_pipeline.sh
```

### Step 5: Evaluate (auto-uses the training prompt)

**No train-test prompt mismatch by design**:
- During training, `data_preprocess_vXX.sh` writes `SYSTEM` to `${REMOTE_DATA_DIR}/_system_prompt.txt`.
- During eval, `eval_vllm_chaingsm.py --method parquet_prompt` reads that file as the system prompt and auto-selects the right `USER_TEMPLATE` + answer extractor from the parquet directory name (`grpo_v12_json` / `grpo_v13_json` / `grpo_v14_reasoning`).

```bash
# METHOD=parquet_prompt is the default — no extra flag needed
STEPS="300 600 900 1200 1500" \
RUN_NAME=qwen2.5-0.5b-grpo-verl-v12-json \
RUN_ID=<your_run_id> \
bash train_scripts/local/eval_v12_5ckpts.sh
```

`RUN_ID` is the timestamp directory under `outputs/train/remote/chaingsm-grpo-verl-v12i/`.

To compare with historical reports using the legacy method (no auto prompt sync):
```bash
METHOD=cot_brackets_v12_json bash train_scripts/local/eval_v12_5ckpts.sh
```

## 2. Design differences across the 4 versions

| Submitter | Default reward | Data | Epochs | USER_TEMPLATE / answer extractor |
|---|---|---|---|---|
| `submit_grpo_v12_pipeline.sh` (V12) | `reward_chaingsm_v12_json_verl.py` (4 soft-core) | `grpo_v12_json.parquet` | 2 | `cot_brackets_v12_json` |
| `submit_grpo_v12_pipeline.sh` (V12i, **default**) | `reward_chaingsm_v12i_json_verl.py` (4 LCS) | `grpo_v12_json.parquet` | 2 | `cot_brackets_v12_json` |
| `submit_grpo_v13_pipeline.sh` | `reward_chaingsm_v13_json_verl.py` (5 hard-match) | `grpo_v13_json.parquet` | 2 | `cot_brackets_v13_json` |
| `submit_grpo_v14_pipeline.sh` | `reward_chaingsm_v14_reasoning_verl.py` (4 reworked) | `grpo_v14_reasoning.parquet` | 3 | `cot_brackets_v14_reasoning` |

**V12 and V12i share the same training data** (V12 prompt + V12i reward). V13 and V14 each have their own.

## 3. Training data internals (6102 rows, ×2 training pairs)

`gsm8k_train_unified_6102.jsonl` merges raw GSM8K with 4 distractor-chain variant types:

| category | count | meaning |
|---|---|---|
| `original` | 3051 | clean original problems (no distractors) |
| `attribute_mismatch` | 910 | attribute-mismatch variants |
| `independent_decoy` | 837 | independent-decoy variants |
| `path_competition` | 814 | path-competition variants |
| `target_scope_misalignment` | 490 | target-scope-misalignment variants |

Each row has eval-time fields (5) + supervision fields:
- Eval-time: `id` / `base_id` / `category` / `question_distracted` / `gold_answer`
- Reward-time: `core_chain` / `distractor_chain` / `gold_expression` / `distractor_expression` / `gold_trace` / `distractor_trace`

**Field naming** (kept stable per collaborator requirement):
- `question_distracted` field: for **variant** rows it stores the **distracted** problem text; for **original** rows it stores the **clean original** problem text.
- The two are distinguished by `category='original'` (and by `distractor_chain=None`).

Training pairs: 3051 variants + 3051 originals = 6102 (×2 trainer sees the full set).

## 4. Environment variables (optional, usually leave default)

| Variable | Default | Meaning |
|---|---|---|
| `REMOTE_ROOT` | `/home/wwq416/snap/wwq/math-chain` | repo root |
| `REMOTE_MODEL_PATH` | `/export/home/asifali/HF_cache/Qwen2.5-0.5B-Instruct` | base model |
| `REMOTE_OUTPUT_DIR` | `${REMOTE_ROOT}/outputs/train/remote` | ckpt output dir |
| `REMOTE_DATA_DIR` | `${REMOTE_ROOT}/chaingsm_data/data/final/grpo` | parquet output dir |
| `DRY_RUN=1` | 0 | print commands only, do not run |
| `SKIP_PREPROCESS=1` | 0 | skip the build step (parquet already exists) |
| `VERSION=v12\|v12i` | `v12i` | V12 pipeline only |
| `V12_TOTAL_EPOCHS` / `V12I_TOTAL_EPOCHS` / `V13_TOTAL_EPOCHS` | 2 / 2 / 2 | training epochs |
| `V14_TOTAL_EPOCHS` | 3 | V14 training epochs |

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Missing input file: SRC=.../gsm8k_train_unified_6102.jsonl` | jsonl not pulled. `git pull` again. |
| `ModuleNotFoundError: verl` | `Reasoning360` conda env not active. `source activate Reasoning360` |
| Eval says "no _system_prompt.txt" | The version's preprocess has not been run. Run `bash data_preprocess_vXX.sh` first. |
| `SLURM partition gpu-all not visible` | Cluster config changed. Edit `SLURM_PARTITION` in `train_scripts/remote/remote_env.sh`. |
| Parquet is 6102 rows but ckpt looks off | Confirm build script not broken: `git diff chaingsm_data/data/final/sft/build_grpo_v*.py` |
| Want to inspect data | `head -1 chaingsm_data/data/final/sft/gsm8k_train_unified_6102.jsonl \| python3 -m json.tool` |
| Want to see current prompt | Open `chaingsm_data/data/final/sft/build_grpo_v12_json.py` and scroll to L24 / L75. |

## 6. Cluster resources (matches `data_preprocess_parsed_v6a_ALL.sh`)

- **1 GPU / 8 CPU / 32 GB / `gpu-all` partition**
- **CUDA module**: `cuda12.4/toolkit`
- **Conda env**: `Reasoning360`
- **HF cache**: `/export/home/asifali/HF_cache`
- **Logs**: `./all_logs/%j-%x-slurm.{out,err}`

## 7. Out of scope (independent submitter chains)

- `submit_sft_verl.sh` / `submit_sft_then_grpo_verl_vllm.sh` / `submit_dpo_trl.sh`: SFT/DPO chains on the legacy V10 path, **independent of the V12+ pipeline**. **Not part of the V12+ one-shot flow.** If you need to run them, read the comment header inside each script.
