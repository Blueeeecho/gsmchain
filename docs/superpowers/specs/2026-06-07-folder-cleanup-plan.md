# math-chain 文件夹整体整理方案

> **生成日期**：2026-06-07
> **作用**:在不动远程/不动训练模型的前提下,把整个项目盘一次,标出"留/移/删",并列出推荐处理顺序。
> **执行原则**:**git 已跟踪的文件默认移到 `_archive/`**(可还原),**git 已忽略的目录可直接删除**。

---

## 1. 顶层总览(扫描结果)

| 类别 | 体积 | 是否 git 跟踪 | 推荐处理 |
|---|---:|---|---|
| `code/results/` (评测产物) | ~880M | ❌ gitignore | 部分移 `_archive/`,部分保留 |
| `outputs/` (训练产物) | ~108G | ❌ gitignore | `outputs/train/` 保留(实训练);`outputs/2026-05-*` / `2026-06-0*` 全是早期试跑,删 |
| `__pycache__/` / `.pytest_cache/` | 几 M | ❌ gitignore | **直接删** |
| `tmp/` | 0 | ❌ gitignore | **直接删(空)** |
| `chaingsm_data/data/raw/{test,train}-*` | ~10M | ✅ tracked | 全部保留(GSM8K 原始数据) |
| `chaingsm_data/data/{final,gsmchain,pilot,reports}/` | ~150M | ✅ tracked | 详见 §3 |
| `chaingsm_data/src/` | ~20K | ✅ tracked | **保留**(数据生成源码) |
| `code/*.py` | ~120K | ✅ tracked | 详见 §4 |
| `docs/` | 几 M | ✅ tracked | 详见 §5 |
| `codebuddy/` | ~32K | ✅ tracked | 5/28-5/29 重构草稿,**移 `_archive/`** |
| `dataset/` | ~2.3M | ✅ tracked | `train-00000-of-00001.parquet` 与 `chaingsm_data/data/raw/` 重复,**删 `dataset/`** |
| `Noise_math_data-main/` | 几 G | ❌ gitignore | **保留**(README 要求:远程验证前作参考) |
| `plan_1.md` / `baseline_readme.md` / `train_readme.md` | 几 M | ✅ tracked | 详见 §5(部分与 README 重叠) |
| `train_pipeline/` / `train_configs/` / `train_scripts/` / `tests/` | 几 M | ✅ tracked | **全部保留** |

---

## 2. 评测产物 `code/results/`(880M,**重点清理**)

### 2.1 推荐删除(已经废)

| 目录 | 体积 | 原因 |
|---|---:|---|
| `code/results/chaingsm_test/` (8 个子目录) | 318M | 5/25-5/27 试跑,使用 6575 未清理集 + 旧版 prompt profile,数据已被 `chaingsm_base_8shot_*` 取代 |
| `code/results/lm_eval_llama_smoke/` | 164K | lm-eval 对接烟测,结论已写入 `docs/gsm8k_gsmplus_official_reproduction_report_zh.md` |
| `code/results/lm_eval_llama_vllm_smoke50/` | 4K | 同上,50 条烟测 |
| `code/results/chaingsm_base_8shot_smoke/` | 244K | 32 条烟测,功能被正式 run 覆盖 |
| `code/results/baseline_test/granular_style_prompting/` | (在大目录内) | GranuLar 一步法烟测,无全量 |
| `code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/{164144,164447,164622,164926}/` | <1M | 4 个 errors-only / 极小烟测,无 predictions.jsonl |
| `code/results/baseline_test/full_run_20260527_*.log` | 几 K | 旧日志 |
| `code/results/baseline_test/postprocess_reports/` | 几 K | 后处理报告,无主表 |

### 2.2 推荐移到 `_archive/code_results_2026-06-07/`

| 目录 | 体积 | 原因 |
|---|---:|---|
| `code/results/chaingsm_base_8shot/20260606_144428/` | 42M | batch=64 + 6575 旧集(5 模型),已被 5467 batch=64 + 164305/173456 取代 |
| `code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_163320/` | <1M | 唯一有 0.5B predictions 的烟测,留作基线 |
| `code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_164117/` | <1M | Math-1.5B 烟测,errors-only,删 |

### 2.3 推荐保留(后续直接对照)

| 目录 | 体积 | 原因 |
|---|---:|---|
| `code/results/chaingsm_base_8shot/20260606_164305/` | ~30M | batch=64 5467 全量(5 模型) |
| `code/results/chaingsm_base_8shot/20260606_173456/` | ~30M | batch=64 5467 全量(4 Qwen,确认 0.5B 一致) |
| `code/results/chaingsm_base_8shot/20260607_140452/` | ~10M | batch=64 5467, Llama-3.2-1B |
| `code/results/chaingsm_base_8shot/20260607_143400/` | ~10M | batch=64 5467, Llama-3.2-3B |
| `code/results/chaingsm_base_8shot_batch16/20260606_174725/` | ~10M | batch=16 5467, Qwen 0.5B + 1.5B |
| `code/results/chaingsm_base_8shot_batch16/20260606_184504/` | ~10M | batch=16 5467, Qwen 3B + Math-1.5B |
| `code/results/chaingsm_base_8shot_batch16/20260607_131942/` | ~10M | batch=16 5467, Llama-3.2-1B |
| `code/results/chaingsm_base_8shot_batch16/20260607_141832/` | ~10M | batch=16 5467, Llama-3.2-3B |
| `code/results/chaingsm_base_8shot_batch16/20260607_151358/` | ~10M | batch=16 5467, Qwen-1.5B 重跑 |
| `code/results/official_gsm/` | 23M | GSM8K Original/Rephrase/Distract + lm_eval 复现,AbstRaL 复现参考 |
| `code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_163320/` | <1M | 唯一 AbstRaL 主方法烟测数据 |

### 2.4 处理命令(默认)

```bash
# 1. 删除(全在 .gitignore,安全)
rm -rf code/results/chaingsm_test
rm -rf code/results/lm_eval_llama_smoke
rm -rf code/results/lm_eval_llama_vllm_smoke50
rm -rf code/results/chaingsm_base_8shot_smoke
rm -rf code/results/baseline_test/granular_style_prompting
rm -rf code/results/baseline_test/postprocess_reports
rm -rf code/results/baseline_test/full_run_20260527_*.log
rm -rf code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_164144
rm -rf code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_164447
rm -rf code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_164622
rm -rf code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_164926

# 2. 移到 _archive/
mkdir -p _archive/code_results_2026-06-07
mv code/results/chaingsm_base_8shot/20260606_144428 _archive/code_results_2026-06-07/
```

可回收 ~520M。

---

## 3. 数据集 `chaingsm_data/data/` + `chaingsm_data/reports/`

### 3.1 推荐保留

| 路径 | 体积 | 原因 |
|---|---:|---|
| `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` | 8.9M | **当前主线干净测试集(5,467)** |
| `chaingsm_data/data/gsmchain/{README.md,cleaning_stats.json}` | 几 K | 干净集元信息 |
| `chaingsm_data/data/raw/test-00000-of-00001.{jsonl,parquet}` | 1.3M | GSM8K 官方 test(1,319) |
| `chaingsm_data/data/raw/train-00000-of-00001.{jsonl,parquet}` | 7M | GSM8K 官方 train(7,473) |
| `chaingsm_data/data/raw/selected_gsm8k_train_balanced_one_variant_all.jsonl` | 5M | 14,946 训练样本(选了 variant) |
| `chaingsm_data/data/final/train_balanced_one_variant/` | ~60M | SFT/DPO/GRPO 训练数据 |
| `chaingsm_data/data/final/rl_preprocessed/` | ~30M | 训练链路依赖,README 明确说不能删 |
| `chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl` | ~30M | 6,575 完整(clean 前) — 留作 deepseek 审计复现 |
| `chaingsm_data/reports/deepseek_gold_audit/` | 几 M | 生成 5,467 的审计结果,**不能删** |
| `chaingsm_data/reports/train_balanced_one_variant/` | 几 M | 训练样本选择报告 |
| `chaingsm_data/src/` | 20K | 数据生成源码 |
| `chaingsm_data/generate_dataset.py` | 30K | 主数据生成入口 |
| `chaingsm_data/README.md` | 15K | 数据集说明 |
| `chaingsm_data/requirements.txt` | 75B | 数据生成依赖 |

### 3.2 推荐删除

| 路径 | 体积 | 原因 |
|---|---:|---|
| `chaingsm_data/data/raw/selected_gsm8k_test_full_all.jsonl` | 848K | 1,319 简单选中,被 `test-00000-of-00001.jsonl` 覆盖 |
| `chaingsm_data/data/raw/selected_gsm8k_train_200.jsonl` | 130K | 200 样本 pilot,功能被 `train_balanced_one_variant` 覆盖 |
| `chaingsm_data/data/raw/selected_gsm8k_train_pilot_2.jsonl` | 730B | 极小 pilot |
| `chaingsm_data/data/raw/failed_requests_gsm8k_test_full_all.jsonl` | 3M | 失败 API 重试记录,已重试成功 |
| `chaingsm_data/data/raw/selected_failed_retry_json_full_no_think_all3090_all.jsonl` | 2.3M | 同上,重试已成功 |
| `chaingsm_data/data/final/test.jsonl` | 14K | 10 条烟测集,被 `gsmchain/gsm8k_test_clean.jsonl` 覆盖 |
| `chaingsm_data/data/final/train_200/` | 几 K | 200 样本 pilot |
| `chaingsm_data/data/final/failed_retry_json_full_no_think_all3090/` | 几 K | 失败重试子集,重试已成功 |
| `chaingsm_data/data/pilot/` | 几 K | pilot 数据 |
| `chaingsm_data/reports/failed_retry_json_full_no_think_all3090/` | 几 K | 失败重试报告 |
| `chaingsm_data/reports/gsm8k_test_full/` | 几 K | 早期 audit 报告(被 `deepseek_gold_audit/full/` 覆盖) |
| `dataset/` (整个目录) | 2.3M | 与 `chaingsm_data/data/raw/train-00000-of-00001.parquet` 重复 |

可回收 ~12M。

---

## 4. 代码 `code/`

### 4.1 推荐保留(全在主线)

- `code/eval_chaingsm_base_8shot.py` — **主线评测入口**
- `code/eval_official_gsm.py` — 官方 GSM8K 复现入口
- `code/eval_abstral_baselines.py` — AbstRaL/GranuLar 评测入口
- `code/gsm_answer_extractor.py` — **统一答案提取器,所有评测依赖**
- `code/rescore_gsm_predictions.py` — 历史结果重算(测试覆盖)

### 4.2 推荐删/移

| 路径 | 体积 | 处理 | 原因 |
|---|---:|---|---|
| `code/eval_chaingsm.py` | 21K | **移到 `_archive/`** | 旧版评测,被 `eval_chaingsm_base_8shot.py` 取代,只剩 `code/README.md` 引用 |
| `code/README.md` | 9K | **移到 `_archive/`** | 描述的是旧版 `eval_chaingsm.py`,且默认数据路径指向 10 条 test.jsonl |

> 后续把"ChainGSM 本地模型评测"重写一份指向 `eval_chaingsm_base_8shot.py` + 5467 干净集。

---

## 5. 文档

### 5.1 推荐保留(主线 / 当前)

- `README.md` — 项目总览,**当前**
- `docs/chaingsm_base_8shot_evaluation_zh.md` — 5/467 8-shot 评测详细说明
- `docs/gsm8k_gsmplus_official_reproduction_report_zh.md` — 官方基线复现
- `docs/llama_one_shot_prompt_vllm_hf_analysis_zh.md` — 1-shot 提示词分析
- `docs/remote-server-settings.md` — 远程配置
- `docs/superpowers/specs/2026-06-07-*.md` × 5 — 今天落盘
- `docs/superpowers/specs/2026-06-06-*.md` × 4 — 6/6 系列 spec(历史)
- `docs/superpowers/plans/2026-06-06-*.md` × 4 — 6/6 系列 plan(历史)
- `docs/superpowers/specs/2026-06-02-remote-chain-alignment-design.md` — 远程对齐 spec
- `docs/superpowers/plans/2026-06-02-local-only-maintenance.md` — 本地维护 plan
- `docs/superpowers/plans/2026-06-02-remote-chain-alignment.md` — 远程对齐 plan
- `train_readme.md` — 训练链路

### 5.2 推荐移到 `_archive/docs/`

| 路径 | 行数 | 原因 |
|---|---:|---|
| `codebuddy/eval_output_refactor.md` | 13K | 5/28 草稿,实际已落地为 `eval_chaingsm_base_8shot.py` |
| `codebuddy/grpo_verl_refactor_review_20260529.md` | 19K | 5/29 GRPO/verl 改造草稿,已落地 |
| `plan_1.md` | 893 | Q&A 风格,内容已被 `chaingsm_data/README.md` + `codebuddy/eval_output_refactor.md` 覆盖 |
| `code/README.md` | 9K | 描述 `eval_chaingsm.py` 旧版 |
| `docs/chaingsm_base_8shot_evaluation_zh.md` § 9(可选) | — | 第 9 节同时含 batch=16 + batch=64,若要"只关注最新"可由今天写的 spec 取代 |
| `baseline_readme.md` | 682 | 4 训练流程说明,与 `train_readme.md` + README 大量重叠 |

> 注:`codebuddy/` 的两个 md 是 5/28-5/29 **重写前的草稿**,实际架构在 6/2-6/6 才定型,它们与今天的状态脱节,只作为历史记录。

### 5.3 推荐删

- `tmp/`(空)

### 5.4 文档冲突检查

- `docs/chaingsm_base_8shot_evaluation_zh.md` vs `docs/superpowers/specs/2026-06-07-chaingsm-base-8shot-batch16-eval-spec.md`:前者更详细(含 AbstRaL 对比表),后者更精炼(只讲最新)。**保留前者**,后者作为"快速 spec"留查。
- `codebuddy/eval_output_refactor.md` vs `plan_1.md` vs `chaingsm_data/README.md`:三处都讲数据格式 + variable 字段,**保留 `chaingsm_data/README.md`**(权威),其余移 `_archive/`。
- `train_readme.md` vs `baseline_readme.md` vs `README.md` § 6-7:三处都讲训练链路。**保留 `train_readme.md` + `README.md`**,`baseline_readme.md` 移 `_archive/`。

---

## 6. 训练产物 `outputs/`

| 路径 | 体积 | 处理 |
|---|---:|---|
| `outputs/train/` | 108G | **保留**(实训练,108G 不能随便动) |
| `outputs/eval_sft20epoch_on_train/` | 8.5M | **保留**(SFT 20 epoch 评测) |
| `outputs/eval_sft2epoch_on_train/` | 9.3M | **保留**(SFT 2 epoch 评测) |
| `outputs/2026-05-29/` | 228K | **删**(早期试跑) |
| `outputs/2026-05-30/` | 116K | **删** |
| `outputs/2026-05-31/` | 284K | **删** |
| `outputs/2026-06-01/` | 228K | **删** |
| `outputs/2026-06-02/` | 116K | **删** |

可回收 ~1M。

---

## 7. 临时 / 缓存文件

| 路径 | 体积 | 处理 |
|---|---:|---|
| `__pycache__/`(多处) | 几 M | **全删**(在 .gitignore) |
| `.pytest_cache/` | 几 M | **删** |
| `tmp/` | 0 | **删** |

---

## 8. 建议执行顺序

1. **先回收 git 已忽略的临时文件**(安全、零风险):`__pycache__`、`.pytest_cache`、`tmp/`、所有 `*.log`。
2. **删除 `outputs/2026-05-*` / `2026-06-0[12]/`**(试跑,1M 总)。
3. **删除评测产物的废目录**:`code/results/{chaingsm_test,lm_eval_*,chaingsm_base_8shot_smoke,baseline_test 的多数}`(320M+)。
4. **删除数据集废料**:`chaingsm_data/data/{raw,final,pilot,reports}/*_pilot_*,failed_*,test.jsonl` 等(12M)。
5. **删除 `dataset/`**(2.3M,与 `chaingsm_data/data/raw/` 重复)。
6. **把 `code/{eval_chaingsm.py,README.md}`、`codebuddy/*`、`plan_1.md`、`baseline_readme.md` 移到 `_archive/`**(可还原,28K+893L+682L+32K)。
7. **把 `code/results/chaingsm_base_8shot/20260606_144428` 移到 `_archive/`**(42M,旧 6575 batch=64)。

---

## 9. 风险与边界

- **不动**:`Noise_math_data-main/`(远程参考,README 要求保留)、`outputs/train/`(108G 实训练)、`chaingsm_data/data/final/{rl_preprocessed,train_balanced_one_variant}/`(训练链路依赖)、`chaingsm_data/src/`、`generate_dataset.py`、`chaingsm_data/README.md`、`chaingsm_data/reports/deepseek_gold_audit/`。
- **可还原**:`_archive/` 不进 git(可手动 git rm -r --cached 处理,或加到 .gitignore),日后真要恢复直接 `mv` 回原位。
- **不可还原**:`code/results/`、`outputs/2026-*`、`chaingsm_data/data/{raw,final,pilot,reports}/*_pilot` 的删除——都是 .gitignore 内的产物,但体积大,删了就没了。

---

## 10. 验收清单(执行后)

- [ ] `code/results/chaingsm_test/` 不存在
- [ ] `code/results/lm_eval_*_smoke*` 不存在
- [ ] `code/results/chaingsm_base_8shot_smoke/` 不存在
- [ ] `code/results/baseline_test/` 仅保留 `abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_163320/`
- [ ] `code/results/chaingsm_base_8shot/` 仅保留 4 个 5467 batch=64 run
- [ ] `code/results/chaingsm_base_8shot_batch16/` 保留全部 5 个 run
- [ ] `outputs/2026-05-*` 与 `outputs/2026-06-0[12]/` 不存在
- [ ] `__pycache__/` / `.pytest_cache/` / `tmp/` 不存在
- [ ] `dataset/` 不存在
- [ ] `chaingsm_data/data/raw/{failed_*,selected_failed_*,selected_gsm8k_test_full_all.jsonl,selected_gsm8k_train_200.jsonl,selected_gsm8k_train_pilot_2.jsonl}` 不存在
- [ ] `chaingsm_data/data/final/{test.jsonl,train_200/,failed_retry_json_full_no_think_all3090/}` 不存在
- [ ] `chaingsm_data/data/pilot/` 不存在
- [ ] `chaingsm_data/reports/{failed_retry_json_full_no_think_all3090,gsm8k_test_full}/` 不存在
- [ ] `_archive/` 含 `code_results_2026-06-07/20260606_144428/`
- [ ] `_archive/docs/` 含 `codebuddy/`,`plan_1.md`,`baseline_readme.md`,`code/eval_chaingsm.py`,`code/README.md`
- [ ] `pytest -q tests/` 仍 45/45 通过
- [ ] `code/eval_chaingsm_base_8shot.py --help` 仍能跑
