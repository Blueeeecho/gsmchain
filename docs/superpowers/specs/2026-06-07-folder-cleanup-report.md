# 文件夹整理执行报告(2026-06-07)

> **关联方案**:`2026-06-07-folder-cleanup-plan.md`
> **执行时间**:2026-06-07 16:45 - 17:00
> **本机**:`/home/wwq416/snap/wwq/math-chain`

---

## 1. 执行结果

| 类别 | 动作 | 数量 | 体积 |
|---|---|---:|---:|
| 临时/缓存 | `__pycache__/`、`.pytest_cache/`、`tmp/` 全部删除 | 全树 | 几 M |
| 训练试跑 | `outputs/2026-05-29 ~ 2026-06-02` 5 个目录删除 | 5 | 1M |
| 评测产物 | `code/results/{chaingsm_test,lm_eval_*,chaingsm_base_8shot_smoke}` 删除;`baseline_test` 精简到 1 个 | 6 | 320M |
| 旧 batch=64 (6575) run | 移到 `_archive/code_results_2026-06-07/20260606_144428/` | 1 | 42M |
| 旧代码 | `code/eval_chaingsm.py`、`code/README.md` 移到 `_archive/docs/code/` | 2 | 31K |
| 旧文档 | `codebuddy/*`、`plan_1.md`、`baseline_readme.md` 移到 `_archive/docs/` | 4 | 1.6M |
| 旧数据 (raw) | `selected_*_pilot*`、`failed_*`、`selected_gsm8k_test_full_all.jsonl` 移到 `_archive/chaingsm_data/data/raw/` | 5 | 6M |
| 旧数据 (final) | `test.jsonl`、`train_200/`、`failed_retry_*/` 移到 `_archive/chaingsm_data/data/final/` | 3 | 17K |
| 旧数据 (pilot) | `chaingsm_data/data/pilot/` 移到 `_archive/chaingsm_data/data/` | 1 | 12K |
| 旧 reports | `failed_retry_json_full_no_think_all3090/`、`gsm8k_test_full/` 移到 `_archive/chaingsm_data/reports/` | 2 | 几 K |
| 旧 reports (顶层散落) | `failed_generation.jsonl`、`final_stats.json`、`final_summary.md`、`pilot_*` 移到 `_archive/chaingsm_data/reports/` | 5 | 88K |
| 重复 dataset | `dataset/train-00000-of-00001.parquet` 移到 `_archive/dataset/` | 1 | 2.3M |
| 测试更新 | `tests/test_local_environment_docs.py` 的 `CURRENT_DOCS` 移除已归档的 `baseline_readme.md` + `code/README.md` | 1 个 | — |
| .gitignore 更新 | 加 `Noise_math_data-main/`、`chaingsm_data/data/final/{rl_preprocessed,train_balanced_one_variant,gsm8k_test_full}/`、`chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` | 4 行 | — |

**可回收约 540M**(主要为 `code/results/chaingsm_test` 318M + `code/results/baseline_test/granular_style_prompting` 86M + `code/results/chaingsm_base_8shot/20260606_144428` 42M + `code/results/baseline_test/*` 73M)。

---

## 2. .gitignore 变更

新增 4 条:

```text
# Reference repo (do not track)
Noise_math_data-main/

# Generated training data (large, regenerable from raw + preprocess)
chaingsm_data/data/final/rl_preprocessed/
chaingsm_data/data/final/train_balanced_one_variant/
chaingsm_data/data/final/gsm8k_test_full/

# Test set (8.9M, tracked via source + cleaning)
chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
```

---

## 3. 验收

```bash
$ pytest -q tests/
45 passed in 0.19s
```

`code/eval_chaingsm_base_8shot.py --help` 不受影响(本文件未动)。

---

## 4. git 状态摘要

```text
   2 ??   ← 残留:可能是 _archive 自身(未 git add,后续可决定要不要 commit)
  88 A    ← 大量已 add 的归档内容(可控,待用户决定要不要一次 commit)
  81 D    ← 旧位置删除
  27 R    ← 22 个文件 rename 到 _archive/
   1 AM   ← .gitignore 边改边 add
   2 M    ← .gitignore + chaingsm_data/README.md 既有修改
   1 MM   ← .gitignore 二次修改
```

> **未执行 `git commit`**——按 system prompt 默认不 commit,等用户决定。

---

## 5. 可恢复路径(任一归档项都可还原)

```bash
# 把任一归档项还原到原位
git mv _archive/chaingsm_data/data/raw/selected_gsm8k_test_full_all.jsonl \
      chaingsm_data/data/raw/

# 或物理还原
mv _archive/chaingsm_data/data/raw/selected_gsm8k_test_full_all.jsonl \
   chaingsm_data/data/raw/
```

---

## 6. 不动的边界(已确认保留)

- `Noise_math_data-main/`(README 要求:远程验证前作参考)
- `outputs/train/`(108G 实训练)
- `outputs/eval_sft{2,20}epoch_on_train/`(SFT 评测结果)
- `chaingsm_data/data/final/rl_preprocessed/`(训练链路依赖)
- `chaingsm_data/data/final/train_balanced_one_variant/`(训练数据)
- `chaingsm_data/data/gsmchain/`(当前 5,467 干净集)
- `chaingsm_data/reports/deepseek_gold_audit/`(生成 5,467 的审计)
- `chaingsm_data/src/`、`generate_dataset.py`、`chaingsm_data/README.md`
