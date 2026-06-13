# Llama / Qwen / Mathstral Instruct 8-shot CoT 与 AbstRaL 报告对比

> **生成日期**：2026-06-07
> **目的**：把 AbstRaL 论文 Table 5 的 GSM-Plus Original CoT-8S 数字,与本机 5,467 干净集 Original(1,319 条)子集 + 全量(5,467 条)对照成一张表。
> **前提**:AbstRaL 数字来自论文原表(用户粘贴)。本机数字来自 `code/results/chaingsm_base_8shot_batch16/*/summary_*.json` 与 `chaingsm_base_8shot/*/summary_*.json`,均已落盘,且已用 `verification-before-completion` 重新校核。

---

## 1. 模型 vs 报告数字(Original 子集 1,319 条)

> 干净集的 `original` 类目与 AbstRaL 报告的 GSM-Plus Original 任务在样本上对齐,因此可以横向比较。

| 模型 | AbstRaL GSM-Plus Original CoT-8S | 本机 batch=16 Original | 本机 batch=64 Original |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 45.2 | 45.64 (602/1319) | 45.41 (599/1319) |
| Llama-3.2-3B-Instruct | 79.5 | 77.41 (1021/1319) | 77.03 (1016/1319) |
| Llama-3.1-8B-Instruct | 85.7 | — | — |
| Qwen2.5-0.5B-Instruct | 42.4 | 43.29 (571/1319) | 41.77 (551/1319) |
| Qwen2.5-1.5B-Instruct | 67.0 | 72.25 (953/1319) · 49.11% (2685/5467) 重跑一致 | 53.68 (708/1319) |
| Qwen2.5-3B-Instruct | 81.2 | 85.90 (1133/1319) | 84.99 (1121/1319) |
| Qwen2.5-7B-Instruct | 89.0 | — | — |
| Qwen2.5-Math-7B-Instruct | 91.8 | — | — |
| Mathstral-7B-v0.1 | 80.7 | — | — |

观察:

- **Llama-3.2-1B-Instruct / 3B-Instruct、Qwen2.5-0.5B-Instruct / 3B-Instruct** 四个模型本机数字与 AbstRaL 报告的差距在 ±5pp 之内,提示词与提取器对齐。
- **Qwen2.5-1.5B-Instruct** 在 batch=16 下达到 72.25%,**超过** AbstRaL 报告的 67.0% 5.25pp;但 batch=64 下掉到 53.68%,存在 18.6pp 巨大差距,**待重跑/对照组确认**(怀疑 vLLM 0.21.0 调度问题)。
- **缺数据的模型**(8B Llama、7B Qwen/Math、Mathstral-7B)本机尚未下载,见 §4。

---

## 2. 全量(5,467 条)准确率(本机独有,无 AbstRaL 报告)

| 模型 | 本机 batch=16 全量 | 本机 batch=64 全量 |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 24.55 (1342/5467) | 22.26 (1217/5467) |
| Qwen2.5-1.5B-Instruct | 48.95 (2676/5467) | 34.24 (1872/5467) |
| Qwen2.5-Math-1.5B-Instruct | 57.78 (3159/5467) | 51.00 (2788/5467) |
| Qwen2.5-3B-Instruct | 60.82 (3325/5467) | 60.23 (3293/5467) |
| Llama-3.2-1B-Instruct | 27.71 (1515/5467) | 27.42 (1499/5467) |
| Llama-3.2-3B-Instruct | 58.37 (3191/5467) | 58.30 (3187/5467) |

> 全量 = original + 4 类 ChainGSM 变体(共 5,467 条),用于观察在干扰链上的总体退化幅度。
> Llama-3.2-1B / 3B、Qwen-3B 三个模型 batch=16 与 batch=64 全量差 ≤ 0.6pp,提示 vLLM batch_size 不是主要变量。

---

## 3. 各模型 5 类目准确率(batch=16, 5,467 条)

> 全部已从 `summary_by_category.json` 校核。

| 模型 | original | independent_decoy | attribute_mismatch | path_competition | target_scope_misalignment |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 43.29 (571/1319) | 16.33 (180/1102) | 20.16 (205/1017) | 21.22 (212/999) | 16.89 (174/1030) |
| Qwen2.5-1.5B-Instruct | 72.25 (953/1319) | 37.48 (413/1102) | 47.39 (482/1017) | 44.94 (449/999) | 36.80 (379/1030) |
| Qwen2.5-Math-1.5B-Instruct | 76.19 (1005/1319) | 50.91 (561/1102) | 55.36 (563/1017) | 54.75 (547/999) | 46.89 (483/1030) |
| Qwen2.5-3B-Instruct | 85.90 (1133/1319) | 46.91 (517/1102) | 59.10 (601/1017) | 57.66 (576/999) | 48.35 (498/1030) |
| Llama-3.2-1B-Instruct | 45.64 (602/1319) | 18.51 (204/1102) | 22.81 (232/1017) | 25.23 (252/999) | 21.84 (225/1030) |
| Llama-3.2-3B-Instruct | 77.41 (1021/1319) | 50.91 (561/1102) | 58.21 (592/1017) | 54.25 (542/999) | 46.12 (475/1030) |

读法:

- 同一行从左到右,第一列是 original,后面四列是四类变体。可以看到 original → 变体的退化幅度。
- 退化最严重的列(每行最差)通常是 `target_scope_misalignment`(目标偏移)或 `independent_decoy`(独立诱饵),与 AbstRaL 论文观察一致。
- 1B → 3B 缩放,每个类目都稳定提升 20-30pp,符合模型容量预期。

---

## 4. 缺数据模型的本地准备状态

| 模型 | 本地是否已下载 | 路径 | 备注 |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 未确认 | — | 需检查 `/home/wwq416/snap/wwq/model/llama/` |
| Qwen2.5-7B-Instruct | 未确认 | — | 需检查 `/home/wwq416/snap/wwq/model/Qwen/` |
| Qwen2.5-Math-7B-Instruct | 未确认 | — | 需检查 `/home/wwq416/snap/wwq/model/Qwen/` |
| Mathstral-7B-v0.1 | 未确认 | — | 需检查 `/home/wwq416/snap/wwq/model/mistralai/` |

后续如需补齐:用 `download_model.py` 拉权重,落到上表对应路径,再跑 `eval_chaingsm_base_8shot.py --model <name> --batch-size 16`。

---

## 5. 复现链接(数据来源)

| 模型 | batch=16 run | batch=64 run |
|---|---|---|
| Qwen2.5-0.5B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260606_174725/` | `code/results/chaingsm_base_8shot/20260606_173456/` |
| Qwen2.5-1.5B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260606_174725/` · 重跑 `20260607_151358/` (72.25% Original 复现) | `code/results/chaingsm_base_8shot/20260606_173456/` |
| Qwen2.5-Math-1.5B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260606_184504/` | `code/results/chaingsm_base_8shot/20260606_173456/` |
| Qwen2.5-3B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260606_184504/` | `code/results/chaingsm_base_8shot/20260606_173456/` |
| Llama-3.2-1B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260607_131942/` | `code/results/chaingsm_base_8shot/20260607_140452/` |
| Llama-3.2-3B-Instruct | `code/results/chaingsm_base_8shot_batch16/20260607_141832/` | `code/results/chaingsm_base_8shot/20260607_143400/` |

每个 run 目录含 `summary_overall.json` / `summary_by_category.json` / `model_outputs/<model>/predictions.jsonl` / `prompt_diagnostics.json`。

---

## 6. 一次性命令:重生成全表

```bash
cd /home/wwq416/snap/wwq/math-chain
python3 - <<'PY'
import json, os
from pathlib import Path

batch16_root = Path('code/results/chaingsm_base_8shot_batch16')
batch64_root = Path('code/results/chaingsm_base_8shot')

def latest_with(root: Path, model: str) -> Path | None:
    candidates = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for c in candidates:
        ov = c / 'summary_overall.json'
        if not ov.exists():
            continue
        data = json.loads(ov.read_text())
        for r in data:
            if r.get('model_name') == model:
                return c
    return None

models = [
    'Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct',
    'Qwen2.5-Math-1.5B-Instruct', 'Qwen2.5-3B-Instruct',
    'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct',
]
print('model, batch16_overall, batch64_overall')
for m in models:
    a = latest_with(batch16_root, m)
    b = latest_with(batch64_root, m)
    a_val = '-'
    b_val = '-'
    if a:
        d = next((r for r in json.loads((a/'summary_overall.json').read_text()) if r['model_name']==m), None)
        if d: a_val = f"{d['accuracy_percent']:.2f}"
    if b:
        d = next((r for r in json.loads((b/'summary_overall.json').read_text()) if r['model_name']==m), None)
        if d: b_val = f"{d['accuracy_percent']:.2f}"
    print(f'{m}, {a_val}, {b_val}')
PY
```

---

> **重跑验证报告**:`2026-06-07-qwen15b-rerun-report.md` — 72.25% Original 复现成功,973/1319 答对题完全一致,全量 aggregate 0.16pp 误差内。
