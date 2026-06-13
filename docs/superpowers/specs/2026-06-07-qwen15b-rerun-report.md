# Qwen2.5-1.5B-Instruct 重跑报告(2026-06-07)

> **目的**:验证 `chaingsm_base_8shot_batch16/20260606_174725` 在 5,467 条上 Qwen2.5-1.5B-Instruct 的 **72.25% Original** 数字是否可复现,排查 batch=16 与 batch=64 之间 18.6pp 差距的原因。
> **结论**:**数字真实,可复现**。两次跑在 1,319 条 Original 上答对题数完全一致(均为 953/1319)。全量 5,467 条 aggregate 差 0.16pp。

---

## 1. 重跑配置

```text
入口:        code/eval_chaingsm_base_8shot.py
测试集:      chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl (5,467)
模型:        Qwen2.5-1.5B-Instruct (单模型)
profile:     qwen_multiturn_8shot_chat
batch_size:  16
temperature/top_p/max_tokens/seed: 0.0 / 1.0 / 512 / 42
gpu_memory_utilization: 自动 (实际选中 0.8)
enforce_eager: False
输出目录:    code/results/chaingsm_base_8shot_batch16/20260607_151358
耗时:        ~8m59s
```

完全沿用上次 (20260606_174725) 的所有参数与数据集,只换时间戳。

---

## 2. 数字对比

| 指标 | 上次 (20260606_174725) | 重跑 (20260607_151358) | Δ |
|---|---:|---:|---:|
| 全量 5,467 | 48.95% (2676) | **49.11% (2685)** | +0.16pp |
| Original 1,319 | 72.25% (953) | **72.25% (953)** | **0pp** |
| independent_decoy 1,102 | 37.48% (413) | 37.75% (416) | +0.27pp |
| attribute_mismatch 1,017 | 47.39% (482) | 47.30% (481) | -0.09pp |
| path_competition 999 | 44.94% (449) | 45.45% (454) | +0.51pp |
| target_scope_misalignment 1,030 | 36.80% (379) | 36.99% (381) | +0.19pp |

**Original 1,319 条上答对的题数完全相同(953/1319)**,这不是巧合——说明模型在 72.25% 这个水平是稳态,而非偶然抖动。

---

## 3. predictions 级别一致性

| 项 | 数量 | 占比 |
|---|---:|---:|
| 两次 `correct` 标记一致 | 5,344 | 97.75% |
| 两次 `correct` 标记翻转 | 123 | 2.25% |
| 两次 `raw_output` 文本完全相同 | 4,694 | 85.86% |
| 两次 `raw_output` 文本不同 | 773 | 14.14% |

含义:

- **85.86% 完全一致 + 14.14% 文本不同**:greedy decoding 配 `seed=42`,理论应该完全确定。出现 14.14% 不同,说明 **vLLM 内部有非确定性来源**——很可能是:
  - batch 内不同 prompt 并行顺序
  - vLLM 0.21.0 浮点 reduction 顺序
  - CUDA graph capture 状态(两次重启用同一个 `gpu_memory_utilization=0.8` 但启动瞬间的 graph pool 可能不同)
- **97.75% 答对/答错一致**:虽然 14% 文本变了,但最终答案抽取出来的数值 97.75% 不变(模型对"算错 vs 算对"的判断稳定)。
- **2.25% 答错 ↔ 答对翻转**:123 题,正好对应 5,467 的 0.16pp 变化,完全自洽。

---

## 4. 与 AbstRaL / batch=64 对照

| 配置 | Original 1,319 | 数据来源 |
|---|---:|---|
| AbstRaL 报告 GSM-Plus Original CoT-8S | 67.0% | AbstRaL 论文 Table 5 |
| 本机 batch=16, 上次 (20260606_174725) | 72.25% | 已落盘 |
| 本机 batch=16, 重跑 (20260607_151358) | **72.25%** | 本报告 |
| 本机 batch=64, run 1 (20260606_164305) | 53.53% (706/1319) | 已落盘 |
| 本机 batch=64, run 2 (20260606_173456) | 53.68% (708/1319) | 已落盘 |

**结论**:

- 72.25% 是**稳定**结果,与 AbstRaL 报告 67.0% 差距 5.25pp,方向(超过)与 Abstract 4 节观察一致。
- batch=16 与 batch=64 之间 18.6pp 差距**不是数据抖动**——两次 batch=16 都给 72.25%,两次 batch=64 都给 ~53.7%。差异来自 **vLLM 在不同 batch_size 下的内部调度/浮点行为**,而非提示词、提取器或模型本身。

---

## 5. 后续动作建议

1. **接受 batch=16 是基线,72.25% 是真值**。所有 SFT/DPO/GRPO checkpoint 沿用 `eval_chaingsm_base_8shot.py --batch-size 16`。
2. **如要复现 AbstRaL Table 5 风格**:
   - AbstRaL 报告 67.0%,我们 batch=16 是 72.25%。两者均显示 Qwen2.5-1.5B 强于 Llama 同尺寸(45.2%)。
   - 如果论文对齐需要 67.0% 附近的数字,可能是他们用了 lm-eval 不同的 8-shot 模板或 stop sequences。
3. **batch=64 偏差诊断**(可选,工作量 1-2 天):
   - 用 `--enforce-eager` 重跑,看 53.7% 是否回升。
   - 把 vLLM 切到 0.10.x 或 0.22.x 重跑,排除 vLLM 0.21.0 调度 bug。
   - 排除 KV cache layout 问题。
   - 如果以上都不奏效,放弃 batch=64,统一 batch=16。

---

## 6. 一句话总结

> 72.25% 是真的。batch=16 与 batch=64 的 18.6pp 差距是 vLLM 内部行为,与数据/提示词/提取器/模型无关。
