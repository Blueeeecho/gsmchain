# LB-PRM v8.2 设计与训练计划 (2026-06-14)

> 接 v7 失败 + v8 reward 缺失 + v8.1 reward 数据管道断开反馈.
> v7 best: step_100 original 0.4428 (距 0.46 -1.72pp).
> v8 (4 子项 0.05/0.85/0.05/0.05) 修 v7 砍 numeric 抢梯度, 但**没适配 ChainGSM 干扰式样**.
> v8.1 (5 机制 + irrelevant_eq 扣分) 设计方向对, 但 `preprocess_chaingsm_8shot_cot.py` 把 `gold_expression` / `core_chain` 等关键字段砍掉, **4 机制在真实训练中空转 + 1 个反扣**.
> v8.2 = 修数据管道 + v8.1 4 机制接通信号 + SFT ckpt 已就绪 + GRPO 1000 步.

---

## 1. 目标与差距

| 指标 | 数值 | 状态 |
|---|---:|---|
| 0.5B base + 8-shot CoT 原生 | 0.4329 (571/1319) | baseline |
| v7 best (step_100) | 0.4428 (+0.99pp) | 失败, 距目标 -1.72pp |
| SFT ckpt 1ep 8-shot CoT | 待测 (期望 0.45-0.50) | eval 死胎待重启 |
| **目标** | **>= 0.46 (607/1319)** | - |

## 2. v8.1 失败根因 (重审)

### 2.1 数据管道: preprocess 砍了 5 个关键字段
`train_pipeline/preprocess_chaingsm_8shot_cot.py:50` 的 `to_verl_grpo_8shot_row()` 写 parquet 时只塞了:
```python
"reward_model": {"style": "rule", "ground_truth": {"gold_answer": str(record.get("answer", ""))}}
```
**漏接** 5 个字段:
- `gold_expression`: 完整 gold 算式 (e.g. `(16-3-4)*2`)
- `core_chain`: 推出 gold 的算式链 (e.g. `[[eggs_per_day, sold_eggs, -3-4], [sold_eggs, daily_income, *2]]`)
- `distractor_chain`: 干扰算式链 (e.g. `[[eggs_per_day, total_eggs_value, *2]]`)
- `question`: 题面完整文本 (target_recognition 用)
- `category`: original / path_competition / independent_decoy / attribute_mismatch / target_scope_misalignment

**这些字段原始 14946.jsonl 100% 都有** (7055/7055 用 `source_augmented_with_traces.jsonl` 验证).

### 2.2 真实训练中 v8.1 的实际表现
我用 `score_response` 在空 reference `{gold_answer: 18}` 下模拟了 4 个 case:

| 算式 (text) | reference | 实际 reward | 预期 |
|---|---|---:|---|
| `16-3=13. 13-4=9. 9*2=18. final 18` (core chain 答对) | `{gold_answer: 18}` | **0.70** | 1.0 |
| `2*12=24. 24*1.5=36. final 18` (distractor+抄对) | `{gold_answer: 18}` | 0.40 | <0.5 |
| `2*12=24. 24*1.5=36. final 36` (纯 distractor) | `{gold_answer: 18}` | **0.05** | <0.1 |
| `16-3=13. 13-4=9. 9*2=18.` (无 marker) | `{gold_answer: 18}` | **0.70** | 1.0 |

**核心 bug**: CASE 1 (core chain 答对) `n_irrelevant=2, penalty=0.20`! 13/9/16/3/4/2 都不在 S (S={18}) → 3 个 core 算式被全判 irrelevant. **v8.1 核心机制在真实训练中反着扣, 把好 chain 干掉**.

### 2.3 其他 3 个机制的空转
- `target_recognition` 0.15: 拿不到 question, 全 return 0.5 中性 → 0.075 固定损失 (信号失真)
- `chain_length_consistency` 0.05: 直接 `return 1.0` (代码注释: "v8.1 v4: 暂不信任 length_consistency, 给中性分") → 全空转
- `chain_to_answer_check` 0.20: 不依赖 reference 字段, **能工作** ✅
- `format` 0.05: 不依赖 reference 字段, **能工作** ✅

**v8.1 真实有效信号**: `format(0.05) + answer(0.55) + chain_to_answer_check(0.20) = 0.80` + 中性 0.125, 扣 `irrelevant_eq` 0~0.30. **跟 v7 几乎没差**.

## 3. v8.2 设计: 修数据 + 调整 reward 权重

### 3.1 数据管道修复 (核心, 推荐路线 A)
`train_pipeline/preprocess_chaingsm_8shot_cot.py:50` 改为:
```python
"reward_model": {
    "style": "rule",
    "ground_truth": {
        "gold_answer": str(record.get("answer", "")),
        "gold_expression": str(record.get("gold_expression", "")),
        "core_chain": record.get("core_chain") or [],
        "distractor_chain": record.get("distractor_chain") or [],
        "question": record.get("question_distracted") or record.get("question_original") or "",
        "category": str(record.get("category", "")),
    },
},
```
**重跑 parquet**: 5 min.

### 3.2 v8.2 reward 公式 (跟 v8.1 同结构, 不改)
```
total = 0.05 * format
      + 0.55 * answer
      + 0.20 * chain_to_answer_check
      + 0.15 * target_recognition
      + 0.05 * chain_length_consistency
      - 0.30 * irrelevant_eq_ratio
```
**5 机制接通信号后**:
| 子项 | 期望信号 | v8.2 真实 |
|---|---|---|
| format 0.05 | 收尾 marker | 跟 v7/v8 同 (能用) |
| answer 0.55 | 答对 gold | 跟 v7/v8 同 (能用) |
| chain_to_answer_check 0.20 | chain Z == extract_answer | 跟 v7/v8 同 (能用) |
| target_recognition 0.15 | unit 匹配问句 | **接通, 0~0.15 区分度** |
| chain_length_consistency 0.05 | 算式数 vs 题面复杂度 | 仍 return 1.0 中性 |
| irrelevant_eq -0.30 | 写 distractor 扣分 | **接通, 0~-0.30 区分度** |

### 3.3 TDD 测试: 加 3 个真实数据 case
在 `test_lbprm_v8_1.py` 加 (因为 v8.2 跟 v8.1 共享 reward 实现):
- Case F: 4 类变体 (path_competition) + 模型写 core chain + 答对 → 满分 1.0
- Case G: 4 类变体 (independent_decoy) + 模型写 distractor 算式 → 扣 0.20-0.30
- Case H: target_scope_misalignment + 模型答 after-donation → target_recognition 0.0

### 3.4 SFT 起点
- 已有: `outputs/sft/sft_8shot_cot_1ep/Qwen2.5-0.5B-Instruct/sft_8shot_cot_1ep/20260614_161054/checkpoints/best/`
- 16:32 训完 1 epoch, PID 1812088 退出码 0
- 16:33 启动 eval (PID 1901384) 死胎, 需要重启

## 4. 训练计划 (3 步)

### 步骤 1: 修数据管道 (5 min)
- 改 `preprocess_chaingsm_8shot_cot.py:50` 拼齐 5 字段
- 重跑 `python train_pipeline/preprocess_chaingsm_8shot_cot.py`
- 验证新 parquet 第一个 row 的 `ground_truth` 含 5 字段

### 步骤 2: 重启 SFT ckpt eval (5-10 min)
- 跑 `code/eval_chaingsm_base_8shot.py` 拿 SFT ckpt original 数字
- 如果 original >= 0.46 → 取 SFT ckpt, 目标达成, 写报告
- 如果 original < 0.46 → 启动 GRPO

### 步骤 3: GRPO 1000 步 (2.5h, 5 节点 eval)
- 起点: SFT ckpt
- reward: v8.1 (4 机制接通)
- MAX_STEPS=1000, SAVE_FREQ=200
- 5 节点 eval: 200/400/600/800/1000
- 监控: `outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/grpo_verl_v8/<RUN_ID>/eval/step_NNN/summary_by_category.json`

## 5. 终止与判断

| 情况 | 动作 |
|---|---|
| SFT ckpt original >= 0.46 | 取 SFT ckpt, 报目标达成 |
| GRPO step 200 original >= 0.46 | 视为达成, 后续观测收敛 |
| GRPO step 200 original < 0.43 (SFT 摧毁) | 改 0.5 epoch SFT, 重训 |
| GRPO step 200-400 持平且 < 0.46 | 反思: penalty 0.30 太重 → 改 0.20; target_recognition 不准 → 改 hard 关键词匹配 |
| GRPO step 600 original >= 0.46 | 强制 stop at 600 |
| GRPO 1000 步仍未达成 | 迭代 v8.3: 加 8-shot 模板相似度 / chain coverage 机制 |

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| SFT 摧毁 8-shot (跟 v6 一样) | step_200 eval < 0.40 → 改 0.5 epoch SFT |
| 修数据后 SFT 数据没改, 只改 GRPO 数据, 训练/推理数据 schema 不一致 | SFT 数据用 `sft_train_v2.jsonl` 自由格式, 跟 GRPO parquet 独立, 无 schema 耦合 |
| irrelevant_eq 误判 (S 集合不全) | `gold_expression` 解析 + 链式 eval + 括号嵌套 + core_chain eval, 5 重覆盖 |
| GRPO 1000 步撞 0.45 | penalty 0.30 → 0.40, 加 8-shot 模板相似度 |
| vLLM 显存不够 (PID 1901384 16:33 死胎) | 改 `enforce_eager=True` + `gpu_memory_utilization=0.3` |

## 7. 落盘物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec (本文件) | `docs/superpowers/specs/2026-06-14-lbprm-v8-2-design.md` | ✅ |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v8-2-plan.md` | ⏳ 步骤 1 |
| preprocess 修复 | `train_pipeline/preprocess_chaingsm_8shot_cot.py` | ⏳ 步骤 2 |
| v8.2 TDD (复用 v8.1 + 3 case) | `train_pipeline/test_lbprm_v8_1.py` | ⏳ 步骤 3 |
| GRPO 入口 (复用 v8 脚本) | `train_scripts/local/run_grpo_verl_lbprm_v8.sh` | ✅ (已切到 v8.1 reward) |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v8-2-report.md` | ⏳ 步骤 6 |

## 8. 决策记录

- ✅ 2026-06-14 13:33-15:21: v7 500 步训练 (失败, best 0.4428)
- ✅ 2026-06-14 15:22-15:49: v7 5 节点 eval
- ✅ 2026-06-14 15:50-15:55: v7 报告 + v8 spec/plan/reward
- ❌ v7 失败 (numeric 抢 answer 梯度)
- ✅ 2026-06-14 15:55-16:05: v8 TDD/reward (4 子项)
- ⚠️ v8 reward 被批评"不配 ChainGSM 干扰"
- ✅ 2026-06-14 16:08-16:10: v8.1 5 机制 reward (TDD 20/20 PASS)
- ✅ 2026-06-14 16:10-16:32: SFT 1 epoch (PID 1812088 退出 0)
- ❌ 2026-06-14 16:33: SFT ckpt eval 死胎 (PID 1901384 退出, vLLM 编译后挂)
- ⚠️ 2026-06-14 16:35: 用户批评 reward "不配数据风格, 没加分心链"
- 🔍 2026-06-14 16:35-16:39: 根因定位: preprocess 砍了 5 字段, v8.1 4 机制空转
- ⏳ 2026-06-14 16:40+: v8.2 实施 (修数据 + 重启 eval + GRPO 1000 步)

---

## 9. 实际跑完结果 (v8.2 final)

- ✅ 2026-06-14 17:00-20:20: v8.2 GRPO 1000 步训练完成 (3.3h)
- ✅ 2026-06-14 20:23-20:36: 5 节点 eval 跑完
- ❌ v8.2 失败: best original = 0.4003 (距 0.46 目标 -5.97pp)
- 关键 trade-off 发现: 4 类变体涨 +10-14pp 同时 original 跌 7.81pp
- 详见 final 报告: `docs/superpowers/reports/2026-06-14-lbprm-v8-2-report.md`
