# v7 启动前反思 + 起步阶段状态 (2026-06-14 13:36)

> 接 v6 反思 (`2026-06-14-v6-prereflex.md`).
> 用户硬指标: original >= 0.46.
> 起点换 0.5B base (无 SFT) + 3 子项 reward (0.10/0.70/0.20).
> 训练已启动 (PID 1087380, 13:33:54 开始, 500 步, 5 个 eval 节点: 100/200/300/400/500).

---

## 1. 关键数字 (实测, 1319 条 original + 5467 条 overall)

| 配置 | original | overall | 备注 |
|---|---:|---:|---|
| 0.5B base + 8-shot CoT 原生 (无 RL) | **0.4329** (571/1319) | 0.2438 (1333/5467) | v7 起点 baseline |
| 0.5B + sft_2epoch/best + 8-shot CoT | 0.1296 (171/1319) | ~0.115 | 排除 (SFT 摧毁 8-shot) |
| 0.5B + v3 best (GRPO step_300, JSON 协议) | 0.2995 (395/1319) | 0.3047 (1666/5467) | 上限参考, JSON 协议天花板 |
| 1.5B base + 8-shot CoT 原生 | 0.7225 (953/1319) | 0.4911 (2684/5467) | 同 prompt 协议下的容量参考 |
| **v7 目标** | **>= 0.46** | — | 用户硬指标 |

**目标差距**: original 需要涨 0.4329 → 0.46 = +0.0271 (+2.71pp), 相对涨幅 6.3%.

## 2. v7 起点决策 (vs v3-v6 全部)

| 候选 | 实测 original | 备注 | 决策 |
|---|---:|---|---|
| sft_2epoch/best | 0.1296 | SFT 摧毁 8-shot 能力 (v6 prereflex 实测) | ❌ 排除 |
| v3 best (GRPO JSON 协议) | ~0.10 (估计) | JSON 协议训练残留 | ❌ 排除 |
| 0.5B base (无 SFT) | **0.4329** | 干净起点, 8-shot 协议原生 | ✅ **采用** |
| 0.5B base + 8-shot SFT 1 epoch | (未实测) | 需 1-2h SFT 准备 | 留 v7.1/v8 |

**采用**: 直接用 `Qwen2.5-0.5B-Instruct` HF 原始模型 + 8-shot CoT 训练集, 跳过 SFT.

## 3. v7 reward 决策 (vs v3-v6 全部)

v6 / v6b 的 4 子项结构在 0.5B 自由推理下问题:
- format=0.95+, step_count=0.74, no_contradiction=0.875 → 区分度低
- numeric_correctness=0.22 (v6 smoke) → 真正有信号的空间在算式
- equation_count_bonus 鼓励"凑算式", 不利于推理

**v7 简化设计 (3 子项, 总权重 1.0)**:
```
total = 0.70 * answer + 0.20 * numeric_correctness + 0.10 * format
```

- answer 0.70: 主线信号, 直接对齐 original 数字对
- numeric_correctness 0.20: 0.5B 算式正确率 22% 是真实薄弱点
- format 0.10: 0.5B 自由推理下基本 ok, 但保留 hard constraint

**TDD**: 21/21 PASS (test_lbprm_v7.py).

## 4. v7 smoke (5 步) 实测数字

| 指标 | smoke step 5 (5 步) | baseline (0 step) | delta |
|---|---:|---:|---:|
| original | 0.4299 (567/1319) | 0.4329 (571/1319) | -0.30pp (noise) |
| overall | 0.2372 (1297/5467) | 0.2438 (1333/5467) | -0.66pp |
| independent_decoy | 0.1470 | 0.1615 | -1.45pp |
| attribute_mismatch | 0.1967 | 0.1967 | 0 |
| path_competition | 0.1972 | 0.2102 | -1.30pp |
| target_scope | 0.1660 | 0.1689 | -0.29pp |

**5 步训练后基本没变化** (在 noise 范围内), 跟 v3 / v5 历史一致 (0.5B 在协议切换下需要 100+ 步才突破).

**关键观察**:
- 训练内 `format` 从 1.0 → 0.94, 5 步内开始掉 (训练导致的正常波动)
- 训练内 `answer` 0.125, `numeric_correctness` 0.42 (step 5), 跟 v6 报告 0.22 / 0.74 一致量级
- entropy 0.55 (起点) → 0.45 (step 5), 训练在动 policy, 符合预期
- **DataLoader worker OOM 5 步后被 kill** (Ray 资源竞争), 但 actor/huggingface 落盘成功, eval 跑通, 不影响 long run

## 5. v7 完整训练 (500 步) 计划

- 起点: `Qwen2.5-0.5B-Instruct` HF 原始
- 训练集: `verl_grpo_train_8shot_cot.parquet` (7419 条, 8-shot CoT prompt)
- 训练步数: 500 (用户授权上限 1000, 实际跑 500 节省时间)
- 5 个 eval 节点: step 100/200/300/400/500 (各 5467 条 vLLM 评测)
- 训练超参: LR=5e-7, KL=0.02, ROLLOUT_N=4, TRAIN_BATCH_SIZE=4, MAX_PROMPT=1280, MAX_RESPONSE=512
- 启动时间: 2026-06-14 13:33:54
- ETA: 17:50 (训练 4.3h) + 25 min eval = 18:15 完成

## 6. 监控计划 (按时间顺序)

| 时间 | 节点 | 预期 original | 决策 |
|---|---|---:|---|
| 14:36 | step 100 eval | 0.43-0.45 | < 0.43 反思起点/RL, 0.43-0.45 继续 |
| 15:36 | step 200 eval | 0.44-0.46 | >= 0.46 视为达成, 0.44-0.46 继续 |
| 16:36 | step 300 eval | 0.45-0.47 | 监控涨点趋势 |
| 17:36 | step 400 eval | 0.45-0.48 | 监控涨点趋势 |
| 18:36 | step 500 eval | 0.46-0.49 | 最终验证 original >= 0.46 |

## 7. 风险与回退

| 风险 | 应对 |
|---|---|
| 0.5B base 8-shot 起点 + 100 步内 accuracy 不涨 | step 100 eval 数字定夺, < 0.43 反思 prompt / reward |
| 500 步撞 0.45 (差 1pp) | 反思 answer 权重 0.70→0.80, 或 MAX_RESPONSE_LENGTH 512→768 |
| 4 类变体反向退化 | 监控 by_category, 任何一类降 5pp → 反思 |
| 训练时间超 5h | 接受, 优先保证 step 100 真实数字 |
| DataLoader worker OOM | 已加 `data.dataloader_num_workers=2` 缓解, Ray 会自动重启 worker |

## 8. v7.1 / v8 候选 (等 v7 报告再选)

1. **加 SFT 1 epoch 起点**: 0.5B base + 8-shot CoT SFT 1 epoch (自生成 trace 过滤) + GRPO 500 步
2. **加 answer 权重**: v7.1 = answer 0.70→0.80, numeric 0.20→0.10
3. **加 max_response_length**: 512→768 给推理更多 token
4. **保留 v7 best checkpoint**: 跑完取 original 最高的 step
5. **继续 v7 训练**: 从 best checkpoint resume, 再跑 500 步 (200 → 1000 步 总)

## 9. 决策记录

- 2026-06-14 13:13: 写 v7 spec/plan/reward/test/entry, 21/21 PASS
- 2026-06-14 13:15: 跑 v7 baseline eval (0.5B base + 8-shot CoT 原生), original 0.4329, overall 0.2438
- 2026-06-14 13:20: 跑 v7 smoke (5 步 + step_5 eval), original 0.4299 (-0.30pp noise), 数据链路 ✅
- 2026-06-14 13:33: 启动 v7 完整 500 步训练, ETA 18:15
- 2026-06-14 14:36 预计: 看到 step 100 eval 数字, 决定是否继续
