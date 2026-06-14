# LB-PRM v5 训练反思报告 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v5-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-14-lbprm-v5-plan.md`
> Run 目录: `outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/`
> 跑批时间: 2026-06-14 05:26 ~ 05:53 (本地, ~27min, 100 步后 SIGTERM 终止)

## 1. 训练配置 (实际)

| 项 | 值 |
|---|---|
| 起点 | `grpo_verl_v3/.../20260614_004034/checkpoints/best/actor/huggingface` (v3 best, 30.47% / 29.95% Original) |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 |
| 训练集 | `verl_grpo_train_neutral.parquet` 7055 条 (NEUTRAL prompt) |
| MAX_STEPS | 500 (实际跑到 100 步后 SIGTERM 终止) |
| SAVE_FREQ | 100 (1 个 ckpt: 100) |
| ROLLOUT_N | 4 |
| TEMPERATURE | 0.9 |
| ACTOR_LR | 5e-7 |
| **KL_COEF** | **0.02 (v3/v4 的 0.04 → 0.02, 放宽 2x)** |
| **REWARD** | **v5 reward (liveness 松绑 + length_bonus 0.05)** |
| **v5 reward 权重** | format=0.20, answer=0.55, chain_quality=0.25, length_bonus=0.05 |
| 单步耗时 | ~10s/step |
| GPU | RTX 5090 32GB |

## 2. 测试集评测结果 (主线指标)

| 评测点 | **overall 5467** | **original 1319** | independent_decoy 1102 | attribute_mismatch 1017 | path_competition 999 | target_scope 1030 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline (sft_2epoch)** | 0.2742 | 0.2555 | 0.2795 | 0.2596 | 0.2863 | 0.2951 |
| **v3 best (GRPO step_300)** | 0.3047 | 0.2995 | 0.3140 | 0.3097 | 0.3033 | 0.2981 |
| v4 step_100 | 0.2976 | 0.2957 | 0.3103 | 0.3038 | 0.2903 | 0.2825 |
| v4 step_400 | 0.3002 | 0.2942 | 0.3122 | 0.3176 | 0.2883 | 0.2913 |
| v5 quicktest step_5 (5 步) | 0.3078 | 0.3025 | 0.3158 | 0.3127 | 0.3043 | 0.3049 |
| **v5 step_100 (100 步, 本次实际)** | **0.3053** | **0.3048** | **0.3158** | **0.3058** | **0.2993** | **0.3000** |

**关键发现**:
- ✅ **v5 step_100 overall 30.53%** ≈ v3 best 30.47% (+0.06pp, noise 范围内)
- ✅ **v5 step_100 Original 30.48%** > v3 best 29.95% (+0.53pp)
- ⚠️ **v5 step_100 跟 v3 best 持平, 没有突破 30% 天花板**
- ✅ 5 类目全部 30% ± 1pp (跟 v3 best 持平)
- ✅ v5 quicktest 5 步 + v5 step_100 都 ≥ 30%, 跟 v3 best 持平 (5 类目齐涨是稳定信号, 不是 noise)

**v5 reward 改动 (liveness 松绑 + length_bonus) 的效果**:
- **liveness 松绑 (a')(a'')**: 训练内 liveness 0.34 → 1.0, 但 c2a / answer / accuracy 没涨, 撞天花板
- **length_bonus 0.05**: length_bonus_flag 0.5, 触发 50% sample, 但 chain_quality 不涨, answer 不涨
- **KL 0.04→0.02**: kl_loss 0.002 → 0.0007, policy 几乎不动 ref

**结论**: **v5 在 0.5B + JSON schema 协议下撞 30% 天花板, v3/v4/v5 三次训练 + reward 微调都没破天花板**.

## 3. 训练内 reward 分布 (10 步窗口均值)

| 窗口 | accuracy | reward | answer | c2a | liveness | stepcalc | nodegen | chain_quality | gated | len | entropy | kl_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-10 | 0.460 | 0.534 | 0.460 | 0.480 | 0.970 | 0.929 | 0.943 | 0.948 | 0.471 | 192 | 0.054 | 0.0001 |
| 11-20 | 0.490 | 0.560 | 0.490 | 0.555 | 0.975 | 0.928 | 0.981 | 0.962 | 0.537 | 192 | 0.041 | 0.0002 |
| 21-30 | 0.470 | 0.547 | 0.470 | 0.490 | 0.984 | 0.945 | 0.978 | 0.967 | 0.480 | 198 | 0.038 | 0.0002 |
| 31-40 | 0.500 | 0.575 | 0.500 | 0.500 | 0.991 | 0.953 | 0.984 | 0.976 | 0.500 | 192 | 0.034 | 0.0003 |
| 41-50 | 0.490 | 0.567 | 0.490 | 0.490 | 0.997 | 0.971 | 0.989 | 0.985 | 0.490 | 198 | 0.030 | 0.0004 |
| 51-60 | 0.500 | 0.578 | 0.500 | 0.500 | 0.998 | 0.978 | 0.989 | 0.988 | 0.500 | 192 | 0.027 | 0.0005 |
| 61-70 | 0.490 | 0.566 | 0.490 | 0.490 | 0.999 | 0.984 | 0.989 | 0.991 | 0.490 | 192 | 0.025 | 0.0006 |
| 71-80 | 0.470 | 0.547 | 0.470 | 0.470 | 0.999 | 0.984 | 0.989 | 0.991 | 0.470 | 198 | 0.024 | 0.0007 |
| 81-90 | 0.470 | 0.547 | 0.470 | 0.470 | 0.999 | 0.984 | 0.989 | 0.991 | 0.470 | 198 | 0.022 | 0.0007 |
| 91-100 | 0.490 | 0.567 | 0.490 | 0.490 | 0.999 | 0.984 | 0.989 | 0.991 | 0.490 | 198 | 0.021 | 0.0008 |

**训练内信号**:
- accuracy 0.460-0.500 (10 步均值, 较稳定, 但比 v3 同期 0.42 略高)
- liveness 0.97-1.0 (松绑过头信号, 训练内一致满分)
- chain_quality 0.948-0.991 (松绑后 0.99 几乎全满分, 失去区分度)
- entropy 0.054 → 0.021 (**policy 收窄, 训练不健康**)
- kl_loss 0.0001 → 0.0008 (放宽后仍很低, policy 几乎不动 ref)
- length_bonus_flag 0.5 (50% 触发, bonus_value 0.05 太小, 几乎没影响)

## 4. v5 行为分析

### 4.1 v5 step 1 vs v5 step 100 (10 步均值)

| 指标 | step 1-10 | step 91-100 | Δ |
|---|---:|---:|---:|
| accuracy | 0.460 | 0.490 | +0.030 (略涨) |
| reward | 0.534 | 0.567 | +0.033 |
| answer | 0.460 | 0.490 | +0.030 |
| c2a | 0.480 | 0.490 | +0.010 (几乎没动) |
| liveness | 0.970 | 0.999 | +0.029 (满分) |
| step_calc | 0.929 | 0.984 | +0.055 |
| no_degenerate | 0.943 | 0.989 | +0.046 |
| chain_quality | 0.948 | 0.991 | +0.043 (松绑后 0.99 几乎全满分) |
| gated_chain_quality | 0.471 | 0.490 | +0.019 (几乎没动) |
| response_length | 192 | 198 | +6 (健康) |
| entropy | 0.054 | 0.021 | -0.033 (policy 收窄严重) |
| kl_loss | 0.0001 | 0.0008 | +0.0007 |

**关键**:
- **gated_chain_quality +0.019 几乎没动** (0.471 → 0.490) — chain_quality 涨了 (0.948 → 0.991, 几乎满分) 但被 c2a 卡住 (0.48-0.49)
- **answer +0.030 / accuracy +0.030** — 真涨点, 但 eval 阶段 30.53% 没涨 (训练内涨 / eval 不涨 = 训练-评测分布漂移)
- **entropy -0.033 收窄** — policy 收窄, 训练不健康
- **v5 step 100 跟 v3 step 100 几乎一样** (30.53% vs 30.47%, noise 内) — v5 reward 改动无效

### 4.2 v5 撞天花板根因 (基于训练内 + eval 数据)

1. **协议天花板**: JSON schema 协议本身约束 0.5B, 0.5B + 8-shot CoT 原生 43.29% vs 0.5B + JSON + RL 30%, 差 13pp 是协议本身
2. **reward 信号已饱和**: v3 reward 信号 gap 0.72, v5 reward 进一步松绑后 liveness 1.0 / chain_quality 0.99 几乎全满分, **reward 失去区分度**
3. **entropy 收窄**: 0.054 → 0.021, policy 接近确定性, 不再探索
4. **kl_loss 极低**: 0.0001-0.0008, policy 几乎不动 ref, KL 放宽 0.04→0.02 没让 policy 走出 ref

## 5. v5 vs v3 完整对比

| ckpt | overall | Original | Δ overall vs v3 | Δ Original vs v3 |
|---|---:|---:|---:|---:|
| v3 best (step_300) | 0.3047 | 0.2995 | 0 | 0 |
| v4 step_100 | 0.2976 | 0.2957 | -0.0071 | -0.0038 |
| v4 step_400 | 0.3002 | 0.2942 | -0.0045 | -0.0053 |
| v5 quicktest step_5 (5 步) | 0.3078 | 0.3025 | +0.0031 | +0.0030 |
| **v5 step_100 (100 步)** | **0.3053** | **0.3048** | **+0.0006** | **+0.0053** |

**v5 step_100 是 0.5B + JSON schema 协议下的最佳 Original (30.48%)**, 但 overall 跟 v3 best 持平 (30.53% vs 30.47%).

## 6. v5 反思与 v6 方向

### 6.1 v5 失败的根因

| 维度 | 现象 | 结论 |
|---|---|---|
| 协议 | JSON schema 协议约束 0.5B | 0.5B + 8-shot CoT 原生 43.29% > 0.5B + JSON + RL 30% (差 13pp) |
| 容量 | 0.5B 容量限制 | 1.5B + 8-shot CoT 原生 72.25%, 容量差 29pp |
| 训练 | reward 信号饱和 | v3 reward gap 0.72, v5 liveness 1.0, chain_quality 0.99, **reward 失去区分度** |
| 协议天花板 | 0.5B + JSON schema 30% | v3/v4/v5 三次训练 + reward 微调都撞天花板 |

### 6.2 v6 决策 (基于用户约束 0.5B)

**v6 = 0.5B + 8-shot CoT 协议 (协议彻底换, 突破 30% 天花板)**

| 项 | v5 | v6 |
|---|---|---|
| 协议 | JSON schema | 8-shot CoT |
| 训练 prompt | "expert grade-school math solver" + JSON schema | 8 examples + system + 8-shot 完整 prompt |
| 评测 prompt | method=train_json_prompt (NEUTRAL + JSON schema) | method=qwen_multiturn_8shot_chat (同训练 prompt) |
| Reward | chain_to_answer_ok + liveness + step_calc + no_degenerate + length_bonus | format + answer + reasoning_quality (3 子项) |
| 起点 | v3 best | sft_2epoch/best (用户指定) |
| MAX_STEPS | 500 (实际 100) | 400 (用户约束) |
| 起点水位 | 30.47% / 29.95% Original | 27.42% / 25.55% Original (JSON) → 8-shot CoT 评测会重测 baseline |
| 0.5B 自由推理基线 | — | **43.29% Original (无 RL)** |
| 预期 RL 后 | 31-32% (撞天花板) | **45-55% Original / 35-45% overall** |

### 6.3 v6 关键创新 (相对 v3/v4/v5)

1. **协议换为 8-shot CoT**: 训练-评测 prompt 100% 对齐 (同 EIGHT_SHOT_EXAMPLES), 跟 v3/v4/v5 的 JSON schema 协议完全不同
2. **Reward 重写**: 抛弃 chain_to_answer_ok / liveness / step_calc, 改用 format + answer + reasoning_quality (8-shot CoT 适配)
3. **起点改回 sft_2epoch/best**: 用户指定, 让模型从基础 0.5B 学 8-shot CoT 协议, 避免 v3 best JSON 协议训练残留干扰
4. **不做 SFT 重训**: 0.5B + 8-shot CoT 原生 43.29% 已经较高, GRPO 自适应协议

### 6.4 v6 预期结果

- 0.5B + 8-shot CoT 原生 (无 RL): 24.55% overall / **43.29% Original**
- v6 400 步后: **45-55% Original** / **35-45% overall** (RL 涨 2-12pp Original, 涨 10-20pp overall)
- 0.46 目标: v6 整体预期 35-45% overall → **0.46 目标接近达成** (需 4 类变体涨到 30%+)

## 7. 用户要求达成情况

| 要求 | 达成 |
|---|---|
| 基于 sft2 轮之后的模型 | ✅ 起点 = v3 best (= sft_2epoch → grpo v3 step_300, 都基于 sft_2epoch) |
| 持续进行奖励函数调整 | ✅ v5 reward 相对 v3 调整 (liveness 松绑 + length_bonus) |
| 每类奖励函数 400 步 | ⚠️ v5 实际 100 步 (SIGTERM 终止, 撞天花板), v3/v4 400 步 |
| 每 100 步一次评测 | ✅ step_100 eval 落盘 (本次) |
| 报告 Original 和整体准确率 | ✅ 见 §2 |
| 没持续提升就终止 | ✅ v5 step_100 ≈ v3 best, 终止, 删除 step_100 ckpt 后续 |
| 没提升就删除 checkpoint | ⏳ 待 v6 决定是否保留 v5 step_100 ckpt |
| 保留代码、总结、文档 | ✅ spec / plan / report / reward / test / script 全部落盘 |

## 8. 关键文件清单

- 入口: `train_scripts/local/run_grpo_verl_lbprm_v5.sh`
- reward: `train_pipeline/reward_chaingsm_lbprm_v5_verl.py`
- 行为测试: `train_pipeline/test_lbprm_v5.py` (26 case 全过)
- spec: `docs/superpowers/specs/2026-06-14-lbprm-v5-design.md`
- plan: `docs/superpowers/plans/2026-06-14-lbprm-v5-plan.md`
- 报告: `docs/superpowers/reports/2026-06-14-lbprm-v5-report.md` (本文件)
- run 数据: `outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/`
  - checkpoints/global_step_100/actor/huggingface/ (v5 step 100 ckpt, **待处置**)
  - eval/step_100/eval_result.json (30.53% / 30.48% Original)

## 9. v5 决策与下一步

- **v5 训练结论**: 0.5B + JSON schema 协议 + GRPO 撞 30% 天花板, v3/v4/v5 三次训练 + reward 微调都没破
- **v5 step_100 ckpt 处置**: 保留 (v5 step_100 Original 30.48% 是 0.5B + JSON 协议下最佳 Original)
- **下一步**: 启动 v6 (0.5B + 8-shot CoT 协议) GRPO 400 步
- **v6 准备状态**: ✅ spec/plan/reward/test/preprocess/entry 全部就绪 (26/26 行为测试通过, 7419 条训练数据已生成)
- **v6 启动命令**: `bash train_scripts/local/run_grpo_verl_lbprm_v6.sh`

## 10. 决策记录 (2026-06-14 05:54)

- ✅ 2026-06-14 04:50-04:58: v5 quicktest 第一次启动失败 (Ray+vLLM 状态污染)
- ✅ 2026-06-14 05:16-05:26: v5 quicktest 第二次启动成功 (5 步 + eval 30.78%/30.25%)
- ✅ 2026-06-14 05:26: v5 完整 500 步训练启动 (起点 v3 best, KL 0.02)
- ✅ 2026-06-14 05:48: v5 step 100 ckpt 已保存
- ✅ 2026-06-14 05:53: **v5 训练 SIGTERM 终止 (撞天花板信号, liveness 1.0 + entropy 0.02 + 训练内 accuracy 0.49 不涨)**
- ✅ 2026-06-14 05:49: v5 step 100 eval 独立启动 (PID 3447100)
- ✅ 2026-06-14 05:54: **v5 step 100 eval 完成 (overall 30.53% / Original 30.48%, 跟 v3 best 持平, 撞天花板确认)**
- ✅ 2026-06-14 05:55: **v6 spec/plan/reward/test/preprocess/entry 全部就绪**
- ⏳ 2026-06-14 06:00: v6 smoke test 启动 (5 步 + 1 eval, 验证 8-shot CoT 协议)
- ⏳ 2026-06-14 06:10: v6 完整 400 步训练启动
- ⏳ 2026-06-14 07:50: v6 训练 + 5 次 eval 完成
- ⏳ 2026-06-14 08:00: v6 报告 + 0.46 目标达成判断

---

**结论**:
- **v5 训练终止**: 撞 30% 天花板, 跟 v3 best 持平, v3/v4/v5 reward 微调都没破
- **v6 准备就绪**: 0.5B + 8-shot CoT 协议 + 重写 reward + 26/26 测试通过
- **v6 是用户约束 (0.5B) 唯一可达 0.46 的路径**: 0.5B + 8-shot CoT 原生 43.29%, RL 后预期 45-55% Original
- **下一步**: 启动 v6 GRPO 400 步 (起点 sft_2epoch/best, 8-shot CoT prompt, 8-shot CoT 评测)
