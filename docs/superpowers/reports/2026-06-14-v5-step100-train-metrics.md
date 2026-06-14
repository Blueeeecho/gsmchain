# v5 step 100 训练内 metrics + v6 准备就绪报告 (2026-06-14 05:50)

> v5 训练在跑 step 101/500, v6 spec/reward/test/preprocess/entry 全部就绪

---

## 1. v5 step 100 训练内 metrics 详细

### 1.1 step 101 详细

| 指标 | v3 step 100 mean | v4 step 100 mean | **v5 step 101** | 解读 |
|---|---:|---:|---:|---|
| accuracy | 0.426 | 0.407 | **0.750** | **+0.32pp vs v3**, +0.34pp vs v4 |
| reward | 0.513 | 0.495 | **0.815** | +0.30pp vs v3 |
| answer | 0.426 | 0.407 | 0.750 | +0.32pp |
| c2a | 0.456 | 0.435 | 0.750 | +0.29pp |
| liveness | 0.346 | 0.346 | **1.000** | **松绑过头** (v3 0.346, v4 0.346, v5 1.0) |
| step_calc | 0.890 | 0.890 | 0.969 | +0.08pp |
| no_degenerate | 0.932 | 0.932 | 0.888 | -0.04pp (略降) |
| chain_quality | 0.627 | 0.627 | 0.968 | +0.34pp |
| gated_chain_quality | 0.313 | 0.295 | 0.746 | +0.45pp |
| response_length | 192 | 197 | 188 | -4 (健康) |
| entropy | 0.056 | 0.060 | 0.026 | -0.030 (policy 收窄, 跟 v3 接近) |
| kl_loss | 0.001 | 0.001 | 0.0007 | -0.0003 |
| length_bonus | (无) | (无) | 0.0156 | n_steps ∈ [3,6] flag 0.31 |

**关键观察**:
- ✅ **训练内 accuracy 0.75** (vs v3 0.43, v4 0.41) — 显著涨, 但**单步 8 中 6 对**仍然 50% noise 范围
- ⚠️ **liveness 1.0 持续满分** — **松绑过头**信号明确, 跟 v2 失败模式 (0.89-0.97) 相似
- ✅ response_length 188 正常 (v2 失败掉到 110, v5 健康)
- ⚠️ entropy 0.026 (v3 step 100 0.056) — **policy 收窄得快**, KL 0.02 可能还是过强

### 1.2 step 103 (后续 2 步)

| 指标 | step 101 | step 103 |
|---|---:|---:|
| accuracy | 0.750 | 0.3125 |
| reward | 0.815 | 0.493 |
| answer | 0.750 | 0.3125 |
| c2a | 0.750 | 0.3125 |
| liveness | 1.000 | 1.000 |
| step_calc | 0.969 | 0.984 |
| no_degenerate | 0.888 | 0.925 |
| chain_quality | 0.968 | 0.980 |
| gated_chain_quality | 0.746 | 0.309 |
| response_length | 188 | 199 |
| length_bonus_flag | 0.3125 | 0.875 |

**震荡巨大**: step 101 0.75 → step 103 0.31. 8 个 sample 内 noise ±0.4 正常, **单步数据 noise 极大**.

**这印证了 v3 报告 §4 的判断**: 单步数据 noise 大, **需要看 5 ckpt eval 才知真撞没撞天花板**.

---

## 2. v6 准备就绪状态

### 2.1 已落盘文件

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v6-design.md` | ✅ 200 行 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md` | ✅ 157 行 |
| reward 函数 | `train_pipeline/reward_chaingsm_lbprm_v6_verl.py` | ✅ 292 行 |
| 行为测试 | `train_pipeline/test_lbprm_v6.py` | ✅ **26/26 全过** |
| 数据 preprocess | `train_pipeline/preprocess_chaingsm_8shot_cot.py` | ✅ 164 行 |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v6.sh` | ✅ |
| 8-shot CoT 训练数据 | `verl_grpo_train_8shot_cot.parquet` | ✅ 7419 条 |

### 2.2 v6 reward 设计

```
公式: total = 0.15·format + 0.60·answer + 0.25·reasoning_quality

format = 1.0 if response 收尾于 "The final answer is N." else 0.0
answer = 1.0 if N 数值 == gold_answer else 0.0
reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction

step_count: 3-7 步满分 1.0, <3 步扣 0.3/步, >7 步扣 0.1/步
numeric_correctness: 算式 "X op Y = Z" 验证正确率
no_contradiction: final N 是否在过程数字中
```

### 2.3 v6 关键参数

- **起点**: `sft_2epoch/best` (27.42% / 25.55% Original, JSON schema 协议)
- **训练 prompt**: 8-shot CoT 完整模板 (8 examples + system, 跟评测完全一致)
- **MAX_PROMPT_LENGTH**: 1280 (验证 8-shot CoT prompt max=1029)
- **MAX_RESPONSE_LENGTH**: 512 (CoT 推理通常 < 300 token)
- **MAX_STEPS**: 400 (用户约束)
- **SAVE_FREQ**: 80 (5 个 eval 节点: 80/160/240/320/400)
- **KL_LOSS_COEF**: 0.02 (跟 v5 一致)
- **ACTOR_LR**: 5e-7

### 2.4 v6 训练 prompt 长度验证

`preprocess_chaingsm_8shot_cot.py` 抽检 100 条: **min=921 max=1029 avg=970**, max=1029 > 1024, **MAX_PROMPT_LENGTH=1280 留余量**.

### 2.5 v6 行为测试 (26/26 全过)

```
[PASS] v6-A-perfect-3steps-with-correct-arithmetic             r=1.000
[PASS] v6-B-perfect-4steps                                     r=1.000
[PASS] v6-C-correct-arithmetic-wrong-final                     r=0.400
[PASS] v6-D-correct-wrong-arithmetic                           r=0.963
[PASS] v6-E-correct-arithmetic-final-not-in-process            r=1.000
[PASS] v6-F-wrong-step-arith-correct                           r=0.400
[PASS] v6-G-wrong-wrong-wrong                                  r=0.325
[PASS] v6-H-correct-no-equations                               r=0.925
[PASS] v6-I-correct-hash-marker                                r=1.000
[PASS] v6-J-correct-boxed-marker                               r=1.000
[PASS] v6-K-correct-multistep-proper                           r=1.000
[PASS] v6-L-empty                                              r=-0.500
[PASS] v6-M-completely-unrelated                               r=0.287
[PASS] v6-N-perfect-5steps                                     r=1.000
[PASS] v6-O-correct-8steps                                     r=0.938
[PASS] v6-P-correct-only-2steps                                r=0.962
[PASS] v6-Q-wrong-only-final-in-process                        r=0.325
[PASS] v6-R-wrong-final-not-in-process                         r=0.325
[PASS] v6-S-classic-8shot-style                                r=1.000
[PASS] v6-T-wrong-final-not-in-process-but-arith-ok            r=0.400
[PASS] 5 schema checks
[PASS] compute_reward entry: score=1.000, all keys present

=== 26/26 PASSED ===
```

### 2.6 v6 启动依赖

- **GPU**: 0.5B 全参 + vLLM + optim 估计 12-15GB, **v5 跑完后有 17-19GB 可用, 够用**
- **数据**: parquet 已生成 7419 条 8-shot CoT prompt
- **Reward**: 26/26 测试通过
- **起点**: sft_2epoch/best 已存在
- **评测**: 复用 `code/eval_chaingsm_base_8shot.py` (跟训练 prompt 完全一致)

---

## 3. 启动 v6 的判断

### 3.1 选项 A: 等 v5 跑完再启动 v6 (推荐)

- v5 跑完预计还要 ~1h10min (step 500, 当前 step 103)
- v5 跑完后 GPU 完全空闲, v6 启动稳定
- v5 报告 + v6 spec 落盘, 完整决策记录
- **风险**: v5 跑完时 v6 才启动, 总时间 ~1.5h+

### 3.2 选项 B: 立即启动 v6 (并行)

- v5 用 14.5GB / 32GB, v6 0.5B 训练用 ~12GB, 留 6GB buffer, **理论能并行**
- **风险**: vLLM 多实例可能争抢显存, OOM, 或两者互相拖慢
- **优势**: 总时间缩短 ~1h

### 3.3 选项 C: 立即 kill v5, 直接启动 v6

- v5 还在 18% 进度, 离 step 100 eval 还有 0 步 (ckpt 已保存)
- v5 step 100 ckpt 已保存, 可独立 eval
- **风险**: v5 报告不完整 (4 步 → 1 步 ckpt)

### 3.4 推荐: 选项 A

理由:
- v5 训练在跑, 已 18%, **不要浪费 18min 已投入的算力**
- v5 step 100 ckpt 已保存, **v5 报告 1 个 eval 也有价值**
- v6 等 v5 跑完再启动, GPU 满速, 不会 OOM
- 总时间差距 ~1h, 不影响 v6 决策质量

### 3.5 v5 step 100 eval 期望

- v5 quicktest 5 步 + 1 次 eval = 30.78% / 30.25% (vs v3 best 30.47% / 29.95%)
- v5 step 101 训练内 accuracy 0.75 (vs v3 step 100 0.43) — 显著涨
- **期望 v5 step 100 eval**: 30-32% overall / 30-32% Original (跟 v3/v4 持平, 5 类目可能齐涨)
- **若 v5 step 100 eval < v3 best**: 确认撞 30% 天花板, **立即决定走 v6 (8-shot CoT)**
- **若 v5 step 100 eval > v3 best**: 等 v5 跑完 500 步看是否继续涨

---

## 4. 决策记录 (2026-06-14 05:50)

- ✅ 2026-06-14 05:48: v5 step 100 ckpt 已保存
- ✅ 2026-06-14 05:48: v5 step 101 训练内 accuracy 0.75 (单步 noise 大)
- ✅ 2026-06-14 05:50: v6 spec/plan/reward/test/preprocess/entry 全部就绪
- ✅ 2026-06-14 05:50: v6 reward 26/26 行为测试全过
- ✅ 2026-06-14 05:50: v6 训练数据 7419 条已 preprocess, prompt 长度验证 921-1029
- ⏳ 2026-06-14 05:55: v5 step 100 自动 eval 触发
- ⏳ 2026-06-14 06:50: v5 500 步训练完
- ⏳ 2026-06-14 07:00: v5 5 ckpt 评测完
- ⏳ 2026-06-14 07:00: v5 报告落盘
- ⏳ 2026-06-14 07:05: v6 smoke test (5 步 + 1 次 eval)
- ⏳ 2026-06-14 07:30: v6 完整 400 步训练启动
- ⏳ 2026-06-14 09:30: v6 400 步训练完 + 5 eval
- ⏳ 2026-06-14 10:00: v6 报告 + 0.46 目标达成判断

---

**结论**:
- **v5 step 101 训练内 accuracy 0.75** (vs v3 0.43, v4 0.41) 显著涨, 但 **liveness 1.0 松绑过头** 风险大
- **v6 准备全部就绪**, 26/26 行为测试通过, 数据已 preprocess, 起点确认
- **推荐策略**: 等 v5 跑完 500 步 + 5 ckpt eval, **然后启动 v6 (8-shot CoT)**
- **v6 是用户约束 (0.5B) 范围内唯一可达 0.46 的路径** (0.5B + 8-shot CoT 原生 43.29% > 0.5B + JSON + RL 30%)
