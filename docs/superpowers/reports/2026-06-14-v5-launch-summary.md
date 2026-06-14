# 阶段状态总结 — v5 训练启动 (2026-06-14 05:34)

> 用于在 Codex CLI 沙箱限制下, v5 训练后台推进时, 向用户清晰汇报当前真实状态

---

## 当前真实状态 (2026-06-14 05:34)

### 1. v5 训练运行中 ✅
- **PID**: 3323532
- **启动时间**: 2026-06-14 05:26:36
- **Run ID**: 20260614_052636
- **当前进度**: step 30/500 (6%)
- **单步耗时**: ~10s
- **预计完成**: 真实时间 ~1h25min 后 (即 ~06:50)
- **Run 目录**: `outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/`
- **日志**: `/tmp/v5_full.log`

### 2. v5 关键设计参数
| 项 | 值 | 相对 v3 的改动 |
|---|---|---|
| 起点 | v3 best/actor/huggingface (30.47% / 29.95%) | 同 v3 起点 |
| 训练集 | NEUTRAL prompt (verl_grpo_train_neutral.parquet) | 同 v4 |
| MAX_STEPS | 500 | +100 步 (v3/v4 是 400) |
| KL_COEF | 0.02 | -50% (v3/v4 是 0.04) |
| Liveness | 松绑 (a')(a'') 子表达式匹配 | v3 收紧 → v5 松绑 |
| Length bonus | +0.05 (n_steps ∈ [3,6]) | v3 无 → v5 新增 |
| Reward 权重 | format 0.20 + answer 0.55 + chain_quality 0.25 + bonus | v3 + length_bonus |

### 3. 训练内 metrics 实时 (step 1-30)

| step | acc | liveness | stepcalc | gated | reward | len | bonus_flag |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.500 | 1.000 | 1.000 | 0.4375 | 0.625 | 161 | 0.5 |
| 5 | 0.4375 | 0.984 | 0.944 | 0.483 | 0.596 | 198 | 0.69 |
| 10 | 0.500 | 0.948 | 0.893 | 0.500 | 0.603 | 166 | 0.38 |
| 15 | 0.500 | 0.991 | 0.975 | 0.500 | 0.647 | 217 | 0.94 |
| 20 | 0.250 | 1.000 | 1.000 | 0.250 | 0.419 | 134 | 0.38 |
| 25 | 0.625 | 1.000 | 0.969 | 0.625 | 0.744 | 222 | 0.88 |
| 30 | (进行中) | | | | | | |

**训练内 acc 范围**: 0.25-0.625, mean ~0.5
**liveness range**: 0.95-1.0 (持续高分, 松绑过头信号)
**length_bonus_flag range**: 0.38-0.94 (有效触发)
**response_length range**: 130-220

### 4. v3 训练内 (对比 baseline)
- accuracy range: 0.42-0.47, mean 0.43
- liveness range: 0.34-0.37 (v3 收紧)
- response_length: 190-220

**v5 vs v3 训练内对比**:
- v5 acc 0.5 (mean) vs v3 acc 0.43 (mean) — **v5 略高**
- v5 liveness 0.95-1.0 vs v3 0.34-0.37 — **v5 松绑效应显现**
- v5 length 130-220 vs v3 190-220 — **v5 更短 (length_bonus 鼓励)**
- v5 length_bonus_flag 0.5-0.94 vs v3 无 — **v5 新信号触发**

### 5. 关键风险评估

| 风险 | 当前状态 | 应对 |
|---|---|---|
| liveness 松绑过头 | 5-30 步内 0.95-1.0 持续满分, **松绑过头** | 监控 step 100 ckpt eval, 看 liveness 1.0 是否真的导致 chain_quality 0.98+ 但 answer 0.5 (答案对不上 chain) |
| KL 0.02 太低 | kl_loss 0.0001-0.0003, **policy 几乎不动 ref** | 监控 accuracy 是否单调涨, 跌则立即停, 反思 |
| length_bonus 触发长 chain hacking | 还没出现 (liveness=1.0 + length 0.875 flag 没让 len>400 持续) | 监控 len max 截断频率 |
| 训练挂 / Ray 状态污染 | 训练 5+ 分钟稳定推进 | 监控 GPU util, 0% 持续>2min 即重启 |

### 6. v5 quicktest 5 步 step_5 eval 数据 (完整)

| 指标 | v3 best | v5 quicktest step_5 | Δ |
|---|---:|---:|---:|
| Overall 5467 | 0.3047 | 0.3078 | **+0.31pp** ✅ |
| Original 1319 | 0.2995 | 0.3025 | **+0.30pp** ✅ |
| attribute_mismatch | 0.3097 | 0.3127 | +0.30pp |
| independent_decoy | 0.3140 | 0.3158 | +0.18pp |
| path_competition | 0.3033 | 0.3043 | +0.10pp |
| target_scope_misalignment | 0.2981 | 0.3049 | **+0.68pp** |

**5 步 noise 内 5 类目齐涨**, 罕见。但 5 步数据 **noise 范围 ±5pp**, 0.31pp **不可靠**。

### 7. v6 方向 (等 v5 跑完决定)

按对 0.46 目标贡献排序:

1. **A. 换 Qwen2.5-1.5B + 8-shot CoT 协议** (首选) — 0.5B + JSON 协议无解, 1.5B 8-shot CoT 上限 72.25%
2. **D. 0.5B + 8-shot CoT 协议** (备选) — 0.5B 自由推理上限 43%, RL 后 45-50%, 仍差 0.46
3. **C. v5.1 = v5 reward 微调** (小幅) — 0.5B + JSON 协议继续挤, 31-32% 封顶
4. **B. 换 1.5B + JSON 协议** (中等风险) — JSON 协议 1.5B 上限未知
5. **E. 数据增强** (长尾) — 0.5B 32-35% 封顶

**若 v5 撞 30-31%**: 直接 v6 = A 方向 (1.5B + 8-shot CoT)
**若 v5 涨到 32%+**: 考虑 v5.1 = C 方向微调 (answer 0.55→0.65 或 length_bonus 0.05→0.10)
**若 v5 撞 30% 以下**: 立即反思 v5 reward 改动是否反而有害, 退到 v3 best

### 8. 下一步操作 (按真实时间)

| 时间 | 操作 |
|---|---|
| **现在 (05:34)** | 写阶段总结 (本文件) + 等 v5 step 100 评测 |
| **~06:00 (真实 25 min 后)** | v5 step 100 ckpt 评测, 看 v5 真实表现 vs v3 best 30.47% |
| **~06:20 (45 min 后)** | v5 step 200 ckpt 评测 |
| **~06:50 (75 min 后)** | v5 训练完 500 步 |
| **~07:20 (105 min 后)** | v5 5 个 ckpt 评测完 |
| **~07:30 (115 min 后)** | v5 报告落盘 + 反思 + 决定 v5.1 or v6 方向 |

### 9. Codex 沙箱限制说明

**关键限制**: Codex CLI 沙箱的 `sleep` 命令**不等价于真实时间**。本对话 sleep 30 分钟, 训练进程只推进了 ~25 步 (~5 min 真实训练时间)。

**这意味着**:
- 我**无法在一个对话窗口内等 v5 跑完**
- 我**只能给阶段性报告**, 后续对话继续
- 训练日志 `/tmp/v5_full.log` 实时更新
- monitor 脚本 `/tmp/monitor_v5.sh` 可随时跑

**后续操作**:
- 用户在后续对话中可继续用 `bash /tmp/monitor_v5.sh` 看 v5 进度
- v5 训练完会**自动触发 eval** (脚本里 for 循环)
- eval 完会有 `eval/step_100/eval_result.json` 等落盘
- 报告文档等用户/agent 继续写

### 10. 监控命令速查

```bash
# v5 训练进度
bash /tmp/monitor_v5.sh

# v5 详细训练 metrics
cat outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/metrics/train_metrics.jsonl

# v5 eval 结果
for d in outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/eval/step_*; do
  echo "=== $d ==="
  cat $d/summary_overall.jsonl 2>/dev/null
done
```

---

## 决策记录 (2026-06-14 05:34)

- ✅ 2026-06-14 04:50-04:58: v5 quicktest 第一次启动失败 (环境)
- ✅ 2026-06-14 05:16-05:26: v5 quicktest 第二次启动成功 (5 步 + eval)
- ✅ 2026-06-14 05:26: 启动 v5 完整 500 步训练 (起点 v3 best, KL 0.02)
- ✅ 2026-06-14 05:30: v5 quicktest 反思文档落盘
- ✅ 2026-06-14 05:34: 本阶段总结文档落盘
- ⏳ 2026-06-14 06:00: 等 v5 step 100 评测
- ⏳ 2026-06-14 06:50: 等 v5 训练完
- ⏳ 2026-06-14 07:20: 等 v5 5 ckpt 评测完
- ⏳ 2026-06-14 07:30: 落 v5 报告 + 决定 v5.1/v6 方向

---

**结论**:
- **现状**: v5 训练在跑, ~6% 进度, 健康, step 30
- **短期**: 持续监控 v5 训练, 等 step 100 ckpt 评测
- **中期**: v5 报告 + 反思 + 决定 v5.1 或 v6 方向
- **长期**: **v6 = 换 1.5B + 8-shot CoT 协议是 0.46 目标的唯一路径**
- **立即**: 报告已落盘, v5 在跑, 等下一步交互
