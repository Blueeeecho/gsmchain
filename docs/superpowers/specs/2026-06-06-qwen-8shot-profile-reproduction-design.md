# Qwen 8-shot 提示模板复现实验设计

## 目标

在已清洗的 5,467 条 ChainGSM 测试集上，恢复此前已经验证有效的 Qwen
8-shot 提示模板，并重新评测以下基础模型：

- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Qwen2.5-Math-1.5B-Instruct

本次不评测 Llama，也不修改模型权重。

## 已确认根因

当前普通 Qwen 使用单个 user 消息承载全部 8 个示例，导致
Qwen2.5-1.5B-Instruct 的 Original 准确率从旧版 71.42% 降到 53.53%。

Qwen2.5-Math-1.5B-Instruct 当前还被普通 Qwen 的 system prompt 与 chat
序列化覆盖。5,467 条输出中有 3,965 条仅复述
`You are a helpful assistant.`，因此 8.18% 的总体准确率不是有效能力结果。

## 提示模板

### 普通 Qwen2.5-Instruct

0.5B、1.5B 和 3B 使用完全相同的多轮 8-shot chat：

1. 一条 system 消息，要求逐步解决数学问题。
2. 八组 user 问题与 assistant 推理解答。
3. 最后一条 user 消息为待测问题，并以
   `A: Let's think step by step.` 引导回答。
4. 使用各 checkpoint 自带的 chat template 渲染。

三个尺寸的普通 Instruct 模型共享同一组消息、生成参数、停止条件和答案抽取。

### Qwen2.5-Math-Instruct

Math 模型使用此前达到 78.17% Original 准确率的纯 completion 8-shot：

```text
Q: 示例问题
A: 示例推理与答案

...

Q: 当前问题
A: Let's think step by step.
```

该路径不调用 chat template，也不注入普通 Qwen system prompt。

## 统一实验条件

- 数据：`chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl`
- 样本数：5,467
- temperature：0
- top-p：1
- max new tokens：512
- seed：42
- 相同数值答案抽取与正确性判定
- 相同自动显存比例回退
- 相同 JSONL checkpoint 与断点续跑

## 验证

单元测试覆盖：

- 普通 Qwen 模型统一路由到多轮 8-shot chat。
- 多轮消息包含 system、八组 user/assistant 示例和最终 user。
- Math 模型独立路由到 completion profile。
- Math completion 不包含 chat role token 或普通 system prompt。
- 普通 Qwen 使用 `<|im_end|>` 停止，Math completion 使用生成下一题的
  `Q:` 等文本边界停止。
- 默认模型列表仅包含四个 Qwen 模型。

烟测覆盖每个模型的少量 Original 样本。Math 烟测必须确认不再大量复述 system
prompt，且能够稳定抽取数值答案。

## 成功标准

- 四个模型均生成 5,467 条预测。
- 没有模型级加载或生成失败。
- 普通 Qwen Original 结果与 AbstRaL CoT-8S 的尺寸趋势一致：
  42.4%、67.0%、81.2% 附近。
- Qwen2.5-1.5B-Instruct 明显恢复到旧版约 71.4% 的水平。
- Qwen2.5-Math-1.5B-Instruct Original 接近旧版约 78.2%，且不再出现大规模
  system prompt 回显。
- 输出总体与各类别准确率，并记录每个模型实际使用的 prompt profile。
