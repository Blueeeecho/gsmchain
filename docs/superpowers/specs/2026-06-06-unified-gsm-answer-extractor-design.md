# GSM 数值答案统一提取器设计

## 目标

将 GSM8K、GSM-Plus 和 ChainGSM 的自由文本数值答案解析固化为仓库唯一标准。
解析过程必须确定、与 gold 答案无关，并且相同输出始终得到相同结果。

## 适用范围

统一提取器供以下评测入口共同使用：

- `code/eval_official_gsm.py`
- `code/eval_chaingsm.py`
- `code/eval_abstral_baselines.py`
- `code/eval_chaingsm_base_8shot.py`
- `train_pipeline/eval_vllm_chaingsm.py`

共享实现放在 `code/gsm_answer_extractor.py`。现有脚本不再维护各自的正则和
数值归一化实现，但可以重新导出同名函数以保持现有调用兼容。

## 确定性解析优先级

提取器不读取题目、类别或 gold，只处理模型输出，并按以下固定顺序返回第一个有效结果：

1. 截断模型自行生成的下一道 `Question:` 或 `Q:`。
2. 取最后一个明确答案标记：`Final answer`、`####`。
3. 取最后一个平衡的 `\boxed{...}`，支持嵌套 LaTeX 分数。
4. 检查最后一句；若包含等式，取最后一个等号右侧的第一个数。
5. 检查最后两句中以 `So`、`Therefore`、`Thus`、`Hence`、
   `Consequently`、`Together` 或 `Finally` 开始的结论：
   - 有等式时取最后一个等号右侧的第一个数；
   - 否则取答案谓词之后的第一个数，例如
     `will take 12 days`、`spends $75 over 30 days`。
6. 检查显式 `Answer:` 标记。
7. 最后才回退到全文最后一个数。

以上规则是确定性语法解析，不根据 gold 选择候选答案。对于没有任何答案结构的自由文本，
最后数字回退仍属于必要的兼容策略；若要彻底消除此回退，生成端必须强制输出
`#### <answer>` 或结构化 JSON。

## 数值语义

- 支持整数、小数、带逗号数字、正负号和分数。
- 支持 `\frac{a}{b}` 与 `-\frac{a}{b}`。
- `$` 和 `%` 只作为展示符号移除，不自动执行百分比缩放。
- 使用 `Fraction` 比较，保留现有相对/绝对容差。
- 无法解析时返回 `None`，不抛出评测级异常。

## 测试标准

回归测试必须覆盖：

- 明确答案标记、boxed、LaTeX 分数和下一题截断；
- 结论中的答案位于背景数字之前；
- 结论中的答案位于持续时间、目标总量或输入数量之前；
- 最后一句等式右侧解析；
- 已正确解析的旧样本不发生回归；
- 0.5B 当前 1,319 条 Original 只重算后达到 `571/1319 = 43.29%`。

## 迁移策略

先建立共享模块和回归测试，再逐个替换四套重复实现。迁移不修改 prompt、生成参数或
已有 raw output。历史预测通过重汇总脚本按统一标准更新 `pred_answer`、`correct` 和摘要，
原始 `raw_output` 保持不变。
