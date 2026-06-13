# ChainGSM 干净测试集

本目录保存经过 DeepSeek 强模型审计后整理的 ChainGSM 测试集。

## 过滤规则

- 完整保留原测试集中的 1,319 条 `original` 记录。
- 仅删除 DeepSeek 审计结果中 `flagged=true` 的变体记录。
- 不修改任何保留记录的字段、字段顺序或字段值。
- 保持原始数据集中的记录顺序。

## 文件

- `gsm8k_test_clean.jsonl`：清理后的测试集。
- `cleaning_stats.json`：清理规则、来源和类别数量。

来源数据：

`chaingsm_data/data/final/gsm8k_test_full/gsm8k_test_all.jsonl`

审计结果：

`chaingsm_data/reports/deepseek_gold_audit/full/audit_records.jsonl`
