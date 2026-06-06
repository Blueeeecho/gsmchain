# GSM 数值答案统一提取器实施计划

> **供自动化执行代理使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项实施本计划。步骤使用复选框跟踪。

**目标：** 建立仓库唯一的 GSM 数值答案解析标准，并迁移所有现有评测入口。

**架构：** 新建无模型依赖的 `code/gsm_answer_extractor.py`，集中实现提取、归一化和判分。现有评测脚本导入并重新导出共享函数，保持调用兼容；历史预测用共享函数离线重汇总。

**技术栈：** Python 3.12、正则表达式、`fractions.Fraction`、pytest、JSONL。

---

### 任务 1：建立共享提取器回归测试

**文件：**
- 新建：`tests/test_gsm_answer_extractor.py`

- [ ] 编写明确标记、boxed、分数、下一题截断测试。
- [ ] 编写 `$75 over 30 days`、`4 hours ... 30 ... 15`、
  `12 days ... 600 inches` 等真实误取测试。
- [ ] 编写最后一句等式和现有正确案例不回归测试。
- [ ] 运行测试并确认因共享模块不存在而失败。

### 任务 2：实现确定性共享解析器

**文件：**
- 新建：`code/gsm_answer_extractor.py`
- 测试：`tests/test_gsm_answer_extractor.py`

- [ ] 实现下一题截断、平衡括号、答案标记和数值解析。
- [ ] 实现最后一句等式与结论谓词解析。
- [ ] 实现 `normalize_answer` 和 `is_correct`。
- [ ] 运行共享模块测试并确认通过。

### 任务 3：迁移所有评测入口

**文件：**
- 修改：`code/eval_official_gsm.py`
- 修改：`code/eval_chaingsm.py`
- 修改：`code/eval_abstral_baselines.py`
- 修改：`train_pipeline/eval_vllm_chaingsm.py`
- 测试：`tests/test_official_gsm_eval.py`

- [ ] 删除四处重复实现，改为导入共享函数。
- [ ] 保持原模块的同名导入接口兼容。
- [ ] 运行全部提取器和评测辅助测试。

### 任务 4：重算历史结果并验证

**文件：**
- 新建：`code/rescore_gsm_predictions.py`
- 测试：`tests/test_rescore_gsm_predictions.py`

- [ ] 实现 JSONL 原地安全重算，保留 `raw_output`。
- [ ] 更新 `pred_answer` 和 `correct`，并重写模型及组合摘要。
- [ ] 对当前 0.5B 结果重算，确认 Original 为 `571/1319 = 43.29%`。
- [ ] 等 1.5B 推理完成后用同一标准重算。

### 任务 5：最终验证

- [ ] 运行：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest \
  tests/test_gsm_answer_extractor.py \
  tests/test_official_gsm_eval.py \
  tests/test_chaingsm_base_8shot_eval.py \
  tests/test_rescore_gsm_predictions.py -q
```

- [ ] 执行 `python -m py_compile` 检查所有修改脚本。
- [ ] 检查 Git diff，确保没有提交模型输出或用户无关改动。
