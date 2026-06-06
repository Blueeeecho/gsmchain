# Qwen 8-shot 提示模板复现实验实施计划

> **供自动化执行代理使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项实施本计划。步骤使用复选框跟踪。

**目标：** 修复 Qwen 基础模型的 8-shot profile 路由，并在 5,467 条干净 ChainGSM 测试集上重新运行四个 Qwen 模型。

**架构：** 在现有 `eval_chaingsm_base_8shot.py` 中增加普通 Qwen 多轮 chat 和 Qwen-Math completion 两个明确 profile。普通 Qwen 三个尺寸共享完全相同的消息与设置；Math 绕过 chat template。评测、自动显存回退、答案抽取和报告逻辑保持不变。

**技术栈：** Python 3.12、pytest、Transformers tokenizer、vLLM、JSONL/CSV。

---

### 任务 1：增加 profile 路由回归测试

**文件：**
- 修改：`tests/test_chaingsm_base_8shot_eval.py`

- [ ] **步骤 1：编写普通 Qwen 路由失败测试**

增加测试，要求 0.5B、1.5B、3B 均返回
`qwen_multiturn_8shot_chat`，并要求默认模型列表不再包含 Llama。

- [ ] **步骤 2：编写 Math 路由失败测试**

增加测试，要求 Qwen2.5-Math-1.5B-Instruct 返回
`qwen_math_completion_8shot`。

- [ ] **步骤 3：运行测试并确认失败**

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest \
  tests/test_chaingsm_base_8shot_eval.py -v
```

预期：旧 `qwen_single_turn_8shot_chat` 路由和五模型列表导致测试失败。

### 任务 2：实现普通 Qwen 多轮 8-shot

**文件：**
- 修改：`code/eval_chaingsm_base_8shot.py`
- 修改：`tests/test_chaingsm_base_8shot_eval.py`

- [ ] **步骤 1：编写多轮消息结构失败测试**

断言普通 Qwen 消息包含 18 条消息：

```python
["system", "user", "assistant", ..., "user", "assistant", "user"]
```

最后一条 user 包含当前问题与 `A: Let's think step by step.`。

- [ ] **步骤 2：实现普通 Qwen 消息构造**

复用旧版 `EIGHT_SHOT_EXAMPLES` 的八组问题和推理解答，并通过 tokenizer
原生 chat template 渲染为字符串。

- [ ] **步骤 3：运行测试并确认普通 Qwen profile 通过**

执行定向 pytest，要求多轮结构、字符串输入和 `<|im_end|>` stop 均通过。

### 任务 3：实现 Qwen-Math completion 8-shot

**文件：**
- 修改：`code/eval_chaingsm_base_8shot.py`
- 修改：`tests/test_chaingsm_base_8shot_eval.py`

- [ ] **步骤 1：编写 completion 输入失败测试**

断言 Math 输入是纯字符串，包含九组 `Q:`/`A:`，不调用
`apply_chat_template`，也不包含 `You are a helpful assistant.` 或
`<|im_start|>`。

- [ ] **步骤 2：实现 Math completion 构造与停止条件**

构造旧版 `eight_shot_cot_completion` 字符串，并使用 `Q:`、
`Question:`、`<|im_end|>` 等边界阻止模型继续生成下一题。

- [ ] **步骤 3：运行全部定向测试**

要求普通 Qwen、Math、自动显存候选和答案抽取相关测试全部通过。

- [ ] **步骤 4：提交 profile 修复**

只提交评测脚本和对应测试，不包含实验输出。

### 任务 4：四模型烟测

**运行输出：**
- `code/results/chaingsm_base_8shot/<timestamp>/`

- [ ] **步骤 1：运行普通 Qwen 烟测**

每个普通 Qwen 模型运行少量样本，确认 prompt diagnostics 均记录
`qwen_multiturn_8shot_chat`，输入为 chat template 渲染字符串。

- [ ] **步骤 2：运行 Math 烟测**

运行少量 Original，确认 profile 为 `qwen_math_completion_8shot`，输出不再大量
复述 system prompt，且大部分样本可抽取答案。

- [ ] **步骤 3：检查烟测结果**

只有四个模型均无加载错误、预测数完整、Math 回显问题消失后才进入全量。

### 任务 5：全量运行与结果对标

**输入：**
- `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl`

**输出：**
- `code/results/chaingsm_base_8shot/<timestamp>/`

- [ ] **步骤 1：运行四个 Qwen 模型**

使用 temperature 0、top-p 1、max tokens 512、seed 42 和自动显存回退。

- [ ] **步骤 2：验证完整性**

每个模型必须各有 5,467 条预测，且没有模型级生成失败。

- [ ] **步骤 3：生成整体与分类汇总**

输出总体准确率、Original 准确率和四类变体准确率。

- [ ] **步骤 4：与参考数据对标**

重点比较：

- 普通 Qwen Original：AbstRaL 42.4%、67.0%、81.2%。
- Qwen2.5-1.5B 旧版：71.42%。
- Qwen2.5-Math-1.5B 旧版：78.17%。
- 同一干净数据上旧预测总体：1.5B 44.12%、Math 52.35%。

- [ ] **步骤 5：输出中文实验报告**

记录精确配置、prompt profile、完整性检查、总体和分类准确率，以及未达到参考值时的剩余差异。
