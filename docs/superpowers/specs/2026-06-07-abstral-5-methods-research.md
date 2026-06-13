# AbstRaL 论文 5 个对照方法调研与重跑决策

> **生成日期**：2026-06-07
> **目标方法**：CoT-RL、PoT-8S、CoA、AoT、SyReLM
> **对应基线**：AbstRaL 论文 Table 5 的 GSM-Plus Original / Rephrase / Distract(CoT-8S 对照组)
> **本地已有**：AbstRaL 主方法 `abstral_style_two_stage_prompting`、GranuLar 一步法 `granular_style_prompting`、仓库自带的 8-shot CoT baseline
> **状态**：方法学调研 + 重跑必要性 + 实现难度评估(本文档不写代码)

---

## 0. 重要前提

> 我没有 AbstRaL 论文原文可读。下面 5 个方法的"实现描述"来自我对方法学本身与各原论文的常识级理解,以及仓库中 `eval_abstral_baselines.py` 已经写好的 AbstRaL 主方法作为间接参照。
> 
> 如果仓库里没有 `papers/abstral.pdf` 或同等材料,最终实现前请人工核对 AbstRaL 论文 Table 5 的方法说明、提示词模板与原论文 cite。本 spec 只用于"是否值得重跑、难度如何"的决策。

仓库当前只有 `code/eval_abstral_baselines.py`(含 `abstral_style_two_stage_prompting` + `granular_style_prompting`)可作为 prompt 风格与解析风格的参考实现。

---

## 1. AbstRaL 主方法(本机已实现,作为对照)

> 此项已实现,本节用于对比 5 个对照方法时作为锚点。

- **本机实现**:`code/eval_abstral_baselines.py:evaluate_abstral_two_stage`(`ABSTRAL_METHOD = "abstral_style_two_stage_prompting"`)
- **两阶段**:
  1. Stage 1:把题面数值替换为 `[in0], [in1], ...` 抽象符号,输出 `<abstract_question>` + `<conditions>` 块。
  2. Stage 2:基于抽象题面,生成 `<subquestions>` / `<abstract_reasoning>` / `<final_var>` / `<final_expression>` / `<answer>`,期间用 sympy 校验 `<<in0+in1=out0>>` 形式推导的可求值性。
- **本机状态**:只在 5,467 干净集上跑过 5 条烟测(`code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_163320/`),**全量未跑**。

---

## 2. 5 个对照方法逐一评估

### 2.1 CoT-RL (Chain-of-Thought with Reinforcement Learning)

- **方法学**(常识级):在标准 CoT 提示上,用 RL(PPO/GRPO/DPO)对模型本身做微调,reward 一般是"最终答案正确"或"答案 + 推理步骤正确"。AbstRaL 论文 Table 5 引用的一般是"同尺寸 Instruct + RL 训练"得到的 checkpoint。
- **本机可复现性**:**几乎不可能直接复现**——需要:
  - AbstRaL 用的具体 RL 算法、KL 系数、reward shaping、训练步数。
  - 同款 RL 训练数据(可能并非公开)。
  - 训练后的 checkpoint 权重。
- **重跑必要性**:**不重跑**。我们要做的是"在同提示词下对比同尺寸 Instruct 模型",而非复现 RL 训练。
- **实现难度**:**N/A**——跳过。
- **建议**:如果一定要对比 RL 效果,改为"用本仓库的 GRPO 训练(在 14,946 条 SFT/DPO 数据上跑若干步),checkpoint 跑同 8-shot 评测"。这与 AbstRaL 的 CoT-RL 不是同一设置,不能放同一列对比。

### 2.2 PoT-8S (Program of Thought, 8-Shot)

- **方法学**(常识级,原论文 Chen et al. 2022):模型被提示生成 Python 代码而不是自然语言 CoT,代码执行后得到最终答案。8-shot 指 8 条代码示例。
- **本机可复现性**:**可复现**。仓库里 `code/eval_official_gsm.py` 已经有 PoT 风格的代码生成示例(可借鉴);`eval_abstral_baselines.py` 的 sympy 解析逻辑可以改造成 Python `ast` 解析。
- **重跑必要性**:**低**。PoT-8S 与 CoT-8S 在 5,467 干净集上的对比"是否能降低 distractor 影响"才是有意义的——而本项目主线是 8-shot CoT 评测,引入 PoT 需要新写评测入口 + sandbox 执行代码(安全风险)。
- **实现难度**:**中高**。难点:
  - 提示词模板与 8 个示例需要手工构造(可参考 `eval_official_gsm.py` 的 few-shot)。
  - 代码执行:必须用 subprocess / RestrictedPython / 容器沙箱,不能直接 `exec`,且要控制超时。
  - 答案提取:从代码最后一行 `print` 或变量赋值中取数值,加 `is_correct`。
  - ChainGSM 的 distractor chain 不会自然出现在 Python 代码里——PoT 在本任务上可能**没有 ablation 意义**(它解决的是"算错了",不是"选错链")。
- **建议**:**先不重跑**。理由:PoT 对 distractor 不敏感,做不出我们关心的"鲁棒性差异"信号。如果用户明确要 PoT,再写 PoT-8S spec,作为"算错"对照而非"选错"对照。

### 2.3 CoA (Chain of Abstraction, Microsoft 2024)

- **方法学**(常识级,原论文 Wang et al. 2024):把推理拆成"先写抽象占位符链 → 再实例化填值"两步。模型先输出带占位符的 CoA 草稿,再在第二轮注入具体数值。
- **本机可复现性**:**可复现**。CoA 论文公开了完整提示词模板与示例。
- **重跑必要性**:**中**。CoA 与 AbstRaL 同属"抽象推理"流派,放在一起对比是论文的核心 ablation。但本项目主线 5,467 干净集还没跑过 AbstRaL 主方法,先做 AbstRaL 主方法才有对照点。
- **实现难度**:**中**。难点:
  - 两阶段调用,与 AbstRaL 主方法共用同一套 vLLM 双发调用框架,改提示词即可。
  - 解析:`<scratchpad>` 里的占位符、`<answer>` 里的最终值。
  - 第二阶段要把占位符替换为真实数值,需要小心 LaTeX/数值边界。
- **建议**:**作为 AbstRaL 之后第二个补做的方法**。与 AbstRaL 主方法共享 80% 框架代码,边际成本低。

### 2.4 AoT (Atom of Thought, 2024)

- **方法学**(常识级,原论文 Tong et al. 2024):把问题递归拆成"原子子问题",每个原子子问题独立解决后合并。常用 `DAG` / 树状结构表示。
- **本机可复现性**:**可复现**。原论文公开了 few-shot 提示模板。
- **重跑必要性**:**中**。AoT 的"分解-合并"思路对"忽略 distractor chain"有帮助,但实现成本高于 CoA,且本项目目标是 distractor 鲁棒性,AOT 不一定带来明显增益。
- **实现难度**:**中高**。难点:
  - 原子分解的输出格式(`<sub-q>` / `<atom>` / `<combine>`)。
  - 合并阶段的去重 / 冲突解决。
  - 单条样本可能产生几十次中间调用,推理吞吐最差。
- **建议**:**低优先级**。先把 AbstRaL + CoA 跑通,再考虑 AoT。

### 2.5 SyReLM (Symbolic Regression Language Model)

- **方法学**(常识级,与 AbstRaL 同期的对比方法):用 LLM 拟合"数值-数值"映射,再用符号回归(SymPy / PySR)反解显式表达式,最后代入求值。
- **本机可复现性**:**难**。需要:
  - 数值表(题目里所有数值 + 答案) → 让 LLM 写 Python/SymPy 拟合代码。
  - 显式表达式 + 数值求值 + 判等。
  - PySR 之类的符号回归器 + 调参。
- **重跑必要性**:**低**。SyReLM 在 AbstRaL Table 5 里是"另类对照组",目的是展示"纯符号方法也能跑"。本项目关心的不是数值拟合,跳过。
- **实现难度**:**高**。需要 PySR/EGGP 之类依赖,本地没装,且调参成本大。
- **建议**:**不重跑**。

---

## 3. 5 个方法汇总表

| 方法 | 类型 | 重跑必要性 | 实现难度 | 建议 |
|---|---|---|---|---|
| CoT-RL | RL 训练 | 不重跑(本项目不是 RL 复现) | N/A | 跳过 |
| PoT-8S | 代码生成 | 低(对 distractor 不敏感) | 中高 | 跳过 |
| CoA | 抽象推理两阶段 | 中 | 中 | 第二个补做 |
| AoT | 原子分解 | 中 | 中高 | 低优先级 |
| SyReLM | 符号回归 | 低 | 高 | 跳过 |
| **AbstRaL 主方法** | 抽象推理两阶段 | **高(本机未全量)** | 已实现 | **本周重跑** |

---

## 4. 本周动作(就 AbstRaL 主方法)

1. 把 `eval_abstral_baselines.py` 在 5,467 干净集 + batch=16 上跑一遍,产物落到 `code/results/abstral_5k_8shot/<timestamp>/`。
2. 沿用现有 `prompt_diagnostics.json` 风格,加 `stage1_diagnostics.json` + `stage2_diagnostics.json`。
3. 用 `gsm_answer_extractor.is_correct` 与 sympy 表达式校验"双判定":既看最终数值,又看 `final_expression` 是否能取到 gold。
4. 跑完生成 `docs/abstral_5k_results_2026-06-XX.md`,与 CoT-8S batch=16 直接对比。

---

## 5. 中期动作(待用户确认)

- CoA spec(预计 1-2 天,与 AbstRaL 共享 80% 框架)。
- AoT spec(预计 3-4 天,需要新框架)。
- PoT-8S spec(若用户明确要"算错"对照)。
- SyReLM:不列入计划。

---

## 6. 决策记录

| 决策 | 结论 | 理由 |
|---|---|---|
| 是否重跑 CoT-RL? | 否 | 不是 RL 复现项目,且无权重与训练数据 |
| 是否重跑 PoT-8S? | 否(默认) | 解决"算错"问题,与本项目"选错链"目标不重叠 |
| 是否重跑 CoA? | 是(在 AbstRaL 主方法后) | 共享框架,边际成本低 |
| 是否重跑 AoT? | 待定 | 实现成本高、信号未必显著 |
| 是否重跑 SyReLM? | 否 | 依赖重、调参成本高、信号不直接 |
| 是否重跑 AbstRaL 主方法? | **是(本周)** | 仓库已实现但只跑过 5 条烟测,论文核心方法必须全量 |
