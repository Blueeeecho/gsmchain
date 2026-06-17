# SFT (AbstRaL 超参) + CoT 双尖括号协议 训练结果报告 (2026-06-17)

> 关联 spec: docs/superpowers/specs/2026-06-17-sft-abstral-params-cot-protocol-design.md
> 关联 plan: docs/superpowers/plans/2026-06-17-sft-abstral-params-cot-protocol-plan.md
> Run 目录: outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/
> Eval 目录: outputs/sft_cot_eval/sft_cot_2ep_<RUN_ID>/

## 1. 训练配置 (实际)

| 项 | 值 |
|---|---|
| 起点 | Qwen2.5-0.5B-Instruct 原生 base |
| 数据 | chaingsm_data/data/final/sft/all_sft_cot.jsonl (6094 条) |
| epoch | 2 |
| per_device_batch_size | 2 |
| gradient_accumulation_steps | 4 (effective 8) |
| learning_rate | 5e-6 (cosine) |
| max_length | 1024 |
| optimizer | AdamW(0.9, 0.999, 1e-8) |
| bf16 | true |
| 训练时长 | <填充> |
| GPU | RTX 5090 32GB |

## 2. 训练 loss 收敛

| 阶段 | loss | 备注 |
|---|---:|---|
| step 10 (epoch 0.1) | <填充> | |
| epoch 1 末 | <填充> | |
| epoch 2 末 | <填充> | |
| ratio last/first | <填充> | <0.7 算收敛 |

## 3. 评测结果 (cot_brackets 协议, 5,467 条干净集)

| 类别 | 数量 | correct | accuracy |
|---|---:|---:|---:|
| overall | 5,467 | <填充> | <填充> |
| original | 1,319 | <填充> | <填充> |
| independent_decoy | 1,102 | <填充> | <填充> |
| attribute_mismatch | 1,017 | <填充> | <填充> |
| path_competition | 999 | <填充> | <填充> |
| target_scope_misalignment | 1,030 | <填充> | <填充> |

## 4. 关键发现

<填充: 跟 0.5B base 8-shot CoT 0.4329 original baseline 对比>

## 5. 已知限制

- **本次只跑了 1 次 eval, 没跑 8-shot 横向对比**
- 训练 ckpt 保留: epoch1, epoch2, current, best
- best == current == epoch2 (train_sft_trl 末段: 不做 best 选取, 直接 copy)

## 6. 关键文件位置

- 训练: outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/
- 评测: outputs/sft_cot_eval/sft_cot_2ep_<RUN_ID>/
- 代码: train_pipeline/train_sft_trl.py, train_pipeline/eval_vllm_chaingsm.py
- 配置: train_configs/local/sft.yaml
- 数据: chaingsm_data/data/final/sft/all_sft_cot.jsonl (不动)
