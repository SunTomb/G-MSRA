# G-MSRA 项目工作流程 v7.1 — 评测进展跟踪 + 剩余任务收尾

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 4 月 13 日 · 基于 [v7.0](PROJECT_WORKFLOW7.0.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 侧重 | 状态 |
|------|------|------|
| v1.0 ~ v5.1 | 项目搭建 → Phase 1 RL → Baseline 评测 | ✅ 背景资料 |
| [v6.0](PROJECT_WORKFLOW6.0.md) | Phase 2 + RL Baseline 完成，全面总结 | ✅ 已归档 |
| [v6.1](PROJECT_WORKFLOW6.1.md) | Phase 3 首轮诊断 + 代码修复 + 重训指南 | ✅ 已归档 |
| [v7.0](PROJECT_WORKFLOW7.0.md) | 全三阶段训练完成，进入评测与论文阶段 | ✅ 已归档 |
| **v7.1（本文档）** | **评测进展跟踪 + Bug 修复 + 剩余任务收尾** | ✅ 当前版本 |

---

## 一、已完成实验结果总览

### 1.1 G-MSRA 主模型评测 ✅ 完成

> Phase 2 与 Phase 3 v5 产生了**完全一致**的 byte-for-byte 预测，因此只报一行。

| Benchmark | F1 | EM | 内存条数 | 耗时 | 日志 |
|-----------|:---:|:---:|:---:|:---:|------|
| LoCoMo (400 examples) | **0.0962** | 0.0450 | 183 | ~38min | `logs/eval_phase2_locomo.log` |
| LongMemEval (500 examples) | **0.2616** | 0.1760 | 183 | ~109min | `logs/eval_phase2_longmemeval.log` |

**LongMemEval 分类详情**：

| Category | F1 | EM | n | 说明 |
|:--------:|:---:|:---:|:---:|------|
| [3] Knowledge Update | **0.5053** | 0.4306 | 72 | ⭐ 旗舰能力 |
| [1] Single-session | 0.3524 | 0.2533 | 150 | 信息提取 |
| [4] Temporal Reasoning | 0.1749 | 0.0945 | 127 | 时间推理 |
| [2] Multi-session | 0.1384 | 0.0826 | 121 | 跨 session |
| [5] Abstain | 0.0869 | 0.0333 | 30 | 弃权检测 |

**LoCoMo 分类详情**：

| Category | F1 | EM | n |
|:--------:|:---:|:---:|:---:|
| [3] | 0.3450 | 0.3500 | 20 |
| [4] | 0.1062 | 0.0437 | 160 |
| [1] | 0.0558 | 0.0290 | 69 |
| [2] | 0.0086 | 0.0000 | 65 |
| [5] | 0.0000 | 0.0000 | 86 |

结果文件：`results/eval_phase2/`, `results/eval_phase3v5/`

---

### 1.2 消融实验 T5-A0（No Memory）✅ 完成

**纯 LLM 推理**，跳过 memory snapshot/restore 和 event ingestion，使用 `--no_memory` 参数。

| Benchmark | F1 | EM | 耗时 | 日志 |
|-----------|:---:|:---:|:---:|------|
| LoCoMo | 0.0272 | 0.0175 | 18.9min | `logs/ablation_no_memory_locomo.log` |
| LongMemEval | 0.0490 | 0.0140 | 24.3min | `logs/ablation_no_memory_longmemeval.log` |

结果文件：`results/ablation_no_memory/`

---

### 1.3 Baseline 评测 T4（部分完成）

| Baseline | LoCoMo F1 | LongMem F1 | 状态 | 说明 |
|----------|:---------:|:----------:|:----:|------|
| Reflexion | **0.0163** | **0.0408** | ✅ 完成 | `results/baselines_v2/reflexion/results.json` |
| EvolveR | — | — | ❌ 被 kill（模型加载中） | 需重跑 |
| Self-Consolidation | — | — | ❌ 未开始 | 需重跑 |
| Memory-R1 | — | — | ❌ OOM @ optimizer.step() | 需 `--eval_only` |
| Mem0+Memory-R1 | — | — | ❌ 前序失败 | 需 `--eval_only` |

---

### 1.4 当前数据汇总（论文 Table 1 + Table 2 进度）

#### Table 1: 主结果

```
Method              | LoCoMo F1 | LoCoMo EM | LongMem F1 | LongMem EM | 状态
────────────────────┼───────────┼───────────┼────────────┼────────────┼──────
Reflexion           |   0.0163  |     —     |   0.0408   |     —      | ✅
EvolveR             |   待 T4   |   待 T4   |   待 T4    |   待 T4    | ❌
Self-Consolidation  |   待 T4   |   待 T4   |   待 T4    |   待 T4    | ❌
Memory-R1           |   待 T4   |   待 T4   |   待 T4    |   待 T4    | ❌
Mem0+Memory-R1      |   待 T4   |   待 T4   |   待 T4    |   待 T4    | ❌
────────────────────┼───────────┼───────────┼────────────┼────────────┼──────
G-MSRA (Ours)       |   0.0962  |   0.0450  |   0.2616   |   0.1760   | ✅
```

#### Table 2: 消融结果

```
Variant             | LoCoMo F1 | LongMem F1 | ΔF1 vs Full (LoCoMo) | ΔF1 vs Full (LongMem) |
────────────────────┼───────────┼────────────┼──────────────────────┼───────────────────────┤
G-MSRA (Full)       |   0.0962  |   0.2616   |         —            |          —            |
A0: No Memory       |   0.0272  |   0.0490   |      −71.7%          |       −81.3%          |
A0.5: Events Only   |   待 T5   |   待 T5    |        待            |         待            |
```

> **A0 消融结论**：Memory store 贡献了 **71.7% 的 LoCoMo 性能** 和 **81.3% 的 LongMemEval 性能**，
> 这是论文最关键的消融证据之一。

---

## 二、Bug 修复记录（v7.0 → v7.1 期间）

### 2.1 `baselines.py` 修复清单

| 修复 | 位置 | 问题描述 | 修复内容 |
|:----:|------|---------|---------|
| ① | L374 | `env_signal_kwargs` 参数名 `prediction`/`ground_truth` 与 `DialogueSignalExtractor.extract()` 签名不匹配 | → `agent_response`/`qa_ground_truth` |
| ② | L226-287 | `train_dialogue()` / `evaluate_dialogue()` 无进度日志，1000+ LLM 调用全程静默 | 每 10 步输出进度 |
| ③ | L441-514 | `MemoryR1Baseline` events 跨 example 累积（不隔离），导致 ~247K agent.step 调用 | per-episode reset + cap=20 events |
| ④ | L441-514 | `MemoryR1Baseline` 训练/评测无进度日志 | 每 10 步输出进度 |

### 2.2 `run_baselines.py` 新功能

| 功能 | 参数 | 用途 |
|------|------|------|
| 仅评测模式 | `--eval_only` | 跳过训练，避免 Memory-R1 OOM |
| 跳过已有结果 | `--skip_existing` | 不重跑已保存结果的 baseline（如 Reflexion） |
| DEBUG 日志抑制 | 自动生效 | 抑制 `memory_manager`/`grounded_reward`/`memory.store`/`agent` 的 DEBUG 输出 |

### 2.3 OOM 根因分析

Memory-R1 在 A40 (44GB 可用) 上 OOM：

```
模型参数 (bf16):     14.5 GB
LoRA adapter:        0.02 GB
Adam 优化器状态:     ~28 GB  ← 2× 模型参数 (exp_avg + exp_avg_sq)
梯度:                ~14.5 GB
──────────────────────────
总计:                ~57 GB  > 44 GB 可用 ❌
```

解决方案：`--eval_only` 跳过训练（不创建优化器），只做评测。

---

## 三、剩余任务与执行指南

### 3.1 任务一览

| # | 任务 | 需要重跑 | 预估耗时 | 优先级 |
|---|------|:--------:|:--------:|:------:|
| T4a | EvolveR baseline eval | ✅ | ~1.5h | ⭐⭐⭐⭐⭐ |
| T4b | SelfConsolidation baseline eval | ✅ | ~1.5h | ⭐⭐⭐⭐⭐ |
| T4c | Memory-R1 eval-only | ✅ | ~1.5h | ⭐⭐⭐⭐⭐ |
| T4d | Mem0+Memory-R1 eval-only | ✅ | ~1.5h | ⭐⭐⭐⭐⭐ |
| T5-A0.5 | Events Only 消融 | ✅ | ~0.5h | ⭐⭐⭐ |
| T6 | 训练消融 A1-A7 | ✅ | ~14-21h | ⭐⭐ |
| T7 | 论文数据汇总 + 画图 | — | ~2h | ⭐⭐⭐⭐ |
| T8 | 论文初稿撰写 | — | 若干天 | ⭐⭐⭐⭐⭐ |

串行总时间：~8h（不含 T6 训练消融和 T8 论文）
2 GPU 并行：**~4h** 即可完成 T4+T5

---

### 3.2 环境准备

```bash
ssh wujcan@Tang2   # 或 Tang3
cd /NAS/yesh/G-MSRA
export PYTHONPATH=/NAS/yesh/G-MSRA
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
```

> ❗ **同步代码**：运行前确保以下文件已从本地同步到服务器：
>
> - `gmsra/baselines.py` （修复 ①②③④）
> - `scripts/run_baselines.py` （新增 `--eval_only` / `--skip_existing` / DEBUG 抑制）

---

### 3.3 T4: Baseline 补全（最高优先级）

#### Part1: EvolveR + SelfConsolidation（训练 + 评测）

Reflexion 已完成，使用 `--skip_existing` 跳过。

```bash
tmux new -s baselines_p1

CUDA_VISIBLE_DEVICES=1 python scripts/run_baselines.py \
    --baselines reflexion,evolver,self_consolidation \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --max_train_episodes 100 \
    --output_dir results/baselines_v2 \
    --skip_existing \
    2>&1 | tee logs/baselines_v2_part1.log
```

- **预计行为**：跳过 Reflexion → 训练+评测 EvolveR (~1.5h) → 训练+评测 SelfConsolidation (~1.5h)
- **总耗时**：~3h
- **显存**：~15GB

#### Part2: Memory-R1 + Mem0（仅评测，避免 OOM）

```bash
tmux new -s baselines_p2

CUDA_VISIBLE_DEVICES=1 python scripts/run_baselines.py \
    --baselines memory_r1,mem0_memory_r1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --max_train_episodes 100 \
    --output_dir results/baselines_v2 \
    --eval_only \
    2>&1 | tee logs/baselines_v2_part2.log
```

- **预计行为**：跳过训练 → 直接评测 LoCoMo + LongMemEval（每个 ~1.5h）
- **总耗时**：~3h
- **显存**：~15GB（无优化器，不会 OOM）

#### 监控

```bash
tail -f logs/baselines_v2_part1.log | grep "progress\|BASELINE\|Skipping"
tail -f logs/baselines_v2_part2.log | grep "progress\|BASELINE\|eval_only"
```

完成标志：日志输出 `BASELINE SUMMARY` 表格。

---

### 3.4 T5-A0.5: Events Only 消融

测试"只有 eval events、无训练 memory"的性能，验证训练 memory 的价值。

> ⚠️ 需确认 `eval_locomo.py` 支持空 checkpoint。如不支持，创建空 checkpoint 目录：
>
> ```bash
> mkdir -p outputs/empty_checkpoint
> echo '[]' > outputs/empty_checkpoint/memory_store.json
> echo '{"step_count": 0}' > outputs/empty_checkpoint/agent_meta.json
> ```

```bash
# A0.5a: LoCoMo
CUDA_VISIBLE_DEVICES=5 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/ablation_events_only \
    --benchmark locomo \
    --no_qlora \
    2>&1 | tee logs/ablation_events_only_locomo.log

# A0.5b: LongMemEval
CUDA_VISIBLE_DEVICES=6 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/ablation_events_only \
    --benchmark longmemeval \
    --no_qlora \
    2>&1 | tee logs/ablation_events_only_longmemeval.log
```

---

### 3.5 T6: 训练消融 A1-A7（低优先级）

> 仅在 T4+T5 完成且有余力时执行。T4+T5+T7 已足够支撑论文。

参见 [v7.0 §3.6](PROJECT_WORKFLOW7.0.md#36-t6-训练消融实验-a1-a7) 的完整指南。

---

### 3.6 T7: 论文数据汇总（T4 完成后执行）

#### Table 1: 预期最终主结果

```
Method              | LoCoMo F1 | LoCoMo EM | LongMem F1 | LongMem EM |
────────────────────┼───────────┼───────────┼────────────┼────────────┤
Reflexion           |   0.0163  |     —     |   0.0408   |     —      |
EvolveR             |   待填    |   待填    |   待填     |   待填     |
Self-Consolidation  |   待填    |   待填    |   待填     |   待填     |
Memory-R1 (eval)    |   待填    |   待填    |   待填     |   待填     |
Mem0+Memory-R1 (e)  |   待填    |   待填    |   待填     |   待填     |
────────────────────┼───────────┼───────────┼────────────┼────────────┤
G-MSRA (Ours)       |   0.0962  |   0.0450  |   0.2616   |   0.1760   |
```

> Memory-R1 和 Mem0 标注 "(eval-only)" 用脚注说明跳过训练的原因
> （7B 全参数 Adam 超出单卡显存，与 G-MSRA 使用 LoRA 不同）。

#### Table 2: 消融结果

```
Variant             | LoCoMo F1 | LongMem F1 | ΔF1 (LoCoMo) | ΔF1 (LongMem) |
────────────────────┼───────────┼────────────┼──────────────┼────────────────┤
G-MSRA (Full)       |   0.0962  |   0.2616   |      —       |       —        |
A0: No Memory       |   0.0272  |   0.0490   |   −71.7%     |    −81.3%      |
A0.5: Events Only   |   待填    |   待填     |     待       |      待        |
```

#### Table 3: Phase 2 Self-Reward 校准

数据来源：`logs/phase2_v4.log`, `outputs/phase2/calibration.json`

#### Per-Category Highlight

```
LongMemEval Category    | G-MSRA F1 | No-Memory F1 | Δ      |
────────────────────────┼───────────┼──────────────┼────────┤
[3] Knowledge Update    |   0.5053  |    0.0237    | +2032% |
[1] Single-session      |   0.3524  |    0.0699    | +404%  |
[4] Temporal Reasoning  |   0.1749  |    0.0717    | +144%  |
[2] Multi-session       |   0.1384  |    0.0126    | +998%  |
[5] Abstain             |   0.0869  |    0.0560    | +55%   |
```

#### 图表清单

| 图编号 | 内容 | 数据来源 | 优先级 |
|:------:|------|---------|:------:|
| Fig 2 | Phase 2 α 退火曲线 | `logs/phase2_v4.log` | ⭐⭐⭐⭐⭐ |
| Fig 3 | Phase 2 τ vs Step | `logs/phase2_v4.log` | ⭐⭐⭐⭐⭐ |
| Fig 4 | Phase 3 R_avg 曲线 | `outputs/phase3_v5/metrics.json` | ⭐⭐⭐⭐ |
| Fig 5 | 消融对比柱状图 | ablation results | ⭐⭐⭐⭐⭐ |
| Fig 6 | Per-category F1 (Full vs No-Memory) | eval results | ⭐⭐⭐⭐ |

---

## 四、已有文件清单

### 4.1 训练产物

```
outputs/
├── phase0/best/                  # Phase 0 SFT checkpoint
├── phase1/best/                  # Phase 1 LoRA adapter
├── phase2/
│   ├── best/                     # Phase 2 agent 状态 (memory_store + agent_meta)
│   └── calibration.json          # τ 校准数据
├── phase3_v5/
│   ├── best/                     # Phase 3 最终 checkpoint (≡ Phase 2)
│   ├── checkpoint_500/ ~ 1500/
│   ├── metrics.json
│   └── diagnostics.json
```

### 4.2 评测结果

```
results/
├── eval_phase2/                  # ✅ G-MSRA 主模型评测
│   ├── locomo_results.json
│   └── longmemeval_results.json
├── eval_phase3v5/                # ✅ (与 phase2 完全一致)
│   ├── locomo_results.json
│   └── longmemeval_results.json
├── ablation_no_memory/           # ✅ T5-A0 消融
│   ├── locomo_results.json
│   └── longmemeval_results.json
├── baselines_v2/                 # 🔄 部分完成
│   ├── reflexion/results.json    # ✅
│   ├── evolver/                  # ❌ 待跑
│   ├── self_consolidation/       # ❌ 待跑
│   ├── memory_r1/                # ❌ 待跑 (eval_only)
│   └── mem0_memory_r1/           # ❌ 待跑 (eval_only)
```

### 4.3 日志

```
logs/
├── eval_phase2_locomo.log            # ✅
├── eval_phase2_longmemeval.log       # ✅
├── eval_phase3v5_locomo.log          # ✅
├── eval_phase3v5_longmemeval.log     # ✅
├── ablation_no_memory_locomo.log     # ✅
├── ablation_no_memory_longmemeval.log # ✅
├── baselines_v2_part1.log            # 🔄 Reflexion ✅, 需重跑其余
└── baselines_v2_part2.log            # ❌ OOM, 需 --eval_only 重跑
```

---

## 五、GPU 与资源规划

### 5.1 剩余任务并行方案（2 张 GPU）

```
时间   GPU 0                         GPU 1
─────┼────────────────────────────┼────────────────────────────
0h   │ T4a: Part1 (EvolveR+SC)    │ T4c: Part2 (MemR1+Mem0)
     │ --skip_existing             │ --eval_only
     │ ~3h                         │ ~3h
3h   │ ✅ 完成                     │ ✅ 完成
     │ T5-A0.5: Events Only       │ (空闲)
     │ ~0.5h                       │
3.5h │ ✅ 完成                     │
─────┴────────────────────────────┴────────────────────────────
总计: ~3.5h
```

### 5.2 显存需求

| 场景 | 显存 | 说明 |
|------|:----:|------|
| Baseline eval (Part1: 3 baselines) | ~15GB | 纯推理 |
| Baseline eval-only (Part2: 2 baselines) | ~15GB | 纯推理，无优化器 |
| 消融 eval | ~15GB | 纯推理 |

所有场景单张 A40 (48GB) 均充足。

---

## 六、论文策略

### 6.1 核心贡献定位

| # | 贡献 | 数据支撑 | 状态 |
|---|------|---------|:----:|
| 1 | **Grounded Self-Reward Mechanism** | τ = 0.4~0.8, α 退火 1.0→0.03 | ✅ |
| 2 | **Memory-Augmented Inference** | Full F1 vs No-Memory: +253% (LoCoMo), +434% (LongMem) | ✅ |
| 3 | **知识更新追踪** | Cat [3] F1 = 0.5053, 远超其他类别 | ✅ |
| 4 | **统一 Baseline 公平比较** | 5 baselines 同一评测管线 | 🔄 待 T4 完成 |

### 6.2 关键发现

1. **Memory store 是核心贡献**：消融实验证明 71-81% 的性能来自 memory store
2. **Knowledge Update 是旗舰场景**：LongMemEval Cat [3] F1=0.5053，比 No-Memory 的 0.0237 提升 **21倍**
3. **RL policy 训练未收敛**：Phase 2 ≡ Phase 3，NOOP 占比 99.9%，在 Discussion 中诚实报告
4. **No-Memory ≈ Reflexion baseline**：验证了消融基线的合理性（都是纯 LLM 推理）

### 6.3 论文表述指引

**✅ 可以声称**：

- "Our memory-augmented approach achieves F1=0.0962 on LoCoMo and 0.2616 on LongMemEval"
- "Memory retrieval contributes 71.7% of LoCoMo and 81.3% of LongMemEval performance"
- "Knowledge update questions show the strongest gains (F1=0.5053, +2032% over no-memory)"
- "Self-reward calibration achieves τ > 0.4 Kendall correlation"

**❌ 不能声称**：

- ~~"RL fine-tuning improves memory management policy"~~
- ~~"The agent learns UPDATE/DELETE operations through training"~~

### 6.4 Limitation Section

> While the grounded self-reward mechanism produces well-calibrated signals
> (τ > 0.4), the sparse nature of the dialogue QA F1 reward (18.8% non-zero)
> makes closed-loop RL policy optimization challenging within 1,586 episodes.
> The policy remains dominated by NOOP (99.9%), and parametric consolidation
> was never triggered. Future work should explore denser reward signals,
> curriculum-based exploration bonuses, or significantly more training episodes.

---

## 七、里程碑更新

| 里程碑 | 状态 | 完成日期 |
|--------|:----:|:--------:|
| M0-M3: 基础建设 + Phase 0/1 + Baseline | ✅ | 2026-03-26 |
| M4: Phase 2 课程退火 | ✅ | 2026-03-30 |
| M5-M6: RL Baseline 训练+评测 | ✅ | 2026-04-02 |
| M7: Phase 3 全闭环 | ✅ | 2026-04-09 |
| M7.1: Phase 2 v4 重训 | ✅ | 2026-04-08 |
| M7.2: Phase 3 v5 训练 | ✅ | 2026-04-12 |
| **M8: G-MSRA 模型评测 (T1/T2/T3)** | **✅** | **2026-04-13** |
| **M9a: 消融实验 A0 (No Memory)** | **✅** | **2026-04-13** |
| M9b: 消融实验 A0.5 (Events Only) | ☐ | — |
| M10a: Reflexion baseline | ✅ | 2026-04-13 |
| **M10b: 剩余 4 个 Baseline** | **☐** | **—** |
| M11: 论文数据汇总 + 画图 | ☐ | — |
| M12: 论文初稿 → 投递 | ☐ | — |

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:----:|:----:|------|
| ~~Baseline TypeError (prediction kwarg)~~ | ~~高~~ | ~~🔴~~ | ✅ 已修复 `baselines.py` L374 |
| ~~Memory-R1 OOM~~ | ~~高~~ | ~~🔴~~ | ✅ 已添加 `--eval_only` 模式 |
| ~~Baseline 无进度日志~~ | ~~高~~ | ~~🟡~~ | ✅ 已添加每 10 步日志 |
| ~~MemoryR1 events 跨 example 累积~~ | ~~高~~ | ~~🔴~~ | ✅ per-example reset + cap=20 |
| SelfConsolidation consolidation 触发 OOM | 低 | 🟡 | 观察日志，如报错用 `--eval_only` |
| eval_only 模式下 baseline F1 ≈ 0 | 中 | 🟡 | 合理：未经训练的文本基线仅做推理 |
| GPU 不空闲 | 低 | 🟡 延迟 | Tang2/Tang3 共 16 张 A40 |

---

> **当前状态总结**：
>
> - ✅ G-MSRA 主模型评测完成（LoCoMo F1=0.0962, LongMemEval F1=0.2616）
> - ✅ 核心消融 A0 完成（Memory 贡献 71-81%）
> - ✅ Reflexion baseline 完成
> - ✅ 全部已知代码 bug 已修复
> - ☐ **下一步：同步代码到服务器 → 并行启动 T4 Part1+Part2 → ~3.5h 完成**
> - ☐ T4 完成后汇总数据开始论文撰写
