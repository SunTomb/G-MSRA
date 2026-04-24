# PROJECT_WORKFLOW 3.1.0 — 实验总结 · 反思 · 方向转型

> 创建时间：2026-04-24 13:40  
> 前序：WORKFLOW 2.2.1 (LoRA 蒸馏诊断 + 方案 B 验证)  
> 状态：**Phase 1 全部实验完成 ✅ → 结论：当前架构的参数化蒸馏路线失败 → 需要方向转型**

---

## 一、修复实验结果

### 1.1 LoRA Merge Sweep（14 组，耗时 ~18h）

将 Phase 1 LoRA（通用能力）和 v11 LoRA（蒸馏知识）按比例混合：

```
merged_lora = alpha × Phase1_LoRA + (1-alpha) × v11_LoRA
```

**与 v11 best LoRA 混合：**

| alpha | Phase1 占比 | v11(best) 占比 | F1 | EM | vs baseline |
|:-----:|:-----------:|:--------------:|:--:|:--:|:-----------:|
| 1.0 | 100% | 0% | **0.0970** | 0.0450 | = baseline |
| 0.95 | 95% | 5% | 0.0950 | 0.0400 | -2.1% |
| 0.9 | 90% | 10% | 0.0875 | 0.0300 | -9.8% |
| 0.8 | 80% | 20% | 0.0833 | 0.0275 | -14.1% |
| 0.7 | 70% | 30% | 0.0832 | 0.0250 | -14.2% |
| 0.6 | 60% | 40% | 0.0870 | 0.0250 | -10.3% |
| 0.5 | 50% | 50% | 0.0847 | 0.0225 | -12.7% |

**与 v11 ckpt500 LoRA 混合（仅 1 次 consolidation）：**

| alpha | Phase1 占比 | v11(ckpt500) 占比 | F1 | EM | vs baseline |
|:-----:|:-----------:|:------------------:|:--:|:--:|:-----------:|
| 1.0 | 100% | 0% | **0.0970** | 0.0450 | = baseline |
| 0.95 | 95% | 5% | 0.0950 | 0.0400 | -2.1% |
| 0.9 | 90% | 10% | 0.0872 | 0.0300 | -10.1% |
| 0.8 | 80% | 20% | 0.0844 | 0.0250 | -13.0% |
| 0.7 | 70% | 30% | 0.0893 | 0.0225 | -7.9% |
| 0.6 | 60% | 40% | 0.0783 | 0.0150 | -19.3% |
| 0.5 | 50% | 50% | 0.0772 | 0.0100 | -20.4% |

**结论：🔴 LoRA merge 完全失败。** 无论任何 alpha 值，混入 v11 蒸馏 LoRA 都导致 F1 单调下降。v11 蒸馏的权重中不包含任何对 OOD 评测有正面贡献的泛化知识。

---

### 1.2 RL 策略评测（400 examples，耗时 ~21h）

用 RL 训练的 memory manager（`decide() + execute_operation()`）处理评测事件，Phase 1 LoRA 保持 QA 能力：

| 指标 | 值 | 对比 |
|------|:--:|:----:|
| **F1** | **0.0259** | 🔴 -73% vs Events Only (0.097) |
| **EM** | **0.0100** | 🔴 -78% vs Events Only (0.045) |
| 平均 memory 保留 | 351-372 条 | 几乎全 NOOP |

**Agent Step 日志分析：**

```
操作分布（从日志采样前 800 行）：
  NOOP:  ~85%  ← 绝大多数事件被忽略
  ADD:   ~14%  ← 偶尔添加，但内容质量极差
  UPDATE: <1%  ← 几乎不存在
  DELETE:  0%  ← 从未执行

ADD 的内容质量问题：
  ❌ "I should ask Evan about his recent trip."  ← 非事实性知识
  ❌ "<content>\nUPDATE <id>: <new_content>\nDELETE <id>\nN..."  ← 模型输出了操作模板，而非内容
  ❌ "No recent information on Evan's broken items."  ← 否定性/无意义信息
  ❌ "Evan's last known vehicle is unknown."  ← 无信息量
```

**结论：🔴 RL 策略在评测中完全失效。**
1. NOOP 主导（85%）：模型拒绝接收评测集的事件
2. ADD 质量极差：输出的不是有效的知识摘要，而是无意义的模板和猜测
3. 最终 F1=0.026，**甚至低于 No Memory baseline（0.027）**
4. 根因：训练时的 NOOP 固化（Phase 3 后期 NOOP 达 96%）迁移到了评测

---

### 1.3 全部实验结果汇总

| 方案 | F1 | 说明 | 结论 |
|------|:--:|------|:----:|
| Events Only (Phase1 LoRA) | **0.097** | 🟢 最优 baseline | 上限 |
| No Memory | 0.027 | 纯 LLM | 下限 |
| LoRA merge (best alpha=0.95) | 0.095 | 95% Phase1 + 5% v11 | ❌ 不超过 baseline |
| G-MSRA v11 (full system) | 0.048 | 完整系统 | ❌ -50% |
| RL 策略评测 | **0.026** | Agent Step + Phase1 LoRA | ❌ 最差 |

---

## 二、诚实的反思

### 2.1 核心失败点

G-MSRA 的三大创新全部在评测中失效：

| 创新点 | 训练指标 | 评测结果 | 失败原因 |
|--------|:--------:|:--------:|----------|
| RL CRUD 策略 | UPDATE > ε-greedy | F1=0.026 (最差) | NOOP 固化 + OOD 泛化失败 |
| 参数化蒸馏 (LoRA) | loss 1.55→0.25 | F1↓50% | 训练集过拟合，覆盖通用能力 |
| 自奖励信号 | success 16%→70% | 未独立验证 | 可能存在 reward hacking |

### 2.2 老师的反馈总结

1. **"不能想着做加法"**：我们一直在加新模块修问题（知识分类、遗忘机制、dual-adapter），但问题出在**框架设计本身**。应该做减法。

2. **LoRA 蒸馏不靠谱**：
   - 高频知识的筛选机制太粗糙（只看 links ≥ 2 + confidence ≥ 0.5）
   - 没有区分"需要固化的持久事实"和"临时上下文"
   - 缺少 HERMES/DREAM 那样的选择性遗忘机制

3. **OOD + 过拟合**：训练集和评测集完全不重叠，蒸馏 = 在训练集上过拟合

4. **Self-reward 可能 reward hacking**：训练时 success rate 很高，但没有在评测集上用 Judge 验证

### 2.3 根本矛盾

G-MSRA 的核心假设是：

> "将高频使用的记忆蒸馏到模型参数中，可以提高长期对话的 QA 能力"

**这个假设在当前实验条件下被否定了。** 原因：

1. **知识蒸馏 ≠ 能力提升**：LoRA 微调的目标是"让模型能生成特定的三元组"，但这与"让模型能回答关于这些三元组的问题"是完全不同的任务
2. **参数空间冲突**：7B 模型的 LoRA（rank=32, q_proj+v_proj）的参数空间太小，无法同时保持通用 QA 能力和编码特定知识
3. **评测范式不匹配**：per-example isolation 意味着每个问题都是"冷启动"，历史 memory 是噪音。这与训练时的连续对话场景完全不同

---

## 三、项目价值的重新审视

### 3.1 仍然有价值的部分

尽管最终指标不理想，项目中以下工作仍有独立价值：

1. **RL CRUD 训练流程**：per-action REINFORCE + weighted exploration 成功教会了模型执行 CRUD 操作。问题不在 RL 方法本身，而在于评测范式（per-example isolation）无法体现连续交互的价值。

2. **Consolidation 触发机制**：基于 conflict index + reward variance + memory growth 的自适应触发器（TriggerConfig）是一个合理的设计，可复用于其他场景。

3. **实验基础设施**：评测脚本（eval_locomo.py）、训练循环（train_phase3_full.py）、诊断框架（17 组 ablation）都是可复用的工具。

4. **负面结果的科学价值**：系统性证明了"LoRA 蒸馏用于长期记忆固化"在当前条件下失败，是有价值的 negative result。

### 3.2 不可挽救的部分

1. **v11 LoRA**：完全过拟合训练集，不可用
2. **NOOP 固化的 RL 策略**：在评测中 NOOP 率 85%，无法提供有效的 memory 管理

---

## 四、后续方向

### 方向 A：做减法——去掉蒸馏，聚焦 RL CRUD + 纯外部 memory（推荐）

> 呼应老师"不能想着做加法"的建议

**核心思路**：放弃参数化蒸馏，保留 RL CRUD 策略 + 纯外部 memory store，聚焦于"智能 memory 管理"本身。

**具体改动**：
- 去掉 `distiller.py` 和整个 consolidation 模块
- 保留 RL 训练的 `MemoryManager`，但重新训练（修复 NOOP 固化）
- 评测时用**连续对话**模式（非 per-example isolation）
- 与 MemoryBank、ReadAgent 等纯外部 memory baseline 对比

**预期**：
- F1 至少与 Events Only 持平（因为不再有蒸馏覆盖能力）
- RL CRUD 可以通过 UPDATE/DELETE 提高 memory 质量，从而超过 raw ADD
- 论文叙事：**"RL 训练的记忆管理策略在连续对话中优于启发式方法"**

**风险**：
- 没有 consolidation 后，论文的"创新点"减少
- 需要重新设计评测流程（连续对话）

---

### 方向 B：修复评测范式——连续对话评测

**问题**：当前 per-example isolation 评测是对 G-MSRA 最不利的设置——每个问题都独立重置 memory，历史训练记忆只是噪音。

**改进**：
- 设计连续对话评测流程：agent 处理一系列对话 session，在 session 之间保持 memory
- 后续 session 的问题基于前序 session 的内容
- 这样 RL 策略的 CRUD 操作和 consolidation 才能体现价值

**但这需要新的评测数据集**，LoCoMo/LongMemEval 不直接支持这种范式。

---

### 方向 C：转向知识更新方向

参考前序对话（conversation 8793c37f）中讨论的 pivoting 策略：

- 不比较"记住更多"，而是比较"知识更新"的能力
- 场景：用户偏好变化、事实纠正、信息过时
- 用 CRUD 操作天然适配知识更新任务
- 与 HERMES 的 dream 机制和选择性遗忘进行对比

---

### 方向 D：发表 Negative Result Paper

如果上述方向都来不及执行，可以将现有结果整理为 **negative result / empirical analysis**：

- **标题方向**：*"Why Parametric Memory Consolidation Fails: An Empirical Study of LoRA Distillation for Long-term Dialogue Agents"*
- **贡献**：
  1. 系统性实验证明 LoRA 蒸馏在 OOD 评测中的失败模式
  2. 17 组 ablation 的诊断方法论
  3. Per-action REINFORCE 在 memory CRUD 任务上的有效性分析
- **适合投稿**：workshop paper 或 Findings

---

## 五、决策建议

| 方向 | 时间成本 | 风险 | 论文可行性 | 建议 |
|------|:--------:|:----:|:----------:|:----:|
| A. 去掉蒸馏 | 1-2 周 | 中 | 中（需新实验） | ⭐ 首选 |
| B. 修复评测范式 | 2-3 周 | 高（需新数据） | 高 | 🟡 如有时间 |
| C. 知识更新方向 | 2-3 周 | 中 | 高 | ⭐ 次选 |
| D. Negative Result | 1 周 | 低 | 中 | 🟢 保底 |

**建议优先级**：先与老师讨论方向 A 和 C 的取舍，同时准备方向 D 作为保底。

---

## 六、待与老师讨论的问题

1. **方向选择**：老师倾向于做减法（方向 A）还是转向知识更新（方向 C）？
2. **论文定位**：是追求在现有框架上做出 positive result，还是接受 negative result 的定位？
3. **评测范式**：是否需要设计新的连续对话评测，还是继续用 LoCoMo/LongMemEval？
4. **时间约束**：还剩多少时间完成实验和论文？这决定了可行的方向。
5. **Self-reward 验证**：老师提到的 reward hacking 检测实验，是否需要优先完成？

---

## 七、文件索引

### 7.1 本轮实验产物

| 路径 | 说明 |
|------|------|
| `results/merge_sweep/merge_sweep_results.json` | 14 组 LoRA merge 评测结果 |
| `results/diag_agent_step/locomo_results.json` | RL 策略评测结果 |
| `logs/merge_sweep.log` | Merge sweep 完整日志（262MB） |
| `logs/diag_agent_step.log` | Agent step 评测日志 |

### 7.2 关键脚本

| 路径 | 说明 |
|------|------|
| `scripts/eval_lora_merge.py` | LoRA merge sweep 脚本 |
| `scripts/eval_locomo.py` | 评测脚本（含 `--use_agent_step`） |

### 7.3 历史版本文档

| 文件 | 内容 |
|------|------|
| `PROJECT_WORKFLOW2.1.0.md` | Phase 3 v6 BUG 修复 |
| `PROJECT_WORKFLOW2.1.2.md` | Phase 3 v8→v11 迭代 |
| `PROJECT_WORKFLOW2.2.0.md` | 评测计划 + 论文规划 |
| `PROJECT_WORKFLOW2.2.1.md` | 评测诊断 + 方案 B 验证 |
| `PROJECT_WORKFLOW3.0.0.md` | **实验总结 + 方向转型** |

---

## 八、时间线

| 阶段 | 状态 | 时间 |
|:----:|------|:----:|
| Phase 1 SL 训练 | ✅ | 早期 |
| Phase 2 RL 过渡 | ✅ | 4/17-18 |
| Phase 3 v6-v11 | ✅ | 4/17-21 |
| 评测 (13+14+1=28 组) | ✅ | 4/22-24 |
| 诊断 + 方案 B 验证 | ✅ | 4/22 |
| LoRA Merge Sweep | ✅ 14 组 | 4/22-23 |
| RL 策略评测 | ✅ | 4/22-24 |
| 📌 **方向决策** | **待老师讨论** | 4/24 |
| 📌 新方向实验 | 待执行 | 4/25+ |
