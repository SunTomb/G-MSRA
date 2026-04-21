# G-MSRA 项目工作流程 v6.1 — Phase 3 诊断修复 + 重训指南

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 4 月 5 日 · 基于 [v6.0](PROJECT_WORKFLOW6.0.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 侧重 | 状态 |
|------|------|------|
| v1.0 ~ v5.1 | 项目搭建 → Phase 1 RL → Baseline 评测 | ✅ 背景资料 |
| [v6.0](PROJECT_WORKFLOW6.0.md) | Phase 2 + RL Baseline 完成，全面总结 | ✅ 已被本文档取代 |
| **v6.1（本文档）** | **Phase 3 首轮训练诊断 + 代码修复 + 重训指南** | ✅ 当前版本 |

---

## 一、Phase 3 首轮训练诊断（2026-04-03 ~ 04-04）

### 1.1 运行概况

Phase 3 在 Tang3 集群上成功完成了全部 1586 个 Episode，耗时 29.4 小时。

```
开始时间: 2026-04-03 03:40:07
结束时间: 2026-04-04 09:06:05
总耗时:   29.4h
```

输出文件：

- `outputs/phase3/best/` — 最终 checkpoint
- `outputs/phase3/checkpoint_500/`, `checkpoint_1000/`, `checkpoint_1500/`
- `outputs/phase3/metrics.json` — 训练指标（每 50 ep）
- `outputs/phase3/diagnostics.json` — 完整诊断数据

### 1.2 关键指标 — ❌ 训练无效

| 指标 | 实际值 | 预期值 | 判定 |
|------|--------|--------|:----:|
| R_env (全程) | **0.000** | 逐步提升至 0.3+ | ❌ |
| avg_reward (R_total) | 0.14 ~ 0.23 | 逐步提升 | ⚠️ |
| NOOP 占比 | **99.9%** (4754/4758) | 30-50% | ❌ |
| ADD 次数 | 4 | 数百次 | ❌ |
| UPDATE / DELETE | 0 / 0 | >0 | ❌ |
| memory_size | 11 → 13 | 50-200 | ❌ |
| consolidation_count | **0** | ≥2 | ❌ |
| avg_success_rate | **0.0** (全程) | 逐步提升 | ❌ |

### 1.3 根因分析

**根因：`agent_response` 始终为空字符串，导致 R_env = compute_f1("", answer) = 0。**

因果链：

```
agent_response="" 传入 DialogueSignalExtractor.extract()
    → compute_f1("", ground_truth) = 0.0
    → R_env = 0 全程
    → R_total = 0 + 0.3×R_mem，上限仅 0.3
    → NOOP 与 ADD 奖励信号无差异 → RL 策略坍缩至 NOOP
    → memory 停滞 (仅 13 条) → Consolidation 触发条件不满足
    → success_rate = 0.0
```

具体代码问题：

1. **`train_phase3_full.py` 的 `load_task_stream()`** 中 `env_kwargs.agent_response = ""`，而 Phase 3 从未调用 `agent.answer_question()` 来生成实际预测
2. **Phase 2 `final_alpha = 0.622`**：500 步训练不够，α 未退火到 0，Phase 3 直接切到 100% self-reward 跳跃过大
3. **Phase 2 的 `agent.step()` 调用** 也传入 `env_kwargs={agent_response: ""}`，导致 step 内部的 R_env 也为 0

---

## 二、代码修复（2026-04-05）

### 2.1 修改清单

| 文件 | 修改内容 | 严重程度 |
|------|---------|:--------:|
| `scripts/train_phase3_full.py` | 每个 episode 结束后调用 `agent.answer_question()` 获取真实预测 | 🔴 必须 |
| `scripts/train_phase3_full.py` | 用真实预测构建 `env_kwargs`，传入最终的 `agent.step()` | 🔴 必须 |
| `scripts/train_phase3_full.py` | 事件处理阶段 `env_signal_kwargs={}` 而非传空 answer | 🟡 推荐 |
| `scripts/train_phase3_full.py` | 新增 `--num_epochs` 参数，默认 max_events 从 3 → 5 | 🟡 推荐 |
| `scripts/train_phase3_full.py` | `load_task_stream()` 不再预填 `env_kwargs`（改在训练循环中动态构建） | 🟡 推荐 |
| `gmsra/reward/env_signals.py` | `agent_response` 改为可选参数，传 `{}` 时返回中性值 0.5 | 🔴 必须 |
| `scripts/train_phase2_transition.py` | 事件处理改用 `memory_manager.decide()+execute()`，跳过 Judge 推理（2.5× 加速） | 🟡 推荐 |
| `scripts/train_phase2_transition.py` | 新增 `--num_epochs`（默认 2）+ `--tau_threshold`（降至 0.15） | 🟡 推荐 |

### 2.2 核心修复逻辑（Phase 3）

**修复前**（每个 episode）：

```python
for event in task_events[:max_events]:
    result = agent.step(
        event=event,
        agent_response="",                    # ← 空字符串!
        env_signal_kwargs={"agent_response": "", "qa_ground_truth": answer},
    )
    last_result = result
# R_env = F1("", answer) = 0 → 策略坍缩
```

**修复后**（每个 episode）：

```python
# 1) 事件处理阶段：不计算 R_env
for event in task_events[:max_events]:
    agent.step(event=event, agent_response="", env_signal_kwargs={})

# 2) 生成真实预测
predicted = agent.answer_question(task_question)

# 3) 最终 step：用真实预测计算 R_env
last_result = agent.step(
    event=task_events[-1],
    agent_response=predicted,                  # ← 真实预测!
    env_signal_kwargs={
        "agent_response": predicted,           # ← F1(predicted, answer) > 0
        "qa_ground_truth": task_answer,
    },
)
```

### 2.3 Phase 2 多轮退火

**修复前**：数据集仅 1586 条，`anneal_steps=3000`，实际只跑了 500 步（因 `min(len(dataset), anneal_steps)`），α 停在 0.622。

**修复后**：新增 `--num_epochs` 参数（默认 2），`full_dataset = dataset × num_epochs`，total_steps 可达 3000，确保 α → 0。

---

## 三、重训操作指南

### 3.0 前置步骤：同步代码到服务器

从本地推送修复后的代码：

```bash
# 方法 A：scp 上传
scp scripts/train_phase2_transition.py wujcan@Tang3:/NAS/yesh/G-MSRA/scripts/
scp scripts/train_phase3_full.py wujcan@Tang3:/NAS/yesh/G-MSRA/scripts/

# 方法 B：git（需要 Tang3 能联网）
cd /NAS/yesh/G-MSRA && git pull
```

### 3.1 Phase 2 重训 ✅ 已启动（v4）

> **迭代记录**：
> - v2 (首次)：`env_signal_kwargs={agent_response: ""}` → `TypeError`
> - v3 (修复 env_signals.py)：可运行，但 `tau_threshold=0.5` 导致退火频繁暂停，α 极慢下降
> - **v4 (当前)**：`--tau_threshold 0.15`，退火持续进行

#### 环境准备

```bash
ssh wujcan@Tang3
tmux new -s phase2
cd /NAS/yesh/G-MSRA
eval "$(conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1
export PYTHONPATH=/NAS/yesh/G-MSRA
```

#### 备份旧数据

```bash
mv outputs/phase2 outputs/phase2_v1_backup
```

#### 启动训练

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_phase2_transition.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --checkpoint outputs/phase1/best \
    --output_dir outputs/phase2 \
    --anneal_steps 3000 \
    --num_epochs 2 \
    --tau_threshold 0.15 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/phase2_v4.log
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--anneal_steps` | 3000 | α 从 1.0 退火到 0.0 需要的总步数 |
| `--num_epochs` | 2 | 数据集 1586 × 2 = 3172，足够走完 3000 步 |
| `--tau_threshold` | **0.15** | 降低阈值，避免 τ 波动导致退火频繁暂停 |
| `--checkpoint` | `outputs/phase1/best` | 加载 Phase 1 的 LoRA adapter |

#### 已观察到的指标（v3 前 90 步）

| 指标 | 实际值 | 判定 |
|------|--------|:----:|
| R_env | 0.000 ~ 0.045 | ✅ **非零，修复生效** |
| R_mem | 0.0 ~ 1.0 | ✅ Judge 正常 |
| τ (Kendall) | -0.247 → **0.612** → 0.487 | ✅ 整体上升 |
| memory_size | 2 → **10** | ✅ 持续增长 |
| ADD 操作 | 频繁出现 | ✅ |
| 速度 | 0.01 step/s (~80s/步) | ✅ 含 answer_question 推理 |

#### 预期最终指标

| 指标 | 预期值 |
|------|--------|
| 总步数 | 3000 |
| final_alpha | ≈ **0.0**（`tau_threshold=0.15` 确保持续退火） |
| τ (Kendall) | 0.3 ~ 0.6 |
| R_ext | 0.01 ~ 0.05 |
| 速度 | ~0.01 step/s |
| 总耗时 | **~56h** |

#### 监控

```bash
tail -f logs/phase2_v4.log | grep "Step "
# 应看到 α 持续下降：
# Step 100/3000 | α=0.967 | τ=0.xxx | R_ext=0.xxx
# Step 1000/3000 | α=0.667 | ...
# Step 2000/3000 | α=0.333 | ...
# Step 3000/3000 | α=0.000 | ...
```

**完成标志**：日志出现 `Phase 2 complete. Final α=0.0000`

---

### 3.2 Phase 3 重训

Phase 2 完成后，继续 Phase 3。

#### 备份旧数据

```bash
mv outputs/phase3 outputs/phase3_v1_backup
```

#### 启动训练

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_phase3_full.py \
    --checkpoint outputs/phase2/best \
    --lora_checkpoint outputs/phase1/best \
    --output_dir outputs/phase3 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --max_episodes 1586 \
    --max_events 5 \
    --num_epochs 2 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/phase3_v2.log
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--checkpoint` | `outputs/phase2/best` | Phase 2 的 agent 状态（memory_store + step_count） |
| `--lora_checkpoint` | `outputs/phase1/best` | Phase 1 的 LoRA adapter（Phase 2 未保存 LoRA） |
| `--max_events` | 5 | 提高至 5，给更多机会执行 ADD |
| `--num_epochs` | 2 | 数据集循环 2 轮 = 3172 episodes |
| `--max_episodes` | 1586 | 每轮的 episode 数 |

#### 预期指标（修复后）

| 指标 | 预期值 |
|------|--------|
| R_env | **> 0**，逐步提升至 0.01 ~ 0.05 |
| R_total | 逐步提升至 0.05 ~ 0.15 |
| NOOP 占比 | 30-70%（而非 99.9%） |
| ADD 次数 | 数十 ~ 数百次 |
| memory_size | 30-200 |
| consolidation_count | ≥ 1 |
| 总耗时 | **~40-50h**（每 episode 多一次 answer_question 推理） |

#### 监控

```bash
tail -f logs/phase3_v2.log | grep "Episode\|CONSOLIDATION"
# 应看到：
# Episode 50/3172 | R_avg=0.050 | success=0.020 | mem=25 | consol=0
# Episode 100/3172 | R_avg=0.070 | success=0.040 | mem=45 | consol=0
# CONSOLIDATION TRIGGERED at step XXXX (score=0.72 > θ=0.6)
# Episode 200/3172 | R_avg=0.090 | success=0.060 | mem=60 | consol=1
```

**完成标志**：日志出现 `Phase 3 complete! Total time: XXh, Episodes: 3172`

---

### 3.3 下载结果到本地

```bash
# Phase 2 v4
scp -r wujcan@Tang3:/NAS/yesh/G-MSRA/outputs/phase2 d:/USTC/2026Winter/G-MSRA/outputs/
scp wujcan@Tang3:/NAS/yesh/G-MSRA/logs/phase2_v4.log d:/USTC/2026Winter/G-MSRA/logs/

# Phase 3 v2
scp -r wujcan@Tang3:/NAS/yesh/G-MSRA/outputs/phase3 d:/USTC/2026Winter/G-MSRA/outputs/
scp wujcan@Tang3:/NAS/yesh/G-MSRA/logs/phase3_v2.log d:/USTC/2026Winter/G-MSRA/logs/
```

---

## 四、重训后续步骤

### 4.1 验证修复效果

Phase 3 完成后，检查以下指标以确认修复成功：

```python
import json

with open("outputs/phase3/metrics.json") as f:
    metrics = json.load(f)

# 检查 R_env > 0
last = metrics[-1]
assert last["avg_reward"] > 0.05, "R_total 应该显著大于 0"
assert last["operation_stats"]["ADD"] > 10, "应该有足够的 ADD 操作"
assert last["memory_size"] > 20, "记忆数量应该持续增长"
print(f"✅ 验证通过: R_avg={last['avg_reward']:.3f}, mem={last['memory_size']}, "
      f"ADD={last['operation_stats']['ADD']}")
```

### 4.2 G-MSRA 模型评测

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=<GPU> \
python baselines/eval_baselines.py \
    --agent gmsra \
    --checkpoint outputs/phase3/best \
    --benchmark locomo \
    2>&1 | tee results/gmsra_eval.log
```

### 4.3 消融实验

```bash
PYTHONPATH=/NAS/yesh/G-MSRA python scripts/run_ablation.py \
    --output_dir results/ablation \
    2>&1 | tee logs/ablation.log
```

---

## 五、里程碑更新

| 里程碑 | 状态 | 完成日期 |
|--------|:----:|:--------:|
| M0 ~ M3: 基础建设 + Phase 0/1 + Baseline 评测 | ✅ | 2026-03-26 |
| M4: Phase 2 课程退火（首轮） | ✅ | 2026-03-30 |
| M5-M6: RL Baseline 训练+评测 | ✅ | 2026-04-02 |
| M7: Phase 3 全闭环（首轮 — **无效，已诊断**） | ⚠️ | 2026-04-04 |
| **M7.1: 代码修复（3 个文件 + 4 轮迭代）** | **✅** | **2026-04-05** |
| **M7.2: Phase 2 v4 重训（`--tau_threshold 0.15`）** | **🔄 运行中** | **2026-04-05 ~** |
| M7.3: Phase 3 重训（修复 R_env、增 max_events） | ☐ | — |
| M8: G-MSRA 模型评测 | ☐ | — |
| M9: 消融实验 (Table 3) | ☐ | — |
| M10: 主表 (Table 1+2) 填入 | ☐ | — |
| M11: 分析图完成 | ☐ | — |
| M12: 论文初稿 → 投递 | ☐ | — |

---

## 六、时间预估（基于实际观测速度更新）

| 步骤 | 预估耗时 | 依赖 | 状态 |
|------|:--------:|------|:----:|
| 代码修复 + 同步 | 已完成 | — | ✅ |
| Phase 2 v4 重训 | **~56h** (0.01 step/s × 3000 步) | GPU | 🔄 运行中 |
| Phase 3 v2 重训 | ~50-60h | Phase 2 完成 | ☐ |
| G-MSRA 评测 | ~6h | Phase 3 完成 | ☐ |
| 消融实验 | ~12h | Phase 3 完成 | ☐ |
| **总计** | **~5-6 天** | — | — |

---

## 七、风险与应对

| 风险 | 概率 | 应对 |
|------|:----:|------|
| ~~Phase 2 α 无法退火到 0~~ | ~~中~~ | ✅ 已解决：`--tau_threshold 0.15` |
| ~~Phase 2 事件处理 TypeError~~ | ~~高~~ | ✅ 已解决：`env_signals.py` agent_response 改可选 |
| ~~Phase 2 速度过慢 (82h)~~ | ~~高~~ | ✅ 已解决：跳过事件阶段 Judge 推理 (→56h) |
| Phase 3 answer_question() 推理太慢 | 低 | 每 episode 仅调用一次，增量 ~10s/ep，可接受 |
| Phase 3 Consolidation 仍不触发 | 中 | 降低 `trigger.py` 中 `min_interval` 和调高 `gamma` |
| GPU 不空闲 | 中 | 可用任意单张 A40（Phase 2/3 仅需单卡 ~22GB） |

---

## 八、已完成的 Baseline 结果（无需重跑）

| Method | LoCoMo F1 | LoCoMo EM | 状态 |
|--------|:---------:|:---------:|:----:|
| Reflexion | 0.030 | 0.000 | ✅ |
| EvolveR | 0.026 | 0.000 | ✅ |
| Self-Consolidation | 0.032 | 0.000 | ✅ |
| Memory-R1 | 0.036 | 0.000 | ✅ |
| Mem0+Memory-R1 | 0.028 | 0.000 | ✅ |
| **G-MSRA (Ours)** | **待重训 + 评测** | — | ☐ |

---

> **当前状态总结**：
>
> - ✅ Phase 0 + Phase 1 训练完成
> - ✅ 5/5 个 Baseline 评测全部完成
> - ✅ 代码修复完成（3 个文件，4 轮迭代调试）
> - 🔄 **Phase 2 v4 正在运行**（`--tau_threshold 0.15`，预计 ~56h）
> - ☐ 下一步：Phase 2 完成 → Phase 3 重训 → 评测 → 消融 → 论文
