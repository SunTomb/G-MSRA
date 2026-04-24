# G-MSRA 项目工作流程 v6.0 — Phase 2 + RL Baseline 完成，全面进展总结

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 4 月 1 日 · 基于 [v5.1](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.5.1.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 侧重 | 状态 |
|------|------|------|
| [v1.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW.md) | 研究问题 + 排期规划 + 代码骨架 | ✅ 背景资料 |
| [v2.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.2.0.md) | 训练脚本补全（Phase 0-3 + 消融 + 数据准备） | ✅ 背景资料 |
| [v3.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.3.0.md) | Baseline 复现（5 个 Agent + 评测框架） | ✅ 背景资料 |
| [v4.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.4.0.md) | 合成数据验证完成 → 全量数据实验执行指南 | ✅ 背景资料 |
| [v5.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.5.0.md) | Phase 1 RL 训练完成 → Phase 2-3 执行指南 | ✅ 背景资料 |
| [v5.1](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.5.1.md) | Baseline 评测性能优化 (`--fast_mode`) | ✅ 背景资料 |
| **v6.0（本文档）** | **Phase 2 + RL Baseline 完成，全面总结 + 下一步** | ✅ 当前版本 |

---

## 一、已完成工作总结

### 1.1 G-MSRA 训练管线

| 阶段 | 状态 | 关键成果 | 时间 |
|------|:----:|---------|------|
| Phase 0: SFT 热启动 | ✅ | `outputs_v1/phase0/` | 2026-03-20 |
| Phase 1: RL 训练 | ✅ | `outputs_v1/phase1/best` (checkpoint-1322) | 2026-03-22 |
| **Phase 2: 课程退火** | **✅** | **`outputs_v1/phase2/best` + `checkpoint_500`** | **2026-03-30** |
| Phase 3: 全闭环 | ☐ | — | — |

#### Phase 2 关键指标

```
Step 500/500 | α=0.622 | τ=0.302
R_ext=0.000 | R_self=0.150 | R_annealed=0.092
mem_size=9 | ADD:9 / NOOP:1491
```

本地已下载完整输出：`outputs_v1/phase2/best`、`checkpoint_100~500`、`calibration.json`

- **α 从 1.0 降至 0.622**：self-reward 正在接管 ext-reward，退火进行中
- **τ (Kendall) = 0.302**：self-reward 与 ext-reward 相关性偏低（阈值 0.5），annealing 多次暂停
- **R_env 出现非零值**（0.010~0.020）：对话信号提取在工作
- **9 个 memory / 1500 步**：NOOP 占绝大多数，符合预期

### 1.2 Baseline 评测

#### 无需训练的 Baseline（fast_mode，3/26 完成）

| Agent | 模式 | Episodes | F1 | EM | 日期 |
|-------|:----:|:--------:|:---:|:--:|------|
| Reflexion | fast_mode | 314 | 0.030 | 0.000 | 3/26 |
| EvolveR | fast_mode | 314 | 0.026 | 0.000 | 3/26 |
| Self-Consolidation | fast_mode | 314 | 0.032 | 0.000 | 3/26 |

结果/日志文件：
- [baseline_results.json](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/baseline_results.json) — 保存了 evolver 的完整评估结果
- [reflexion_fast_eval.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/reflexion_fast_eval.log)
- [evolver_fast_eval.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/evolver_fast_eval.log)
- [selfconsolidation_fast_eval.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/selfconsolidation_fast_eval.log)

#### RL Baseline（需训练 → 评测）

| Agent | 训练 Epochs | Train Episodes/Epoch | Avg F1 (Train) | Eval F1 | Eval Episodes | 状态 |
|-------|:-----------:|:--------------------:|:--------------:|:-------:|:-------------:|:----:|
| **Memory-R1** | 3 | 1228 | 0.002 | **0.036** | 314 | ✅ 完成 |
| **Mem0+Memory-R1** | 3 | 1228 | 0.025 | **0.028** | 314 | ✅ 完成 |

日志/数据文件：
- [memory_r1_eval_results.json](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/memory_r1_eval_results.json) — Memory-R1 完整评估结果（314 ep, F1=0.036）
- [rl_baselines_v2.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/rl_baselines_v2.log) — Memory-R1 训练日志
- [rl_eval_v2.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/rl_eval_v2.log) — Memory-R1 评估日志
- [memory_r1_train_metrics.json](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/checkpoints/memory_r1_train_metrics.json)
- [mem0_rl_v2.log](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/mem0_rl_v2.log) — Mem0+R1 训练+评估日志
- [rl_baselines_combined1.json](file:///d:/USTC/2026Winter/G-MSRA/results_v1/baselines/rl_baselines_combined1.json) — Mem0+R1 完整结果（F1=0.028, 314 ep）

### 1.3 性能优化措施（v5.1→v6.0 期间实施）

| 优化 | 修改文件 | 效果 |
|------|---------|------|
| **事件上限 max_events=3** | `train_phase2_transition.py`, `train_and_eval_rl_baselines.py`, `eval_baselines.py` | 训练/评估速度提升 250× |
| **BitsAndBytesConfig 懒加载** | `gmsra/utils.py` | 不装 bitsandbytes 也能运行 |
| **base_agent 显式 use_qlora=False** | `baselines/base_agent.py` | 避免 QLoRA 默认开启 |
| **setup_lora 先冻结再 LoRA** | `train_and_eval_rl_baselines.py` | 修复 OOM（44GB→15GB） |
| **进度日志 Speed/ETA** | `train_and_eval_rl_baselines.py` | 可监控训练进度 |

---

## 二、当前存在的问题

### ~~问题 1：Phase 2 输出未下载到本地~~ ✅ 已解决

`outputs_v1/phase2/` 已下载，包含 `best`、`checkpoint_100~500`、`calibration.json`。

### ~~问题 2：Mem0+Memory-R1 Baseline 不完整~~ ✅ 已解决

已重跑完成（3 epochs × 1228 ep，评估 314 ep，F1=0.028，总耗时 ~29 小时）。
- 完整结果：`rl_baselines_combined1.json`
- 训练日志：`mem0_rl_v2.log`

### ~~问题 3：baseline_results.json 被覆盖~~ ✅ 已解决

- `baseline_results.json` — 保存 evolver 评估结果 (F1=0.026)
- `memory_r1_eval_results.json` — 保存 memory_r1 完整评估结果 (F1=0.036)

---

## 三、下一步操作计划

### 3.1 Phase 3：全闭环训练（纯 self-reward）

#### 修复概要（2026-04-03）

`train_phase3_full.py` 已修复以下问题：

| 修复项 | 说明 |
|--------|------|
| 🔴 事件上限 `--max_events 3` | 避免逐事件 LLM 推理瓶颈（同 Phase2/Baseline 修复） |
| 🔴 不再双重 LoRA | 删除了 `distiller.setup_dual_lora()`，避免与 Phase 2 LoRA 冲突/OOM |
| 🔴 LoRA 参数解冻 | `PeftModel` 加载后显式 `param.requires_grad = True` |
| 🟡 Speed/ETA 日志 | 每 50 episode 打印速度和预计剩余时间 |
| 🟡 `--no_qlora` 参数 | 与 Phase 2 保持一致 |
| 🟡 默认 3000 episodes | 从 10000 降低，预计 ~14h |
| 🟡 异常处理 | `agent.step()` 和 RL 更新均有 try/except，不会因单 episode 崩溃 |

#### Step 1: 预检（本地）

同步修改后的脚本到服务器：
```bash
scp scripts/train_phase3_full.py wujcan@Tang2:/NAS/yesh/G-MSRA/scripts/
```

#### Step 2: 检查 GPU（服务器）

```bash
nvidia-smi
# 确认至少一张 GPU 有 ≥25GB 空闲显存
# Phase 3 预估显存：~20-24GB（bf16 + LoRA + AdamW + forward/backward）
```

#### Step 3: 启动训练（服务器）

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=2 \
python scripts/train_phase3_full.py \
    --checkpoint outputs_v1/phase2/best \
    --lora_checkpoint outputs_v1/phase1/best \
    --output_dir outputs_v1/phase3 \
    --max_episodes 3000 \
    --max_events 3 \
    --no_qlora --no_wandb \
    2>&1 | tee logs_v1/phase3.log
```

> 📝 `--checkpoint` 指向 Phase 2 的 agent 状态（memory_store + step_count），`--lora_checkpoint` 指向 Phase 1 的 LoRA adapter（因为 Phase 2 没有保存 LoRA 权重）。

#### 预期运行指标

| 指标 | 预期值 |
|------|--------|
| Speed | ~0.05-0.06 ep/s |
| 总时间 | ~14 小时（3000 ep） |
| 显存 | ~20-24 GB |
| 日志输出 | 每 50 ep 含 Speed/ETA |
| Checkpoint | 每 500 ep 自动保存 |

#### Step 4: 监控进度（Shell 2）

```bash
tail -f logs_v1/phase3.log
# 应看到：
# Episode 50/3000 | R_avg=0.xxx | success=0.xxx | mem=N | Speed: 0.05 ep/s | ETA: 13.5h
```

#### Step 5: 下载结果到本地

训练完成后：
```bash
scp -r wujcan@Tang2:/NAS/yesh/G-MSRA/outputs_v1/phase3 d:/USTC/2026Winter/G-MSRA/outputs/
scp wujcan@Tang2:/NAS/yesh/G-MSRA/logs_v1/phase3.log d:/USTC/2026Winter/G-MSRA/logs/
```

预期输出文件：
- `outputs_v1/phase3/best/` — 最优 checkpoint（LoRA + memory_store + agent_meta）
- `outputs_v1/phase3/checkpoint_500/`, `checkpoint_1000/`, ... — 中间 checkpoint
- `outputs_v1/phase3/metrics.json` — 训练指标（每 50 ep）
- `outputs_v1/phase3/diagnostics.json` — 完整诊断（memory 分布、操作统计）

> ⚠️ **不支持多卡并行**：agent.step() 内部维护有状态的 memory_store，不兼容数据并行。单 GPU 即可。

---

### 3.2 G-MSRA 模型评测

Phase 3 完成后，用训练好的模型在 LoCoMo 测试集上评测：

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=<GPU> \
python baselines/eval_baselines.py \
    --agent gmsra \
    --checkpoint outputs_v1/phase3/best \
    --benchmark locomo \
    2>&1 | tee results/gmsra_eval.log
```

> 📝 如果 `eval_baselines.py` 不支持 `gmsra` agent，需要先添加 G-MSRA agent 的评测代码。

---

### 3.3 消融实验（Table 3）

验证各组件贡献（在 Phase 3 完成后运行）：

```bash
PYTHONPATH=/NAS/yesh/G-MSRA python scripts/run_ablation.py \
    --output_dir results/ablation \
    2>&1 | tee logs/ablation.log
```

---

### 3.4 合并结果 → 填论文表格

所有实验完成后，将分散的结果填入论文 Table 1/2：

| Method | LoCoMo F1 | LoCoMo EM |
|--------|:---------:|:---------:|
| Reflexion | 0.030 | 0.000 |
| EvolveR | 0.026 | 0.000 |
| Self-Consolidation | 0.032 | 0.000 |
| Memory-R1 | 0.036 | 0.000 |
| Mem0+Memory-R1 | 0.028 | 0.000 |
| **G-MSRA (Ours)** | **待评测** | — |

---

## 四、里程碑更新

| 里程碑 | 状态 | 完成日期 |
|--------|:----:|:--------:|
| M0: 核心库完成 | ✅ | 2026-03-15 |
| M0.5: 训练脚本补全 + 冒烟测试 27/27 | ✅ | 2026-03-16 |
| M0.8: Baseline 代码完成 | ✅ | 2026-03-16 |
| M0.9: 合成数据 Baseline 全通 | ✅ | 2026-03-17 |
| M1: Phase 0 SFT 训练完成 | ✅ | 2026-03-20 |
| M2: Phase 1 RL 训练完成 | ✅ | 2026-03-22 |
| M2.5: Baseline 评测性能优化 | ✅ | 2026-03-26 |
| M3: 无训练 Baseline 评测完成 | ✅ | 2026-03-26 |
| **M4: Phase 2 课程退火完成** | **✅** | **2026-03-30** |
| **M5: Memory-R1 Baseline 训练+评测完成** | **✅** | **2026-04-01** |
| **M6: Mem0+Memory-R1 Baseline 完成** | **✅** | **2026-04-02** |
| M7: Phase 3 全闭环训练 | ☐ | — |
| M8: G-MSRA 模型评测 | ☐ | — |
| M9: 消融实验 (Table 3) | ☐ | — |
| M10: 主表 (Table 1+2) 填入 | ☐ | — |
| M11: 分析图完成 | ☐ | — |
| M12: 论文初稿 → 投递 | ☐ | — |

---

## 五、本地文件清单

### 日志文件 (`logs_v1/`)

| 文件 | 说明 | 大小 |
|------|------|------|
| `phase1_tang.log` | Phase 1 RL 训练日志 | 4.1 MB |
| `phase2_tang.log` | Phase 2 旧版运行（未限事件，已废弃） | 1.6 MB |
| `phase2_v2.log` | **Phase 2 完整训练日志（500/500 步）** | 485 KB |

### 训练输出 (`outputs_v1/`)

| 目录 | 说明 | 状态 |
|------|------|:----:|
| `outputs_v1/phase0/` | SFT 热启动 checkpoint | ✅ 本地 |
| `outputs_v1/phase1/best` | Phase 1 RL 最优 checkpoint | ✅ 本地 |
| `outputs_v1/phase2/` | Phase 2 退火 checkpoint (best + checkpoint_100~500 + calibration.json) | ✅ 本地 |

### Baseline 结果 (`results_v1/baselines/`)

| 文件 | 说明 | 完整？ |
|------|------|:------:|
| `baseline_results.json` | EvolveR 评估结果 (F1=0.026, 314 ep) | ✅ |
| `memory_r1_eval_results.json` | Memory-R1 评估结果 (F1=0.036, 314 ep) | ✅ |
| `reflexion_fast_eval.log` | Reflexion fast_mode 评估 (F1=0.030) | ✅ |
| `evolver_fast_eval.log` | EvolveR fast_mode 评估 (F1=0.026) | ✅ |
| `selfconsolidation_fast_eval.log` | Self-Consolidation fast_mode 评估 (F1=0.032) | ✅ |
| `rl_baselines_v2.log` | Memory-R1 训练日志（3 epoch 完成） | ✅ |
| `rl_eval_v2.log` | Memory-R1 评估日志 | ✅ |
| `memory_r1_train_metrics.json` | Memory-R1 训练指标 | ✅ |
| `mem0_rl_v2.log` | Mem0+R1 训练+评估日志（完整） | ✅ |
| `rl_baselines_combined1.json` | Mem0+R1 完整结果 (F1=0.028, 314 ep) | ✅ |
| `rl_baselines_combined.json` | Mem0+R1 旧结果（44 ep 训练） | 🗄️ 归档 |

---

## 六、风险与应对

| 风险 | 概率 | 应对 |
|------|:----:|------|
| ~~训练代码瓶颈~~ | — | ✅ 已解决：max_events=3 |
| ~~OOM~~ | — | ✅ 已解决：setup_lora 修复 |
| ~~bitsandbytes 依赖~~ | — | ✅ 已解决：懒加载 |
| Phase 3 训练不收敛 | 中 | τ=0.302 表明 self-reward 相关性仍偏低 |
| 所有 Baseline F1 < 0.04 | 高 | 但这恰好为 G-MSRA 的自奖励方法提供对比空间 |

---

> **当前状态总结**：
>
> - ✅ Phase 0 + Phase 1 + **Phase 2 训练全部完成**
> - ✅ **5/5 个 Baseline 评测全部完成**（Reflexion、EvolveR、Self-Consolidation、Memory-R1、Mem0+R1）
> - ✅ 所有训练输出已下载到本地
> - ☐ 下一步：Phase 3 全闭环 → G-MSRA 模型评测 → 消融 → 填表 → 论文
