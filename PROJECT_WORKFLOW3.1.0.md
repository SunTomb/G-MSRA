# PROJECT_WORKFLOW 3.1.0 — G-MSRA v2: RL 记忆策展 + LLM Compaction

> 创建时间：2026-04-24 15:30  
> 前序：WORKFLOW 3.0.0 (实验总结 + 方向转型决策)  
> 状态：**Phase 0 完成 ✅ → Phase 1 训练脚本就绪 → 待部署训练**

---

## 一、方向决策总结

### 1.1 老师讨论结论

- **充足时间**完成实验和论文
- 目标：**高水平会议/期刊，Positive Result**
- 同意设计新的连续对话评测

### 1.2 核心转型：做减法

```
v1 (失败):  Event → RL CRUD → Memory Store → LoRA 蒸馏 → QA
                                                ↑
                                         问题在这里（覆盖通用能力）

v2 (新方案): Event → RL CRUD → Memory Store → (直接) → QA
                                     ↕
                              LLM Compaction
                          (聚类 + 摘要合并，不碰模型权重)
```

**删掉的模块**：LoRA 蒸馏、EWC 正则化、参数化 Consolidation  
**保留的模块**：RL CRUD 策略、Memory Store、语义检索 QA  
**新增的模块**：LLM Compaction（聚类合并）、EvoMemory 评测

### 1.3 论文标题方向

> **"Learning to Curate: RL-Trained Memory Management for Continual Dialogue Agents"**

---

## 二、Phase 0 完成进度 ✅

### 2.1 代码变更清单

| 文件 | 变更 | 说明 |
|------|:----:|------|
| `gmsra/config.py` | MODIFY | 新增 `CompactionConfig`（相似度阈值、聚类参数、触发条件） |
| `gmsra/consolidation/compaction.py` | **NEW** | `MemoryCompactor`: 聚类 → LLM 摘要 → DELETE+ADD |
| `gmsra/consolidation/__init__.py` | MODIFY | 导出 `MemoryCompactor` |
| `gmsra/agent.py` | MODIFY | `use_v2=True` 双模式支持；`_run_consolidation()` 支持 compaction |
| `scripts/eval_evomemory.py` | **NEW** | EvoMemory 知识演化评测（4 种模式） |
| `scripts/train_phase_v2.py` | **NEW** | v2 RL 训练脚本（delayed QA F1 奖励） |

### 2.2 验证状态

```
✅ GMSRAConfig + CompactionConfig 导入正常
✅ MemoryCompactor 导入正常
✅ GMSRAAgent(use_v2=True) 导入正常
✅ 聚类数学逻辑验证通过（cosine similarity）
✅ train_phase_v2.py 语法检查通过
⏳ 端到端 smoke test 需在集群运行（依赖 sentence_transformers + GPU）
```

---

## 三、Phase 1: RL 训练 — 详细指导

### 3.1 v2 训练 vs v11 训练的关键差异

| 维度 | v11 (旧) | v2 (新) |
|------|---------|---------|
| **奖励信号** | self-reward (Judge) | delayed QA F1 (ground truth) |
| **NOOP 惩罚** | -0.10 | **-0.30** (-0.15 base - 0.15 extra) |
| **UPDATE 奖励** | +0.35 | **+0.40** |
| **Consolidation** | LoRA 蒸馏（每 500 ep） | LLM Compaction（可选，每 100 ep） |
| **训练数据** | LoCoMo only (44 条) | **LoCoMo + EvoMemory (144 条)** |
| **每 episode events** | 5 | **10** |
| **epsilon** | 0.15 → 0.05 | **0.20 → 0.05**（更多探索） |
| **Memory 初始化** | 从 Phase 2 checkpoint 加载 | **每 episode 空 store**（无历史包袱） |
| **模型权重** | LoRA 被蒸馏修改 | **LoRA 只做 RL 梯度更新** |

### 3.2 资源估算基准

> 以下估算基于 v11 实测数据：3172 episodes × 5 events/ep → 19.1h @ 0.05 ep/s（单卡 A40 48GB, bf16）。  
> v2 每 episode 10 events（2× v11），但删除了 LoRA 蒸馏开销，净效速率约 0.03-0.04 ep/s。

| 资源项 | bf16 (`--no_qlora`) | QLoRA 4-bit |
|--------|:-------------------:|:-----------:|
| 模型权重 | ~14 GB | ~5 GB |
| LoRA 适配器 | ~0.1 GB | ~0.1 GB |
| 优化器状态 (AdamW) | ~2 GB | ~1 GB |
| KV Cache + 生成 | ~2 GB | ~1.5 GB |
| 梯度 + 激活 | ~2 GB | ~1 GB |
| **合计峰值** | **~20-22 GB** | **~9-11 GB** |
| 适合显卡 | A40 (48GB) ✅ / A100 (80GB) ✅ | RTX 3090 (24GB) ✅ |

### 3.3 训练命令

#### 实验 1: 基础 v2 训练（推荐先跑这个）

```bash
cd /NAS/yesh/G-MSRA

PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=0 \
python scripts/train_phase_v2.py \
    --lora_checkpoint outputs/phase1/best \
    --output_dir outputs/v2_base \
    --max_episodes 3000 --max_events 10 --num_epochs 3 \
    --epsilon_start 0.20 --epsilon_end 0.05 \
    --noop_penalty 0.15 --compactness_weight 0.10 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/v2_base.log
```

| 指标 | 值 |
|------|----|
| **显存占用** | **~20-22 GB** (bf16, 单卡) |
| **总 episodes** | 3000 × 3 epochs = **9000** |
| **预估速率** | ~0.03-0.04 ep/s (A40) |
| **预计时间** | **~65-85h** (A40) / **~50-65h** (A100) |

**关键监控指标**：

- NOOP% < 50%（vs v11 的 96%）→ 训练成功的必要条件
- UPDATE% > 10% → 证明 RL 学到了知识更新
- avg_f1 持续上升 → 奖励信号有效

#### 实验 2: 带 Compaction 的 v2 训练

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=4 \
python scripts/train_phase_v2.py \
    --lora_checkpoint outputs/phase1/best \
    --output_dir outputs/v2_compact \
    --max_episodes 3000 --max_events 10 --num_epochs 3 \
    --epsilon_start 0.20 --epsilon_end 0.05 \
    --noop_penalty 0.15 --compactness_weight 0.10 \
    --enable_compaction --compact_interval 100 \
    --compact_threshold 0.80 --compact_trigger_size 30 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/v2_compact.log
```

| 指标 | 值 |
|------|----|
| **显存占用** | **~22-25 GB** (bf16, 需额外 embedding 空间用于聚类) |
| **总 episodes** | 3000 × 3 epochs = **9000** |
| **预计时间** | **~70-95h** (A40) / **~55-75h** (A100) |
| **额外开销** | 每 100 ep 触发 LLM compaction (~30s/次, 共 ~90 次) |

#### 实验 3: 高 NOOP 惩罚消融

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=7 \
python scripts/train_phase_v2.py \
    --lora_checkpoint outputs/phase1/best \
    --output_dir outputs/v2_high_penalty \
    --max_episodes 3000 --max_events 10 --num_epochs 3 \
    --noop_penalty 0.25 --compactness_weight 0.15 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/v2_high_penalty.log
```

| 指标 | 值 |
|------|----|
| **显存占用** | **~20-22 GB** (与实验 1 相同) |
| **总 episodes** | 3000 × 3 epochs = **9000** |
| **预计时间** | **~65-85h** (A40) / **~50-65h** (A100) |

### 3.4 训练过程监控

训练脚本每 25 个 episode 输出一次日志，格式：

```
Ep 500/9000 | R=0.180 | F1=0.120 | eps=0.178 | mem=45 | 
NOOP=35%(win) 40%(all) | ops_win={'ADD':'30%','UPDATE':'20%','DELETE':'15%','NOOP':'35%'} |
explore=250 | compact=2 | baselines={ADD:0.12,UPDATE:0.45,DELETE:0.32,NOOP:-0.28} |
1.5 ep/s | ETA: 1.6h
```

**健康训练的特征**：

| 指标 | 健康值 | 不健康值（需干预） |
|------|--------|-------------------|
| NOOP% (window) | < 50% | > 70%（NOOP 固化重现） |
| UPDATE% (window) | > 10% | < 5%（没学到 UPDATE） |
| avg_f1 | 趋势上升 | 持续 0（奖励信号失效） |
| avg_reward | > 0 | 持续 < -0.2（惩罚过重） |
| memory_size | 10-100 | > 400（ADD 泛滥）或 0（全删了） |

### 3.5 如果训练不健康

**情况 1: NOOP 仍然 > 70%**

- 增大 NOOP 惩罚: `--noop_penalty 0.30`
- 增大初始探索: `--epsilon_start 0.30`

**情况 2: avg_f1 = 0（奖励信号失效）**

- 检查数据加载是否正常（QA pair 是否存在）
- 检查 `answer_question()` 是否能正确生成回答

**情况 3: memory_size 一直增长到 500**

- 增大 compactness_weight: `--compactness_weight 0.20`
- 降低 OPERATION_BONUS["ADD"] 到 0.00

**情况 4: 训练 NaN / loss 爆炸**

- 降低学习率: 修改 `config.rl.learning_rate` 到 `5e-6`
- 增大 grad_accum: 修改 `config.rl.gradient_accumulation_steps` 到 `4`

---

## 四、Phase 2: 评测 — 训练完成后执行

### 4.1 EvoMemory 评测（主战场）

> EvoMemory 共 100 条样例，每条 3-5 个 events + 1 QA。纯推理，无梯度。  
> **显存：~15-16 GB** (bf16 推理) | 可与训练共享同一张卡（串行）

```bash
# Raw ADD baseline （预计 ~15min）
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_evomemory.py \
    --mode raw_add \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/evomemory_raw_add \
    --no_qlora \
    2>&1 | tee logs/eval_evo_raw_add.log

# Heuristic CRUD baseline （预计 ~20min）
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_evomemory.py \
    --mode heuristic_crud \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/evomemory_heuristic \
    --update_threshold 0.85 \
    --no_qlora \
    2>&1 | tee logs/eval_evo_heuristic.log

# RL CRUD (v2 trained) （预计 ~25min，含 RL decide 开销）
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_evomemory.py \
    --mode rl_crud \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/evomemory_rl_crud \
    --no_qlora \
    2>&1 | tee logs/eval_evo_rl_crud.log
```

### 4.2 LoCoMo per-example 评测（泛化验证）

> LoCoMo 44 条 × 多轮 QA。  
> **显存：~15-16 GB** (bf16 推理) | **预计时间：~30-45min**

```bash
# v2 在传统评测下必须 ≥ Events Only baseline (0.097)
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=0 \
python scripts/eval_locomo.py \
    --lora_checkpoint outputs/phase1/best \
    --output_dir results/v2_locomo \
    --no_qlora --benchmark locomo \
    2>&1 | tee logs/eval_v2_locomo.log
```

### 4.3 预期结果

| 方法 | EvoMemory F1 (预期) | LoCoMo F1 (预期) |
|------|:-------------------:|:----------------:|
| Raw ADD | 低（旧事实干扰检索） | ~0.097 (baseline) |
| Heuristic CRUD | 中（规则 UPDATE） | ~0.097 |
| **RL CRUD (v2)** | **高（学到的 UPDATE）** | **≥ 0.097** |
| v11 (旧系统) | — | 0.048 (已知失败) |

---

## 五、文件索引

### 5.1 v2 新增文件

| 路径 | 说明 |
|------|------|
| `gmsra/consolidation/compaction.py` | LLM Compaction 模块 |
| `scripts/train_phase_v2.py` | v2 训练脚本 |
| `scripts/eval_evomemory.py` | EvoMemory 评测脚本 |

### 5.2 v2 修改文件

| 路径 | 说明 |
|------|------|
| `gmsra/config.py` | 新增 `CompactionConfig` |
| `gmsra/agent.py` | `use_v2` 双模式支持 |
| `gmsra/consolidation/__init__.py` | 导出 `MemoryCompactor` |

### 5.3 历史版本文档

| 文件 | 内容 |
|------|------|
| `PROJECT_WORKFLOW2.2.1.md` | 评测诊断 + 方案 B 验证 |
| `PROJECT_WORKFLOW3.0.0.md` | 实验总结 + 方向转型决策 |
| `PROJECT_WORKFLOW3.1.0.md` | **本文档：v2 实现 + 训练指导** |

---

## 六、训练后 Checklist

训练完成后，按以下顺序执行：

```
1. [ ] 检查 logs/v2_base.log
       - NOOP% < 50%? → 继续
       - NOOP% > 70%? → 增大惩罚重训

2. [ ] 检查 outputs/v2_base/metrics.json
       - avg_f1 趋势上升? → 继续
       - avg_f1 = 0? → 检查数据/QA pipeline

3. [ ] 跑 EvoMemory 评测 (4.1 中的 3 个命令)
       - RL CRUD > Raw ADD? → 核心 positive result!
       - RL CRUD > Heuristic CRUD? → RL 价值验证

4. [ ] 跑 LoCoMo per-example 评测 (4.2)
       - v2 F1 ≥ 0.097? → 泛化验证通过
       - v2 F1 < 0.097? → 分析原因

5. [ ] 撰写 PROJECT_WORKFLOW3.2.0.md 记录结果

6. [ ] 开始论文写作
```

---

## 七、GPU 规划总览

### 7.1 显卡分配建议

| GPU | 任务 | 显存占用 | 预计时间 |
|:---:|------|:--------:|:--------:|
| GPU 1 (A40/A100) | 实验 1: v2_base 训练 | ~20-22 GB | ~65-85h |
| GPU 2 (A40/A100) | 实验 2: v2_compact 训练 | ~22-25 GB | ~70-95h |
| GPU 3 (A40/A100) | 实验 3: v2_high_penalty 训练 | ~20-22 GB | ~65-85h |
| GPU 0 (任意) | Phase 2: 评测（训练后串行） | ~15-16 GB | ~2h |

> **3 个训练实验可并行**，各占一张卡。评测在训练结束后串行跑即可。  
> 如果只有 2 张卡，建议先跑实验 1 + 实验 3（并行），实验 2 后补。

### 7.2 时间线

| 阶段 | 状态 | 时间 |
|:----:|------|:----:|
| Phase 1-3 v11 训练 | ✅ | 4/17-21 |
| 28 组评测 + 诊断 | ✅ | 4/22-24 |
| 方向转型决策 | ✅ | 4/24 |
| Phase 0: 架构精简 | ✅ | 4/24 |
| Phase 1: v2 训练脚本 | ✅ | 4/24 |
| 📌 **v2 训练 (3 实验并行)** | **待部署** | 4/24-28 (~3-4天) |
| ⏳ Phase 2: 评测 | 待训练完成 | 4/28-29 (~2h) |
| ⏳ Phase 3: Baselines | 待实现 | 4/29-30 |
| ⏳ Phase 4: 分析+论文 | 待开始 | 4/30+ |
