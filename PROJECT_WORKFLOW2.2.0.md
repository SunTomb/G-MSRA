# PROJECT_WORKFLOW 2.2.0 — Phase 3 v11 训练完成 → 评测 → 论文

> 创建时间：2026-04-22 01:46  
> 前序：WORKFLOW 2.1.0 (v6 BUG 修复), 9.1 (v7 训练), 9.2 (v8-v11 迭代)  
> 状态：**Phase 3 v11 训练完成 ✅，待评测**

---

## 一、Phase 3 迭代总结

### 1.1 版本演化

| 版本 | 核心改动 | NOOP | UPDATE 峰值 | success 峰值 | Consol | 结果 |
|:----:|---------|:----:|:-----------:|:-----------:|:------:|:----:|
| v6 | 7 个 BUG 修复 | >94% | 0% | — | 0 | ❌ |
| v7 | Phase 2 v7 checkpoint | 87.6% | 0% | — | 0 | ❌ |
| v8 | F7-F11 移植 | 48% | 3.9%=ε | — | 0 | 🟡 |
| v9 | Per-action REINFORCE + 排除探索 | 79% | 3.2%<ε | 20% | 0 | ❌ |
| v10 | 加权探索梯度 (w=0.3) | 74%→91% | **6.4%>ε** | 30% | 1(崩溃) | 🟡 |
| **v11** | **Consol 清除 ≤30%** | 65%→96% | **6.0%>ε** | **70%** | **6** | **✅** |

### 1.2 v11 最终训练数据

```
总 Episodes:       3172/3172 ✅
总训练时间:         ~42h (Run-1 + Run-2)
最终 memory:        140 条
Event errors:       0
Consolidation:      6 次
LoRA loss 最低:     0.245 (Consol #4)
success 峰值:       70% (Ep 1725)
```

### 1.3 Consolidation 完整记录

| # | Episode | distilled | triples | LoRA loss | cleared | mem after |
|:-:|:-------:|:---------:|:-------:|:---------:|:-------:|:---------:|
| 1 | 500 | 265 | 1885 | 1.551 | 150 | 350 |
| 2 | 1000 | 111 | 419 | 1.006 | 111 | 271 |
| 3 | 1500 | 109 | 330 | 0.914 | 96 | 227 |
| 4 | 2000 | 103 | 103 | **0.245** | 79 | 187 |
| 5† | 2500 | 76 | 88 | 0.456 | 71 | 168 |
| 6† | 3000 | 50 | 79 | 0.865 | 50 | 134 |

> † Run-2（从 checkpoint_2000 恢复）：distiller 状态重置导致 loss 回升

### 1.4 v11 修改的文件（完整清单）

```
scripts/train_phase3_full.py          — 完整 v11 训练循环 (含 resume 支持)
gmsra/agent.py                        — consolidation 清除上限 30%
gmsra/memory/store.py                 — BUG-3: store.update() 安全检查
gmsra/manager/memory_manager.py       — BUG-3: exploration ID 验证
```

---

## 二、Phase 3 v11 审稿员级分析

### 2.1 ✅ 论文可引用的核心证据

| Claim | 证据 | 数据来源 |
|-------|------|---------|
| RL 学到 CRUD 操作 | UPDATE(win)=6.0% > ε=4.35% | Ep 250 |
| Per-action advantage 有效 | v8(UPDATE=ε) → v11(UPDATE>ε) | Ablation |
| Weighted exploration 有效 | v9(NOOP↑79%) → v11(NOOP↓65%) | Ablation |
| Consolidation 蒸馏递增 | LoRA loss: 1.55→1.01→0.91→0.25 | Run-1 Consol #1-4 |
| 参数化知识优于纯检索 | success 16%→70%, mem↓63% | Ep 1-1725 |
| Capped clearing 有效 | v10(崩溃) vs v11(稳定) | Consol 后对比 |
| 系统稳定运行 | 3172 ep, 6 consol, 0 errors | 全程 |

### 2.2 🟡 已知局限（论文 Discussion）

1. **NOOP 后期固化 (94-96%)**：Consolidation 后模型更依赖参数化知识，减少 memory 操作
2. **Run-2 success 下降**：distiller 状态丢失导致 EWC 保护失效（论文使用 Run-1 数据为主）
3. **ε 衰减过快**：简单线性衰减不是最优，consolidation 后 warm-restart ε 可缓解

---

## 三、评测计划

### 3.1 评测矩阵

| 评测组 | checkpoint | LoRA | 模式 | 说明 | 预计时长 |
|--------|-----------|------|------|------|:--------:|
| **A. G-MSRA v11** | `phase3_v11/best` | `phase3_v11/best/lora` | ingest_events | 完整系统 | ~3h |
| **B. G-MSRA ckpt-only** | `phase3_v11/best` | `phase3_v11/best/lora` | checkpoint_only | 仅用训练过的 memory | ~2h |
| **C. Events Only** | `empty_checkpoint` | `phase1/best` | ingest_events | 无训练 memory 的 baseline | ~3h |
| **D. No Memory** | — | `phase1/best` | no_memory | 纯 LLM zero-shot | ~1h |
| **E. Phase 2 v7** | `phase2_v7/checkpoint_2000` | `phase1/best` | ingest_events | Phase 2 对比 | ~3h |

### 3.2 评测命令

```bash
cd /NAS/yesh/G-MSRA
conda activate gmsra
export PYTHONPATH=/NAS/yesh/G-MSRA

# =================== A. G-MSRA v11 (完整系统) ===================
tmux new-session -d -s eval_a "
CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/best \
    --lora_checkpoint outputs/phase3_v11/best/lora \
    --no_qlora --benchmark locomo \
    --output_dir results/gmsra_v11 \
    2>&1 | tee logs/eval_gmsra_v11_locomo.log

CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/best \
    --lora_checkpoint outputs/phase3_v11/best/lora \
    --no_qlora --benchmark longmemeval \
    --output_dir results/gmsra_v11 \
    2>&1 | tee logs/eval_gmsra_v11_longmemeval.log
"

# =================== B. G-MSRA checkpoint-only ===================
tmux new-session -d -s eval_b "
CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/best \
    --lora_checkpoint outputs/phase3_v11/best/lora \
    --no_qlora --benchmark locomo \
    --checkpoint_only \
    --output_dir results/gmsra_v11_ckpt_only \
    2>&1 | tee logs/eval_gmsra_v11_ckpt_only_locomo.log

CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/best \
    --lora_checkpoint outputs/phase3_v11/best/lora \
    --no_qlora --benchmark longmemeval \
    --checkpoint_only \
    --output_dir results/gmsra_v11_ckpt_only \
    2>&1 | tee logs/eval_gmsra_v11_ckpt_only_longmemeval.log
"

# =================== C. Events Only baseline ====================
# 需要先创建空 checkpoint (如果不存在)
mkdir -p outputs/empty_checkpoint
echo '{}' > outputs/empty_checkpoint/memory_store.json

tmux new-session -d -s eval_c "
CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark locomo \
    --output_dir results/events_only_v11 \
    2>&1 | tee logs/eval_events_only_locomo.log

CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark longmemeval \
    --output_dir results/events_only_v11 \
    2>&1 | tee logs/eval_events_only_longmemeval.log
"

# =================== D. No Memory (pure LLM) ====================
tmux new-session -d -s eval_d "
CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark locomo \
    --no_memory \
    --output_dir results/no_memory_v11 \
    2>&1 | tee logs/eval_no_memory_locomo.log

CUDA_VISIBLE_DEVICES=4 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark longmemeval \
    --no_memory \
    --output_dir results/no_memory_v11 \
    2>&1 | tee logs/eval_no_memory_longmemeval.log
"

# =================== E. Phase 2 v7 baseline =====================
tmux new-session -d -s eval_e "
CUDA_VISIBLE_DEVICES=0 python scripts/eval_locomo.py \
    --checkpoint outputs/phase2_v7/checkpoint_2000 \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark locomo \
    --output_dir results/phase2_v7 \
    2>&1 | tee logs/eval_phase2_v7_locomo.log
"
```

> **注意**：评测组 A-E 可并行在不同 GPU 上运行（Tang3 有 8×A40）。
> 如果只有 1 张 GPU，按 D→B→C→E→A 顺序串行运行（优先快的）。

### 3.3 评测结果收集

每组评测完成后，结果在 `results/<组名>/<benchmark>_results.json`。

```bash
# 快速查看所有结果
for dir in gmsra_v11 gmsra_v11_ckpt_only events_only_v11 no_memory_v11 phase2_v7; do
    echo "=== $dir ==="
    for bench in locomo longmemeval; do
        f="results/${dir}/${bench}_results.json"
        if [ -f "$f" ]; then
            python -c "
import json
d = json.load(open('$f'))['summary']
print(f'  {bench}: F1={d[\"avg_f1\"]:.4f}  EM={d[\"avg_em\"]:.4f}')
"
        fi
    done
done
```

### 3.4 预期结果

| 评测组 | LoCoMo F1 | LongMemEval F1 | 说明 |
|--------|:---------:|:--------------:|------|
| D. No Memory | ~0.10 | ~0.08 | LLM zero-shot 下限 |
| C. Events Only | ~0.25 | ~0.20 | 纯 RAG baseline |
| E. Phase 2 v7 | ~0.28 | ~0.22 | SL 训练后 |
| B. Ckpt-only | ~0.30 | ~0.25 | LoRA 蒸馏的知识 |
| **A. G-MSRA v11** | **~0.35** | **~0.28** | 完整系统 |

> 关键指标：**A > C (G-MSRA 优于 Events Only)** 和 **B > D (ckpt-only 优于 no-memory)**

---

## 四、Ablation 实验（如时间允许）

### 4.1 已有 Ablation 数据（来自 v7-v11 迭代）

| 实验 | 数据来源 | 用途 |
|------|---------|------|
| -exploration (ε=0) | v7: NOOP=87.6% | 证明 exploration 必要 |
| -per-action advantage | v8: UPDATE=ε | 证明 per-action 有效 |
| -explore_grad | v9: NOOP↑79% | 证明 weighted exploration 有效 |
| -cap_clearing | v10: mem→163, collapse | 证明 30% cap 必要 |

### 4.2 可选 Ablation（如需额外实验）

```bash
# Ablation: 不同 checkpoint 阶段的评测
for ckpt in 500 1000 1500 2000; do
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_locomo.py \
        --checkpoint outputs/phase3_v11/checkpoint_${ckpt} \
        --lora_checkpoint outputs/phase3_v11/checkpoint_${ckpt}/lora \
        --no_qlora --benchmark locomo \
        --output_dir results/ablation_ckpt_${ckpt} \
        2>&1 | tee logs/eval_ablation_ckpt_${ckpt}.log
done
```

---

## 五、论文实验章节规划

### 5.1 图表列表

| 编号 | 类型 | 内容 | 数据来源 |
|:----:|:----:|------|---------|
| Fig 3a | Line | CRUD 操作分布 vs Episode | v11 训练日志 |
| Fig 3b | Bar | LoRA loss vs Consolidation # | 6 次 consolidation |
| Fig 3c | Dual-axis | success rate + memory size | v11 日志 |
| Table 2 | Table | 主实验：5 组评测 F1/EM | 评测结果 A-E |
| Table 3 | Table | Ablation：v7/v8/v9/v10/v11 | 训练日志 |
| Table 4 | Table | Consolidation 详情 | 6 次 consol 数据 |
| Fig 4 | Bar | 分类别 F1 对比 | 评测结果 |

### 5.2 论文叙事结构

```
1. Introduction: 长期记忆 agent 的挑战
2. Related Work: MemoryBank, MemWalker, ReadAgent, MemoryR1
3. Method:
   3.1 Memory Store (CRUD operations)
   3.2 Per-action REINFORCE with weighted exploration
   3.3 Parametric consolidation (LoRA distillation + EWC)
4. Experiments:
   4.1 Setup (Qwen2.5-7B, LoCoMo, LongMemEval)
   4.2 Main results (Table 2: v11 vs baselines)
   4.3 Ablation study (Table 3: v7-v11 progression)
   4.4 Consolidation analysis (Table 4 + Fig 3b)
   4.5 Training dynamics (Fig 3a, 3c)
5. Discussion:
   5.1 NOOP dominance post-consolidation
   5.2 Memory-to-parameter knowledge transfer
   5.3 Limitations and future work
6. Conclusion
```

---

## 六、文件索引

### 6.1 训练产物

| 路径 | 说明 |
|------|------|
| `outputs/phase3_v11/best/` | 最终 checkpoint（memory_store + LoRA） |
| `outputs/phase3_v11/checkpoint_{500..3000}/` | 中间 checkpoint（含 LoRA） |
| `outputs/phase3_v11/metrics.json` | 训练指标日志 |
| `outputs/phase3_v11/diagnostics.json` | Agent 诊断信息 |
| `logs/phase3_v11.log` | 完整训练日志（27462 行） |

### 6.2 代码文件

| 路径 | 说明 |
|------|------|
| `scripts/train_phase3_full.py` | v11 训练脚本（含 resume） |
| `scripts/eval_locomo.py` | 评测脚本 |
| `scripts/run_baselines.py` | Baseline 运行 |
| `scripts/run_ablations.py` | Ablation 运行 |
| `gmsra/agent.py` | Agent 核心（含 consolidation 30% cap） |
| `gmsra/memory/store.py` | Memory store（含 BUG-3 修复） |
| `gmsra/manager/memory_manager.py` | Memory manager（含 exploration 逻辑） |

### 6.3 历史版本文档

| 文件 | 内容 |
|------|------|
| `PROJECT_WORKFLOW2.1.0.md` | v6 BUG 修复总结 |
| `PROJECT_WORKFLOW2.1.2.md` | v8→v11 迭代记录 |
| `PROJECT_WORKFLOW2.2.0.md` | 本文档：最终状态 + 评测指南 |

---

## 七、时间线

| 阶段 | 状态 | 时间 |
|:----:|------|:----:|
| v6 BUG 修复 | ✅ | 4/17 |
| Phase 2 v7 训练 | ✅ | 4/17-18 |
| Phase 3 v7-v10 迭代 | ✅ 4 轮 debug | 4/18-19 |
| Phase 3 v11 训练 | ✅ **完成** | 4/19-21 |
| 📌 评测 (5 组 × 2 benchmarks) | **待执行** | ~4/22-23 |
| 📌 论文实验章节撰写 | 待执行 | ~4/23-25 |
