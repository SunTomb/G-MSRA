# PROJECT_WORKFLOW 9.2 — Phase 3 迭代：v7→v8→v9→v10→v11

> 创建时间：2026-04-18 22:00  
> 最后更新：2026-04-19 14:20  
> 前序：WORKFLOW 9.1  
> 状态：**Phase 3 v11 代码修复已完成，等待上传部署**

---

## 一、Phase 2 v7 最终训练结果（✅完成）

> 完成时间：2026-04-18 20:06 | 总耗时：15.9h

| 操作 | 次数 | 占比 |
|:----:|:----:|:----:|
| ADD | 15265 | 63.6% |
| NOOP | 6863 | 28.6% |
| UPDATE | 1260 | 5.2% |
| DELETE | 612 | 2.5% |

- Phase 3 使用 `checkpoint_2000`（Step 2800+ 模型退化）

---

## 二、Phase 3 版本演化总览

| 版本 | 核心改动 | NOOP | UPDATE | 结果 | 失败原因 |
|:----:|---------|:----:|:------:|:----:|---------|
| v7 | 未携带 F7-F11 | 87.6% | 0% | ❌ | 脚本遗留 v6 代码 |
| v8 | 移植 F7-F11 | 48% | 3.9%=ε | 🟡 | Episode-level advantage |
| v9 | 排除探索梯度 | 79% | 1.4% | ❌ | FIX-3 矫枉过正 |
| v10 | 加权探索梯度 | 74%→91% | **6.4%>ε** ✅ | 🟡 | Consolidation 清除 67% memory |
| **v11** | **限制清除 ≤30%** | ⏳ | ⏳ | ⏳ | — |

---

## 三、Phase 3 v10 关键结果

### 3.1 ✅ RL 首次学到 UPDATE

```
UPDATE(win) 峰值 = 6.4%  >  ε 理论值 4.0%
UPDATE baseline: 0.00 (v9) → 0.38 (v10)
Event errors: 30 (v9) → 0 (v10)
```

### 3.2 🔴 Consolidation 灾难

```
Ep 500: mem=500 → 清除 337 条(67%) → mem=163
       NOOP 74% → 91%, success 30% → 2%
       恢复速度：每 25 ep +3 条 → 需 ~2700 ep 恢复
```

---

## 四、Phase 3 v11 修复

### 4.1 修改的文件

| 文件 | 改动 |
|------|------|
| `gmsra/agent.py` | **1 处**：`_run_consolidation()` 限制清除 ≤30% |
| `scripts/train_phase3_full.py` | 版本号 v10→v11（RL 逻辑不变） |

### 4.2 核心改动

```python
# agent.py: _run_consolidation()
# v11: 限制最多清除 30% 的记忆
max_clear = int(len(self.memory_store.entries) * 0.3)
cleared = 0
for entry_id in self.distiller.distilled_entries[-stats.get("distilled", 0):]:
    if cleared >= max_clear:
        break
    if entry_id in self.memory_store.entries:
        self.memory_store.delete(entry_id)
        cleared += 1
```

**v10 vs v11 对比（Ep 500 consolidation 时）**：

| 指标 | v10 | v11 (预期) |
|------|:---:|:---------:|
| 蒸馏 candidates | 337 | 337 (不变) |
| 清除数量 | 337 (100%) | **≤150 (30%)** |
| 清除后 mem | 163 | **≥350** |

---

## 五、部署步骤

### 5.0 停止旧训练

```bash
tmux kill-session -t phase3_v10 2>/dev/null
```

### 5.1 上传

```
本地                                 → 集群
gmsra/agent.py                       → /NAS/yesh/G-MSRA/gmsra/agent.py
scripts/train_phase3_full.py         → /NAS/yesh/G-MSRA/scripts/train_phase3_full.py
```

### 5.2 验证

```bash
cd /NAS/yesh/G-MSRA
conda activate gmsra
python -c "import ast; ast.parse(open('gmsra/agent.py').read()); print('OK')"
python -c "import ast; ast.parse(open('scripts/train_phase3_full.py').read()); print('OK')"
```

### 5.3 启动 Phase 3 v11

```bash
tmux new-session -d -s phase3_v11 "
export PYTHONPATH=/NAS/yesh/G-MSRA
CUDA_VISIBLE_DEVICES=1 python scripts/train_phase3_full.py \
    --checkpoint outputs/phase2_v7/checkpoint_2000 \
    --lora_checkpoint outputs/phase1/best \
    --max_episodes 3000 \
    --max_events 5 \
    --num_epochs 2 \
    --epsilon_start 0.15 \
    --epsilon_end 0.05 \
    --no_qlora --no_wandb \
    --output_dir outputs/phase3_v11 \
    2>&1 | tee logs/phase3_v11.log
"
```

### 5.4 监控

```bash
tmux attach -t phase3_v11
tail -f logs/phase3_v11.log

# 关键检查：consolidation 后 mem 是否保持 >350
grep -E 'Consolidation at|Cleared|Episode.*/3172' logs/phase3_v11.log | tail -5
```

---

## 六、v11 预期指标

### 6.1 Consolidation 前（Ep 1-475，应复现 v10）

| 指标 | v10 实际 | v11 预期 |
|------|:--------:|:--------:|
| NOOP(win) | 74-78% | ≈同 |
| UPDATE(win) | 5-6.4% | ≈同 |
| UPDATE baseline | 0.38 | ≈同 |
| success | 10-30% | ≈同 |

### 6.2 Consolidation 后（Ep 500+，关键差异点）

| 指标 | v10 实际 | **v11 预期** |
|------|:--------:|:----------:|
| mem after | 163 | **≥350** |
| NOOP(win) Ep 600 | 89% | **<80%** |
| success Ep 600 | 6% | **>15%** |
| ADD 恢复速度 | 极慢 | 正常 |

### 6.3 最终（Ep 3172）

| 指标 | 预期 |
|------|:----:|
| NOOP | **<50%** |
| UPDATE | **>8%** |
| Consolidation | **≥6 次** |
| mem | **>400** |

---

## 七、时间线

| 阶段 | 状态 | 时间 |
|:----:|------|:----:|
| Phase 2 v7 | ✅ | 4/18 |
| Phase 3 v7 | ❌ NOOP=87% | 4/18 |
| Phase 3 v8 | ❌ UPDATE=ε | 4/19 01:48 |
| Phase 3 v9 | ❌ NOOP↑ | 4/19 06:06 |
| Phase 3 v10 | 🟡 RL✅ consol❌ | 4/19 13:53 |
| v11 代码修复 | ✅ | 4/19 14:20 |
| 📌 上传 + 启动 v11 | 待执行 | ~4/19 14:30 |
| ⏳ Phase 3 v11 训练 | — | **~4/21** |
| ⏳ 评测 | — | **~4/22** |

---

## 八、评测（v11 完成后）

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/best \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark longmemeval \
    --output_dir results/gmsra_v11 \
    2>&1 | tee logs/eval_gmsra_v11.log
```

---

## 九、Ablation 计划（论文 Table）

| 实验 | 描述 | 已有数据 |
|------|------|:--------:|
| Full v11 | Per-action + weighted explore + capped consolidation | ⏳ |
| -exploration | ε=0 | v7: NOOP=87.6% |
| -per-action | Episode-level advantage | v8: UPDATE=ε |
| -explore_grad | 排除探索梯度 | v9: NOOP↑ |
| -cap_clearing | 不限制清除比例 | v10: mem→163 |
| -consolidation | 完全不触发 | 可从 v11 checkpoint 设置 |
