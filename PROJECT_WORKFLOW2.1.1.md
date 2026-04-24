# PROJECT_WORKFLOW 2.1.1 — v7 RL 信号链根因修复 & Phase 2 重训

> 创建时间：2026-04-18 04:00  
> 前序：WORKFLOW 2.1.0  
> 状态：**代码修复已完成，等待上传集群并启动 Phase 2 v7 重训**

---

## 一、v6 训练结果复盘

### 1.1 v6 日志数据（Step 1500/3000，~50%进度）

Phase 2 v6 训练了 ~21.4 小时（2026-04-17 05:54 → 04-18 03:18），数据如下：

| 指标 | v6 实际值 | v9.0 预期 | 判定 |
|------|:---------:|:---------:|:---:|
| NOOP 占比 | **96.6%**（4357/4512） | <60% | 🔴 未达标 |
| ADD 操作 | 155 | 渐进增长 | 🟡 |
| UPDATE 操作 | **0** | >100 | 🔴 完全失败 |
| DELETE 操作 | **0** | >100 | 🔴 完全失败 |
| Memory 条数 | 2 → **155** | 200-400 | 🟡 |
| α 退火 | 0.997 → **0.677** | → 0.0 | 🟡 频繁暂停 |
| τ (Kendall) | −0.22 ~ 0.93 震荡 | 稳步上升 | 🔴 |
| R_mem 分布 | 0.3(59%) / 0.5(38%) / 0.8(0.5%) | 连续 | 🔴 |
| R_env 非零率 | 14.6%，均值 0.191 | ↑ | ✅ 改善 |
| Consolidation | 0 | >2 | 🔴 |

**核心结论**：v6 的 7 个 BUG 修复（F1-F6）只修了 loss 公式和 Judge rubric，但没有解决 RL 信号链上的 **5 处根本性断裂**。

### 1.2 根因分析

| # | 断裂点 | 详细描述 | 严重性 |
|---|--------|---------|:------:|
| ① | REINFORCE 与在线决策脱节 | Phase 2 先用 `decide()` 处理 events，然后对最后 event **重新** `decide()` 做 REINFORCE。前面 events 的决策不参与梯度，且重新 decide 的上下文不一致 | 🔴 致命 |
| ② | R_mem 只有 3 个离散档 | Judge LLM 输出集中在 0.3/0.5/0.8，97.2% 的样本在这三个值。advantage 信号被噪声淹没 | 🔴 致命 |
| ③ | 零探索机制 | temperature=0.7 无法突破 NOOP 先验。模型从未生成过 UPDATE/DELETE，因此永远学不到它们的价值 | 🔴 致命 |
| ④ | Prompt 太长 (1024) 稀释梯度 | Response 只有 1-2 tokens（"NOOP"），但 prompt 有 1024 tokens，梯度极度稀释 | 🟡 |
| ⑤ | max_events=3 太少 | 每步只处理 3 个 events，memory 增长极慢（155/1500步），反馈循环太弱 | 🟡 |

---

## 二、v7 代码修复总结

| # | 修复 | 文件 | 描述 |
|---|------|------|------|
| **F7** | REINFORCE 梯度链修复 | `train_phase2_transition.py` | 每个 event 的 `decide()` 输出**立即**计算 `log π(a|s)`，episode 结束后用 episode-level reward 对所有 action 的 log_prob 统一 REINFORCE 更新 |
| **F8** | 移除 Judge R_mem，改用 QA F1 + operation bonus | `train_phase2_transition.py` | 避免离散三档 R_mem 的噪声。奖励改为：`R_episode = R_env(QA F1) + mean(operation_bonus)`。Judge R_mem 仍计算，但仅用于 τ 监控 |
| **F9** | ε-greedy 探索 | `memory_manager.py` | ε 从 0.3 → 0.05 线性衰减。探索时按 `[ADD=0.35, UPDATE=0.30, DELETE=0.15, NOOP=0.20]` 权重随机选择操作 |
| **F10** | Prompt 截断至 512 tokens | `memory_manager.py` | `compute_action_log_prob()` 将 prompt 截断至 `512 - len(action_ids)` tokens，集中梯度在决策区域 |
| **F11** | max_events 增至 8 | `train_phase2_transition.py` | CLI 参数 `--max_events 8`（默认值从 3 → 8），加速 memory 增长和反馈循环 |

### Operation Bonus 设计（F8）

```python
OPERATION_BONUS = {
    "ADD":    +0.20,   # 奖励存储新信息
    "UPDATE": +0.30,   # 最高奖励 — UPDATE 最难学习
    "DELETE": +0.20,   # 奖励清理过时信息
    "NOOP":   -0.10,   # 惩罚不作为
}
```

### ε-greedy 探索设计（F9）

```python
# ε 线性衰减
progress = step_idx / (total_steps - 1)
epsilon = 0.3 + (0.05 - 0.3) * progress  # 0.3 → 0.05

# 探索时的操作权重
weights = [ADD=0.35, UPDATE=0.30, DELETE=0.15, NOOP=0.20]
```

---

## 三、变更的文件清单

```
gmsra/manager/memory_manager.py       — F9, F10 (添加 decide_with_exploration + compute_action_log_prob)
scripts/train_phase2_transition.py     — F7, F8, F9, F10, F11 (训练循环完整重写)
```

---

## 四、部署步骤

### 4.0 上传文件

```
本地                                          → 集群
gmsra/manager/memory_manager.py               → /NAS/yesh/G-MSRA/gmsra/manager/memory_manager.py
scripts/train_phase2_transition.py             → /NAS/yesh/G-MSRA/scripts/train_phase2_transition.py
```

### 4.1 停止 v6 训练

```bash
tmux kill-session -t phase2_v6 2>/dev/null
```

### 4.2 验证文件完整性

```bash
cd /NAS/yesh/G-MSRA
conda activate gmsra
python -c "import ast; ast.parse(open('scripts/train_phase2_transition.py').read()); print('OK')"
python -c "import ast; ast.parse(open('gmsra/manager/memory_manager.py').read()); print('OK')"
```

### 4.3 启动 Phase 2 v7 训练

```bash
tmux new-session -d -s phase2_v7 "
export PYTHONPATH=/NAS/yesh/G-MSRA
CUDA_VISIBLE_DEVICES=1 python scripts/train_phase2_transition.py \
    --checkpoint outputs/phase1/best \
    --output_dir outputs/phase2_v7 \
    --anneal_steps 3000 \
    --max_events 8 \
    --epsilon_start 0.3 \
    --epsilon_end 0.05 \
    --no_qlora --no_wandb \
    2>&1 | tee logs/phase2_v7.log
"
```

---

## 五、监控指标与早停判断

### 5.1 健康指标（前 100 步）

| 指标 | 健康范围 | 如果不在范围 |
|------|:--------:|------------|
| NOOP 占比 | **<80%** | ε=0.3 应保证 ≤70% |
| UPDATE 次数 | >5 | 检查 `store.entries` 是否为空 |
| DELETE 次数 | >2 | 同上 |
| 训练无报错 | 无 Exception | 检查 `compute_action_log_prob` |

### 5.2 最终指标（Step 3000）

| 指标 | 目标 |
|------|:----:|
| NOOP 占比 | **<70%** |
| UPDATE+DELETE | **>50** |
| R_ext 均值 | **>0.25** |

---

## 六、Phase 3（Phase 2 完成后执行）

Phase 3 也需要同步应用 F7/F9 的思路（探索+正确梯度），但可等 Phase 2 v7 结果出来后再决定是否需要额外修改。

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/train_phase3_full.py \
    --checkpoint outputs/phase2_v7/best \
    --lora_checkpoint outputs/phase1/best \
    --max_episodes 3000 --max_events 5 --num_epochs 2 \
    --no_qlora --no_wandb \
    --output_dir outputs/phase3_v7 \
    2>&1 | tee logs/phase3_v7.log
```

---

## 七、预期效果

| 指标 | v5 (旧) | v6 (失败) | v7 (预期) |
|------|:-------:|:---------:|:---------:|
| NOOP 占比 | >94% | 96.6% | **<70%** |
| UPDATE+DELETE | 0 | 0 | **>50** |
| R_env 均值 | 0.044 | 0.191 | **>0.25** |
| G-MSRA vs Events Only | 0% | — | **>5% F1** |

---

## 八、时间线估算

| 阶段 | 任务 | 预计时长 | 预计完成 |
|:----:|------|:--------:|:--------:|
| ✅ | v7 代码修复 | 已完成 | 4/18 04:00 |
| 📌 | 上传至集群 + 启动 Phase 2 v7 | 10 min | 4/18 04:30 |
| ⏳ | Phase 2 v7 训练 | ~40-55h | **4/20 06:00** |
| ⏳ | Phase 3 v7 训练 | ~50-70h | **4/23 04:00** |
| ⏳ | 评测 (3 组) | ~10-15h | **4/23 20:00** |
| **总计** | | | **~5.5 天** |
