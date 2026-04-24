# PROJECT_WORKFLOW 2.1.0 — v6 BUG 修复 & 重训计划

> 创建时间：2026-04-17 01:00  
> 前序：WORKFLOW 1.8.0  
> 状态：**Phase 0 代码修复已完成，等待 Phase 1-3 重训**

---

## 一、问题回顾

T11 Scalability 实验发现 G-MSRA 与 Events Only 结果字节级一致。深度代码审查发现 **7 个关键 BUG** 导致 RL 训练彻底失败（NOOP > 94%）。

## 二、v6 代码修复总结

| # | 修复 | 文件 | 状态 |
|---|------|------|:----:|
| F1 | REINFORCE loss: `teacher-forcing CE × advantage` → 正确的 `log π(a\|s) × advantage` | `train_phase2_transition.py`, `train_phase3_full.py` | ✅ |
| F2 | 移除 `MAX_EVENT_ADDS=1` 硬限制，改用 memory store 容量自动淘汰 | `train_phase3_full.py` | ✅ |
| F3 | 添加 NOOP penalty (-0.1)、UPDATE/DELETE bonus (+0.15)、ADD bonus (+0.05) | `train_phase3_full.py` | ✅ |
| F4 | GRPO reward_fn 的 index mapping: `i % len` → `i // num_gens`，并添加 prompt matching | `train_phase1_rl.py` | ✅ |
| F5 | 评测添加 `--checkpoint_only` 模式，不注入 per-example events | `eval_locomo.py` | ✅ |
| F6 | Judge rubric 重写：NOOP 默认评分从 0.6-0.8 降至 0.1-0.3，Active ops 奖励 0.5-1.0 | `grounded_reward.py` | ✅ |

## 三、资源与时间估算

### 硬件参考

- **模型**：Qwen/Qwen2.5-7B-Instruct, bf16 (no_qlora)
- **服务器**：Tang3 (A40 48GB × 8)

### 各 Phase 资源需求

| Phase | GPU 配置 | 显存估算 | 预计时长 | 速度参考 | 说明 |
|:-----:|---------|:--------:|:--------:|:--------:|------|
| **Phase 1** | 6× A40 (可复用) | ~22-30 GB/卡 | ⏭ 跳过 | 0.07 ep/s | 复用已有 LoRA（格式学习正确） |
| **Phase 2** | 1× A40 | **~28-32 GB** | **~40-55h** | 0.02 step/s | 3000 steps, 含 decide() + Judge LLM 推理 |
| **Phase 3** | 1× A40 | **~30-35 GB** | **~60-80h** | 0.01-0.02 ep/s | 5000 eps × 2 epochs, 含 step() + Judge + RL |
| **评测** | 1× A40 | ~20-24 GB | **~3-5h/组** | 0.14 ex/s | 500 examples × 3 组 ≈ 9-15h 总评测 |

> **v6 vs v5 速度变化**：v6 的 RL 更新不再调用额外的 `decide()`（省 1 次 LLM 推理），但移除了 MAX_EVENT_ADDS=1 后每 episode 可能执行更多 ADD（多 embedding 计算）。总体速度预计 **与 v5 持平或略快**（~10%）。

### 显存详细分解

```
Qwen2.5-7B bf16 模型         ≈ 14 GB
LoRA adapter (r=16)          ≈  0.1 GB
Sentence-Transformers encoder ≈  0.3 GB
KV Cache (ctx=1024)          ≈  2-4 GB
Optimizer states (AdamW)     ≈  4-6 GB
Activations + Gradients      ≈  6-10 GB
────────────────────────────────────────
总计                         ≈ 27-35 GB
```

> **安全裕量**：A40 有 48GB 显存，最坏情况下占用 35GB，仍有 13GB 余量。如果 OOM，减小 `max_events` 从 5 → 3 可降低 ~3GB。

### 时间线规划

| 时段 | 任务 | 开始 → 结束（预估） |
|:----:|------|:------------------:|
| Day 1-3 | Phase 2 重训 (3000 steps) | 4/17 午 → 4/19 晚 |
| Day 3-6 | Phase 3 重训 (5000 eps × 2) | 4/19 晚 → 4/22 午 |
| Day 6-7 | 评测 (3 组 × 2 benchmarks) | 4/22 午 → 4/23 午 |
| **总计** | | **~6-7 天** |

> **加速方案**：如果时间紧迫，可将 Phase 3 `--max_episodes` 从 5000 减到 3000（省 ~30h），或 `--num_epochs` 从 2 减到 1（省 ~50%），但可能影响收敛质量。

---

## 四、重训命令

### 4.1 Phase 2 重训

```bash
# Tang3, 单卡, 预计 40-55h, ~28-32 GB VRAM
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=2 python scripts/train_phase2_transition.py \
    --checkpoint outputs_v1/phase1/best \
    --anneal_steps 3000 \
    --no_qlora --no_wandb \
    --output_dir outputs_v1/phase2_v6 \
    2>&1 | tee logs2/phase2_v6.log
```

**监控要点**：
- 观察 `operation_str` 日志，NOOP 占比应逐步下降到 < 70%
- R_ext 和 R_self 的 Kendall τ 应逐步上升
- 内存大小 (mem_size) 应有增长趋势

### 4.2 Phase 3 重训

```bash
# Tang3, 单卡, 预计 60-80h, ~30-35 GB VRAM
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/train_phase3_full.py \
    --checkpoint outputs_v1/phase2_v6/best \
    --lora_checkpoint outputs_v1/phase1/best \
    --max_episodes 5000 \
    --max_events 5 \
    --num_epochs 2 \
    --no_qlora --no_wandb \
    --output_dir outputs_v1/phase3_v6 \
    2>&1 | tee logs_v1/phase3_v6.log
```

**监控要点**：
- Operation stats 中 ADD/UPDATE/DELETE 占比 > 30%
- Consolidation trigger 是否触发（count > 0）
- R_avg 是否有上升趋势
- mem_size 在 200-400 之间波动（不会失控，因为 max_entries=500 自动淘汰）

**⚠ 早停检查点（前 100 episodes）**：
- 如果 NOOP > 90% → 仍有未发现的 bug，需排查
- 如果 mem_size 爆炸到 > 450 → 需要重新引入更温和的 ADD 控制
- 如果 R_avg 持续下降 → advantage 计算可能有问题

### 4.3 Phase 4 评测

```bash
# 每组约 3-5h, ~20-24 GB VRAM

# G-MSRA 主模型 (事件注入模式 — 与之前对比)
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs_v1/phase3_v6/best \
    --lora_checkpoint outputs_v1/phase1/best \
    --no_qlora --benchmark longmemeval \
    --output_dir results/gmsra_v6 \
    2>&1 | tee logs_v1/eval_gmsra_v6.log

# G-MSRA (checkpoint-only 模式 — 验证 memory 质量)
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs_v1/phase3_v6/best \
    --lora_checkpoint outputs_v1/phase1/best \
    --no_qlora --benchmark longmemeval \
    --checkpoint_only \
    --output_dir results/gmsra_v6_ckpt_only \
    2>&1 | tee logs_v1/eval_gmsra_v6_ckpt_only.log

# Events Only baseline (空 checkpoint 对比)
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs/empty_checkpoint \
    --lora_checkpoint outputs_v1/phase1/best \
    --no_qlora --benchmark longmemeval \
    --output_dir results/events_only_v6 \
    2>&1 | tee logs_v1/eval_events_only_v6.log
```

---

## 五、预期效果

| 指标 | 旧版 (v5) | 预期 v6 |
|------|:---------:|:-------:|
| NOOP 占比 | >94% | <60% |
| UPDATE/DELETE 次数 | 0 | >100 |
| Consolidation 触发 | 0 | >2 |
| G-MSRA vs Events Only (F1 差) | 0% | >5% |
| checkpoint_only 模式有效 | ❌ | ✅ |

---

## 六、变更的文件清单

```
scripts/train_phase3_full.py      — F1, F2, F3 (REINFORCE, MAX_ADD, NOOP penalty)
scripts/train_phase2_transition.py — F1 (REINFORCE fix)
scripts/train_phase1_rl.py         — F4 (GRPO index mapping)
gmsra/reward/grounded_reward.py    — F6 (Judge NOOP scoring)
scripts/eval_locomo.py             — F5 (checkpoint-only mode)
```

