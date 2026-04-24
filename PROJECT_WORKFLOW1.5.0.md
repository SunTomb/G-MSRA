# G-MSRA 项目工作流程 v5.0 — Phase 1 完成 · Phase 2-3 执行指南

> **Closing the Loop: Environment-Grounded Self-Reward for Autonomous Memory Management in Lifelong LLM Agents**
>
> 更新日期：2026 年 3 月 25 日 · 基于 [v4.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.4.0.md) 更新

---

## 〇、与前序版本的关系

| 版本 | 侧重 | 状态 |
|------|------|------|
| [v1.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW.md) | 研究问题 + 排期规划 + 代码骨架 | ✅ 背景资料 |
| [v2.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.2.0.md) | 训练脚本补全（Phase 0-3 + 消融 + 数据准备） | ✅ 背景资料 |
| [v3.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.3.0.md) | Baseline 复现（5 个 Agent + 评测框架） | ✅ 背景资料 |
| [v4.0](file:///d:/USTC/2026Winter/G-MSRA/PROJECT_WORKFLOW1.4.0.md) | 合成数据验证完成 → 全量数据实验执行指南 | ✅ 背景资料 |
| **v5.0（本文档）** | **Phase 1 RL 训练完成 → Phase 2-3 + 评测执行指南** | ✅ 当前版本 |

---

## 一、里程碑总览

| 里程碑 | 状态 | 完成日期 |
|--------|:----:|:--------:|
| M0: 核心库完成 | ✅ | 2026-03-15 |
| M0.5: 训练脚本补全 + 冒烟测试 27/27 | ✅ | 2026-03-16 |
| M0.8: Baseline 代码完成 | ✅ | 2026-03-16 |
| M0.9: 合成数据 Baseline 全通 | ✅ | 2026-03-17 |
| **M1: Phase 0 SFT 训练完成** | **✅** | **2026-03-20** |
| **M2: Phase 1 RL 训练完成** | **✅** | **2026-03-22** |
| M3: Phase 2 课程退火 | ☐ | — |
| M4: Phase 3 全闭环训练 | ☐ | — |
| M5: 全量 Baseline 评测 | ☐ | — |
| M6: 消融实验 (Table 3) | ☐ | — |
| M7: 主表 (Table 1+2) 填入 | ☐ | — |
| M8: 分析图完成 | ☐ | — |
| M9: 论文初稿 → 投递 | ☐ | — |

---

## 二、已完成：Phase 1 RL 训练结果

### 2.1 训练配置

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen/Qwen2.5-7B-Instruct |
| 训练方法 | GRPOTrainer (TRL 0.15.2) |
| 精度 | bf16 (full) + LoRA (非 QLoRA) |
| GPU | 6× NVIDIA A40 48GB (Tang3 节点) |
| `CUDA_VISIBLE_DEVICES` | 1,2,3,4,6,7 |
| `per_device_batch_size` | 6 |
| `num_generations` | 6 |
| `max_completion_length` | 192 |
| `gradient_accumulation_steps` | 1 |
| 数据集 | 1586 LoCoMo episodes → 7930 GRPO prompts |
| W&B | 关闭 (`--no_wandb`) |

### 2.2 训练结果

| 指标 | 值 |
|------|-----|
| 总步数 | 1322/1322 (100%) |
| 总耗时 | 17 小时 27 分钟 |
| 平均速度 | 47.52 s/step |
| 最终 Loss | 3.494 |
| 最终 Reward | 0.232 |
| 最终 KL | 100.36 |
| 最终 Completion Length | 160.2 tokens |

### 2.3 训练过程观察

| 观察 | 详情 | 建议 |
|------|------|------|
| Reward 平坦 | 全程 ~0.23，无明显提升 | 调整 reward shaping / 增加 reward 分辨率 |
| KL 偏高 | 从 75 升至 100 | 考虑增大 KL 惩罚系数 |
| 梯度 Norm 飙升 | 最后阶段达 9335 | 降低 lr / 加强 gradient clipping |
| 完成长度缩短 | 184 → 160 tokens | 正常：模型学习更简洁表达 |

### 2.4 输出文件

```
outputs/phase1/
├── best/                          # 最终模型
│   ├── adapter_model.safetensors  # LoRA 权重 (20 MB)
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── ...
├── checkpoint-1200/               # 中间 checkpoint
├── checkpoint-1300/
└── checkpoint-1322/               # 最终 checkpoint (含 optimizer state)
```

---

## 三、集群执行环境详情

### 3.1 Tang3 节点配置

| 项目 | 配置 |
|------|------|
| GPU | 8× NVIDIA A40 48GB |
| 可用 GPU | 1,2,3,4,6,7（GPU 0 和 5 被占用） |
| 用户 | `wujcan@Tang3` |
| 项目路径 | `/NAS/yesh/G-MSRA` |
| Conda 环境 | `gmsra` (Python 3.10) |
| HF 缓存 | `/NAS/yesh/hf_cache/hub` |
| 模型缓存 | 离线模式 (`HF_HUB_OFFLINE=1`) |

### 3.2 关键配置文件

| 文件 | 用途 |
|------|------|
| `cluster/accelerate_a40.yaml` | 6-GPU Accelerate 配置 |
| `cluster/accelerate_a40_2gpu.yaml` | 2-GPU 配置 |
| `cluster/ds_zero2_a40.json` | DeepSpeed ZeRO-2 配置 |
| `cluster/run_tang.sh` | Tang 节点启动脚本 |
| `cluster/run_song.sh` | Song 节点启动脚本 |

### 3.3 环境 Workaround 记录

| 问题 | 解决方案 |
|------|---------|
| `trl` 导入报错（optional deps） | 手动 patch `trl` 源码中 optional import |
| `transformers` 版本冲突 | 固定 `transformers==4.48.3` |
| W&B 超时 (90s) | 添加 `--no_wandb` 强制禁用 |
| `nohup` + `accelerate launch` 收 SIGHUP | **使用 `tmux` 替代 `nohup`** |
| `num_generations` 不整除 `global_batch_size` | 设 `--num_generations 6`（= per_device_bs × 1） |

---

## 四、下一步执行指南

### 4.1 总体路线图

```
[Phase 0 SFT ✅] → [Phase 1 RL ✅] → [Phase 2 退火] → [Phase 3 全闭环] → [评测] → [消融] → [论文]
                                       ~~~~~~~~~~~     ~~~~~~~~~~~~~~     ~~~~~~~   ~~~~~~
                                        下一步           最耗时            可并行    可并行
```

### 4.2 Phase 2: 课程退火

Phase 2 在 Phase 1 基础上进行 reward 课程退火，逐步从外部 reward 过渡到自奖励。

```bash
# 在 tmux 中执行：
tmux new -s phase2

cd /NAS/yesh/G-MSRA
eval "$(conda shell.bash hook)"
conda activate gmsra
export HF_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 accelerate launch \
    --config_file cluster/accelerate_a40.yaml \
    --num_processes 6 \
    scripts/train_phase2_transition.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --checkpoint outputs/phase1/best \
    --output_dir outputs/phase2 \
    --anneal_steps 3000 \
    --no_qlora --gpu_preset a40 \
    --per_device_batch_size 6 \
    --num_generations 6 \
    --no_wandb \
    2>&1 | tee logs/phase2_tang.log

# 脱离 tmux: Ctrl+B → D
# 预期输出: outputs/phase2/best/ + calibration.json
# 预估耗时: ~1-2 天
```

### 4.3 Phase 3: 全闭环

Phase 3 在 Phase 2 基础上进行完全自奖励 + 巩固操作训练。

```bash
tmux new -s phase3

# ... (同上 conda/env 设置)

CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 accelerate launch \
    --config_file cluster/accelerate_a40.yaml \
    --num_processes 6 \
    scripts/train_phase3_full.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --checkpoint outputs/phase2/best \
    --output_dir outputs/phase3 \
    --num_episodes 10000 \
    --no_qlora --gpu_preset a40 \
    --per_device_batch_size 6 \
    --num_generations 6 \
    --no_wandb \
    2>&1 | tee logs/phase3_tang.log

# 预期输出: outputs/phase3/best/ + metrics.json + diagnostics.json
# 预估耗时: ~3-5 天
```

### 4.4 评测（可在训练完成后执行）

```bash
# G-MSRA 评测
CUDA_VISIBLE_DEVICES=7 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3/best --benchmark locomo

# Baseline 评测（可与训练并行，用别的 GPU）
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py \
    --agent reflexion --benchmark locomo
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py \
    --agent self_consolidation --benchmark locomo
CUDA_VISIBLE_DEVICES=7 python baselines/eval_baselines.py \
    --agent evolver --benchmark locomo

# RL Baseline 训练 + 评测
CUDA_VISIBLE_DEVICES=7 python baselines/train_and_eval_rl_baselines.py \
    --train_epochs 10
```

### 4.5 消融实验

在 Phase 1/3 checkpoint 可用后执行：

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/run_ablations.py \
    --base_checkpoint outputs/phase1/best \
    --num_episodes 1000
```

---

## 五、tmux 使用速查

> ⚠️ **所有长时间训练必须在 tmux 中运行**。`nohup` 无法保护 `accelerate launch` 的子进程。

| 操作 | 命令 |
|------|------|
| 创建新会话 | `tmux new -s <名称>` |
| 脱离会话（训练继续） | `Ctrl+B` → `D` |
| 重新接入会话 | `tmux attach -t <名称>` |
| 列出所有会话 | `tmux ls` |
| 杀死会话 | `tmux kill-session -t <名称>` |
| 滚动查看历史 | `Ctrl+B` → `[`，然后用方向键/PgUp/PgDn，按 `Q` 退出 |

---

## 六、GPU 并行策略建议

| GPU | 任务 | 预估时间 |
|:---:|------|:--------:|
| GPU 1-4,6,7 | G-MSRA Phase 2/3 训练 (6×A40) | Phase 2: 1-2天 / Phase 3: 3-5天 |
| GPU 7 (空闲时) | Baseline 评测（逐个跑） | 各 ~10 min |
| GPU 7 (过夜) | RL Baseline 训练 | ~1 天 |

---

## 七、中断恢复流程

如果训练因任何原因中断（但有 checkpoint 存在）：

```bash
# 1. 查看现有 checkpoint
ls outputs/phase1/  # 例如看到 checkpoint-1200

# 2. 从 checkpoint 恢复
CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 accelerate launch \
    --config_file cluster/accelerate_a40.yaml \
    --num_processes 6 \
    scripts/train_phase1_rl.py \
    ... (原参数) \
    --resume_from_checkpoint outputs/phase1/checkpoint-1200
```

当前 `save_steps=100`，约每 1.5 小时自动保存一次 checkpoint（最多保留 3 个）。

---

## 八、结果收集 → 论文填充

所有结果汇总到 `results/` 目录后，填入 `paper/main.tex`：

| 论文表格 | 数据来源 | 状态 |
|----------|---------|:----:|
| Table 1 (主表) | `results/baselines/` + G-MSRA eval | ☐ |
| Table 2 (ALFWorld) | ALFWorld 评测结果 | ☐ |
| Table 3 (消融) | `results/ablations/` | ☐ |
| Fig 3 (Reward Drift) | Phase 2 `calibration.json` | ☐ |
| Fig 4 (Trigger 3D) | Phase 3 `diagnostics.json` | ☐ |

---

## 九、代码资产清单增量（v5.0 新增/修改）

### 新增文件

| 文件 | 用途 |
|------|------|
| `cluster/accelerate_a40.yaml` | 6×A40 Accelerate 多 GPU 配置 |
| `cluster/accelerate_a40_2gpu.yaml` | 2×A40 配置 |
| `cluster/ds_zero2_a40.json` | DeepSpeed ZeRO-2 内存优化配置 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `scripts/train_phase1_rl.py` | 新增 `--gpu_preset a40`、`--no_wandb`、`--resume_from_checkpoint`、`--deepspeed` 参数；`save_steps` 100→安全保存 |
| `cluster/run_tang.sh` | Tang3 节点 6-GPU 启动脚本 |

### 输出文件

| 路径 | 内容 |
|------|------|
| `outputs/phase0/best/` | Phase 0 SFT 模型 |
| `outputs/phase1/best/` | Phase 1 RL LoRA adapter (20MB) |
| `outputs/phase1/checkpoint-{1200,1300,1322}/` | 训练中间 checkpoints |
| `logs/phase1_tang.log` | Phase 1 完整训练日志 (4.3MB, 34403 行) |

---

## 十、风险与应对（更新）

| 风险 | 概率 | 应对 |
|------|:----:|------|
| ~~代码骨架不完整~~ | — | ✅ 已解决 |
| ~~Baseline 缺失~~ | — | ✅ 已解决 |
| ~~管线跑不通~~ | — | ✅ 已解决 |
| ~~Phase 1 RL 训练发散~~ | — | ✅ 已完成（reward 平坦但完成） |
| ~~nohup 无法保护 accelerate~~ | — | ✅ 已解决：使用 tmux |
| Phase 2/3 脚本适配多 GPU | 中 | 可能需要类似 Phase 1 的参数调整 |
| Reward 信号过粗导致训练效果差 | 中 | 优化 reward shaping；增加 Judge 指标 |
| Phase 3 训练时间过长 | 中 | 减少 episodes 或用 DeepSpeed ZeRO-2 |
| Baseline 全量评测 F1 仍低 | 中 | 检查 prompt 格式、答案后处理 |

---

> **当前状态总结**：
> - ✅ Phase 0 (SFT) 和 Phase 1 (RL) 的训练已全部完成
> - ✅ 集群多 GPU 环境已调通（tmux + accelerate + 6×A40）
> - ☐ 下一步：Phase 2 课程退火 → Phase 3 全闭环 → 评测 → 消融 → 论文
>
> 技术栈和运行环境都已稳定，后续训练应该可以顺利执行。
