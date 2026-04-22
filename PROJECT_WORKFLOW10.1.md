# PROJECT_WORKFLOW 10.1 — 评测诊断 → LoRA 蒸馏修复

> 创建时间：2026-04-22 09:44  
> 前序：WORKFLOW 10.0 (Phase 3 v11 完成 + 评测计划)  
> 状态：**评测完成 ❌ → 根因定位 ✅ → 方案 B 验证中**

---

## 一、评测结果总表（13 组）

### 1.1 主评测（5 组 × 2 benchmark）

| 组 | 描述 | LoCoMo F1 | LongMemEval F1 | mem |
|:--:|------|:---------:|:--------------:|:---:|
| D | No Memory (纯 LLM) | 0.027 | 0.049 | 0 |
| B | Ckpt-only (v11 best) | 0.019 | 0.036 | 140 |
| A | **G-MSRA v11** | **0.048** | **0.146** | 140 |
| E | Phase 2 v7 | 0.096 | — | 500 |
| C | **Events Only** | **0.097** | **0.262** | 0 |

### 1.2 诊断评测（4 组，LoCoMo only）

| 组 | 描述 | F1 | mem | LoRA |
|:--:|------|:--:|:---:|------|
| diag-1 | ckpt500 + v11 LoRA | 0.071 | 350 | v11 蒸馏 |
| diag-2 | ckpt2000 + v11 LoRA | 0.043 | 187 | v11 蒸馏 |
| diag-3 | **空 mem + v11 LoRA** | **0.048** | **0** | v11 蒸馏 |
| diag-4 | ckpt500 memory only | 0.019 | 350 | v11 蒸馏 |

### 1.3 Checkpoint Ablation（4 组，LoCoMo only）

| Checkpoint | Consol 次数 | F1 | mem |
|:----------:|:-----------:|:--:|:---:|
| ckpt500 | 1 | 0.071 | 350 |
| ckpt1000 | 2 | 0.052 | 271 |
| ckpt1500 | 3 | 0.040 | 227 |
| ckpt2000 | 4 | 0.042 | 187 |

---

## 二、根因诊断

### 2.1 铁证链

```
证据 1: lora_only (空 mem + v11 LoRA) = 0.048
         Events Only (空 mem + Phase1 LoRA) = 0.097
         → v11 LoRA 使 F1 降了 50%

证据 2: G-MSRA best (140 mem + v11 LoRA) = 0.048
         lora_only   (0 mem   + v11 LoRA) = 0.048
         → 140 条 checkpoint memory 贡献 = 0

证据 3: Phase2 v7 (500 mem + Phase1 LoRA) = 0.096
         Events Only (0 mem   + Phase1 LoRA) = 0.097
         → 500 条 checkpoint memory 贡献 ≈ 0

证据 4: ckpt500 (1 consol) = 0.071
         ckpt1000 (2 consol) = 0.052
         ckpt1500 (3 consol) = 0.040
         → Consolidation 单调破坏 F1
```

### 2.2 根因

| # | 问题 | 影响 | 严重性 |
|:-:|------|:----:|:------:|
| 1 | **LoRA 蒸馏覆盖 Phase 1 通用能力** | F1 从 0.097 降到 0.048 | 🔴 致命 |
| 2 | Checkpoint memory 与评测数据无关 | 贡献 = 0 | 🟡 预期内 |
| 3 | 评测流程绕过 RL 策略 | RL 训练成果不体现 | 🟡 设计选择 |

### 2.3 为什么 LoRA 蒸馏有害？

```
Phase 1 LoRA: 通用指令跟随 + QA 推理能力
              ↓
Consolidation 蒸馏：用训练集的 semantic triples 训练 LoRA
              ↓
v11 LoRA: 过拟合训练数据（"Sam 养了狗"等特定事实）
          + 覆盖了通用推理能力
          = QA 退化
```

EWC 保护理论上应该防止灾难性遗忘，但 6 次蒸馏后保护不足。

---

## 三、修复方案

### 方案 B: 快速验证（已部署，10 分钟）

**用 Phase 1 LoRA 替换 v11 蒸馏 LoRA**，保留 ckpt500 的 memory：

```bash
PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_locomo.py \
    --checkpoint outputs/phase3_v11/checkpoint_500 \
    --lora_checkpoint outputs/phase1/best \
    --no_qlora --benchmark locomo \
    --output_dir results/diag_ckpt500_phase1lora \
    2>&1 | tee logs/diag_ckpt500_phase1lora.log
```

**预期结果**：
- F1 ≈ 0.096（与 Events Only 持平） → **确认 LoRA 蒸馏是唯一问题**
- F1 < 0.090 → checkpoint memory 本身也有干扰

**下载**：`results/diag_ckpt500_phase1lora/locomo_results.json`

---

### 方案 A: 修复 LoRA 蒸馏（如方案 B 确认根因）

#### A1: LoRA 权重混合（最简单）

不重训。将 Phase 1 LoRA 与 v11 蒸馏 LoRA 按权重混合：

```python
# 在 eval 脚本中加入 LoRA merge
from peft import PeftModel
import torch

model = PeftModel.from_pretrained(base_model, "outputs/phase1/best")
phase1_state = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}

model = PeftModel.from_pretrained(base_model, "outputs/phase3_v11/best/lora")
v11_state = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}

# 混合：保留 70% Phase 1 + 30% v11 蒸馏
alpha = 0.7
for k in phase1_state:
    if k in v11_state:
        merged = alpha * phase1_state[k] + (1 - alpha) * v11_state[k]
        model.state_dict()[k].copy_(merged)
```

评测矩阵：alpha ∈ {0.5, 0.6, 0.7, 0.8, 0.9}

#### A2: 增强 EWC 约束后重训 Consolidation

修改 `gmsra/consolidation/distiller.py`，将 EWC 的 lambda 从当前值提高 10-50 倍：

```python
# 当前: ewc_lambda = 100 (猜测值，需确认)
# 修改: ewc_lambda = 5000  # 大幅增强保护
```

然后从 Phase 2 v7 checkpoint 重新训练 Phase 3，只需关注：
- Consolidation LoRA loss 是否仍下降
- 评测 F1 是否不再下降

**时间成本**：~2 天

#### A3: 独立 LoRA Adapter（最安全）

蒸馏时不修改 Phase 1 的 LoRA，而是添加一个**独立的第二层 LoRA adapter**：

```python
# 冻结 Phase 1 LoRA
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = False

# 添加新的 LoRA-B adapter 用于蒸馏
from peft import LoraConfig, get_peft_model
consol_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, consol_config, adapter_name="consolidation")
```

**时间成本**：需要修改 distiller.py + 重训，~3 天

---

### 方案 C: 修改评测流程

让 RL 策略参与评测，替换 `store.add()` 为 `agent.step()`：

```python
# eval_locomo.py 修改
# 当前（绕过 RL）:
agent.memory_store.add(text, env_reward=0.5)

# 修改为（使用 RL 策略）:
op_str, _, _ = agent.memory_manager.decide_with_exploration(
    text, task_context, epsilon=0.0  # 纯策略，不探索
)
agent.memory_manager.execute_operation(op_str, text)
```

**风险**：NOOP=96% 意味着 RL 策略会跳过大量事件 → 可能更差

---

### 方案 D: 转变论文叙事

将论文从 "consolidation 提升 QA" 转为 **"RL 驱动的 memory CRUD 学习 + consolidation 的挑战与教训"**：

**叙事结构**：
1. RL 成功学到 CRUD 策略（UPDATE>ε 的训练证据）
2. Consolidation 实现了知识压缩（LoRA loss↓，memory↓）  
3. **关键发现**：蒸馏的参数化知识与通用推理能力存在冲突
4. 评测显示 per-example isolation 下 checkpoint memory 无效
5. 提出 LoRA merge / dual-adapter 等改进方向

**目标期刊/会议**：分析型论文可投 Findings of ACL/EMNLP

---

## 四、优先级排序

```
┌──────────────────────────────────────────────┐
│ 方案 B (10 min)                              │ ← 立即执行
│   验证 Phase1 LoRA + ckpt500                 │
│   如果 F1≈0.096 → 确认 LoRA 是唯一问题       │
├──────────────────────────────────────────────┤
│ 方案 A1 (2h)                                 │ ← 第二步
│   LoRA 权重混合，扫描 alpha                   │
│   如果 F1>0.097 → 论文有 positive result     │
├──────────────────────────────────────────────┤
│ 方案 C (4h)                                  │ ← 第三步
│   修改 eval 让 RL 策略参与                    │
│   如果 RL 过滤有效 → 论文更强                │
├──────────────────────────────────────────────┤
│ 方案 A2/A3 (2-3 天)                          │ ← 如果以上不足
│   增强 EWC / 独立 adapter 重训               │
├──────────────────────────────────────────────┤
│ 方案 D (备选)                                │ ← 最后手段
│   转变论文叙事                               │
└──────────────────────────────────────────────┘
```

---

## 五、当前状态 & 文件索引

### 5.1 评测结果文件

| 路径 | 说明 |
|------|------|
| `results/gmsra_v11/` | A. G-MSRA v11 完整系统 |
| `results/gmsra_v11_ckpt_only/` | B. checkpoint-only |
| `results/events_only_v11/` | C. Events Only baseline |
| `results/no_memory_v11/` | D. No Memory |
| `results/phase2_v7/` | E. Phase 2 v7 |
| `results/diag_ckpt500/` | 诊断：ckpt500 + v11 LoRA |
| `results/diag_ckpt2000/` | 诊断：ckpt2000 + v11 LoRA |
| `results/diag_lora_only/` | 诊断：空 mem + v11 LoRA |
| `results/diag_ckpt500_only/` | 诊断：ckpt500 memory only |
| `results/ablation_ckpt_500/` | Ablation：ckpt500 |
| `results/ablation_ckpt_1000/` | Ablation：ckpt1000 |
| `results/ablation_ckpt_1500/` | Ablation：ckpt1500 |
| `results/ablation_ckpt_2000/` | Ablation：ckpt2000 |
| `results/diag_ckpt500_phase1lora/` | **方案 B 验证**（待完成） |

### 5.2 关键代码文件

| 文件 | 修复方案涉及 |
|------|:------------:|
| `scripts/eval_locomo.py` | 方案 C (eval RL 策略) |
| `gmsra/consolidation/distiller.py` | 方案 A2/A3 (EWC/dual adapter) |
| `gmsra/agent.py` | 方案 A3 (adapter 管理) |

---

## 六、时间线

| 阶段 | 状态 | 时间 |
|:----:|------|:----:|
| Phase 3 v11 训练 | ✅ | 4/19-21 |
| 5 组主评测 | ✅ F1 < baseline | 4/22 02:00 |
| 4 组诊断评测 | ✅ LoRA 根因定位 | 4/22 04:00 |
| 4 组 checkpoint ablation | ✅ consol 单调↓ | 4/22 06:00 |
| 📌 方案 B 验证 | **执行中** | 4/22 09:45 |
| ⏳ 方案 A1 LoRA merge | 待 B 确认后 | ~4/22 下午 |
| ⏳ 方案 C eval 修改 | 待 A1 后 | ~4/22 晚 |
| ⏳ 论文实验章节 | 待最终数据 | ~4/23+ |
