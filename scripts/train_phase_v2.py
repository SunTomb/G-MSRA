"""
G-MSRA v2 Training: RL Memory Curation with Delayed QA F1 Reward.

Key differences from Phase 3 v11:
  1. NO LoRA distillation / consolidation — model weights stay frozen after Phase 1
  2. Reward signal: delayed QA F1 (NOT self-reward) — directly tied to downstream task
  3. NOOP penalty: prevents the NOOP-fixation seen in v11 (96% NOOP rate)
  4. Compactness bonus: encourages DELETE/UPDATE to keep memory store lean
  5. LLM Compaction (optional): periodically merge similar memories via LLM summarization
  6. Multi-source training data: LoCoMo + EvoMemory for knowledge update scenarios

Architecture:
  - Phase 1 LoRA stays FROZEN for QA capability
  - Only the RL policy (same LoRA) is trained for CRUD decisions
  - Memory store is external (FAISS + embeddings), never parametric

Usage:
    PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/train_phase_v2.py \\
        --lora_checkpoint outputs/phase1/best \\
        --output_dir outputs/v2 \\
        --max_episodes 3000 --max_events 10 --num_epochs 3 \\
        --epsilon_start 0.20 --epsilon_end 0.05 \\
        --noop_penalty 0.15 --compactness_weight 0.1 \\
        --enable_compaction \\
        --no_qlora --no_wandb \\
        2>&1 | tee logs/v2_train.log
"""

import argparse
import os
import json
import time
import random
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from loguru import logger

from gmsra.config import GMSRAConfig, CompactionConfig
from gmsra.memory.store import MemoryStore
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1


# ======================================================================
# v2 Reward Design
# ======================================================================

# Operation-level shaping bonuses (v2: increased UPDATE/DELETE, harsher NOOP)
OPERATION_BONUS = {
    "ADD":    +0.05,
    "UPDATE": +0.40,   # ↑ from 0.35: heavily reward UPDATE (knowledge evolution)
    "DELETE": +0.30,   # ↑ from 0.25: reward memory cleanup
    "NOOP":   -0.15,   # ↓ from -0.10: stronger NOOP penalty (anti-fixation)
}

# Exploration weights (bias toward UPDATE for knowledge update learning)
EXPLORE_WEIGHTS = {
    "ADD": 0.30,
    "UPDATE": 0.35,    # ↑ from 0.30: more UPDATE exploration
    "DELETE": 0.15,
    "NOOP": 0.20,
}

# Exploration gradient weight
EXPLORE_GRAD_WEIGHT = 0.3


def get_op_type(operation_str: str) -> str:
    upper = operation_str.strip().upper()
    for op in ["ADD", "UPDATE", "DELETE"]:
        if upper.startswith(op):
            return op
    return "NOOP"


def compute_v2_reward(
    r_qa: float,
    op_type: str,
    store_size: int,
    max_entries: int,
    noop_penalty: float,
    compactness_weight: float,
) -> float:
    """Compute the v2 composite reward.

    Components:
      1. Delayed QA F1 (primary signal, directly tied to downstream task)
      2. Operation bonus/penalty (shapes behavior toward CRUD usage)
      3. Compactness bonus (rewards keeping memory lean)

    Args:
        r_qa: QA F1 score from delayed evaluation
        op_type: CRUD operation type
        store_size: Current memory store size
        max_entries: Maximum memory entries
        noop_penalty: Additional NOOP penalty (v2 anti-fixation)
        compactness_weight: Weight for compactness bonus

    Returns:
        Total reward scalar
    """
    # 1. Delayed QA F1 (main signal)
    r_task = r_qa

    # 2. Operation bonus
    r_op = OPERATION_BONUS[op_type]

    # 3. Extra NOOP penalty (v2: configurable, stronger than v11)
    if op_type == "NOOP":
        r_op -= noop_penalty

    # 4. Compactness bonus: reward smaller stores
    if max_entries > 0:
        utilization = store_size / max_entries
        r_compact = compactness_weight * max(0, 1 - utilization)
    else:
        r_compact = 0.0

    return r_task + r_op + r_compact


# ======================================================================
# Data Loading
# ======================================================================

def load_training_data(data_dir: str, max_episodes: int, include_evomemory: bool = True) -> list[dict]:
    """Load training data from LoCoMo + EvoMemory.

    v2 combines:
      - LoCoMo: general dialogue memory (preference, facts, temporal)
      - EvoMemory: knowledge evolution tracking (contradictory updates)

    Returns:
        List of episodes, each with 'events', 'question', 'answer', 'category'
    """
    episodes = []

    # LoCoMo train data
    locomo_path = os.path.join(data_dir, "locomo_train.json")
    if os.path.exists(locomo_path):
        with open(locomo_path, "r", encoding="utf-8") as f:
            locomo_data = json.load(f)
        episodes.extend(locomo_data)
        logger.info(f"Loaded {len(locomo_data)} LoCoMo training episodes")
    else:
        logger.warning(f"LoCoMo train data not found at {locomo_path}")

    # EvoMemory data (knowledge evolution — the v2 differentiator)
    if include_evomemory:
        evo_path = os.path.join(data_dir, "evomemory_test.json")
        if os.path.exists(evo_path):
            with open(evo_path, "r", encoding="utf-8") as f:
                evo_data = json.load(f)
            episodes.extend(evo_data)
            logger.info(f"Loaded {len(evo_data)} EvoMemory training episodes")
        else:
            logger.warning(f"EvoMemory data not found at {evo_path}")

    if not episodes:
        logger.error("No training data found!")
        raise FileNotFoundError("No training data in data_dir")

    # Shuffle and limit
    random.shuffle(episodes)
    episodes = episodes[:max_episodes]
    logger.info(f"Total training episodes: {len(episodes)}")
    return episodes


# ======================================================================
# Main Training Loop
# ======================================================================

def main(args):
    set_seed(42)
    logger.info("=" * 60)
    logger.info("G-MSRA v2: RL Memory Curation Training")
    logger.info("=" * 60)
    logger.info(f"  LoRA (frozen QA): {args.lora_checkpoint}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  NOOP penalty: {args.noop_penalty}")
    logger.info(f"  Compactness weight: {args.compactness_weight}")
    logger.info(f"  Compaction: {'enabled' if args.enable_compaction else 'disabled'}")
    logger.info(f"  EvoMemory training data: {'included' if not args.no_evomemory else 'excluded'}")

    config = GMSRAConfig()

    # --- Load model ---
    use_qlora = not args.no_qlora
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_qlora=use_qlora
    )
    logger.info(f"Model precision: {'bf16 (full)' if args.no_qlora else 'QLoRA 4-bit'}")

    # --- Load Phase 1 LoRA ---
    lora_path = args.lora_checkpoint
    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info(f"Loaded Phase 1 LoRA adapter from: {lora_path}")

        # Enable LoRA gradient for RL training
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA training enabled: {trainable_count:,} / {total_count:,} params "
            f"({100*trainable_count/total_count:.2f}%)"
        )
    else:
        logger.warning(f"No LoRA found at '{lora_path}', training from base model")

    # --- Initialize v2 agent (with compaction, no distillation) ---
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config, use_v2=True)
    agent.initialize(model, tokenizer, env_type="dialogue")

    # --- Initialize compactor (if enabled) ---
    compactor = None
    if args.enable_compaction:
        from gmsra.consolidation.compaction import MemoryCompactor
        compact_config = CompactionConfig(
            similarity_threshold=args.compact_threshold,
            trigger_memory_threshold=args.compact_trigger_size,
        )
        compactor = MemoryCompactor(config=compact_config)
        logger.info(
            f"Compactor initialized: threshold={compact_config.similarity_threshold}, "
            f"trigger_size={compact_config.trigger_memory_threshold}"
        )

    # --- Setup optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = config.rl.learning_rate * 0.3  # Reduced LR for stability
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    logger.info(f"Optimizer: AdamW, lr={lr:.2e}, {len(trainable_params)} param groups")

    # --- Load training data ---
    single_epoch = load_training_data(
        args.data_dir, args.max_episodes,
        include_evomemory=not args.no_evomemory
    )
    num_epochs = args.num_epochs
    task_stream = single_epoch * num_epochs
    total_episodes = len(task_stream)
    logger.info(f"{len(single_epoch)} episodes x {num_epochs} epochs = {total_episodes} total")

    max_events = args.max_events
    grad_accum = config.rl.gradient_accumulation_steps

    # --- Metrics ---
    metrics_log = []
    reward_window = deque(maxlen=50)
    f1_window = deque(maxlen=50)
    op_counts_global = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    op_counts_window = deque(maxlen=500)

    # Per-action-type baselines (exponential moving average)
    action_baselines = {"ADD": 0.0, "UPDATE": 0.0, "DELETE": 0.0, "NOOP": 0.0}

    explore_count = 0
    compaction_count = 0
    LOG_INTERVAL = args.log_interval
    COMPACT_INTERVAL = args.compact_interval

    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    logger.info(f"Starting training: epsilon {args.epsilon_start} -> {args.epsilon_end}")
    logger.info(f"Max events per episode: {max_events}")

    for ep_idx, task in enumerate(task_stream):
        task_events = task.get("events", [task.get("instruction", "")])
        task_question = task.get("question", "")
        task_answer = task.get("answer", "")

        # Fresh memory store per episode (v2: no checkpoint carryover)
        agent.memory_store = MemoryStore(config.memory)
        agent.memory_manager.store = agent.memory_store

        # Epsilon-greedy schedule
        progress = ep_idx / max(total_episodes - 1, 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        # === Phase A: Process events with RL policy ===
        episode_actions = []  # (log_prob, op_type, was_explore)
        selected_events = task_events[:max_events]

        for event in selected_events:
            try:
                op_str, prompt, was_explore = agent.memory_manager.decide_with_exploration(
                    event, task_question, epsilon=epsilon
                )

                agent.memory_manager.execute_operation(op_str, event, env_reward=0.5)
                agent.step_count += 1

                op_type = get_op_type(op_str)
                op_counts_global[op_type] += 1
                op_counts_window.append(op_type)
                if was_explore:
                    explore_count += 1

                # Compute log_prob for REINFORCE
                log_prob = agent.memory_manager.compute_action_log_prob(prompt, op_str)
                episode_actions.append((log_prob, op_type, was_explore))

            except Exception as e:
                logger.debug(f"Event processing error: {e}")

        # === Phase B: Delayed QA F1 reward ===
        r_qa = 0.0
        predicted = ""
        if task_question and task_answer:
            try:
                predicted = agent.answer_question(task_question)
                r_qa = compute_f1(predicted, task_answer)
            except Exception as e:
                logger.debug(f"QA evaluation error: {e}")
        f1_window.append(r_qa)

        # === Phase C: Per-action REINFORCE with v2 reward ===
        policy_loss = torch.tensor(0.0, device=model.device, requires_grad=False)
        effective_count = 0

        for log_prob, op_type, was_explore in episode_actions:
            if log_prob is None:
                continue

            # v2 composite reward
            action_reward = compute_v2_reward(
                r_qa=r_qa,
                op_type=op_type,
                store_size=agent.memory_store.size(),
                max_entries=config.memory.max_entries,
                noop_penalty=args.noop_penalty,
                compactness_weight=args.compactness_weight,
            )

            # Update per-action baseline
            action_baselines[op_type] = (
                0.95 * action_baselines[op_type] + 0.05 * action_reward
            )
            advantage = action_reward - action_baselines[op_type]

            # Exploration actions get reduced gradient weight
            grad_weight = EXPLORE_GRAD_WEIGHT if was_explore else 1.0

            policy_loss = policy_loss + (-advantage * grad_weight * log_prob)
            effective_count += grad_weight

        # Apply gradients
        if effective_count > 0:
            policy_loss = policy_loss / effective_count / grad_accum
            policy_loss.backward()

            if (ep_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.rl.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        # === Phase D: Optional compaction ===
        if compactor and (ep_idx + 1) % COMPACT_INTERVAL == 0:
            if agent.memory_store.size() >= compactor.config.trigger_memory_threshold:
                try:
                    compact_stats = compactor.run(
                        agent.memory_store, model, tokenizer
                    )
                    if not compact_stats.get("skipped", True):
                        compaction_count += 1
                        logger.info(
                            f"Compaction #{compaction_count} at ep {ep_idx+1}: "
                            f"{compact_stats['initial_size']} -> {compact_stats['final_size']}"
                        )
                except Exception as e:
                    logger.debug(f"Compaction error: {e}")

        # === Tracking ===
        r_total = r_qa + sum(
            OPERATION_BONUS[op_type] for _, op_type, _ in episode_actions
        ) / max(len(episode_actions), 1)
        reward_window.append(r_total)

        # Log periodically
        if (ep_idx + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            speed = (ep_idx + 1) / elapsed
            remaining = (total_episodes - ep_idx - 1) / speed if speed > 0 else 0
            eta_hours = remaining / 3600

            avg_reward = sum(reward_window) / len(reward_window)
            avg_f1 = sum(f1_window) / len(f1_window)

            # Operation breakdown
            total_ops = sum(op_counts_global.values()) or 1
            op_pct = {k: f"{100*v/total_ops:.1f}%" for k, v in op_counts_global.items()}
            noop_pct = 100 * op_counts_global["NOOP"] / total_ops

            # Window breakdown
            if op_counts_window:
                wc = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
                for op in op_counts_window:
                    wc[op] += 1
                tw = sum(wc.values()) or 1
                win_pct = {k: f"{100*v/tw:.1f}%" for k, v in wc.items()}
                noop_win = 100 * wc["NOOP"] / tw
            else:
                win_pct = op_pct
                noop_win = noop_pct

            log_entry = {
                "episode": ep_idx + 1,
                "avg_reward": round(avg_reward, 4),
                "avg_f1": round(avg_f1, 4),
                "memory_size": agent.memory_store.size(),
                "op_counts_global": op_counts_global.copy(),
                "op_pct_window": dict(win_pct),
                "baselines": {k: round(v, 3) for k, v in action_baselines.items()},
                "compaction_count": compaction_count,
                "epsilon": round(epsilon, 3),
            }
            metrics_log.append(log_entry)

            logger.info(
                f"Ep {ep_idx+1}/{total_episodes} | "
                f"R={avg_reward:.3f} | F1={avg_f1:.3f} | "
                f"eps={epsilon:.3f} | "
                f"mem={agent.memory_store.size()} | "
                f"NOOP={noop_win:.0f}%(win) {noop_pct:.0f}%(all) | "
                f"ops_win={win_pct} | "
                f"explore={explore_count} | "
                f"compact={compaction_count} | "
                f"baselines={{{','.join(f'{k}:{v:.2f}' for k,v in action_baselines.items())}}} | "
                f"{speed:.2f} ep/s | ETA: {eta_hours:.1f}h"
            )

        # Checkpoint
        if (ep_idx + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{ep_idx+1}")
            agent.save_checkpoint(ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # === Final save ===
    elapsed_total = time.time() - start_time
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "best"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    final_stats = {
        "total_episodes": total_episodes,
        "total_time_hours": round(elapsed_total / 3600, 2),
        "op_counts_global": op_counts_global,
        "explore_count": explore_count,
        "compaction_count": compaction_count,
        "final_baselines": {k: round(v, 3) for k, v in action_baselines.items()},
        "noop_penalty": args.noop_penalty,
        "compactness_weight": args.compactness_weight,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(final_stats, f, indent=2)

    noop_pct_final = 100 * op_counts_global["NOOP"] / max(sum(op_counts_global.values()), 1)
    logger.info("=" * 60)
    logger.info(f"v2 Training Complete!")
    logger.info(f"  Time: {elapsed_total/3600:.1f}h")
    logger.info(f"  Episodes: {total_episodes}")
    logger.info(f"  Final NOOP%: {noop_pct_final:.1f}% (target: <50%)")
    logger.info(f"  Ops: {op_counts_global}")
    logger.info(f"  Compactions: {compaction_count}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G-MSRA v2: RL Memory Curation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train_phase_v2.py --no_qlora --no_wandb

  # Full training with compaction
  python scripts/train_phase_v2.py \\
      --lora_checkpoint outputs/phase1/best \\
      --max_episodes 3000 --num_epochs 3 \\
      --enable_compaction \\
      --no_qlora --no_wandb

  # High NOOP penalty experiment
  python scripts/train_phase_v2.py \\
      --noop_penalty 0.25 --compactness_weight 0.15 \\
      --no_qlora --no_wandb
""",
    )

    # --- Model & checkpoints ---
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_checkpoint", default="outputs/phase1/best",
                        help="Phase 1 LoRA adapter (frozen QA capability)")
    parser.add_argument("--output_dir", default="outputs/v2")
    parser.add_argument("--data_dir", default="data")

    # --- Training ---
    parser.add_argument("--max_episodes", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=10,
                        help="Max events per episode (v2 default: 10, up from v11's 5)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--save_interval", type=int, default=500)

    # --- Exploration ---
    parser.add_argument("--epsilon_start", type=float, default=0.20,
                        help="Initial exploration rate (v2: higher than v11's 0.15)")
    parser.add_argument("--epsilon_end", type=float, default=0.05)

    # --- v2 Reward Design ---
    parser.add_argument("--noop_penalty", type=float, default=0.15,
                        help="Additional NOOP penalty (v2 anti-fixation, default: 0.15)")
    parser.add_argument("--compactness_weight", type=float, default=0.10,
                        help="Weight for memory compactness bonus (default: 0.10)")

    # --- Compaction ---
    parser.add_argument("--enable_compaction", action="store_true",
                        help="Enable LLM-based memory compaction during training")
    parser.add_argument("--compact_interval", type=int, default=100,
                        help="Run compaction every N episodes")
    parser.add_argument("--compact_threshold", type=float, default=0.80,
                        help="Cosine similarity threshold for clustering")
    parser.add_argument("--compact_trigger_size", type=int, default=50,
                        help="Only compact when store has > N entries")

    # --- Data ---
    parser.add_argument("--no_evomemory", action="store_true",
                        help="Exclude EvoMemory data from training")

    # --- Hardware ---
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    main(args)
