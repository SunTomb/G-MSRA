"""
Phase 2 v7: RL Policy Warmup with Shaped Reward.

Trains the memory manager's CRUD policy using QA F1 + operation bonuses.
Gradually anneals α from 1.0→0.0 (external → self-reward) while maintaining
Kendall τ calibration.

Key v7 fixes:
  F7:  REINFORCE computes log π(a|s) per-event inline (no re-decide)
  F8:  Reward = R_env (QA F1) + operation_bonus (no Judge R_mem)
  F9:  ε-greedy exploration: ε = 0.3 → 0.05 linear decay
  F10: Prompt truncated to 512 tokens for gradient concentration
  F11: max_events increased to 8

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=2 python scripts/train_phase2_transition.py \
        --checkpoint outputs/phase1/best \
        --output_dir outputs/phase2_v7 \
        --no_qlora --no_wandb
"""

import argparse
import os
import json
import time
import random

import torch
import torch.nn.functional as F
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1, compute_kendall_tau


# --- Operation Bonus Shaping (F8) ---
OPERATION_BONUS = {
    "ADD":    +0.20,   # Reward storing new information
    "UPDATE": +0.30,   # Highest reward — updating is hardest to learn
    "DELETE": +0.20,   # Reward removing outdated info
    "NOOP":   -0.10,   # Penalize inaction
}


def get_operation_bonus(operation_str: str) -> float:
    """Extract operation type and return bonus."""
    upper = operation_str.strip().upper()
    if upper.startswith("ADD"):
        return OPERATION_BONUS["ADD"]
    elif upper.startswith("UPDATE"):
        return OPERATION_BONUS["UPDATE"]
    elif upper.startswith("DELETE"):
        return OPERATION_BONUS["DELETE"]
    else:
        return OPERATION_BONUS["NOOP"]


def get_op_type(operation_str: str) -> str:
    """Extract clean operation type from raw string."""
    upper = operation_str.strip().upper()
    for op in ["ADD", "UPDATE", "DELETE"]:
        if upper.startswith(op):
            return op
    return "NOOP"


def main(args):
    set_seed(42)
    logger.info(f"Phase 2 v7: RL Policy Warmup | checkpoint={args.checkpoint}")

    config = GMSRAConfig()
    config.reward.anneal_steps = args.anneal_steps
    config.reward.tau_threshold = args.tau_threshold
    if args.learning_rate is not None:
        config.rl.learning_rate = args.learning_rate

    # --- Load model from Phase 1 ---
    use_qlora = not args.no_qlora
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_qlora=use_qlora, use_accelerate=False
    )
    logger.info(f"Model precision: {'bf16 (full)' if args.no_qlora else 'QLoRA 4-bit'}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        logger.info(f"Loaded Phase 1 checkpoint: {args.checkpoint}")

        # Enable LoRA training
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Enabled LoRA training: {trainable_count:,} / {total_count:,} params "
            f"({100*trainable_count/total_count:.2f}%)"
        )

    # Initialize agent
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="dialogue")

    # Load checkpoint state
    if os.path.exists(os.path.join(args.checkpoint, "memory_store.json")):
        agent.load_checkpoint(args.checkpoint)

    # Load dataset
    from scripts.train_phase1_rl import load_locomo_data
    dataset = load_locomo_data(args.data_dir)

    # --- Setup optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = args.learning_rate or (config.rl.learning_rate * 0.5)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.reward.anneal_steps
    )
    grad_accum = args.gradient_accumulation_steps or config.rl.gradient_accumulation_steps

    # --- Annealing schedule ---
    alpha = config.reward.anneal_start_alpha  # 1.0
    alpha_step = (config.reward.anneal_start_alpha - config.reward.anneal_end_alpha) / config.reward.anneal_steps
    paused = False

    # --- Tracking ---
    ext_rewards = []
    self_rewards = []
    annealed_rewards = []
    reward_baseline = 0.0
    op_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    explore_count = 0
    start_time = time.time()

    # --- Multi-epoch dataset ---
    num_epochs = args.num_epochs
    full_dataset = dataset * num_epochs
    total_steps = min(len(full_dataset), config.reward.anneal_steps)
    logger.info(
        f"Phase 2 training: {total_steps} steps "
        f"(dataset={len(dataset)} × {num_epochs} epochs, "
        f"max_events={args.max_events}, ε_start={args.epsilon_start})"
    )

    model.train()
    optimizer.zero_grad()

    for step_idx, episode in enumerate(full_dataset[:total_steps]):
        events = episode.get("events", [])
        question = episode.get("question", "")
        answer = episode.get("answer", "")

        # === ε-greedy schedule (F9): linear decay ===
        progress = step_idx / max(total_steps - 1, 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        # === Phase A: Process events through Memory Manager ===
        # For each event, decide (with exploration) and record log π(a|s)
        episode_log_probs = []
        episode_op_bonuses = []
        selected_events = events[:args.max_events]

        for event in selected_events:
            try:
                # Decide with ε-greedy exploration (F9)
                op_str, prompt, was_explore = agent.memory_manager.decide_with_exploration(
                    event, question, epsilon=epsilon
                )

                # Execute the operation
                agent.memory_manager.execute_operation(op_str, event)
                agent.step_count += 1

                # Record operation stats
                op_type = get_op_type(op_str)
                op_counts[op_type] += 1
                if was_explore:
                    explore_count += 1

                # Compute log π(a|s) for this specific action (F7)
                log_prob = agent.memory_manager.compute_action_log_prob(prompt, op_str)
                episode_log_probs.append(log_prob)

                # Record operation bonus (F8)
                op_bonus = get_operation_bonus(op_str)
                episode_op_bonuses.append(op_bonus)

            except Exception as e:
                logger.debug(f"Event processing error: {e}")

        # === Phase B: Compute episode-level reward ===
        # R_env: QA F1 (external signal)
        predicted = agent.answer_question(question) if question and answer else ""
        r_ext = compute_f1(predicted, answer) if predicted and answer else 0.0

        # Compute self-reward (for τ calibration tracking only, NOT for RL)
        r_self = 0.0
        if question and answer:
            try:
                r_self_result = agent.reward_generator.compute_reward(
                    agent_response=predicted,
                    task_context=question,
                    memory_operation="NOOP",
                    env_signal_kwargs={"agent_response": predicted, "qa_ground_truth": answer},
                )
                r_self = r_self_result.r_total
            except Exception:
                pass

        # Episode reward = R_env + mean(operation_bonuses) (F8)
        mean_op_bonus = (
            sum(episode_op_bonuses) / len(episode_op_bonuses)
            if episode_op_bonuses else 0.0
        )
        r_episode = r_ext + mean_op_bonus

        # Annealed reward (for τ tracking)
        r_annealed = alpha * r_ext + (1 - alpha) * r_self

        ext_rewards.append(r_ext)
        self_rewards.append(r_self)
        annealed_rewards.append(r_annealed)

        # === Phase C: REINFORCE policy update (F7) ===
        if episode_log_probs:
            # Baseline update (EMA)
            reward_baseline = 0.95 * reward_baseline + 0.05 * r_episode
            advantage = r_episode - reward_baseline

            # Sum log probs over all events in this episode
            total_log_prob = torch.stack(episode_log_probs).sum()

            # REINFORCE loss: -advantage * Σ log π(a_t | s_t)
            policy_loss = -advantage * total_log_prob

            # Scale by gradient accumulation
            policy_loss = policy_loss / grad_accum
            policy_loss.backward()

            if (step_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.rl.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # === Phase D: Calibration check & annealing ===
        if (step_idx + 1) % 10 == 0 and len(ext_rewards) >= 10:
            window = min(50, len(ext_rewards))
            tau = compute_kendall_tau(ext_rewards[-window:], self_rewards[-window:])
            avg_annealed = sum(annealed_rewards[-window:]) / window
            elapsed = time.time() - start_time
            eps = (step_idx + 1) / elapsed
            remaining = (total_steps - step_idx - 1) / eps if eps > 0 else 0

            # Operation breakdown
            total_ops = sum(op_counts.values()) or 1
            op_pct = {k: f"{100*v/total_ops:.1f}%" for k, v in op_counts.items()}
            noop_pct = 100 * op_counts["NOOP"] / total_ops

            logger.info(
                f"Step {step_idx+1}/{total_steps} | α={alpha:.3f} | τ={tau:.3f} | "
                f"R_ext={r_ext:.3f} | R_ep={r_episode:.3f} | "
                f"ε={epsilon:.3f} | mem={agent.memory_store.size()} | "
                f"NOOP={noop_pct:.0f}% | ops={op_pct} | "
                f"explore={explore_count} | "
                f"Speed: {eps:.2f} step/s | ETA: {remaining/3600:.1f}h"
            )

            # Pause annealing if τ < threshold
            if tau < config.reward.tau_threshold:
                if not paused:
                    logger.warning(
                        f"ANNEALING PAUSED: Kendall τ={tau:.3f} < "
                        f"threshold={config.reward.tau_threshold}"
                    )
                    paused = True
            else:
                paused = False

        # Anneal alpha
        if not paused and alpha > config.reward.anneal_end_alpha:
            alpha = max(config.reward.anneal_end_alpha, alpha - alpha_step)

        # Periodic save
        if (step_idx + 1) % 100 == 0:
            agent.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{step_idx+1}")
            )

    # === Final save ===
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "best"))

    # Save calibration + operation stats
    calib_data = {
        "ext_rewards": ext_rewards,
        "self_rewards": self_rewards,
        "annealed_rewards": annealed_rewards,
        "final_alpha": alpha,
        "total_steps": len(ext_rewards),
        "operation_counts": op_counts,
        "explore_count": explore_count,
    }
    with open(os.path.join(args.output_dir, "calibration.json"), "w") as f:
        json.dump(calib_data, f)

    logger.info(
        f"Phase 2 v7 complete. Final α={alpha:.4f}, "
        f"ops={op_counts}, explore={explore_count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 2 v7: RL Policy Warmup")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint", default="outputs/phase1/best")
    parser.add_argument("--output_dir", default="outputs/phase2_v7")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--anneal_steps", type=int, default=3000)
    parser.add_argument("--tau_threshold", type=float, default=0.5)

    # v7 new args
    parser.add_argument("--max_events", type=int, default=8,
                        help="Max events per episode (F11, was 3)")
    parser.add_argument("--epsilon_start", type=float, default=0.3,
                        help="ε-greedy exploration start (F9)")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="ε-greedy exploration end (F9)")

    # Training
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate (default: config * 0.5)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of passes over dataset")

    # Infrastructure
    parser.add_argument("--no_qlora", action="store_true",
                        help="Use bf16 full-precision + LoRA instead of QLoRA")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")

    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    main(args)
