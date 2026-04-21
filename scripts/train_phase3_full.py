"""
Phase 3 v11: Per-Action REINFORCE + Weighted Exploration + Capped Consolidation.

v11: Consolidation clearing capped at 30% of memory store (in agent.py).
v10 fixes over v9:
  FIX-3b: Exploration actions participate in gradient with reduced weight (0.3)
          instead of being fully excluded. This solves the chicken-and-egg problem
          where UPDATE can only be learned through exploration experience.
  FIX-6:  Forced consolidation fallback every 500 episodes (regardless of trigger).
  FIX-7:  UPDATE/DELETE baselines updated for ALL actions (including exploration)
          so baselines track actual reward statistics, not just on-policy.
  BUG-3b: store.update() crash fixed in store.py (complete fix).

Retained from v9:
  FIX-1: Per-action REINFORCE with per-action-type baselines
  FIX-2: Explicit consolidation check in training loop
  FIX-4: Sliding window operation stats
  FIX-5: Pass env_reward to memory ADD

Usage:
    PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/train_phase3_full.py \\
        --checkpoint outputs/phase2_v7/checkpoint_2000 \\
        --lora_checkpoint outputs/phase1/best \\
        --output_dir outputs/phase3_v11 \\
        --max_episodes 3000 --max_events 5 --num_epochs 2 \\
        --epsilon_start 0.15 --epsilon_end 0.05 \\
        --no_qlora --no_wandb \\
        2>&1 | tee logs/phase3_v11.log
"""

import argparse
import torch.nn.functional as F
import os
import json
import time
import random
from collections import defaultdict, deque

import torch
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1


# --- Operation Bonus Shaping ---
OPERATION_BONUS = {
    "ADD":    +0.05,
    "UPDATE": +0.35,
    "DELETE": +0.25,
    "NOOP":   -0.10,
}

# --- Exploration weights (must match decide_with_exploration) ---
EXPLORE_WEIGHTS = {
    "ADD": 0.35,
    "UPDATE": 0.30,
    "DELETE": 0.15,
    "NOOP": 0.20,
}

# v10: exploration gradient weight (FIX-3b)
EXPLORE_GRAD_WEIGHT = 0.3


def get_operation_bonus(operation_str: str) -> float:
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
    upper = operation_str.strip().upper()
    for op in ["ADD", "UPDATE", "DELETE"]:
        if upper.startswith(op):
            return op
    return "NOOP"


def main(args):
    set_seed(42)
    logger.info(f"Phase 3 v11: Per-Action REINFORCE + Capped Consolidation | checkpoint={args.checkpoint}")
    logger.info(f"  LoRA weights: {args.lora_checkpoint}")

    config = GMSRAConfig()

    # --- Load model ---
    use_qlora = not args.no_qlora
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_qlora=use_qlora
    )
    logger.info(f"Model precision: {'bf16 (full)' if args.no_qlora else 'QLoRA 4-bit'}")

    # --- Load LoRA weights ---
    # v11 resume: prefer checkpoint's LoRA if resuming
    lora_path = args.lora_checkpoint
    if args.start_episode > 0:
        resume_lora = os.path.join(args.checkpoint, "lora")
        if os.path.exists(os.path.join(resume_lora, "adapter_config.json")):
            lora_path = resume_lora
            logger.info(f"RESUME: using LoRA from checkpoint: {resume_lora}")
    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info(f"Loaded LoRA adapter from: {lora_path}")

        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Enabled LoRA training: {trainable_count:,} / {total_count:,} params "
            f"({100*trainable_count/total_count:.2f}%)"
        )
    else:
        logger.warning(
            f"No adapter_config.json found at '{lora_path}'. "
            f"Training from base model without LoRA checkpoint."
        )

    # Initialize agent
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type=args.env_type)

    # --- Load Phase 2 agent state ---
    agent_state_path = args.checkpoint
    if os.path.exists(os.path.join(agent_state_path, "memory_store.json")):
        agent.load_checkpoint(agent_state_path)
        logger.info(f"Loaded agent state from: {agent_state_path}")
    else:
        logger.warning(f"No memory_store.json at '{agent_state_path}', starting fresh.")

    # --- Setup RL optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable params found! LoRA may not be loaded correctly.")
    else:
        logger.info(f"RL optimizer: {len(trainable_params)} param groups, "
                    f"lr={config.rl.learning_rate * 0.3:.2e}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.rl.learning_rate * 0.3,
        weight_decay=0.01,
    )

    # Load task stream
    single_epoch = load_task_stream(args.env_type, args.data_dir, args.max_episodes)
    num_epochs = args.num_epochs
    task_stream = single_epoch * num_epochs
    total_episodes = len(task_stream)
    logger.info(f"Loaded {len(single_epoch)} tasks × {num_epochs} epochs = {total_episodes} total episodes")

    max_events = args.max_events
    logger.info(f"Event cap per episode: {max_events}")
    logger.info(f"ε-greedy: {args.epsilon_start} → {args.epsilon_end}")
    logger.info(f"Exploration gradient weight: {EXPLORE_GRAD_WEIGHT}")

    grad_accum = config.rl.gradient_accumulation_steps

    # Metrics tracking
    metrics_log = []
    success_window = deque(maxlen=50)
    reward_window = deque(maxlen=50)

    # v9/v10: Per-action-type baselines
    # v11 resume: warm-start from last known values
    if args.start_episode > 0:
        action_baselines = {"ADD": 0.07, "UPDATE": 0.37, "DELETE": 0.28, "NOOP": -0.10}
        logger.info(f"RESUME: warm-started baselines from Ep {args.start_episode}: {action_baselines}")
    else:
        action_baselines = {"ADD": 0.0, "UPDATE": 0.0, "DELETE": 0.0, "NOOP": 0.0}

    # v9: Global + sliding window op counts
    op_counts_global = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
    op_counts_window = deque(maxlen=500)

    explore_count = 0
    consolidation_forced = 0
    LOG_INTERVAL = args.log_interval

    # v10: forced consolidation fallback
    CONSOLIDATION_CHECK_INTERVAL = 100
    CONSOLIDATION_FORCE_INTERVAL = 500

    last_r_ext = 0.0

    # v11 resume: skip completed episodes
    start_episode = args.start_episode

    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    for ep_idx, task in enumerate(task_stream):
        # v11 resume: skip completed episodes
        if ep_idx < start_episode:
            continue

        task_events = task.get("events", [task.get("instruction", "")])
        task_context = task.get("context", task.get("question", ""))
        task_question = task.get("question", task_context)
        task_answer = task.get("answer", "")

        # ε-greedy schedule: linear decay (global, not reset)
        progress = ep_idx / max(total_episodes - 1, 1)
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress

        # === Phase A: Process events ===
        episode_actions = []  # (log_prob, op_type, was_explore)
        selected_events = task_events[:max_events]

        for event in selected_events:
            try:
                op_str, prompt, was_explore = agent.memory_manager.decide_with_exploration(
                    event, task_context, epsilon=epsilon
                )

                agent.memory_manager.execute_operation(
                    op_str, event, env_reward=last_r_ext
                )
                agent.step_count += 1

                op_type = get_op_type(op_str)
                op_counts_global[op_type] += 1
                op_counts_window.append(op_type)
                if was_explore:
                    explore_count += 1

                # v10 FIX-3b: compute log_prob for ALL actions (on-policy and exploration)
                log_prob = agent.memory_manager.compute_action_log_prob(prompt, op_str)
                episode_actions.append((log_prob, op_type, was_explore))

            except Exception as e:
                logger.debug(f"Event processing error: {e}")

        # === Phase B: Compute episode-level reward ===
        predicted = agent.answer_question(task_question) if task_question and task_answer else ""
        r_ext = compute_f1(predicted, task_answer) if predicted and task_answer else 0.0
        last_r_ext = r_ext

        # Self-reward for τ monitoring
        if task_question and task_answer:
            try:
                env_kwargs = {
                    "agent_response": predicted,
                    "qa_ground_truth": task_answer,
                }
                agent.step(
                    event=task_events[-1] if task_events else "",
                    task_context=task_context,
                    agent_response=predicted,
                    env_signal_kwargs=env_kwargs,
                )
            except Exception as e:
                logger.debug(f"self-reward step error: {e}")

        # === Phase C: v10 Per-action REINFORCE with weighted exploration ===
        policy_loss = torch.tensor(0.0, device=model.device, requires_grad=False)
        effective_count = 0

        for log_prob, op_type, was_explore in episode_actions:
            if log_prob is None:
                continue

            # Per-action reward = R_ext + operation bonus
            action_reward = r_ext + OPERATION_BONUS[op_type]

            # v10 FIX-7: update baselines for ALL actions (incl. exploration)
            action_baselines[op_type] = (
                0.95 * action_baselines[op_type] + 0.05 * action_reward
            )
            advantage = action_reward - action_baselines[op_type]

            # v10 FIX-3b: exploration actions get reduced gradient weight
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

        # === Phase D: Consolidation check ===
        if (ep_idx + 1) % CONSOLIDATION_CHECK_INTERVAL == 0:
            try:
                triggered = agent.trigger.should_trigger(agent.step_count)
                # v10 FIX-6: forced consolidation fallback
                if not triggered and (ep_idx + 1) % CONSOLIDATION_FORCE_INTERVAL == 0:
                    triggered = True
                    consolidation_forced += 1
                    logger.info(f"FORCED consolidation at ep {ep_idx+1} (fallback)")
                if triggered:
                    consol_stats = agent._run_consolidation()
                    logger.info(f"Consolidation at ep {ep_idx+1}: {consol_stats}")
            except Exception as e:
                logger.debug(f"Consolidation error: {e}")

        # === Tracking ===
        r_total = r_ext + sum(
            OPERATION_BONUS[op_type] for _, op_type, _ in episode_actions
        ) / max(len(episode_actions), 1)
        task_success = r_ext > 0.0
        success_window.append(float(task_success))
        reward_window.append(r_total)

        # Log periodically
        if (ep_idx + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            speed = (ep_idx + 1) / elapsed
            remaining = (total_episodes - ep_idx - 1) / speed if speed > 0 else 0
            eta_hours = remaining / 3600

            avg_success = sum(success_window) / len(success_window)
            avg_reward = sum(reward_window) / len(reward_window)
            diagnostics = agent.get_full_diagnostics()

            # Global operation breakdown
            total_ops_global = sum(op_counts_global.values()) or 1
            op_pct_global = {k: f"{100*v/total_ops_global:.1f}%" for k, v in op_counts_global.items()}
            noop_pct_global = 100 * op_counts_global["NOOP"] / total_ops_global

            # Sliding window operation breakdown
            if op_counts_window:
                window_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0}
                for op in op_counts_window:
                    window_counts[op] += 1
                total_window = sum(window_counts.values()) or 1
                op_pct_window = {k: f"{100*v/total_window:.1f}%" for k, v in window_counts.items()}
                noop_pct_window = 100 * window_counts["NOOP"] / total_window
            else:
                op_pct_window = op_pct_global
                noop_pct_window = noop_pct_global

            log_entry = {
                "episode": ep_idx + 1,
                "avg_success_rate": avg_success,
                "avg_reward": avg_reward,
                "memory_size": diagnostics["memory_size"],
                "consolidation_count": diagnostics["consolidation_count"],
                "consolidation_forced": consolidation_forced,
                "operation_stats_global": op_counts_global.copy(),
                "operation_stats_window": dict(op_pct_window),
                "baselines": {k: f"{v:.3f}" for k, v in action_baselines.items()},
            }
            metrics_log.append(log_entry)

            logger.info(
                f"Episode {ep_idx+1}/{total_episodes} | "
                f"R_avg={avg_reward:.3f} | R_ext={r_ext:.3f} | "
                f"success={avg_success:.3f} | ε={epsilon:.3f} | "
                f"mem={diagnostics['memory_size']} | "
                f"NOOP={noop_pct_window:.0f}%(win) {noop_pct_global:.0f}%(all) | "
                f"ops_win={op_pct_window} | "
                f"ops_all={op_pct_global} | "
                f"explore={explore_count} | "
                f"consol={diagnostics['consolidation_count']}(+{consolidation_forced}forced) | "
                f"baselines={{{','.join(f'{k}:{v:.2f}' for k,v in action_baselines.items())}}} | "
                f"Speed: {speed:.2f} ep/s | ETA: {eta_hours:.1f}h"
            )

        # Save checkpoint
        if (ep_idx + 1) % 500 == 0:
            agent.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{ep_idx+1}")
            )

    # Save final results
    elapsed_total = time.time() - start_time
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "best"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    diagnostics = agent.get_full_diagnostics()
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    logger.info(
        f"Phase 3 v11 complete! Total time: {elapsed_total/3600:.1f}h, "
        f"Episodes: {total_episodes}, "
        f"Final mem_size: {diagnostics['memory_size']}, "
        f"ops_global={op_counts_global}, explore={explore_count}, "
        f"consolidations={diagnostics['consolidation_count']}(+{consolidation_forced}forced)"
    )


def load_task_stream(env_type: str, data_dir: str,
                     max_episodes: int) -> list[dict]:
    """Load task stream based on environment type."""
    if env_type == "agent_task":
        data_path = os.path.join(data_dir, "alfworld_tasks.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                tasks = json.load(f)
            return tasks[:max_episodes]

        logger.warning(f"Task data not found at {data_path}, using placeholder")
        import random
        random.seed(42)
        return [
            {
                "instruction": f"Complete task {i}",
                "events": [f"Observation: you are in room {i % 5}"],
                "context": "Navigate and complete household tasks.",
                "env_kwargs": {
                    "task_result": {
                        "success": random.random() > 0.6,
                        "partial_score": random.uniform(0.2, 0.8),
                        "steps_taken": random.randint(3, 20),
                        "max_steps": 30,
                    }
                },
            }
            for i in range(max_episodes)
        ]
    else:
        from scripts.train_phase1_rl import load_locomo_data
        data = load_locomo_data(data_dir)
        tasks = []
        for episode in data[:max_episodes]:
            tasks.append({
                "events": episode.get("events", []),
                "context": episode.get("question", ""),
                "question": episode.get("question", ""),
                "answer": episode.get("answer", ""),
            })
        return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 3 v11: Per-Action REINFORCE + Capped Consolidation")
    parser.add_argument("--checkpoint", default="outputs/phase2/best",
                        help="Phase 2 agent state dir")
    parser.add_argument("--lora_checkpoint", default="outputs/phase1/best",
                        help="LoRA adapter dir")
    parser.add_argument("--output_dir", default="outputs/phase3")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--env_type", default="dialogue", choices=["dialogue", "agent_task"])
    parser.add_argument("--max_episodes", type=int, default=3000)
    parser.add_argument("--max_events", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=25)

    parser.add_argument("--epsilon_start", type=float, default=0.15)
    parser.add_argument("--epsilon_end", type=float, default=0.05)

    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Resume from this episode (skip earlier episodes)")
    args = parser.parse_args()
    main(args)
