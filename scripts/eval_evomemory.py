"""
EvoMemory Evaluation: Knowledge Evolution Tracking.

Tests the agent's ability to track evolving knowledge through CRUD operations.
Each example contains a sequence of events with contradictory updates
(e.g., "I live in Beijing" → "I'm moving to Chengdu" → "I'm moving to Shanghai"),
and the question tests whether the agent returns the LATEST state.

This is the primary evaluation for G-MSRA v2, where CRUD operations
(especially UPDATE) provide a clear advantage over raw ADD.

Evaluation modes:
  1. raw_add: All events → store.add() (no management)
  2. heuristic_crud: Rule-based UPDATE (cosine > threshold)
  3. rl_crud: RL-trained memory manager decide() + execute()
  4. rl_crud_compact: RL-trained + LLM compaction

Usage:
    python scripts/eval_evomemory.py \\
        --mode rl_crud \\
        --lora_checkpoint outputs/phase1/best \\
        --no_qlora
"""

import argparse
import copy
import json
import os
import time

import numpy as np
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.memory.store import MemoryStore
from gmsra.utils import (
    compute_exact_match,
    compute_f1,
    load_model_and_tokenizer,
    set_seed,
)


def run_raw_add(store: MemoryStore, events: list[str]) -> MemoryStore:
    """Baseline: ADD all events without any management."""
    for event in events:
        store.add(content=event, env_reward=0.5)
    return store


def run_heuristic_crud(
    store: MemoryStore,
    events: list[str],
    update_threshold: float = 0.85,
) -> MemoryStore:
    """Heuristic baseline: UPDATE if cosine > threshold, else ADD."""
    for event in events:
        # Check if any existing memory is very similar
        results = store.retrieve(event, topk=1)
        if results:
            entry, score = results[0]
            if score >= update_threshold:
                # UPDATE: replace the old entry
                store.update(entry.id, event, env_reward=0.5)
                continue
        # ADD new entry
        store.add(content=event, env_reward=0.5)
    return store


def run_rl_crud(
    store: MemoryStore,
    events: list[str],
    memory_manager,
) -> MemoryStore:
    """RL-trained CRUD: use memory manager's decide() + execute()."""
    for event in events:
        op_str, _ = memory_manager.decide(event)
        memory_manager.execute_operation(op_str, event, env_reward=0.5)
    return store


def answer_question(model, tokenizer, question: str, store: MemoryStore) -> str:
    """Answer a question using retrieved memories."""
    from gmsra.utils import generate_text

    relevant = store.retrieve(question, topk=5)
    memory_context = "\n".join([
        f"- {entry.content}" for entry, score in relevant
    ]) if relevant else "(no relevant memories)"

    prompt = (
        "Based on the following memory entries, answer the question concisely. "
        "If the information has been updated multiple times, use the LATEST version.\n\n"
        f"Memory entries:\n{memory_context}\n\n"
        f"Question: {question}\n\n"
        "Answer (short, factual):"
    )

    answer = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=64, temperature=0.1,
    )

    # Post-process: extract first line, strip
    answer = answer.strip().split("\n")[0].strip()
    return answer


def main(args):
    set_seed(42)
    logger.info(f"EvoMemory Evaluation | mode={args.mode}")

    # ---- Load model ----
    config = GMSRAConfig()
    use_qlora = not args.no_qlora
    model, tokenizer = load_model_and_tokenizer(
        config.model.model_name,
        use_qlora=use_qlora,
        load_in_4bit=args.load_in_4bit,
    )

    lora_path = args.lora_checkpoint
    if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info(f"Loaded LoRA adapter from: {lora_path}")

    # ---- Load RL memory manager (if needed) ----
    memory_manager = None
    if args.mode in ("rl_crud", "rl_crud_compact"):
        from gmsra.manager.memory_manager import MemoryManager
        memory_manager = MemoryManager(
            model=model,
            tokenizer=tokenizer,
            rl_config=config.rl,
            memory_config=config.memory,
        )
        logger.info("Initialized RL memory manager")

    # ---- Load evaluation data ----
    data_path = os.path.join(args.data_dir, "evomemory_test.json")
    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} EvoMemory examples")

    # ---- Evaluation loop ----
    results = []
    f1_scores = []
    em_scores = []
    start_time = time.time()

    for i, example in enumerate(eval_data):
        events = example["events"]
        question = example["question"]
        gold_answer = example["answer"]
        num_updates = int(example.get("num_updates", len(events)))

        # Fresh memory store for each example
        store = MemoryStore(config.memory)

        # Process events according to mode
        if args.mode == "raw_add":
            run_raw_add(store, events)
        elif args.mode == "heuristic_crud":
            run_heuristic_crud(store, events, args.update_threshold)
        elif args.mode in ("rl_crud", "rl_crud_compact"):
            memory_manager.store = store
            run_rl_crud(store, events, memory_manager)

            # Optional: run compaction after processing
            if args.mode == "rl_crud_compact":
                from gmsra.consolidation.compaction import MemoryCompactor
                compactor = MemoryCompactor(config.compaction)
                compactor.config.trigger_memory_threshold = 2  # Low threshold for small examples
                compactor.run(store, model, tokenizer)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Answer the question
        predicted = answer_question(model, tokenizer, question, store)

        # Compute metrics
        f1 = compute_f1(predicted, gold_answer)
        em = compute_exact_match(predicted, gold_answer)
        f1_scores.append(f1)
        em_scores.append(em)

        result = {
            "example_id": i,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted,
            "f1": f1,
            "em": em,
            "num_events": len(events),
            "num_updates": num_updates,
            "memory_size": store.size(),
        }
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(eval_data):
            avg_f1 = np.mean(f1_scores)
            avg_em = np.mean(em_scores)
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {i+1}/{len(eval_data)} | "
                f"F1={avg_f1:.4f} | EM={avg_em:.4f} | "
                f"mem={store.size()} | "
                f"Time: {elapsed:.1f}s"
            )

    # ---- Summary ----
    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)
    elapsed = time.time() - start_time

    summary = {
        "mode": args.mode,
        "num_examples": len(eval_data),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "elapsed_seconds": elapsed,
        "lora_checkpoint": args.lora_checkpoint,
    }

    logger.info("=" * 60)
    logger.info(f"EvoMemory Results ({args.mode})")
    logger.info(f"  F1:  {avg_f1:.4f}")
    logger.info(f"  EM:  {avg_em:.4f}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evomemory_results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output_dir}/evomemory_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G-MSRA v2: EvoMemory Knowledge Evolution Evaluation",
    )
    parser.add_argument("--mode", default="raw_add",
                        choices=["raw_add", "heuristic_crud", "rl_crud", "rl_crud_compact"],
                        help="Evaluation mode")
    parser.add_argument("--lora_checkpoint", default="outputs/phase1/best",
                        help="LoRA adapter directory")
    parser.add_argument("--output_dir", default="results/evomemory")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--no_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--update_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for heuristic UPDATE")
    args = parser.parse_args()
    main(args)
