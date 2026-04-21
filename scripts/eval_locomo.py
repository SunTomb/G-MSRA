"""
Evaluation on LoCoMo / LongMemEval benchmarks.

For each eval example, the evaluation flow is:
  1. RESTORE memory store to checkpoint state (clean per example)
  2. Fast-ADD that example's events to memory (embedding only, no LLM)
  3. Answer the question using memory-augmented RAG (retrieve top-5 + LLM)
  4. Compute F1 / EM

This matches baseline evaluation behavior in baselines.py:
  - Simple baselines: each example sees ONLY its own events (current_events
    is REPLACED per episode via ingest_episode, not accumulated)
  - MemoryR1: events accumulate via LLM-based step(), but the LLM decides
    what to store (selective ADD/NOOP/UPDATE/DELETE)
  - G-MSRA: each example's events are added to checkpoint memory, then
    retrieved by semantic similarity for QA

Usage:
    python scripts/eval_locomo.py \\
        --checkpoint outputs/phase2/best \\
        --lora_checkpoint outputs/phase1/best \\
        --output_dir results/eval_phase2 \\
        --benchmark locomo \\
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
from gmsra.utils import (
    compute_exact_match,
    compute_f1,
    load_model_and_tokenizer,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helper: event text extraction
# ---------------------------------------------------------------------------

# Default max events to ingest per example. Overridable via --max_events CLI.
# We take the LAST N events since questions tend to be about recent content.
DEFAULT_MAX_EVENTS = 250


def _extract_event_text(event) -> str:
    """Extract meaningful text from an event for embedding.

    LoCoMo events can be:
      - Plain strings: "User says: I just adopted a puppy."
      - Dicts: {"speaker": "Sam", "dia_id": "D1:1", "text": "Hey Evan!!"}
      - Dicts with image: {"speaker": "Sam", "blip_caption": "a photo of..."}
      - Dicts with image URL only: {"speaker": "Evan", "img_url": [...]}
      - Metadata strings: "1:47 pm on 18 May, 2023"
      - String-encoded dicts: "{'speaker': 'Sam', 'text': 'Hey'}" (from JSON)
    """
    # If already a dict, extract fields directly
    if isinstance(event, dict):
        return _extract_from_dict(event)

    # String event — may be a string-encoded dict
    s = str(event).strip()
    if not s:
        return ""

    # Try to parse string-encoded Python dicts
    if s.startswith("{") and s.endswith("}"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return _extract_from_dict(parsed)
        except (ValueError, SyntaxError):
            pass  # Not a valid dict literal, use as-is

    return s


def _extract_from_dict(d: dict) -> str:
    """Extract readable text from a LoCoMo event dict."""
    speaker = d.get("speaker", "")
    text = d.get("text", "")
    caption = d.get("blip_caption", "")

    if text:
        return f"{speaker}: {text}" if speaker else text
    if caption:
        return f"{speaker}: [image] {caption}" if speaker else f"[image] {caption}"
    if "img_url" in d:
        return f"{speaker}: [shared an image]" if speaker else "[shared an image]"
    # Fallback
    return str(d)


# ---------------------------------------------------------------------------
# Helper: memory store snapshot / restore
# ---------------------------------------------------------------------------

def _snapshot_memory(store) -> dict:
    """Create an in-memory snapshot of the memory store's data.

    Only copies the mutable data (entries, embeddings, id_list).
    Does NOT copy the encoder or FAISS index (those are shared/rebuilt).
    """
    snapshot = {
        "entries": copy.deepcopy(store.entries),
        "id_list": list(store._id_list),
        # _np_embeddings may not exist if FAISS mode is used
        "np_embeddings": (
            [e.copy() for e in store._np_embeddings]
            if hasattr(store, "_np_embeddings") and store._np_embeddings
            else []
        ),
        "use_faiss": getattr(store, "_use_faiss", False),
        "index_sentinel": store._index,  # "numpy" sentinel or FAISS obj
    }
    return snapshot


def _restore_memory(store, snapshot):
    """Restore memory store from a snapshot (fast, in-memory).

    Overwrites entries, embeddings, and id_list. Preserves encoder.
    """
    store.entries = copy.deepcopy(snapshot["entries"])
    store._id_list = list(snapshot["id_list"])
    store._use_faiss = snapshot["use_faiss"]
    store._index = snapshot["index_sentinel"]

    if not snapshot["use_faiss"]:
        # Numpy mode: restore embedding arrays
        store._np_embeddings = [e.copy() for e in snapshot["np_embeddings"]]
    else:
        # FAISS mode: rebuild index from entries
        store._rebuild_index()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main(args):
    set_seed(42)
    no_memory = getattr(args, "no_memory", False)
    max_events = getattr(args, "max_events", DEFAULT_MAX_EVENTS)
    mode_str = "no_memory (pure LLM)" if no_memory else (
        "checkpoint_only (no events)" if getattr(args, "checkpoint_only", False)
        else "ingest_events (per-example isolation)"
    )
    events_str = (
        "N/A (checkpoint_only)" if getattr(args, "checkpoint_only", False)
        else ("no cap (all events)" if max_events <= 0 else f"max {max_events}")
    )
    logger.info(f"Evaluating on {args.benchmark} | checkpoint={args.checkpoint}")
    logger.info(f"  Mode: {mode_str} | Events: {events_str}")

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
    else:
        logger.info("No LoRA adapter loaded (using base model)")

    # ---- Initialize agent ----
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="dialogue")

    # ---- Load checkpoint memory ----
    initial_mem_size = 0
    ckpt_path = args.checkpoint
    if ckpt_path and os.path.exists(os.path.join(ckpt_path, "memory_store.json")):
        agent.load_checkpoint(ckpt_path)
        initial_mem_size = agent.memory_store.size()
        logger.info(f"Loaded agent state from: {ckpt_path} "
                     f"(memories={initial_mem_size})")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}. "
                        "Agent starts with empty memory.")

    # ---- Create memory snapshot for per-example restoration ----
    checkpoint_snapshot = None
    if not no_memory:
        checkpoint_snapshot = _snapshot_memory(agent.memory_store)
        logger.info(f"Created memory snapshot (size={initial_mem_size}) for "
                     "per-example restoration")

        # Sanity check: verify restore works
        _restore_memory(agent.memory_store, checkpoint_snapshot)
        assert agent.memory_store.size() == initial_mem_size, (
            f"Snapshot sanity check failed: "
            f"restored {agent.memory_store.size()} != expected {initial_mem_size}"
        )
        logger.info("Snapshot restore sanity check passed")
    else:
        logger.info("no_memory mode: skipping memory snapshot/restore and event ingestion")

    # ---- Load evaluation data ----
    eval_data = load_eval_data(args.data_dir, args.benchmark)
    total_events = sum(len(ex.get("events", [])) for ex in eval_data)
    logger.info(f"Loaded {len(eval_data)} eval examples from {args.benchmark} "
                 f"(total events={total_events})")

    # ---- Evaluation loop ----
    results = []
    start_time = time.time()

    # Suppress DEBUG-level log spam from memory store during event ingestion
    import logging
    mem_logger = logging.getLogger("gmsra.memory.store")
    original_level = mem_logger.level

    for idx, example in enumerate(eval_data):
        n_ingested = 0

        if not no_memory:
            # 1) Restore memory to checkpoint state (per-example isolation)
            _restore_memory(agent.memory_store, checkpoint_snapshot)

            # 2) Ingest THIS example's events into memory (embedding only)
            #    UNLESS checkpoint_only mode is enabled
            if not getattr(args, "checkpoint_only", False):
                all_events = example.get("events", [])
                if max_events > 0:
                    events = all_events[-max_events:]  # Take last N
                else:
                    events = all_events  # No cap — use all events

                # Temporarily suppress DEBUG logs for bulk ingestion
                mem_logger.setLevel(logging.WARNING)
                for event in events:
                    text = _extract_event_text(event)
                    if text and len(text) > 3:
                        agent.memory_store.add(
                            text,
                            env_reward=0.5,
                            tags=["eval_event"],
                            source=f"eval_{idx}",
                        )
                        n_ingested += 1
                mem_logger.setLevel(original_level)

        # 3) Answer question using memory-augmented RAG
        question = example["question"]
        ground_truth = example["answer"]
        prediction = agent.answer_question(question)

        # 4) Compute metrics
        f1 = compute_f1(prediction, ground_truth)
        em = compute_exact_match(prediction, ground_truth)

        results.append({
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "f1": f1,
            "em": em,
            "category": example.get("category", "unknown"),
            "events_ingested": n_ingested,
            "mem_size": agent.memory_store.size(),
        })

        # Log progress every 10 examples or at the end
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_data):
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(eval_data) - idx - 1) / speed if speed > 0 else 0
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            mem_info = "disabled" if no_memory else f"{initial_mem_size}+{n_ingested}={agent.memory_store.size()}"
            logger.info(
                f"Progress: {idx+1}/{len(eval_data)} | "
                f"F1={avg_f1:.4f} | EM={avg_em:.4f} | "
                f"mem={mem_info} | "
                f"Speed: {speed:.2f} ex/s | ETA: {eta/60:.1f}min"
            )

    # ---- Aggregate metrics ----
    elapsed_total = time.time() - start_time
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
    avg_em = sum(r["em"] for r in results) / len(results) if results else 0

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    category_metrics = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_f1 = sum(r["f1"] for r in cat_results) / len(cat_results)
        cat_em = sum(r["em"] for r in cat_results) / len(cat_results)
        cat_nz = sum(1 for r in cat_results if r["f1"] > 0)
        category_metrics[cat] = {
            "f1": cat_f1,
            "em": cat_em,
            "count": len(cat_results),
            "nonzero_f1": cat_nz,
        }

    # Excluding-abstain metrics (cat 5)
    non_abstain = [r for r in results if r["category"] != "5"]
    f1_excl = sum(r["f1"] for r in non_abstain) / len(non_abstain) if non_abstain else 0

    mode_label = "no_memory" if no_memory else "ingest_events_per_example"
    summary = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "lora_checkpoint": args.lora_checkpoint,
        "mode": mode_label,
        "num_examples": len(results),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "avg_f1_excl_abstain": f1_excl,
        "memory_size_checkpoint": initial_mem_size,
        "elapsed_seconds": elapsed_total,
        "category_breakdown": category_metrics,
    }

    # ---- Save results ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.benchmark}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2,
                  ensure_ascii=False)

    # ---- Print final report ----
    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTS: {args.benchmark}")
    logger.info(f"  F1:  {avg_f1:.4f}  (excl. abstain: {f1_excl:.4f})")
    logger.info(f"  EM:  {avg_em:.4f}")
    logger.info(f"  Checkpoint memory: {initial_mem_size}")
    logger.info(f"  Time: {elapsed_total/60:.1f} min")
    for cat, m in sorted(category_metrics.items()):
        logger.info(
            f"  [{cat}] F1={m['f1']:.4f} EM={m['em']:.4f} "
            f"(n={m['count']}, nonzero={m['nonzero_f1']})"
        )
    logger.info(f"  Saved to: {out_file}")
    logger.info(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(data_dir: str, benchmark: str) -> list[dict]:
    """Load evaluation data from JSON."""
    path = os.path.join(data_dir, f"{benchmark}_test.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate structure
        for i, ex in enumerate(data[:3]):
            if "question" not in ex or "answer" not in ex:
                logger.warning(f"Example {i} missing 'question' or 'answer' keys: {list(ex.keys())}")
        return data
    logger.warning(f"Eval data not found at {path}, using placeholder")
    return [
        {"events": ["User likes AI"], "question": "What does user like?",
         "answer": "AI", "category": "preference"}
    ] * 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="G-MSRA Evaluation (per-example event ingestion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 2 eval
  python scripts/eval_locomo.py \\
      --checkpoint outputs/phase2/best \\
      --lora_checkpoint outputs/phase1/best \\
      --no_qlora --benchmark locomo

  # Phase 3 v5 eval
  python scripts/eval_locomo.py \\
      --checkpoint outputs/phase3_v5/best \\
      --lora_checkpoint outputs/phase1/best \\
      --no_qlora --benchmark locomo
""",
    )
    parser.add_argument("--checkpoint", default="outputs/phase3_v5/best",
                        help="Agent state dir (memory_store.json + agent_meta.json)")
    parser.add_argument("--lora_checkpoint", default="outputs/phase1/best",
                        help="LoRA adapter dir (must contain adapter_config.json)")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--benchmark", default="locomo",
                        choices=["locomo", "longmemeval"])
    parser.add_argument("--no_qlora", action="store_true",
                        help="Disable QLoRA (use full bf16)")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_memory", action="store_true",
                        help="Ablation: skip memory retrieval and event ingestion "
                             "(pure LLM zero-shot QA)")
    parser.add_argument("--max_events", type=int, default=DEFAULT_MAX_EVENTS,
                        help="Max events to ingest per example. "
                             "0 = no cap (use all events). Default: 250")
    parser.add_argument("--checkpoint_only", action="store_true",
                        help="v6: Only use checkpoint memories for QA, DO NOT ingest "
                             "per-example events. This reveals the true value of "
                             "the trained memory policy vs empty baseline.")
    args = parser.parse_args()
    main(args)
