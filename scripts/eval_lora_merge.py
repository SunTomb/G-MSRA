"""
LoRA Merge Sweep: find the optimal mixing ratio between Phase 1 LoRA and v11 LoRA.

merged = alpha * Phase1_LoRA + (1-alpha) * v11_LoRA

Usage:
    PYTHONPATH=/NAS/yesh/G-MSRA CUDA_VISIBLE_DEVICES=1 python scripts/eval_lora_merge.py \
        --phase1_lora outputs/phase1/best \
        --v11_lora outputs/phase3_v11/best/lora \
        --checkpoint outputs/phase3_v11/checkpoint_500 \
        --no_qlora
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from loguru import logger


def merge_lora_weights(phase1_dir: str, v11_dir: str, output_dir: str, alpha: float):
    """Merge two LoRA adapters by linear interpolation.
    
    merged = alpha * phase1 + (1 - alpha) * v11
    """
    from safetensors.torch import load_file, save_file
    
    p1_weights = load_file(os.path.join(phase1_dir, "adapter_model.safetensors"))
    v11_weights = load_file(os.path.join(v11_dir, "adapter_model.safetensors"))
    
    merged = {}
    all_keys = set(p1_weights.keys()) | set(v11_weights.keys())
    
    for key in all_keys:
        if key in p1_weights and key in v11_weights:
            merged[key] = alpha * p1_weights[key] + (1 - alpha) * v11_weights[key]
        elif key in p1_weights:
            merged[key] = p1_weights[key] * alpha
        else:
            merged[key] = v11_weights[key] * (1 - alpha)
    
    os.makedirs(output_dir, exist_ok=True)
    save_file(merged, os.path.join(output_dir, "adapter_model.safetensors"))
    
    # Copy adapter_config.json from phase1 (same architecture)
    shutil.copy(
        os.path.join(phase1_dir, "adapter_config.json"),
        os.path.join(output_dir, "adapter_config.json")
    )
    
    logger.info(f"Merged LoRA (alpha={alpha}): {len(merged)} tensors -> {output_dir}")


def run_eval(checkpoint: str, lora_dir: str, output_dir: str, benchmark: str = "locomo"):
    """Run eval_locomo.py and return F1 score."""
    cmd = [
        sys.executable, "scripts/eval_locomo.py",
        "--checkpoint", checkpoint,
        "--lora_checkpoint", lora_dir,
        "--no_qlora",
        "--benchmark", benchmark,
        "--output_dir", output_dir,
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error(f"Eval failed (exit code {result.returncode})")
        return None
    
    # Parse result
    result_file = os.path.join(output_dir, f"{benchmark}_results.json")
    if os.path.exists(result_file):
        with open(result_file, encoding="utf-8") as f:
            data = json.load(f)
        return data["summary"]
    return None


def main(args):
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    
    v11_lora_dirs = {
        "best": args.v11_lora,
    }
    if args.v11_lora_ckpt500:
        v11_lora_dirs["ckpt500"] = args.v11_lora_ckpt500
    
    results = []
    
    for v11_label, v11_dir in v11_lora_dirs.items():
        for alpha in alphas:
            label = f"alpha={alpha}_v11={v11_label}"
            output_dir = os.path.join(args.output_base, f"merge_{v11_label}_a{alpha}")
            
            if alpha == 1.0:
                # Pure Phase 1 — no merge needed
                lora_dir = args.phase1_lora
            else:
                # Merge
                lora_dir = os.path.join(args.output_base, f"merged_lora_{v11_label}_a{alpha}")
                merge_lora_weights(args.phase1_lora, v11_dir, lora_dir, alpha)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {label}")
            logger.info(f"{'='*60}")
            
            summary = run_eval(args.checkpoint, lora_dir, output_dir, args.benchmark)
            
            if summary:
                f1 = summary["avg_f1"]
                em = summary["avg_em"]
                results.append({
                    "alpha": alpha,
                    "v11_source": v11_label,
                    "f1": f1,
                    "em": em,
                    "f1_excl": summary["avg_f1_excl_abstain"],
                    "label": label,
                })
                logger.info(f"  >> {label}: F1={f1:.4f} EM={em:.4f}")
            else:
                logger.error(f"  >> {label}: FAILED")
    
    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Label':<35} {'F1':>8} {'EM':>8} {'F1_ex':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["v11_source"], x["alpha"])):
        print(f"{r['label']:<35} {r['f1']:>8.4f} {r['em']:>8.4f} {r['f1_excl']:>8.4f}")
    print("=" * 70)
    
    # Save
    out_file = os.path.join(args.output_base, "merge_sweep_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_file}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x["f1"])
        print(f"\n*** BEST: {best['label']} → F1={best['f1']:.4f} ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Merge Sweep Evaluation")
    parser.add_argument("--phase1_lora", default="outputs/phase1/best")
    parser.add_argument("--v11_lora", default="outputs/phase3_v11/best/lora",
                        help="v11 best LoRA")
    parser.add_argument("--v11_lora_ckpt500", default="outputs/phase3_v11/checkpoint_500/lora",
                        help="v11 ckpt500 LoRA (1 consolidation)")
    parser.add_argument("--checkpoint", default="outputs/phase3_v11/checkpoint_500",
                        help="Memory checkpoint to use")
    parser.add_argument("--output_base", default="results/merge_sweep")
    parser.add_argument("--benchmark", default="locomo")
    parser.add_argument("--no_qlora", action="store_true")
    args = parser.parse_args()
    main(args)
