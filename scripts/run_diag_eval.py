"""
Supplementary evaluation script for diagnosing v11 eval underperformance.

Runs 4 diagnostic evaluations:
1. checkpoint_500 (mem=350, pre-consolidation, most memory)
2. checkpoint_2000 (mem=187, 4x consolidated LoRA)  
3. Empty memory + v11 LoRA (pure LoRA distillation effect)
4. checkpoint_500 + checkpoint_only (no event injection, pure trained memory)
"""
import subprocess
import os

PYTHON = "python"
SCRIPT = "scripts/eval_locomo.py"
BASE_CMD = f"{PYTHON} {SCRIPT} --no_qlora --benchmark locomo"

experiments = [
    {
        "name": "v11_ckpt500",
        "desc": "checkpoint_500 (mem=350, best memory count)",
        "checkpoint": "outputs/phase3_v11/checkpoint_500",
        "lora": "outputs/phase3_v11/checkpoint_500/lora",
        "output": "results/diag_ckpt500",
        "extra": "",
    },
    {
        "name": "v11_ckpt2000",
        "desc": "checkpoint_2000 (mem=187, 4x LoRA distill)",
        "checkpoint": "outputs/phase3_v11/checkpoint_2000",
        "lora": "outputs/phase3_v11/checkpoint_2000/lora",
        "output": "results/diag_ckpt2000",
        "extra": "",
    },
    {
        "name": "v11_lora_only",
        "desc": "Empty memory + v11 best LoRA (pure distillation)",
        "checkpoint": "outputs/empty_checkpoint",
        "lora": "outputs/phase3_v11/best/lora",
        "output": "results/diag_lora_only",
        "extra": "",
    },
    {
        "name": "v11_ckpt500_only",
        "desc": "checkpoint_500 memory only (no events)",
        "checkpoint": "outputs/phase3_v11/checkpoint_500",
        "lora": "outputs/phase3_v11/checkpoint_500/lora",
        "output": "results/diag_ckpt500_only",
        "extra": "--checkpoint_only",
    },
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['desc']}")
    print(f"{'='*60}")
    
    cmd = (
        f"{BASE_CMD} "
        f"--checkpoint {exp['checkpoint']} "
        f"--lora_checkpoint {exp['lora']} "
        f"--output_dir {exp['output']} "
        f"{exp['extra']}"
    ).strip()
    
    print(f"CMD: {cmd}")
    os.makedirs(exp['output'], exist_ok=True)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    print(f"Exit code: {result.returncode}")

print("\n\nAll diagnostic evaluations complete!")
print("Parse results with: python parse_results_diag.py")
