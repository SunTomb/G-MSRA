import json, os, glob

# All result directories to compare
dirs = [
    ('D. No Memory',          'results/no_memory_v11'),
    ('B. Ckpt-only (best)',   'results/gmsra_v11_ckpt_only'),
    ('A. G-MSRA v11 (best)',  'results/gmsra_v11'),
    ('C. Events Only',        'results/events_only_v11'),
    ('E. Phase 2 v7',         'results/phase2_v7'),
    ('-- diag: ckpt500',      'results/diag_ckpt500'),
    ('-- diag: ckpt2000',     'results/diag_ckpt2000'),
    ('-- diag: lora_only',    'results/diag_lora_only'),
    ('-- diag: ckpt500_only', 'results/diag_ckpt500_only'),
]

print(f"{'Group':<28} {'Benchmark':<12} {'F1':>8} {'EM':>8} {'F1_ex':>8} {'mem':>5}")
print("-" * 75)

for label, d in dirs:
    if not os.path.isdir(d):
        continue
    for f in sorted(glob.glob(os.path.join(d, '*_results.json'))):
        try:
            s = json.load(open(f, encoding='utf-8'))['summary']
            bench = s['benchmark']
            print(f"{label:<28} {bench:<12} {s['avg_f1']:>8.4f} {s['avg_em']:>8.4f} {s['avg_f1_excl_abstain']:>8.4f} {s['memory_size_checkpoint']:>5}")
        except Exception as e:
            print(f"{label:<28} ERROR: {e}")
