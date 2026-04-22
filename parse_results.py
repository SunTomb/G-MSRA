import json, os, glob

dirs = [
    'results/gmsra_v11',
    'results/gmsra_v11_ckpt_only', 
    'results/events_only_v11',
    'results/no_memory_v11',
    'results/phase2_v7',
]

for d in dirs:
    print(f'\n=== {d} ===')
    for f in sorted(glob.glob(os.path.join(d, '*_results.json'))):
        try:
            s = json.load(open(f, encoding='utf-8'))['summary']
            cats = s.get('category_breakdown', {})
            parts = []
            for k, v in sorted(cats.items()):
                parts.append(f"[{k}]F1={v['f1']:.3f}(n={v['count']})")
            cat_str = '  '.join(parts)
            bench = s['benchmark']
            print(f"  {bench}: F1={s['avg_f1']:.4f}  EM={s['avg_em']:.4f}  F1_ex={s['avg_f1_excl_abstain']:.4f}  n={s['num_examples']}  mem={s['memory_size_checkpoint']}  {s['elapsed_seconds']/60:.0f}min")
            print(f"    {cat_str}")
        except Exception as e:
            print(f"  ERROR: {f}: {e}")
