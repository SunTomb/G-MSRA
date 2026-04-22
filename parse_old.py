import json, os, glob

dirs = ['results/baselines', 'results/baselines_v2', 'results/eval_phase2', 'results/eval_phase3v5']
for d in dirs:
    if not os.path.isdir(d):
        continue
    print(f'\n=== {d} ===')
    for f in sorted(glob.glob(os.path.join(d, '*_results.json'))):
        try:
            data = json.load(open(f, encoding='utf-8'))
            if 'summary' in data:
                s = data['summary']
                mem = s.get('memory_size_checkpoint', '?')
                print(f"  {os.path.basename(f)}: F1={s['avg_f1']:.4f}  EM={s['avg_em']:.4f}  n={s['num_examples']}  mem={mem}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict) and 'f1' in v:
                        print(f"  {os.path.basename(f)} [{k}]: F1={v['f1']:.4f}")
        except Exception as e:
            print(f"  ERROR: {os.path.basename(f)}: {e}")
