# -*- coding: utf-8 -*-
"""Fig. 5: reliability-diagram bins for camera_shake HIGH and fog HIGH after T* = 0.4580.

    python scripts/reliability_bins.py     (from the benchmark root; run make_table2.py first)

For each cell and each of the 15 equal-width bins (first bin closed on the
left): count, mean top-1 confidence, accuracy. Self-check: the ECE15 (T*)
rebuilt from these bins equals the value in results/table2.csv (the comparison
with ars_data.ece15 repeats the same formula and only guards against edits).
Writes results/fig5_reliability_bins.json (no timestamps, so a re-run gives the
same bytes).
"""
import csv
import hashlib
import json

import numpy as np

import ars_data as A

CELLS = [('camera_shake', 'HIGH'), ('fog', 'HIGH')]
N_BINS = 15


def r(x):
    return round(float(x), 10)


def main():
    cells = A.load_cells()
    with open(A.path('results', 'table2.csv'), encoding='utf-8') as f:
        table2 = {(row['condition'], row['severity']): float(row['ece15_Tstar_0.4580']) for row in csv.DictReader(f)}
    out = {'script': 'scripts/reliability_bins.py', 't_star': A.T_STAR, 'n_bins': N_BINS, 'inputs': {}, 'cells': {}}
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    ok_all = True
    for key in CELLS:
        src = A.source_of(key)
        out['inputs'][src] = hashlib.sha256(open(A.path(*src.split('/')), 'rb').read()).hexdigest()
        lg, y, _ = cells[key]
        probs = A.softmax(lg / A.T_STAR)
        conf = probs.max(axis=-1)
        correct = (probs.argmax(axis=-1) == y).astype(np.float64)
        n = len(y)
        bins, ece_bins = [], 0.0
        for i in range(N_BINS):
            lo, hi = edges[i], edges[i + 1]
            m = ((conf >= lo) & (conf <= hi)) if i == 0 else ((conf > lo) & (conf <= hi))
            c = int(m.sum())
            b = {'bin': i + 1, 'lo': r(lo), 'hi': r(hi), 'count': c}
            if c:
                b['mean_conf'] = r(conf[m].mean())
                b['accuracy'] = r(correct[m].mean())
                ece_bins += (c / n) * abs(correct[m].mean() - conf[m].mean())
            bins.append(b)
        ece_fn = A.ece15(probs, y)
        top = [b for b in bins if b['count']][-1]
        ok = abs(ece_bins - ece_fn) < 1e-9 and abs(ece_fn - table2[key]) < 5e-7
        ok_all &= ok
        out['cells']['%s_%s' % key] = {
            'n': n, 'accuracy': r(correct.mean()), 'mean_conf': r(conf.mean()), 'min_conf': r(conf.min()), 'bins': bins,
            'nonempty_bins': sum(1 for b in bins if b['count']),
            'top_bin': {'bin': top['bin'], 'count': top['count'], 'mean_conf': top['mean_conf'], 'accuracy': top['accuracy']},
            'count_conf_ge_0.8': int((conf > edges[12]).sum()), 'count_conf_below_0.8': int((conf <= edges[12]).sum()),
            'ece15_from_bins': r(ece_bins), 'ece15': r(ece_fn), 'self_check_pass': ok}
    with open(A.path('results', 'fig5_reliability_bins.json'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + '\n')
    for name, c in out['cells'].items():
        print(name, 'n', c['n'], 'acc', c['accuracy'], 'ECE15', c['ece15'], 'self-check', c['self_check_pass'])
        for b in c['bins']:
            if b['count']:
                print('   bin %2d (%.3f-%.3f] n=%3d conf=%.4f acc=%.4f' % (b['bin'], b['lo'], b['hi'], b['count'],
                                                                     b['mean_conf'], b['accuracy']))
    print('wrote results/fig5_reliability_bins.json')
    if not ok_all:
        raise SystemExit('self-check failed')


if __name__ == '__main__':
    main()
