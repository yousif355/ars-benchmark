# -*- coding: utf-8 -*-
"""Sensitivity of the collapse count to the temperature T and to the collapse threshold.

    python -B scripts/sensitivity.py       (from the benchmark root; run make_table2.py first)

For each of the 15 degraded cells and for the clean row, ECE15 is computed from the
stored logits at three temperatures:
  T = 1.0                 raw
  T = 0.4580              the temperature used throughout (T*)
  T = 0.5281383991241455  a refit on 417 validation clips, using the epoch-112
                          checkpoint (the checkpoint that produced these logits), frame-cache
                          flavour (full model on cached video frames).
                          Source of the value: the fitting record of that refit,
                          field results.framecache.T_optimal; the record is not part
                          of this release. It is used here only as a sensitivity point.
  T = 0.3397591710090637  the same refit, features flavour (classifier applied to precomputed
                          features), field results.features.T_optimal of the same record.
                          This is the flavour in which the shipped T* = 0.4580 was fitted (at
                          epoch 40). Also a sensitivity point only.
A degraded cell counts as collapsed when ECE15 > threshold, for thresholds 0.15, 0.20
and 0.25. The thresholds are exploratory: none was derived from a criterion.
Self-checks: the T = 1 and T = 0.4580 values equal results/table2.csv (the raw and
post-T* columns) at its printed precision.
Writes results/sensitivity.csv. The last three rows hold the counts over the 15 degraded cells.
"""
import csv
import sys

import ars_data as A

TEMPS = (('T1', 1.0), ('Tstar_0.4580', A.T_STAR), ('Trefit_0.5281', 0.5281383991241455),
         ('Trefit_features_0.3398', 0.3397591710090637))
THRESHOLDS = (0.15, 0.20, 0.25)


def main():
    cells = A.load_cells()
    t2 = {(r['condition'], r['severity']): r for r in csv.DictReader(open(A.path('results', 'table2.csv'), encoding='utf-8'))}
    rows = []
    counts = {(s, th): 0 for s, _ in TEMPS for th in THRESHOLDS}
    for key, (lg, y, _ids) in cells.items():
        for stage, T in TEMPS:
            e = A.ece15(A.softmax(lg / T), y)
            row = {'condition': key[0], 'severity': key[1], 'stage': stage, 'T': '%.6f' % T, 'n': len(y), 'ece15': '%.6f' % e}
            for th in THRESHOLDS:
                if key[0] == 'clean':
                    row['gt_%.2f' % th] = ''
                else:
                    flag = e > th
                    counts[(stage, th)] += flag
                    row['gt_%.2f' % th] = int(flag)
            rows.append(row)
            # self-check against Table 2 at printed precision
            if stage in ('T1', 'Tstar_0.4580'):
                col = 'ece15_raw_T1' if stage == 'T1' else 'ece15_Tstar_0.4580'
                assert round(e, 4) == round(float(t2[key][col]), 4), (key, stage, e, t2[key][col])
    for stage, T in TEMPS:
        row = {'condition': 'count_of_15_degraded', 'severity': '', 'stage': stage, 'T': '%.6f' % T, 'n': 15, 'ece15': ''}
        for th in THRESHOLDS:
            row['gt_%.2f' % th] = counts[(stage, th)]
        rows.append(row)
    fields = ['condition', 'severity', 'stage', 'T', 'n', 'ece15'] + ['gt_%.2f' % th for th in THRESHOLDS]
    with open(A.path('results', 'sensitivity.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('stage'.ljust(15), *['count ECE15 > %.2f' % th for th in THRESHOLDS])
    for stage, T in TEMPS:
        print(stage.ljust(15), *[('%d of 15' % counts[(stage, th)]).rjust(len('count ECE15 > 0.00')) for th in THRESHOLDS])
    print('rows written:', len(rows))
    return 0


if __name__ == '__main__':
    sys.exit(main())
