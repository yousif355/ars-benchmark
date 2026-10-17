# -*- coding: utf-8 -*-
"""Table 2: accuracy and ECE15 per cell, raw (T = 1) and after T* = 0.4580.

    python scripts/make_table2.py          (from the benchmark root)

Writes results/table2.csv. Per row, accuracy and both ECE values come from one
logit array (scripts/ars_data.py names the file of each row).
"""
import csv
import os

import numpy as np

import ars_data as A


def main():
    cells = A.load_cells()
    rows = []
    for key, (lg, y, _) in cells.items():
        acc = float((lg.argmax(-1) == y).mean() * 100)
        pre, post = A.ece15(A.softmax(lg), y), A.ece15(A.softmax(lg / A.T_STAR), y)
        rows.append({'condition': key[0], 'severity': key[1], 'n': len(y), 'accuracy_pct': f'{acc:.6f}',
                     'ece15_raw_T1': f'{pre:.6f}', 'ece15_Tstar_0.4580': f'{post:.6f}',
                     'accuracy_pct_2dp': f'{round(acc, 2):.2f}', 'ece15_raw_T1_4dp': f'{round(pre, 4):.4f}',
                     'ece15_Tstar_0.4580_4dp': f'{round(post, 4):.4f}', 'source': A.source_of(key)})
    os.makedirs(A.path('results'), exist_ok=True)
    with open(A.path('results', 'table2.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['condition']:12s} {r['severity']:6s} acc {r['accuracy_pct_2dp']:>6s}  "
              f"ECE15 raw {r['ece15_raw_T1_4dp']}  ECE15 T* {r['ece15_Tstar_0.4580_4dp']}")
    print('numpy', np.__version__, '| wrote results/table2.csv')


if __name__ == '__main__':
    main()
