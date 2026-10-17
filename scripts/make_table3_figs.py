# -*- coding: utf-8 -*-
"""Table 3 (per-class Wilson-CI vulnerability flags at HIGH severity) and Figs 2-4.

    python scripts/make_table3_figs.py     (from the benchmark root)

Table 3: confusion matrix rows = true class. A class is flagged in a HIGH cell if
  (a) the Wilson 95 % lower bound of its precision is below 0.40
      (z = 1.959963984540054, k = TP, n = TP + FP), or
  (b) its recall is more than 15 percentage points below its clean recall.
Writes results/table3_flags.csv (60 rows: 5 conditions x 12 classes).

Figs 2-4: accuracy (%) and ECE15 after T* = 0.4580 per cell, written as TikZ
pictures (colours collgood, collmid, collbad are defined in figures/README.md):
  figures/fig2_acc_bars.tex   accuracy bars
  figures/fig3_acc_heat.tex   accuracy heatmap; fill collgood!round(acc)!collbad,
                              white text when acc < 50
  figures/fig4_ece_heat.tex   ECE15 (T*) heatmap; red when ECE15 > 0.20 (collapse)
and results/figs_2_4_values.csv.
"""
import csv
import math
import os

import numpy as np

import ars_data as A

Z95 = 1.959963984540054
PREC_LO = 0.40
RECALL_DROP = 0.15
COLLAPSE = 0.20


def wilson_ci(k, n, z=Z95):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def confusion(y, pred, k=12):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def tikz_fig2(acc):
    name = {'fog': 'fog', 'rain': 'rain', 'low_light': 'low light', 'motion_blur': 'motion blur', 'camera_shake': 'camera shake'}
    fills = {'LOW': 'collgood!60', 'MEDIUM': 'collmid!70', 'HIGH': 'collbad!70'}
    out = ['\\begin{tikzpicture}', '\\begin{axis}[',
           '  ybar, bar width=4pt, width=\\columnwidth, height=5.2cm,',
           '  enlarge x limits=0.12, ymin=0, ymax=100,',
           '  ylabel={Accuracy (\\%)}, font=\\scriptsize,',
           '  symbolic x coords={fog,rain,low light,motion blur,camera shake},',
           '  xtick=data, x tick label style={rotate=20,anchor=east},',
           '  legend style={at={(0.5,1.05)},anchor=south,legend columns=3,draw=none},',
           '  ymajorgrids, grid style={gray!20}', ']']
    for s in A.SEVS:
        out.append('\\addplot+[fill=%s] coordinates' % fills[s])
        out.append('  {' + ''.join('(%s,%.2f)' % (name[c], acc[(c, s)]) for c in A.CONDS) + '};')
    out += ['\\legend{LOW,MEDIUM,HIGH}', '\\end{axis}', '\\end{tikzpicture}']
    return out


def _heat_head():
    return ['\\begin{tikzpicture}[font=\\scriptsize]', '\\def\\cs{1.18}',
            '\\node at (1*\\cs,0.75) {LOW};', '\\node at (2*\\cs,0.75) {MEDIUM};', '\\node at (3*\\cs,0.75) {HIGH};',
            '\\foreach \\r/\\name in {1/fog,2/rain,3/low\\_light,4/motion\\_blur,5/camera\\_shake}{',
            '  \\node[anchor=east] at (0.45,-\\r*\\cs+\\cs) {\\name};', '}']


def _ypos(r):
    return '0' if r == 0 else ('-\\cs' if r == 1 else '-%d*\\cs' % r)


def tikz_fig3(acc):
    out = _heat_head()
    for r, c in enumerate(A.CONDS):
        for j, s in enumerate(A.SEVS, 1):
            a = acc[(c, s)]
            style = 'fill=collgood!%d!collbad' % round(a) + (',text=white' if a < 50 else '')
            out.append('\\node[draw,minimum size=11.2mm,%s] at (%d*\\cs,%s) {%.2f};' % (style, j, _ypos(r), a))
    return out + ['\\end{tikzpicture}']


def tikz_fig4(ece):
    out = _heat_head()
    for r, c in enumerate(A.CONDS):
        for j, s in enumerate(A.SEVS, 1):
            e = ece[(c, s)]
            style = 'fill=collbad!75,text=white' if e > COLLAPSE else 'fill=collgood!70'
            out.append('\\node[draw,minimum size=11.2mm,%s] at (%d*\\cs,%s) {%.4f};' % (style, j, _ypos(r), e))
    return out + ['\\end{tikzpicture}']


def main():
    cells = A.load_cells()
    acc, ece, cms = {}, {}, {}
    for k, (lg, y, _) in cells.items():
        pred = lg.argmax(-1)
        acc[k] = round(float((pred == y).mean() * 100), 2)
        ece[k] = round(A.ece15(A.softmax(lg / A.T_STAR), y), 4)
        cms[k] = confusion(y, pred)

    cm0 = cms[('clean', 'clean')]
    clean_recall = {n: cm0[i, i] / cm0[i, :].sum() for i, n in enumerate(A.CLASS_NAMES)}
    rows = []
    for c in A.CONDS:
        cm = cms[(c, 'HIGH')]
        for i, n in enumerate(A.CLASS_NAMES):
            tp = int(cm[i, i]); fn = int(cm[i, :].sum()) - tp; fp = int(cm[:, i].sum()) - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            lo = wilson_ci(tp, tp + fp)[0]
            drop = clean_recall[n] - rec
            fa, fb = lo < PREC_LO, drop > RECALL_DROP
            rows.append({'condition': c, 'severity': 'HIGH', 'class': n, 'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn,
                         'precision': prec, 'precision_wilson95_lo': lo, 'recall': rec, 'clean_recall': clean_recall[n],
                         'recall_drop_pp': 100 * drop, 'flag_a_precision': fa, 'flag_b_recall': fb, 'flagged': fa or fb})
    os.makedirs(A.path('results'), exist_ok=True)
    os.makedirs(A.path('figures'), exist_ok=True)
    with open(A.path('results', 'table3_flags.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in r.items()})
    with open(A.path('results', 'figs_2_4_values.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'severity', 'accuracy_pct_2dp', 'ece15_Tstar_0.4580_4dp', 'fig3_fill_pct', 'fig4_collapse'])
        for c in A.CONDS:
            for s in A.SEVS:
                w.writerow([c, s, f'{acc[(c, s)]:.2f}', f'{ece[(c, s)]:.4f}', round(acc[(c, s)]), ece[(c, s)] > COLLAPSE])
    for name, lines in (('fig2_acc_bars', tikz_fig2(acc)), ('fig3_acc_heat', tikz_fig3(acc)), ('fig4_ece_heat', tikz_fig4(ece))):
        with open(A.path('figures', name + '.tex'), 'w', newline='\n', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    print('flagged classes per condition at HIGH:',
          {c: int(sum(r['flagged'] for r in rows if r['condition'] == c)) for c in A.CONDS})
    print('flagged cells:', int(sum(r['flagged'] for r in rows)), 'of', len(rows),
          '| on the recall criterion:', int(sum(r['flag_b_recall'] for r in rows)))
    print('cells with ECE15 (T*) > 0.20:', sum(ece[(c, s)] > COLLAPSE for c in A.CONDS for s in A.SEVS), 'of 15')
    print('wrote results/table3_flags.csv, results/figs_2_4_values.csv, figures/fig2-4 .tex')


if __name__ == '__main__':
    main()
