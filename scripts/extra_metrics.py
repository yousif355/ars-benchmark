# -*- coding: utf-8 -*-
"""Extra calibration metrics per cell and stage, with paired clip-level bootstrap CIs.

    python scripts/extra_metrics.py        (from the benchmark root; run make_table2.py first)

Stages: 'raw_T1' = softmax(logits); 'Tstar_0.4580' = softmax(logits / 0.4580).
Metrics per cell and stage:
  accuracy_pct        top-1 (identical for both stages; asserted)
  mean_top1_conf      mean top-1 confidence
  nll                 mean -log p_y, p clipped at 1e-12
  brier               multiclass Brier, mean over clips of sum_k (p_k - 1[y=k])^2, range 0..2
  ece{10,15,20,30}    top-label ECE, equal-width bins on [0, 1], (lo, hi], first bin closed on the left
  cw_ece15_mass       classwise ECE: one-vs-rest per class, 15 equal-mass bins, mean over the
                      12 classes (the reported classwise value)
  cw_ece15_width      the same with 15 equal-width bins (for comparison)
Bootstrap (T* rows only): 2,000 resamples of the 423 clips with replacement,
numpy default_rng(20261015); one index array per resample, drawn once and
applied to all 16 cells, so the cells stay paired. The resampling unit is the
clip. Clips cut from the same source video are not grouped, so these intervals
may be too narrow (README, "Limitations"). 95 % percentile intervals for
accuracy and ECE15 (T*).
Self-checks: the vectorised ECE equals the loop ECE on every cell, stage and
bin count; ECE15 equals results/table2.csv.
Writes results/extra_metrics.csv.
"""
import csv
import sys

import numpy as np

import ars_data as A

BINS = (10, 15, 20, 30)
N_BOOT = 2000
SEED = 20261015
COLLAPSE = 0.20  # collapse threshold on ECE15 (T*), used only in the printed summary


def nll(probs, y, eps=1e-12):
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece_loop(probs, y, n_bins):
    """Reference rule (loop form), any bin count."""
    conf = probs.max(-1)
    corr = (probs.argmax(-1) == y).astype(np.float64)
    n = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = ((conf >= lo) & (conf <= hi)) if i == 0 else ((conf > lo) & (conf <= hi))
        if m.any():
            out += m.sum() / n * abs(corr[m].mean() - conf[m].mean())
    return float(out)


def ece_fast(conf, corr, n_bins):
    """Same rule, vectorised: bin = searchsorted(edges, conf, 'left') - 1, conf == 0 -> bin 0."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.searchsorted(edges, conf, side='left') - 1, 0, n_bins - 1)
    s_conf = np.bincount(b, weights=conf, minlength=n_bins)
    s_corr = np.bincount(b, weights=corr, minlength=n_bins)
    return float(np.abs(s_corr - s_conf).sum() / len(conf))


def _ece_binary_mass(p, t, n_bins):
    order = np.argsort(p, kind='stable')
    return float(sum(len(b) / len(p) * abs(t[b].mean() - p[b].mean())
                     for b in np.array_split(order, n_bins) if len(b)))


def classwise_ece(probs, y, n_bins, scheme):
    f = _ece_binary_mass if scheme == 'mass' else ece_fast
    return float(np.mean([f(probs[:, k], (y == k).astype(np.float64), n_bins) for k in range(probs.shape[1])]))


def ranks(x):
    """Average ranks (1 = smallest), ties shared."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        idx = x == v
        r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main():
    print('python', sys.version.split()[0], '| numpy', np.__version__)
    cells = A.load_cells()
    keys = list(cells)
    n = len(cells[keys[0]][1])

    table2 = {}
    with open(A.path('results', 'table2.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            table2[(r['condition'], r['severity'])] = (float(r['ece15_raw_T1']), float(r['ece15_Tstar_0.4580']))

    rows, post = [], {}
    for k in keys:
        lg, y, _ = cells[k]
        accs = []
        for stage, T in (('raw_T1', 1.0), ('Tstar_0.4580', A.T_STAR)):
            p = A.softmax(lg / T)
            conf = p.max(-1)
            corr = (p.argmax(-1) == y).astype(np.float64)
            accs.append(corr.mean())
            row = {'condition': k[0], 'severity': k[1], 'stage': stage, 'T': T, 'n': n,
                   'accuracy_pct': 100 * corr.mean(), 'mean_top1_conf': conf.mean(),
                   'nll': nll(p, y), 'brier': brier(p, y)}
            for nb in BINS:
                e = ece_fast(conf, corr, nb)
                assert abs(e - ece_loop(p, y, nb)) < 1e-12, (k, stage, nb)
                row[f'ece{nb}'] = e
            ref = table2[k][0 if stage == 'raw_T1' else 1]
            assert abs(row['ece15'] - ref) < 5e-7, (k, stage, row['ece15'], ref)  # table2.csv holds 6 decimals
            row['cw_ece15_mass'] = classwise_ece(p, y, 15, 'mass')
            row['cw_ece15_width'] = classwise_ece(p, y, 15, 'width')
            rows.append(row)
            if stage == 'Tstar_0.4580':
                post[k] = (conf, corr, row)
        assert accs[0] == accs[1], k  # temperature scaling keeps the argmax

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    for k in keys:
        conf, corr, row = post[k]
        b_acc = corr[idx].mean(axis=1) * 100
        b_ece = np.array([ece_fast(conf[i], corr[i], 15) for i in idx])
        row['acc_ci95_lo'], row['acc_ci95_hi'] = np.percentile(b_acc, [2.5, 97.5])
        row['ece15_ci95_lo'], row['ece15_ci95_hi'] = np.percentile(b_ece, [2.5, 97.5])
        row['ece15_boot_mean'] = b_ece.mean()

    cols = ['condition', 'severity', 'stage', 'T', 'n', 'accuracy_pct', 'acc_ci95_lo', 'acc_ci95_hi',
            'mean_top1_conf', 'nll', 'brier', 'ece10', 'ece15', 'ece15_ci95_lo', 'ece15_ci95_hi', 'ece15_boot_mean',
            'ece20', 'ece30', 'cw_ece15_mass', 'cw_ece15_width']
    with open(A.path('results', 'extra_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f'{r[c]:.6f}' if isinstance(r.get(c), (float, np.floating)) else r.get(c, '')) for c in cols})

    print('\nT* rows: accuracy [95% CI] | ECE15 [95% CI]')
    for k in keys:
        r = post[k][2]
        print(f'{k[0]:12s} {k[1]:6s} acc {r["accuracy_pct"]:6.2f} [{r["acc_ci95_lo"]:6.2f}, {r["acc_ci95_hi"]:6.2f}] | '
              f'ECE15 {r["ece15"]:.4f} [{r["ece15_ci95_lo"]:.4f}, {r["ece15_ci95_hi"]:.4f}]')
    collapsed = [k for k in keys if k[0] != 'clean' and post[k][2]['ece15'] > COLLAPSE]
    borderline = [k for k in collapsed if post[k][2]['ece15_ci95_lo'] < COLLAPSE]
    print(f'\ncells with ECE15 (T*) > {COLLAPSE}: {len(collapsed)} of 15; of these, 95% CI lower bound < {COLLAPSE}: '
          + (', '.join(f'{c} {s}' for c, s in borderline) or 'none'))
    for stage in ('Tstar_0.4580', 'raw_T1'):
        sub = [r for r in rows if r['stage'] == stage and r['condition'] != 'clean']
        e = [r['ece15'] for r in sub]
        print(f'Spearman over the 15 cells, {stage}: (ECE15, NLL) {spearman(e, [r["nll"] for r in sub]):.4f}; '
              f'(ECE15, Brier) {spearman(e, [r["brier"] for r in sub]):.4f}')
    print('wrote results/extra_metrics.csv')


if __name__ == '__main__':
    main()
