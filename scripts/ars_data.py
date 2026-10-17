# -*- coding: utf-8 -*-
"""Shared loader and metrics for the ARS benchmark scripts.

Paths are relative to the benchmark root (the parent of scripts/).

Source of every per-cell number (one logit array per row):
  clean       predictions/clean/clean_logits.npz
  15 cells    predictions/arm1/logits_arm1_<condition>_<severity>.npz
All 16 arrays hold the 423 clips of splits/test.txt in the same order (asserted).

Confidence stages:
  raw  = softmax(logits)            (T = 1)
  T*   = softmax(logits / 0.4580)   (temperature fitted on validation logits;
                                     see README, "Provenance")
ECE15: top-1 confidence, 15 equal-width bins on [0, 1], bins (lo, hi] with the
first bin closed on the left.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_STAR = 0.4580
CONDS = ['fog', 'rain', 'low_light', 'motion_blur', 'camera_shake']
SEVS = ['LOW', 'MEDIUM', 'HIGH']
CLASS_NAMES = [  # index order of the logits
    'collision_with_motorcycle', 'collision_with_stationary_object', 'head_on_collision', 'rear_collision',
    'side_collision', 'rollover', 'other_crash', 'pedestrian_hit', 'drifting_or_skidding',
    'fire_or_explosions', 'objects_falling', 'negative_samples']


def path(*parts):
    return os.path.join(ROOT, *parts)


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece15(probs, y, n_bins=15):
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


def source_of(key):
    if key == ('clean', 'clean'):
        return 'predictions/clean/clean_logits.npz'
    return 'predictions/arm1/logits_arm1_%s_%s.npz' % key


def load_cells():
    """{(condition, severity): (logits float64 (423, 12), targets int64, clip_ids)}; clean is ('clean', 'clean')."""
    test = [l.strip() for l in open(path('splits', 'test.txt'), encoding='utf-8') if l.strip()]
    cells = {}
    for key in [('clean', 'clean')] + [(c, s) for c in CONDS for s in SEVS]:
        z = np.load(path(*source_of(key).split('/')))
        cells[key] = (z['logits'].astype(np.float64), z['targets'].astype(np.int64), [str(x) for x in z['clip_ids']])
    y0 = cells[('clean', 'clean')][1]
    for key, (lg, y, ids) in cells.items():
        assert lg.shape == (len(test), len(CLASS_NAMES)), key
        assert ids == test and np.array_equal(y, y0), key  # same clips, same order, same labels
    return cells
