# -*- coding: utf-8 -*-
"""Severity table and per-instance seeds of ARS, generated from ars/ars_transforms.py.

    python scripts/make_ars_tables.py      (from the benchmark root)

Writes:
  ars/severity_table.csv   one row per (condition, severity, parameter) from SEVERITY_TABLE
  ars/seeds.csv            make_instance_seed(42, clip_id, condition, severity) for every
                           clip of splits/test.txt x 5 conditions x 3 severities
  ars/SEEDS.md             how the seeds are derived, with the counts of this run
Global seed 42 is the seed of the inference runs that produced predictions/.
"""
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ars'))
import ars_transforms  # noqa: E402

import ars_data as A  # noqa: E402

GLOBAL_SEED = 42


def main():
    with open(A.path('ars', 'severity_table.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'severity', 'parameter', 'value'])
        n_par = 0
        for c in ars_transforms.CONDITIONS:
            for s in ars_transforms.SEVERITIES:
                for p, v in ars_transforms.SEVERITY_TABLE[(c, s)].items():
                    w.writerow([c, s, p, v])
                    n_par += 1

    test = [l.strip() for l in open(A.path('splits', 'test.txt'), encoding='utf-8') if l.strip()]
    rows = [(cid, c, s, ars_transforms.make_instance_seed(GLOBAL_SEED, cid, c, s))
            for cid in test for c in ars_transforms.CONDITIONS for s in ars_transforms.SEVERITIES]
    with open(A.path('ars', 'seeds.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['clip_id', 'condition', 'severity', 'instance_seed'])
        w.writerows(rows)
    n_unique = len({r[3] for r in rows})
    seeds_sha = hashlib.sha256(open(A.path('ars', 'seeds.csv'), 'rb').read()).hexdigest()

    md = f"""# Per-instance seeds

Every ARS instance (one test clip under one condition at one severity) is
generated with its own seed, so any instance can be rebuilt alone and in any
order:

    payload = f"{{global_seed}}|{{clip_id}}|{{condition}}|{{severity}}"   (UTF-8)
    seed    = int.from_bytes(sha256(payload)[:8], "little") & 0x7FFFFFFFFFFFFFFF

(`make_instance_seed` in `ars/ars_transforms.py`), with global seed {GLOBAL_SEED}.

`seeds.csv` lists every seed of the benchmark. It was written by
`scripts/make_ars_tables.py`, which also wrote the counts below:

| Item | Value |
|---|---|
| Test clips (`splits/test.txt`) | {len(test)} |
| Conditions x severities | {len(ars_transforms.CONDITIONS)} x {len(ars_transforms.SEVERITIES)} |
| Instances (rows of `seeds.csv`) | {len(rows)} |
| Distinct seeds | {n_unique} |
| sha256 of `seeds.csv` | `{seeds_sha}` |
"""
    with open(A.path('ars', 'SEEDS.md'), 'w', newline='\n', encoding='utf-8') as f:
        f.write(md)
    print(f'severity_table.csv: {n_par} parameter rows | seeds.csv: {len(rows)} instances, {n_unique} distinct seeds')
    if n_unique != len(rows):
        raise SystemExit('seed collision')


if __name__ == '__main__':
    main()
