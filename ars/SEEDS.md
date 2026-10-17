# Per-instance seeds

Every ARS instance (one test clip under one condition at one severity) is
generated with its own seed, so any instance can be rebuilt alone and in any
order:

    payload = f"{global_seed}|{clip_id}|{condition}|{severity}"   (UTF-8)
    seed    = int.from_bytes(sha256(payload)[:8], "little") & 0x7FFFFFFFFFFFFFFF

(`make_instance_seed` in `ars/ars_transforms.py`), with global seed 42.

`seeds.csv` lists every seed of the benchmark. It was written by
`scripts/make_ars_tables.py`, which also wrote the counts below:

| Item | Value |
|---|---|
| Test clips (`splits/test.txt`) | 423 |
| Conditions x severities | 5 x 3 |
| Instances (rows of `seeds.csv`) | 6345 |
| Distinct seeds | 6345 |
| sha256 of `seeds.csv` | `5e08042fdc50236abf23305994d8f381281078e05ff770eae3a9a5581cfdf506` |
