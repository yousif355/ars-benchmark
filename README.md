# ARS: adverse-condition robustness and calibration benchmark

This folder holds the code, the stored predictions and the scripts that rebuild
Tables 2 and 3 and Figures 2 to 5 of the accompanying paper.

ARS crosses the 423 clips of a fixed test split of the HWID12 highway-incident
video dataset (12 classes) with five synthetic degradations (fog, rain,
low light, motion blur, camera shake), each at three severities (LOW, MEDIUM,
HIGH): 423 x 5 x 3 = 6,345 instances. Each instance has its own seed, so it can
be rebuilt alone. One frozen video classifier scores every instance. For each
cell the scripts compute top-1 accuracy and calibration error, both before
and after temperature scaling.

## Contents

| Path | What it is |
|---|---|
| `ars/ars_transforms.py` | The five degradations, `SEVERITY_TABLE` and `make_instance_seed` |
| `ars/severity_table.csv` | Severity parameters, written from `SEVERITY_TABLE` by `scripts/make_ars_tables.py` |
| `ars/seeds.csv`, `ars/SEEDS.md` | The 6,345 per-instance seeds and how they are derived (same script) |
| `tests/` | Determinism and bounds tests for `ars_transforms.py` |
| `splits/test.txt` | The 423 test clip IDs, in the order of every prediction file |
| `predictions/clean/clean_logits.npz` | Classifier logits on the 423 clean clips |
| `predictions/arm1/logits_arm1_<condition>_<severity>.npz` | Logits on the degraded clips, no restoration (15 files) |
| `predictions/restoration/` | Logits and per-cell metrics of three restoration variants (see below) |
| `calibration/temperature_scaling_v24.json` | The fitting record of the temperature T*, as written by the fitting script (see "Provenance") |
| `weights/tiny_cnn.pt` | Weights of the degradation detector (see "Model weights") |
| `scripts/` | The scripts that write `results/`, `figures/` and the tables in `ars/` |
| `results/`, `figures/` | Outputs of the scripts, as shipped (`results/sensitivity.csv`: collapse count against T and cut-off) |
| `MANIFEST.sha256` | sha256 of every other file in this folder |

Each `.npz` holds `logits` (423 x 12, float32), `targets` (class index),
`clip_ids` and, for the degraded cells, per-clip flags of the run. The class
index order is `CLASS_NAMES` in `scripts/ars_data.py`.

The HWID12 videos are not included. HWID12 is a public dataset (Kezebou et al.,
2022, "Highway accident detection and classification from live traffic
surveillance cameras: a comprehensive dataset and video action recognition
benchmarking"); obtain it from its source under its own terms. The split into
training, validation and test clips was made for this work (fixed seed 42):
1,942 real training clips plus 70 augmented copies of training clips, 417
validation clips and 423 test clips. `splits/test.txt` lists the IDs of the 423
test clips; the training and validation ID lists are not part of this release.

## Rebuild

Python 3.13.5 was used; `requirements.txt` gives the package versions.

```
pip install -r requirements.txt
python scripts/make_table2.py        # results/table2.csv            (Table 2)
python scripts/extra_metrics.py      # results/extra_metrics.csv     (NLL, Brier, ECE bins, classwise ECE, bootstrap CIs)
python scripts/make_table3_figs.py   # results/table3_flags.csv, figures/fig2-4 .tex (Table 3, Figs 2-4)
python scripts/reliability_bins.py   # results/fig5_reliability_bins.json (Fig. 5)
python scripts/sensitivity.py        # results/sensitivity.csv       (collapse count at four T and three cut-offs)
python scripts/make_ars_tables.py    # ars/severity_table.csv, ars/seeds.csv, ars/SEEDS.md
python -m pytest tests
sha256sum -c MANIFEST.sha256
```

Run the scripts from this folder, in this order (`extra_metrics.py`,
`reliability_bins.py` and `sensitivity.py` check their values against
`results/table2.csv`). The
scripts write no timestamps. A re-run gave files equal to the shipped ones on
one machine (Windows 11, Intel x86-64 CPU, Python 3.13.5, NumPy 2.2.6); results
on other CPUs or NumPy builds may differ in the last digits.
`figures/README.md` explains how to compile the TikZ figures.

## Provenance and confidence stages

Every calibration number in `results/` names its stage:

| Label | Probabilities | Where |
|---|---|---|
| `raw_T1`, `ece15_raw_T1` | softmax(logits), T = 1 | Table 2, raw column; `extra_metrics.csv` rows `raw_T1` |
| `Tstar_0.4580`, `ece15_Tstar_0.4580` | softmax(logits / 0.4580) | Table 2, post-T* column; Figs 4 and 5 (`fig5_reliability_bins.json`, field `t_star`); `extra_metrics.csv` rows `Tstar_0.4580` |
| `T1` | the same as `raw_T1` | `sensitivity.csv` |
| `Trefit_0.5281` | softmax(logits / 0.5281): refit on validation clips, epoch-112 checkpoint, frame-cache flavour | `sensitivity.csv` only |
| `Trefit_features_0.3398` | softmax(logits / 0.3398): refit on validation clips, epoch-112 checkpoint, features flavour | `sensitivity.csv` only |

ECE15 is the top-label expected calibration error over 15 equal-width
confidence bins on [0, 1]. Each bin is (lo, hi], and the first bin is closed
on the left.

- **Temperature.** T* = 0.4580 was fitted by minimising the cross-entropy on 417
  validation clips (`calibration/temperature_scaling_v24.json`, `T_optimal`
  0.45798, rounded to 0.4580 in every script). The fit used the epoch-40
  checkpoint, fed with precomputed per-clip features (the features flavour).
  The predictions here come from the epoch-112 checkpoint, run on the video
  frames. The two checkpoints come from different training runs: epoch 40 is
  the best checkpoint of one run and epoch 112 the best checkpoint of a later
  run. T* is therefore applied to logits it was not fitted on. The `test`
  entries in that file refer to the epoch-40 checkpoint, not to the predictions
  here. T* cannot be reproduced from this release: it holds neither the
  validation logits nor the epoch-40 weights. A refit of T on the same 417
  validation clips with the epoch-112 checkpoint gave 0.5281 in the frame-cache
  flavour (the flavour of the shipped logits) and 0.3398 in the features
  flavour (the flavour in which T* was fitted). The fitting record of the refit
  is not part of this release. Both values are constants in
  `scripts/sensitivity.py`, so the rows of `results/sensitivity.csv` that use
  them can be recomputed but the refit cannot be rerun. `results/sensitivity.csv`
  shows how the collapse count changes with T (see "Reading the results").
- **Calibration file.** `calibration/temperature_scaling_v24.json` is shipped
  unaltered, as the fitting script wrote it. Its `targets` block
  (`master_target_*` and `user_target_*`) holds pass/fail checks of the test
  ECE, computed after the fit. They played no part in fitting T*, which
  minimises the validation cross-entropy only (`method.fit_split`,
  `method.objective`).
- **Two inference passes.** The clean row comes from an inference pass separate
  from the one that produced the 15 degraded cells. The file of the clean row
  records the checkpoint hash prefix `5aa9b577`. The 15 degraded-cell files
  record no hash; their run log (not shipped) names the same checkpoint path
  and epoch 112, so they are attributed to this checkpoint on that basis, not
  by hash. For the degraded clips, the logits of this pass and of an earlier
  pass (not shipped) differ by up to 4.9e-3 (largest absolute difference over
  the 15 cells). Between the two passes ECE15 differs by up to 0.0013 at T = 1
  and by less than 0.0001 at T*. This rests on two passes only.
- **Training overlap.** The classifier was initialised from an earlier model.
  A replay of that model's data split (its clip membership was not stored, so
  it is reconstructed from the code) puts 332 of the 423 test clips (78.5 %)
  and 302 of the 417 validation clips (72.4 %) in the earlier model's training
  data. Accuracy and calibration on this split are therefore not independent
  held-out estimates for the classifier. T* and the two refits were fitted on
  those validation clips, so the temperature was fitted on data that was mostly
  in the training lineage of the classifier it calibrates.
- **Duplicate clips across splits.** The raw source videos of all 2,782
  non-augmented clips of the three splits (1,942 training, 417 validation, 423
  test) were compared by SHA-256; the 70 augmented training clips have no raw
  file and were not hashed. Eight pairs of byte-identical files lie in
  different splits. Four involve a test clip. Three test clips have a
  byte-identical clip in the training split:
  `negative_samples__negative_samples_1105` (training clip
  `negative_samples__negative_samples_1106`),
  `negative_samples__negative_samples_473` (`negative_samples__negative_samples_512`)
  and `rear_collision__rear_collision_127` (`rear_collision__rear_collision_126`).
  One test clip, `collision_with_stationary_object__collision_with_stationary_object_266`,
  is byte-identical to a validation clip,
  `collision_with_stationary_object__collision_with_stationary_object_267`.
  The other four pairs are between training and validation clips:
  `negative_samples__negative_samples_355` and `_366` (training) with `_377`
  (validation), `_458` (training) with `_459` (validation), and `_759`
  (training) with `_758` (validation). None of the five `pedestrian_hit` test
  clips is byte-identical to a training or validation clip. Nothing here is
  corrected for these clips.
  A comparison of 16 frames per clip (a 64-bit difference hash, at a strict
  threshold) also flagged 28 of the 423 test clips as sharing footage with a
  clip of another split. Four of the 28 are the byte-identical test clips above,
  and 65 of the 115 flagged training-test pairs match only at the clip boundary.
  The strict threshold under-detects cropped and brightened copies. A looser
  threshold flags 114 test clips, a list meant for review and not a count of
  leaks. Only the byte-identical pairs were confirmed by file hash.
- **Seeds.** The degraded clips were generated with global seed 42
  (`ars/SEEDS.md`).

## Reading the results

- **Collapse.** A cell is counted as collapsed when ECE15 after T* exceeds 0.20.
  Eight of the 15 cells do. For two of them, rain LOW and low_light HIGH, the
  95 % bootstrap interval of ECE15 reaches below 0.20
  (`results/extra_metrics.csv`, columns `ece15_ci95_lo` and `ece15_ci95_hi`).
  The cut-off 0.20 is exploratory: no criterion in this release derives it. In
  the fitting script the value 0.20 is a pass/fail target for the test ECE
  after T* (field `user_target_post_test_ece` of the calibration file), and the
  collapse cut-off is the same number. This release holds no evidence of how
  the cut-off was chosen for the ARS cells, so counts at 0.15 and 0.25 are
  given. `results/sensitivity.csv` (`scripts/sensitivity.py`) gives the number
  of collapsed cells, out of the 15 degraded cells, at four temperatures and
  three cut-offs:

  | Temperature | ECE15 > 0.15 | ECE15 > 0.20 | ECE15 > 0.25 |
  |---|---|---|---|
  | T = 1 (raw) | 12 | 9 | 9 |
  | T = 0.4580 (T*) | 8 | 8 | 6 |
  | T = 0.5281 (refit, epoch 112, frame-cache flavour) | 8 | 6 | 4 |
  | T = 0.3398 (refit, epoch 112, features flavour) | 8 | 8 | 8 |

  The direction of the error differs between rows. At T = 1, seven of the nine
  cells above 0.20 are under-confident (mean top-1 confidence below accuracy;
  for camera shake HIGH, 0.5600 against 88.42 %). The two cells that T = 1
  shares with T*, fog HIGH and motion_blur HIGH, are over-confident, and so are
  all eight cells above 0.20 at T*. A large ECE15 therefore means
  under-confidence in the T = 1 row and over-confidence in the T* row. The nine
  cells at T = 1 include all three camera-shake cells, where accuracy is 88 %
  to 91 %. The two refit rows are sensitivity points, not second calibrations.
- **Classwise ECE.** The reported classwise ECE is `cw_ece15_mass`: one-vs-rest
  per class, 15 equal-mass bins, averaged over the 12 classes.
  `cw_ece15_width` uses equal-width bins and is given for comparison.
- **Table 3.** A class is flagged in a HIGH cell when the Wilson 95 % lower
  bound of its precision is below 0.40, or when its recall falls more than 15
  points below its clean recall. Per-class counts, both criteria and the
  intervals are in `results/table3_flags.csv`.

## Limitations of the uncertainty estimates

The bootstrap resamples clips, 2,000 times, with one draw shared by all 16
cells. Several clips can come from the same source video, and they are not
grouped, so the intervals may be narrower than intervals that resample source
videos would be. The three rare classes (tier R in Table 3) have five or six
test clips each.

## Restoration variants

`predictions/restoration/` holds the outputs of three variants that place a
degradation detector and an image-restoration step before the same classifier.
They are scored with the same T*.

| Variant | Route | Cells |
|---|---|---|
| `guarded` | detector label -> restoration with a 5 ms wall-clock guard -> classifier | 15 |
| `deployed` | detector label -> restoration without the guard -> classifier | 15 |
| `oracle` | true condition -> restoration without the guard -> classifier | fog and rain, 6 |

The detector and restoration code are not included. The `guarded` variant
depends on wall-clock timing, so a re-run on other hardware can route clips
differently. Each `.npz` records the detector label (`e1_label`) and the flags
`guard_fired`, `sky_skipped`, `restored` and `nonfinite_fixed` per clip.

## Model weights

The classifier checkpoint is published as a release asset, not in this folder;
download location: [ANONYMOUS-HOST-URL, to be inserted before submission].
The detector weights are in this folder:

| File | Where | Size (B) | sha256 |
|---|---|---|---|
| `hvt_best_ema_v24.pt` | release asset | 104,838,745 | `5aa9b57745d671ff78fb4ea47800e87d856f4f3597ee7de14336e8213513dfdd` |
| `tiny_cnn.pt` (degradation detector) | `weights/tiny_cnn.pt` | 22,529 | `81367393dc2882961ce53c0e714553900d206e0a80d9e770152f5e87552baad4` |

`tiny_cnn.pt` is a PyTorch state dict. `hvt_best_ema_v24.pt` is a checkpoint
dict: the EMA weights (`ema_model`), the classification head (`pred_head`) and
training metadata (epoch 112). The model and inference code are not part of
this release, so the scripts use the stored predictions, not the weights.

## Licence

The code (`ars/`, `scripts/`, `tests/`) is under the MIT licence (`LICENSE`).
The data files (`splits/`, `predictions/`, `calibration/`, `results/`) are
derived from HWID12 and follow the terms of that dataset.
