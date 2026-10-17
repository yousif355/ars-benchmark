"""tests/test_ars_transforms.py — determinism + numerical-bounds tests.

Per Protocol E-03-B Red Flag 2 binding decision:
    "A byte-identity determinism test for rain SHALL be included
     in tests/test_ars_transforms.py and MUST PASS before Phase C
     is authorized."

The remaining four conditions also receive byte-identity tests; all five
are required to pass green before any Phase C grid is launched.

This file is authorized but optional on the critical path (it is NOT
imported by eval_ars_v24.py at runtime).

Run from the repo root:
    python -m tests.test_ars_transforms
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable when invoked as a script.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

import ars_transforms


def _fixture_clip(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand((16, 3, 224, 224), generator=g, dtype=torch.float32)


def _byte_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a, b))


def _check_bounds_and_shape(out: torch.Tensor, label: str) -> None:
    assert out.shape == (16, 3, 224, 224), f"[{label}] shape={tuple(out.shape)}"
    assert out.dtype == torch.float32, f"[{label}] dtype={out.dtype}"
    assert torch.isfinite(out).all(), f"[{label}] contains NaN/Inf"
    assert out.min().item() >= 0.0 - 1e-6, f"[{label}] min={out.min().item()}"
    assert out.max().item() <= 1.0 + 1e-6, f"[{label}] max={out.max().item()}"


def test_determinism_byte_identity_all_conditions() -> None:
    clip = _fixture_clip(seed=0)
    cid = "rear_collision__rear_collision_4"
    fails = []
    for cond in ars_transforms.CONDITIONS:
        for sev in ars_transforms.SEVERITIES:
            instance_seed = ars_transforms.make_instance_seed(42, cid, cond, sev)
            out1 = ars_transforms.apply_perturbation(clip.clone(), cond, sev, instance_seed)
            out2 = ars_transforms.apply_perturbation(clip.clone(), cond, sev, instance_seed)
            _check_bounds_and_shape(out1, f"{cond}/{sev}")
            if not _byte_identical(out1, out2):
                fails.append(f"NON-BYTE-IDENTICAL: {cond}/{sev}  "
                             f"max_abs_diff={(out1 - out2).abs().max().item():.2e}")
    assert not fails, "determinism failures:\n  " + "\n  ".join(fails)
    print("OK  determinism: 5 conditions x 3 severities = 15 instances byte-identical")


def test_rain_determinism_byte_identity() -> None:
    """Red Flag 2 dedicated test — rain MUST be byte-identical."""
    clip = _fixture_clip(seed=1)
    cid = "side_collision__side_collision_2"
    for sev in ars_transforms.SEVERITIES:
        instance_seed = ars_transforms.make_instance_seed(42, cid, "rain", sev)
        out1 = ars_transforms.apply_perturbation(clip.clone(), "rain", sev, instance_seed)
        out2 = ars_transforms.apply_perturbation(clip.clone(), "rain", sev, instance_seed)
        _check_bounds_and_shape(out1, f"rain/{sev}")
        assert _byte_identical(out1, out2), (
            f"rain/{sev} NOT byte-identical "
            f"(max_abs_diff={(out1 - out2).abs().max().item():.2e})"
        )
    print("OK  rain byte-identity (Red Flag 2 binding satisfied)")


def test_seed_changes_output() -> None:
    """Different (clip, condition, severity) → different outputs."""
    clip = _fixture_clip(seed=2)
    s_a = ars_transforms.make_instance_seed(42, "clipA", "fog", "LOW")
    s_b = ars_transforms.make_instance_seed(42, "clipB", "fog", "LOW")
    assert s_a != s_b
    out_a = ars_transforms.apply_perturbation(clip.clone(), "fog", "LOW", s_a)
    out_b = ars_transforms.apply_perturbation(clip.clone(), "fog", "LOW", s_b)
    assert not torch.equal(out_a, out_b), "different seeds produced identical output"
    print("OK  distinct seeds produce distinct outputs")


def test_invalid_inputs_raise() -> None:
    seed = 0
    try:
        ars_transforms.apply_perturbation(
            torch.rand((16, 1, 224, 224)), "fog", "LOW", seed
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-3-channel input")
    try:
        ars_transforms.apply_perturbation(
            torch.rand((16, 3, 224, 224)), "snow", "LOW", seed
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown condition")
    try:
        ars_transforms.apply_perturbation(
            torch.rand((16, 3, 224, 224)), "fog", "EXTREME", seed
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown severity")
    print("OK  invalid inputs raise ValueError")


def test_instance_seed_collision_resistance() -> None:
    """SHA-256-derived seeds should not collide across the 6,345-instance grid."""
    seeds = set()
    n = 0
    for clip_idx in range(423):
        cid = f"dummy__clip_{clip_idx}"
        for cond in ars_transforms.CONDITIONS:
            for sev in ars_transforms.SEVERITIES:
                seeds.add(
                    ars_transforms.make_instance_seed(42, cid, cond, sev)
                )
                n += 1
    assert n == 6345, f"expected 6345 instances, got {n}"
    assert len(seeds) == 6345, f"seed collision detected: {n - len(seeds)} duplicates"
    print(f"OK  instance-seed uniqueness across {n} instances")


if __name__ == "__main__":
    test_determinism_byte_identity_all_conditions()
    test_rain_determinism_byte_identity()
    test_seed_changes_output()
    test_invalid_inputs_raise()
    test_instance_seed_collision_resistance()
    print("\nALL ARS DETERMINISM TESTS PASSED")
