"""ars_transforms.py — Adversarial Robustness Suite (ARS) pixel-space transforms.

Self-contained perturbation module for Protocol E-03 (ARS), Innovation #3.
Operates on the canonical Stage-2 (T, 3, H, W) float32 tensor in [0, 1]
BEFORE the Temporal Difference channel is appended and BEFORE the
ImageNet+TD normalisation in eval_v24.infer_framecache().

Five conditions x three severities. Every (clip_id, condition, severity)
triple maps deterministically (SHA-256) to a 63-bit instance seed;
identical seeds produce byte-identical perturbed outputs.

Protocol bindings (E-03-B):
  * Red Flag 3 — camera_shake affine fill=0.0 (explicit float zero) to
    match the [0,1] tensor range and avoid dtype promotion in
    torchvision.transforms.v2.functional.affine.

References (per E-03 §A2):
  * fog          : Koschmieder 1924; Narasimhan & Nayar 2002.
  * rain         : Garg & Nayar 2007 (streak luminance ~ background + Delta).
  * low_light    : Reinhard et al. 2002 (gamma-compression sensor model).
  * motion_blur  : Boracchi & Foi 2012 (line kernel ~ motion magnitude).
  * camera_shake : Koehler et al. 2012 (small affine ~ handheld jitter).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import affine as tv_affine


CONDITIONS: Tuple[str, ...] = (
    "fog",
    "rain",
    "low_light",
    "motion_blur",
    "camera_shake",
)
SEVERITIES: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")


SEVERITY_TABLE: Dict[Tuple[str, str], Dict[str, float]] = {
    ("fog",          "LOW"):    {"fog_coef": 0.30, "alpha_coef": 0.08},
    ("fog",          "MEDIUM"): {"fog_coef": 0.55, "alpha_coef": 0.08},
    ("fog",          "HIGH"):   {"fog_coef": 0.80, "alpha_coef": 0.08},
    ("rain",         "LOW"):    {"density": 0.02, "drop_length": 10, "blur_value": 3, "slant_max_deg": 5.0},
    ("rain",         "MEDIUM"): {"density": 0.05, "drop_length": 20, "blur_value": 5, "slant_max_deg": 8.0},
    ("rain",         "HIGH"):   {"density": 0.08, "drop_length": 30, "blur_value": 7, "slant_max_deg": 10.0},
    ("low_light",    "LOW"):    {"gain": 0.50},
    ("low_light",    "MEDIUM"): {"gain": 0.33},
    ("low_light",    "HIGH"):   {"gain": 0.20},
    ("motion_blur",  "LOW"):    {"kernel_k": 5},
    ("motion_blur",  "MEDIUM"): {"kernel_k": 9},
    ("motion_blur",  "HIGH"):   {"kernel_k": 15},
    ("camera_shake", "LOW"):    {"t_max_px": 3,  "r_max_deg": 1.0},
    ("camera_shake", "MEDIUM"): {"t_max_px": 6,  "r_max_deg": 2.0},
    ("camera_shake", "HIGH"):   {"t_max_px": 10, "r_max_deg": 3.0},
}


_AIRLIGHT: float = 0.9          # Koschmieder atmospheric airlight A
_STREAK_LUM: float = 0.85       # rain streak luminance
_CAMERA_FILL: float = 0.0       # Red Flag 3 binding decision (E-03-B)


def make_instance_seed(
    global_seed: int, clip_id: str, condition: str, severity: str
) -> int:
    """SHA-256-derived 63-bit deterministic seed per (clip, condition, severity).

    Args:
        global_seed: Project-wide integer seed (e.g. 42).
        clip_id:     Test split identifier (e.g. ``rear_collision__rear_collision_4``).
        condition:   One of :data:`CONDITIONS`.
        severity:    One of :data:`SEVERITIES`.

    Returns:
        Non-negative ``int`` in ``[0, 2**63 - 1]``, byte-identical across re-runs.
    """
    payload = f"{int(global_seed)}|{clip_id}|{condition}|{severity}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def apply_perturbation(
    clip: torch.Tensor, condition: str, severity: str, instance_seed: int
) -> torch.Tensor:
    """Apply ``condition`` at ``severity`` to a ``(T, 3, H, W)`` float tensor.

    Input is expected in ``[0, 1]``. Output is the same shape, same dtype,
    clipped to ``[0, 1]``.
    """
    if clip.dim() != 4 or clip.size(1) != 3:
        raise ValueError(f"Expected (T,3,H,W); got {tuple(clip.shape)}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity {severity!r}; expected one of {SEVERITIES}")

    params = SEVERITY_TABLE[(condition, severity)]
    if condition == "fog":
        out = _apply_fog(clip, params, instance_seed)
    elif condition == "rain":
        out = _apply_rain(clip, params, instance_seed)
    elif condition == "low_light":
        out = _apply_low_light(clip, params, instance_seed)
    elif condition == "motion_blur":
        out = _apply_motion_blur(clip, params, instance_seed)
    elif condition == "camera_shake":
        out = _apply_camera_shake(clip, params, instance_seed)
    else:
        raise AssertionError("unreachable")
    return out.clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Fog — per-clip deterministic Koschmieder scattering.
# --------------------------------------------------------------------------- #
def _apply_fog(
    clip: torch.Tensor, params: Dict[str, float], seed: int
) -> torch.Tensor:
    T, C, H, W = clip.shape
    rng = np.random.default_rng(seed)
    fog_coef = float(params["fog_coef"])

    cx = float(rng.uniform(0.2, 0.8)) * W
    cy = float(rng.uniform(0.0, 0.5)) * H
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    rmax = radial.max() if radial.max() > 0 else 1.0
    radial = radial / rmax

    noise_small = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)
    noise_t = torch.from_numpy(noise_small)[None, None]
    noise = F.interpolate(
        noise_t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    depth = 0.6 * radial + 0.4 * noise

    beta = 2.0 + 4.0 * fog_coef
    transmission = np.exp(-beta * depth * fog_coef).astype(np.float32)
    t_torch = torch.from_numpy(transmission)[None, None, :, :]
    out = clip * t_torch + _AIRLIGHT * (1.0 - t_torch)
    return out


# --------------------------------------------------------------------------- #
# Rain — per-clip deterministic streak overlay (frame-coherent shift).
# --------------------------------------------------------------------------- #
def _apply_rain(
    clip: torch.Tensor, params: Dict[str, float], seed: int
) -> torch.Tensor:
    T, C, H, W = clip.shape
    rng = np.random.default_rng(seed)
    density = float(params["density"])
    drop_length = int(params["drop_length"])
    blur_value = int(params["blur_value"])
    slant_max = float(params["slant_max_deg"])

    n_streaks = max(1, int(density * H * W / max(drop_length, 1)))
    slant_deg = float(rng.uniform(-slant_max, slant_max))
    dx_per_px = math.sin(math.radians(slant_deg))

    mask = np.zeros((H, W), dtype=np.float32)
    x0s = rng.integers(0, W, size=n_streaks)
    y0s = rng.integers(0, H, size=n_streaks)
    for x0, y0 in zip(x0s, y0s):
        for k in range(drop_length):
            x = int(round(int(x0) + k * dx_per_px))
            y = int(y0) + k
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1.0

    if blur_value > 1:
        b = int(blur_value)
        pad = b // 2
        mask_t = torch.from_numpy(mask)[None, None]
        mask_t = F.avg_pool2d(
            F.pad(mask_t, (pad, pad, pad, pad), mode="replicate"),
            kernel_size=b, stride=1,
        )
        mask = np.minimum(mask_t[0, 0].numpy() * b, 1.0)

    out = clip.clone()
    dy_per_frame = max(1, drop_length // 4)
    for t in range(T):
        shifted = np.roll(mask, shift=t * dy_per_frame, axis=0)
        m_t = torch.from_numpy(shifted).unsqueeze(0)
        out[t] = out[t] * (1.0 - m_t) + _STREAK_LUM * m_t
    return out


# --------------------------------------------------------------------------- #
# Low-light — gamma compression x' = x^gamma, gamma = 1/gain.
# --------------------------------------------------------------------------- #
def _apply_low_light(
    clip: torch.Tensor, params: Dict[str, float], seed: int
) -> torch.Tensor:
    gain = float(params["gain"])
    if gain <= 0.0:
        raise ValueError(f"gain must be > 0; got {gain}")
    gamma = 1.0 / gain
    return clip.clamp(0.0, 1.0).pow(gamma)


# --------------------------------------------------------------------------- #
# Motion blur — per-frame directional line-kernel convolution.
# --------------------------------------------------------------------------- #
def _line_kernel(k: int, angle_deg: float) -> torch.Tensor:
    if k % 2 == 0 or k < 3:
        raise ValueError(f"kernel size must be odd >= 3; got {k}")
    kernel = np.zeros((k, k), dtype=np.float32)
    angle = math.radians(angle_deg)
    cx = cy = (k - 1) / 2.0
    dx, dy = math.cos(angle), math.sin(angle)
    half = (k - 1) / 2.0
    for s in np.linspace(-half, half, max(k * 4, 2)):
        x = int(round(cx + s * dx))
        y = int(round(cy + s * dy))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[k // 2, k // 2] = 1.0
    return torch.from_numpy(kernel)


def _apply_motion_blur(
    clip: torch.Tensor, params: Dict[str, float], seed: int
) -> torch.Tensor:
    T, C, H, W = clip.shape
    k = int(params["kernel_k"])
    rng = np.random.default_rng(seed)
    angles_deg = rng.uniform(0.0, 180.0, size=T)

    out = torch.empty_like(clip)
    pad = k // 2
    for t in range(T):
        kern = _line_kernel(k, float(angles_deg[t])).to(clip.dtype)
        kern_3 = kern.view(1, 1, k, k).expand(C, 1, k, k).contiguous()
        x = clip[t:t + 1]
        x_p = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.conv2d(x_p, kern_3, groups=C)
        out[t] = y[0]
    return out


# --------------------------------------------------------------------------- #
# Camera shake — per-frame affine jitter, fill=0.0 (Red Flag 3 binding).
# --------------------------------------------------------------------------- #
def _apply_camera_shake(
    clip: torch.Tensor, params: Dict[str, float], seed: int
) -> torch.Tensor:
    T, C, H, W = clip.shape
    t_max = int(params["t_max_px"])
    r_max = float(params["r_max_deg"])
    rng = np.random.default_rng(seed)

    out = torch.empty_like(clip)
    for t in range(T):
        tx = int(rng.integers(-t_max, t_max + 1))
        ty = int(rng.integers(-t_max, t_max + 1))
        ang = float(rng.uniform(-r_max, r_max))
        out[t] = tv_affine(
            clip[t],
            angle=ang,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=[_CAMERA_FILL] * C,
        )
    return out
