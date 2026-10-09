"""Array geometry for the DDL 8-mic array + a classical SRP-PHAT direction finder.

The DDL paper describes the array only qualitatively ("8 channels, microphone heads at the four corners of a box,
in pairs"); no coordinates are published. We therefore *calibrate* the geometry from TRAINING data only:

  far-field model:  tau_ij = (p_i - p_j) . u(bearing, elevation) / c
     tau_ij  measured GCC-PHAT time-difference for pair (i, j)   (positive = channel j lags channel i)
     u       unit vector from the array towards the drone (east, north, up); bearing is clockwise from north
             (GPS forward azimuth), elevation = atan(altitude / horizontal range)
     c       speed of sound from the logged air temperature: 331.3 * sqrt(T / 273.15)
  Positions p (8 x 3, mic 0 fixed at the origin) are fitted by robust (Huber) least squares on clips whose
  GCC-PHAT peaks are strong. The fitted geometry also absorbs the array's (unknown) compass orientation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .features import MAX_LAG, PAIRS, SR, UPS

ROOT = Path(__file__).resolve().parents[2]
GEOM_PATH = ROOT / "localization" / "results" / "array_geometry.json"


def unit_vectors(bearing_deg, elev_deg):
    b, e = torch.deg2rad(torch.as_tensor(bearing_deg, dtype=torch.float32)), \
        torch.deg2rad(torch.as_tensor(elev_deg, dtype=torch.float32))
    return torch.stack([torch.cos(e) * torch.sin(b), torch.cos(e) * torch.cos(b), torch.sin(e)], -1)


def speed_of_sound(temp_k):
    return 331.3 * torch.sqrt(torch.as_tensor(temp_k, dtype=torch.float32) / 273.15)


def predicted_tdoa(P, u, c):
    """P: (8,3) metres, u: (N,3), c: (N,) -> (N, 28) seconds."""
    i, j = zip(*PAIRS)
    d = P[list(i)] - P[list(j)]                     # (28, 3)
    return (u @ d.T) / c[:, None]


def peak_tdoa(gcc):
    """(N, 28, lags) GCC -> (TDOA seconds, peak height) with parabolic sub-bin refinement."""
    k = gcc.argmax(-1)
    L = gcc.shape[-1]
    kc = k.clamp(1, L - 2)
    y0, y1, y2 = (gcc.gather(-1, (kc + o)[..., None])[..., 0] for o in (-1, 0, 1))
    delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-9)
    lag = (kc.float() + delta.clamp(-0.5, 0.5) - (L // 2)) / (SR * UPS)
    return lag, gcc.max(-1).values


def fit_geometry(tdoa, weight, u, c, iters=3000, seed=0):
    torch.manual_seed(seed)
    P = (torch.randn(7, 3) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([P], lr=0.01)
    for it in range(iters):
        full = torch.cat([torch.zeros(1, 3), P])
        r = (predicted_tdoa(full, u, c) - tdoa) * SR       # residual in samples
        loss = (torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=2.0, reduction="none") * weight).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters // 2:
            for g in opt.param_groups:
                g["lr"] = 0.002
    full = torch.cat([torch.zeros(1, 3), P.detach()])
    full = full - full.mean(0)                              # centre the array
    r = ((predicted_tdoa(full, u, c) - tdoa) * SR).abs()
    return full, float(loss), r


def save_geometry(P, info):
    GEOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEOM_PATH.write_text(json.dumps({"positions_m": P.tolist(), **info}, indent=2))


def load_geometry():
    d = json.loads(GEOM_PATH.read_text())
    return torch.tensor(d["positions_m"]), d


class SRPPHAT:
    """Steered-response power over a (bearing x elevation) grid using GCC-PHAT and the calibrated geometry."""

    def __init__(self, P, c=340.0, elev_grid=range(0, 90, 5), dev="cuda"):
        b = torch.arange(360.0)
        e = torch.tensor(list(elev_grid), dtype=torch.float32)
        B, E = torch.meshgrid(b, e, indexing="ij")
        self.bearings, self.elevs = B.reshape(-1), E.reshape(-1)
        u = unit_vectors(self.bearings, self.elevs)
        tau = predicted_tdoa(P, u, torch.full((len(u),), c))                 # (D, 28) s
        self.idx = (tau * SR * UPS + MAX_LAG * UPS).round().long().clamp(0, 2 * MAX_LAG * UPS).to(dev)  # (D, 28)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, gcc):
        """gcc (N, 28, lags) -> bearing estimate in degrees (N,)"""
        g = gcc.to(self.dev).float()
        N = g.shape[0]
        # gather GCC values at each direction's expected lags: (N, D, 28)
        vals = torch.gather(g[:, None].expand(N, self.idx.shape[0], *g.shape[1:]), 3,
                            self.idx[None, :, :, None].expand(N, -1, -1, 1))[..., 0]
        best = vals.sum(-1).argmax(1).cpu()
        return self.bearings[best], self.elevs[best]


def angular_error_deg(pred, true):
    d = (torch.as_tensor(pred, dtype=torch.float32) - torch.as_tensor(true, dtype=torch.float32)) % 360
    return torch.minimum(d, 360 - d)


def elevation_deg(alt, rng):
    return np.degrees(np.arctan2(np.asarray(alt, float), np.maximum(np.asarray(rng, float), 1)))

