"""Train the bearing (+ range) estimator on DDL.

Uses the shared safety harness (src/acousticwatch/trainloop.py): thermal pause > 80 C until < 70 C (polled every
epoch and between batches), fp16 autocast, batch 64, checkpoint every epoch, batch halving on OOM.

Loss: (1 - cos(bearing error))  +  0.5 * |log(range_pred) - log(range_true)|
Augmentation (train only): one global gain U(-6, +6) dB, per-channel gain jitter +-1 dB (mic mismatch),
spatially-white noise at SNR U(0, 30) dB with p = 0.5.

Usage: python localization/scripts/train_loc.py --features both|timing|spectral [--epochs 30]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "localization"), str(ROOT / "src")]
from acousticwatch.trainloop import fit                          # noqa: E402
from loc.data import PROC, load_audio                            # noqa: E402
from loc.features import Spectral, gcc_phat                      # noqa: E402
from loc.model import DOANet                                     # noqa: E402

RES = ROOT / "localization" / "results"
DEV = "cuda"


def targets(df):
    b = np.radians(df.bearing.values)
    return (torch.tensor(np.stack([np.cos(b), np.sin(b)], 1), dtype=torch.float32),
            torch.tensor(np.log(df.range.values), dtype=torch.float32))


def batches(audio, idx, bs, shuffle, rng=None):
    idx = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(idx) - (bs - 1 if shuffle else 0), bs):
        b = idx[s:s + bs]
        yield b, torch.from_numpy(audio[b].astype(np.float32))


def augment(x):
    B, C, _ = x.shape
    g = 10 ** ((torch.empty(B, 1, 1, device=x.device).uniform_(-6, 6)
                + torch.empty(B, C, 1, device=x.device).uniform_(-1, 1)) / 20)
    x = x * g
    m = torch.rand(B, device=x.device) < 0.5
    if m.any():
        p = x[m].pow(2).mean(dim=(1, 2), keepdim=True)
        snr = torch.empty(int(m.sum()), 1, 1, device=x.device).uniform_(0, 30)
        x[m] = x[m] + torch.randn_like(x[m]) * torch.sqrt(p / 10 ** (snr / 10))
    return x


def featurize(x, spec):
    return gcc_phat(x), spec(x)


@torch.no_grad()
def predict(model, spec, audio, idx, bs=256, transform=None):
    model.eval()
    V, R = [], []
    for _, x in batches(audio, idx, bs, False):
        x = x.to(DEV)
        if transform is not None:
            x = transform(x)
        g, s = featurize(x, spec)
        with torch.autocast("cuda", dtype=torch.float16):
            v, r = model(g, s)
        V.append(v.float().cpu()); R.append(r.float().cpu())
    V, R = torch.cat(V), torch.cat(R)
    bearing = (torch.rad2deg(torch.atan2(V[:, 1], V[:, 0])) % 360).numpy()
    return bearing, torch.exp(R).numpy()


def ang_err(p, t):
    d = np.abs(p - t) % 360
    return np.minimum(d, 360 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="both", choices=["both", "timing", "spectral"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    meta = pd.read_parquet(PROC / "meta.parquet")
    sub = meta[meta.split.isin(["train", "val"])].reset_index(drop=True)
    audio = load_audio(sub)
    tr_idx = np.where(sub.split == "train")[0]
    va_idx = np.where(sub.split == "val")[0]
    vec, logr = targets(sub)
    print(f"[loc:{args.features}] train {len(tr_idx)}  val {len(va_idx)}", flush=True)

    spec = Spectral().to(DEV)
    model = DOANet(args.features).to(DEV)
    # spectral normalisation stats from training clips only
    s_idx = rng.choice(tr_idx, min(4000, len(tr_idx)), replace=False)
    S = torch.cat([spec(x.to(DEV)) for _, x in batches(audio, np.sort(s_idx), 256, False)])
    model.s_mu.copy_(S.mean(dim=(0, 3), keepdim=True)); model.s_sd.copy_(S.std(dim=(0, 3), keepdim=True) + 1e-3)
    del S

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = math.ceil(len(tr_idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    def make_loader(bs):
        return batches(audio, tr_idx, bs, True, rng)

    def train_step(data):
        b, x = data
        x = augment(x.to(DEV, non_blocking=True))
        g, s = featurize(x, spec)
        tv, tr_ = vec[b].to(DEV), logr[b].to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            v, r = model(g, s)
        loss = (1 - (v.float() * tv).sum(1)).mean() + 0.5 * (r.float() - tr_).abs().mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
        if sched.last_epoch < sched.total_steps - 1:
            sched.step()
        return loss.item()

    def evaluate():
        b, r = predict(model, spec, audio, va_idx)
        e = ang_err(b, sub.bearing.values[va_idx])
        re = np.abs(r - sub.range.values[va_idx])
        return {"val_bearing_mae_deg": float(e.mean()), "val_bearing_median_deg": float(np.median(e)),
                "val_range_mae_m": float(re.mean())}

    summary = fit(model=model, opt=opt, sched=sched, make_loader=make_loader, train_step=train_step,
                  evaluate=evaluate, ckpt_dir=ROOT / "checkpoints" / "localization" / args.features,
                  epochs=args.epochs, batch=args.batch, select="val_bearing_mae_deg", mode="min",
                  resume=args.resume, log_prefix=f"[loc:{args.features}] ")
    summary["args"] = vars(args)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"training_history_{args.features}.json").write_text(json.dumps(summary, indent=2))
    print(f"[loc:{args.features}] done; best val bearing MAE {summary['best_val_bearing_mae_deg']:.2f} deg")


if __name__ == "__main__":
    main()
