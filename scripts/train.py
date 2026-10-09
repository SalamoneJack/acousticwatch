"""Train the drone-vs-not CNN.

Safety features (project rules, implemented in src/acousticwatch/trainloop.py): fp16 autocast, modest batch (128),
checkpoint every epoch (resumable),
automatic batch halving on CUDA OOM, thermal pause >80 C until <70 C, everything logged.

Usage:  python scripts/train.py --tag detect_v2 --aug distance [--epochs 20] [--batch 128] [--resume]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from acousticwatch.data import WindowDataset, batched, load_store  # noqa: E402
from acousticwatch.features import SR, WIN, LogMel, absorption_db_per_m  # noqa: E402
from acousticwatch.model import DroneCNN                           # noqa: E402
from acousticwatch.trainloop import fit                            # noqa: E402

ABS = absorption_db_per_m()
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


def mix_at_snr(sig, noise, snr_db):
    """Return sig + noise scaled so that 10*log10(P_sig / P_noise) == snr_db (per row)."""
    ps = sig.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
    pn = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
    k = torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return sig + k * noise


def distance_channel_aug(x):
    """'distance' augmentation (v2): simulate unknown range and unknown microphone/recording chain.
    Random air absorption for a 10-600 m path (ISO 9613-1), random spectral tilt (-6..+3 dB/octave around 1 kHz),
    random low-pass (2.5-7 kHz) and high-pass (50-300 Hz) band limits. Applied to both classes identically."""
    B, dev = x.shape[0], x.device
    f = torch.fft.rfftfreq(WIN, 1 / SR, device=dev)[None]
    gain_db = torch.zeros(B, f.shape[1], device=dev)
    m = (torch.rand(B, 1, device=dev) < 0.5).float()
    d = torch.exp(torch.empty(B, 1, device=dev).uniform_(np.log(10), np.log(600)))
    gain_db -= m * ABS.to(dev)[None] * d
    m = (torch.rand(B, 1, device=dev) < 0.5).float()
    slope = torch.empty(B, 1, device=dev).uniform_(-6, 3)
    gain_db += m * slope * torch.log2(f.clamp_min(20) / 1000)
    g = 10 ** (gain_db / 20)
    lp = torch.rand(B, 1, device=dev) < 0.3
    g = torch.where(lp & (f > torch.empty(B, 1, device=dev).uniform_(2500, 7000)), torch.zeros_like(g), g)
    hp = torch.rand(B, 1, device=dev) < 0.3
    g = torch.where(hp & (f < torch.empty(B, 1, device=dev).uniform_(50, 300)), torch.zeros_like(g), g)
    return torch.fft.irfft(torch.fft.rfft(x) * g, n=WIN)


def augment(x, bg_pool, gen, strong=False):
    """GPU waveform augmentation, applied identically to both classes so it can't become a label cue."""
    B, dev = x.shape[0], x.device
    if strong:
        x = distance_channel_aug(x)
    # 1) background mixing: add a random *negative* training window (p=0.5, SNR U(-5, 20) dB; v2: p=0.8, U(-10, 20))
    m = torch.rand(B, device=dev) < (0.8 if strong else 0.5)
    if m.any():
        j = torch.randint(0, bg_pool.shape[0], (int(m.sum()),), generator=gen).numpy()
        bg = torch.from_numpy(np.asarray(bg_pool[np.sort(j)], np.int16).astype(np.float32) / 32768).to(dev)
        snr = torch.empty(int(m.sum()), 1, device=dev).uniform_(-10 if strong else -5, 20)
        x[m] = mix_at_snr(x[m], bg, snr)
    # 2) white noise with p=0.3 at SNR U(5, 40) dB
    m = torch.rand(B, device=dev) < 0.3
    if m.any():
        snr = torch.empty(int(m.sum()), 1, device=dev).uniform_(5, 40)
        x[m] = mix_at_snr(x[m], torch.randn_like(x[m]), snr)
    # 3) random gain +-12 dB (removes any absolute-level cue)
    x = x * 10 ** (torch.empty(B, 1, device=dev).uniform_(-12, 12) / 20)
    return x


def spec_augment(s):
    """One frequency mask (<=8 mel bins) and one time mask (<=10 frames = 100 ms) per example."""
    B, _, Fq, T = s.shape
    dev = s.device
    f0 = torch.randint(0, Fq - 8, (B, 1), device=dev); fw = torch.randint(0, 9, (B, 1), device=dev)
    t0 = torch.randint(0, T - 10, (B, 1), device=dev); tw = torch.randint(0, 11, (B, 1), device=dev)
    fr = torch.arange(Fq, device=dev)[None]; tr = torch.arange(T, device=dev)[None]
    fm = ((fr >= f0) & (fr < f0 + fw))[:, None, :, None]
    tm = ((tr >= t0) & (tr < t0 + tw))[:, None, None, :]
    fill = s.mean(dim=(2, 3), keepdim=True)
    return torch.where(fm | tm, fill, s)


@torch.no_grad()
def evaluate(model, mel, wav, idx, labels, dev):
    model.eval()
    lab = dict(zip(idx, labels))
    probs, ys = [], []
    for b, x in batched(wav, idx, 512):
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(mel(x.to(dev))).float())
        probs.append(p.cpu().numpy()); ys.append([lab[i] for i in b])
    p, y = np.concatenate(probs), np.concatenate(ys)
    loss = float(F.binary_cross_entropy(torch.tensor(p).clamp(1e-6, 1 - 1e-6), torch.tensor(y, dtype=torch.float32)))
    pred = p >= 0.5
    rec = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    fpr = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {"val_loss": loss, "val_auc": float(roc_auc_score(y, p)), "val_recall@0.5": float(rec),
            "val_fpr@0.5": float(fpr), "val_bal_acc@0.5": float((rec + 1 - fpr) / 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples-per-epoch", type=int, default=80_000)
    ap.add_argument("--hard-neg-weight", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", default="base", choices=["base", "distance"])
    ap.add_argument("--tag", default="detect_v1")
    args = ap.parse_args()
    CKPT = ROOT / "checkpoints" / args.tag

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    CKPT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    dev = torch.device("cuda")

    man, wav = load_store()
    tr = man[man.split == "train"]
    va = man[man.split == "val"]
    print(f"train windows {len(tr)} (pos {int(tr.label.sum())}), val windows {len(va)} (pos {int(va.label.sum())})")

    # Balanced sampling: each class gets half of every epoch; inside negatives, hard negatives
    # (engines, aircraft, helicopters, power tools) are drawn `hard_neg_weight` times more often.
    w = np.where(tr.label == 1, 1.0 / (tr.label == 1).sum(), 0.0)
    neg_w = np.where(tr.hard_negative, args.hard_neg_weight, 1.0) * (tr.label == 0)
    w = w + neg_w / neg_w.sum()
    sampler = torch.utils.data.WeightedRandomSampler(w, args.samples_per_epoch, replacement=True)
    ds = WindowDataset(wav, tr.index.values, tr.label.values)
    # The background pool for mixing is restricted to negative *training* windows only.
    bg_pool_idx = np.sort(tr.index.values[tr.label.values == 0])

    class _Pool:
        shape = (len(bg_pool_idx),)
        def __getitem__(self, j):
            return wav[bg_pool_idx[j]]
    bg_pool = _Pool()
    gen = torch.Generator().manual_seed(args.seed)

    model = DroneCNN().to(dev)
    mel = LogMel().to(dev)
    # Per-mel-bin normalisation stats from the training split only (no val/test information).
    rs = np.random.default_rng(1).choice(tr.index.values, min(8000, len(tr)), replace=False)
    S = torch.cat([mel(x.to(dev)) for _, x in batched(wav, rs, 512)])
    model.mu.copy_(S.mean(dim=(0, 3), keepdim=True)[0:1]); model.sd.copy_(S.std(dim=(0, 3), keepdim=True)[0:1] + 1e-3)
    del S

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    def make_loader(bs):
        return torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=4,
                                           pin_memory=True, drop_last=True)

    def train_step(data):
        x, y, _ = data
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        s = spec_augment(mel(augment(x, bg_pool, gen, strong=args.aug == "distance")))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(s).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
        if sched.last_epoch < sched.total_steps - 1:   # extra steps can appear if the batch is halved on OOM
            sched.step()
        return loss.item()

    summary = fit(model=model, opt=opt, sched=sched, make_loader=make_loader, train_step=train_step,
                  evaluate=lambda: evaluate(model, mel, wav, va.index.values, va.label.values, dev),
                  ckpt_dir=CKPT, epochs=args.epochs, batch=args.batch, select="val_loss", mode="min",
                  resume=args.resume, log_prefix=f"[{args.tag}] ")
    summary["args"] = vars(args)
    (RESULTS / f"training_history_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print("done; best val loss", summary["best_val_loss"])


if __name__ == "__main__":
    main()
