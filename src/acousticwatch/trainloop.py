"""Shared, safety-wrapped training loop used by both stages (detection and localization).

Handles the project's GPU rules in one place:
  * thermal guard polled between batches (and at least once per epoch): pause > 80 C until < 70 C
  * fp16 autocast + GradScaler (the caller's step function runs inside autocast)
  * checkpoint every epoch (epoch_XX.pt + last.pt, resumable) and best.pt by a validation metric
  * on CUDA out-of-memory: halve the batch size, roll back to the last finished epoch, retry
"""
import json
import time
from pathlib import Path

import torch

from . import gpu_guard


def fit(*, model, opt, sched, make_loader, train_step, evaluate, ckpt_dir, epochs, batch,
        select="val_loss", mode="min", resume=False, min_batch=8, log_prefix=""):
    """train_step(batch_data) -> float loss (does forward/backward/step itself).
    evaluate() -> dict of validation metrics (must contain `select`).
    Returns a summary dict (history, best, oom events, thermal pauses)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gpu_guard.ensure_power_limit()
    guard = gpu_guard.ThermalGuard()
    history, oom_events = [], []
    best = float("inf") if mode == "min" else -float("inf")
    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    start = 0
    if resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        start, history, best, batch = ck["epoch"] + 1, ck["history"], ck["best"], ck["batch"]
        print(f"{log_prefix}resumed after epoch {ck['epoch']}", flush=True)

    epoch = start
    while epoch < epochs:
        guard._last = 0.0
        guard.check(context=f"{log_prefix}epoch {epoch} start")   # explicit per-epoch temperature poll
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        try:
            for step, data in enumerate(make_loader(batch)):
                guard.check(context=f"{log_prefix}epoch {epoch} step {step}")
                tot += train_step(data); n += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new = max(min_batch, batch // 2)
            gpu_guard.log_event("oom_reduce_batch", where=log_prefix, epoch=epoch, old_batch=batch, new_batch=new)
            oom_events.append({"epoch": epoch, "old": batch, "new": new})
            if new == batch:
                raise
            batch = new
            if (ckpt_dir / "last.pt").exists():
                ck = torch.load(ckpt_dir / "last.pt", map_location="cuda")
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                if sched is not None and ck.get("sched"):
                    sched.load_state_dict(ck["sched"])
            continue
        model.eval()
        metrics = evaluate()
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"], "batch": batch,
               "seconds": round(time.time() - t0, 1), "max_gpu_temp": guard.max_temp, **metrics}
        history.append(rec)
        print(log_prefix + json.dumps(rec), flush=True)
        is_best = better(metrics[select], best)
        if is_best:
            best = metrics[select]
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict() if sched is not None else None,
                 "epoch": epoch, "history": history, "best": best, "batch": batch}
        torch.save(state, ckpt_dir / f"epoch_{epoch:02d}.pt")
        torch.save(state, ckpt_dir / "last.pt")
        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": metrics}, ckpt_dir / "best.pt")
        epoch += 1
    return {"history": history, "best_" + select: best, "final_batch": batch, "oom_events": oom_events,
            "thermal_pauses": guard.pauses, "max_gpu_temp": guard.max_temp}
