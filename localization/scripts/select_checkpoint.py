"""Re-select each localization model's checkpoint by VALIDATION MEDIAN bearing error.

Why: the validation mean error is dominated by a heavy tail of large misses that barely changes during training
(about 33 degrees from epoch 0 to 24), so "best mean" picked an under-trained epoch-0 model. The median tracks the
typical clip and improves steadily. Selection still uses validation data only. Logged in DECISIONS.md.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "localization" / "results"

for fs in ["both", "timing", "spectral"]:
    d = ROOT / "checkpoints" / "localization" / fs
    if not (d / "last.pt").exists():
        continue
    hist = torch.load(d / "last.pt", map_location="cpu", weights_only=False)["history"]
    best = min(hist, key=lambda h: h["val_bearing_median_deg"])
    ck = torch.load(d / f"epoch_{best['epoch']:02d}.pt", map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "epoch": best["epoch"], "val": {k: v for k, v in best.items() if k.startswith("val")},
                "selected_by": "lowest validation median bearing error"}, d / "best.pt")
    p = RES / f"training_history_{fs}.json"
    if p.exists():
        s = json.loads(p.read_text())
        s["selected_epoch"], s["selected_by"] = best["epoch"], "lowest validation median bearing error"
        p.write_text(json.dumps(s, indent=2))
    print(f"{fs}: epoch {best['epoch']}  val median {best['val_bearing_median_deg']:.2f}  "
          f"val MAE {best['val_bearing_mae_deg']:.2f}  val range MAE {best['val_range_mae_m']:.1f} m", file=sys.stdout)
