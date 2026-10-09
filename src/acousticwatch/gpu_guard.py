"""GPU safety: power-limit check, thermal pause, event logging.

Rules (from project brief):
  * power limit 220 W (nvidia-smi -pl 220; needs admin, so we only verify/try here)
  * if GPU temp > 80 C, pause work until it drops below 70 C
  * every event is appended to logs/gpu_events.jsonl
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENT_LOG = ROOT / "logs" / "gpu_events.jsonl"

PAUSE_ABOVE_C = 80
RESUME_BELOW_C = 70
POWER_LIMIT_W = 220


def log_event(kind, **info):
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": kind, **info}
    with open(EVENT_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[gpu_guard] {kind} {info}", flush=True)


def query():
    """Return dict(temp, power, power_limit, mem_used) or None if nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        t, p, pl, m = (float(x) for x in out.split(","))
        return {"temp": t, "power": p, "power_limit": pl, "mem_used": m}
    except Exception:
        return None


def ensure_power_limit():
    q = query()
    if q is None:
        log_event("nvidia_smi_unavailable")
        return
    if q["power_limit"] <= POWER_LIMIT_W + 1:
        log_event("power_limit_ok", power_limit=q["power_limit"])
        return
    r = subprocess.run(["nvidia-smi", "-pl", str(POWER_LIMIT_W)], capture_output=True, text=True)
    q2 = query() or {}
    log_event("power_limit_set_attempt", ok=r.returncode == 0,
              power_limit=q2.get("power_limit"), msg=(r.stdout + r.stderr).strip()[:200])


class ThermalGuard:
    """Call .check() between batches; it is cheap (polls nvidia-smi at most every `interval` s)."""

    def __init__(self, interval=5.0):
        self.interval = interval
        self._last = 0.0
        self.pauses = []          # list of dicts: start temp, seconds paused
        self.max_temp = 0.0

    def check(self, context=""):
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now
        q = query()
        if q is None:
            return
        self.max_temp = max(self.max_temp, q["temp"])
        if q["temp"] <= PAUSE_ABOVE_C:
            return
        start, t0 = q["temp"], time.time()
        log_event("thermal_pause_start", temp=start, context=context)
        while True:
            time.sleep(10)
            q = query()
            if q is not None and q["temp"] < RESUME_BELOW_C:
                break
        dur = round(time.time() - t0, 1)
        self.pauses.append({"start_temp": start, "seconds": dur, "context": context})
        log_event("thermal_pause_end", temp=q["temp"], seconds=dur, context=context)
        self._last = time.time()
