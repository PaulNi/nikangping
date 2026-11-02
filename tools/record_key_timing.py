"""Poll game keys 20s with GetAsyncKeyState; auto-stop and save."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import ctypes
import sys

OUT_DIR = Path(__file__).resolve().parents[1] / "_debug_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DURATION = 20.0

VK = {
    "a": 0x41, "s": 0x53, "z": 0x5A, "w": 0x57, "x": 0x58,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "shift": 0x10, "ctrl": 0x11, "end": 0x23,
    "f1": 0x70, "f2": 0x71,
}
user32 = ctypes.windll.user32

def down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)

def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"key_timing_{stamp}.json"
    live = OUT_DIR / "key_timing_live.json"
    log = OUT_DIR / "key_timing_run.log"

    def logline(msg: str) -> None:
        print(msg, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log.write_text("", encoding="utf-8")
    for i in range(3, 0, -1):
        print(f"Countdown {i}...", flush=True)
        time.sleep(1)
    logline(f"RECORD {DURATION:.0f}s — PLAY NOW")
    logline(f"save -> {path}")

    prev = {k: False for k in VK}
    last_t = {}
    intervals = defaultdict(list)
    sequence = []
    t0 = time.perf_counter()
    last_print = 0.0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= DURATION:
            break
        now = time.perf_counter()
        if now - last_print >= 1.0:
            print(f"... {DURATION - elapsed:.0f}s left  events={len(sequence)}", flush=True)
            last_print = now
        for name, vk in VK.items():
            is_down = down(vk)
            if is_down and not prev[name]:
                gap = None
                if name in last_t:
                    gap = now - last_t[name]
                    intervals[name].append(gap)
                last_t[name] = now
                sequence.append({"t": round(now - t0, 4), "key": name, "gap": None if gap is None else round(gap, 4)})
                msg = f"+{now - t0:6.2f}s  {name}" + (f"  gap={gap:.3f}s" if gap is not None else "")
                logline(msg)
            prev[name] = is_down
        time.sleep(0.008)

    summary = {}
    for k, gaps in sorted(intervals.items()):
        if gaps:
            summary[k] = {
                "count": len(gaps),
                "min": round(min(gaps), 4),
                "max": round(max(gaps), 4),
                "avg": round(statistics.mean(gaps), 4),
                "median": round(statistics.median(gaps), 4),
            }
    payload = {"reason": "timed_20s", "summary": summary, "sequence": sequence}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    live.write_text(text, encoding="utf-8")
    logline("=== Summary ===")
    if not summary:
        logline("NO KEYS captured. Keep this black window visible, then play in game.")
    for k, s in summary.items():
        logline(f"  {k}: n={s['count']} avg={s['avg']} median={s['median']} min={s['min']} max={s['max']}")
    logline(f"DONE saved {path} events={len(sequence)}")
    return 0 if sequence else 2

if __name__ == "__main__":
    sys.exit(main() or 0)
