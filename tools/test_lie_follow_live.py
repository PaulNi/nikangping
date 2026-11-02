# -*- coding: utf-8 -*-
"""
Live LieAlert screen test — capture primary monitor, follow shape, show overlay.

Usage:
  .venv\\Scripts\\python.exe tools\\test_lie_follow_live.py
  .venv\\Scripts\\python.exe tools\\test_lie_follow_live.py --force

Keys (preview window focused):
  Q / Esc  quit
  F        force-start follow session (skip OCR wait)
  R        reset session / unlock
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.LieAlert import LieAlert  # noqa: E402


class _MonitorCapture:
    """Minimal capture stub so LieAlert can map frame→screen coords."""

    def __init__(self, region: dict):
        self.region = region
        self.window_title = "LIVE_MONITOR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="start follow immediately")
    ap.add_argument(
        "--monitor",
        type=int,
        default=-1,
        help="mss monitor index (-1 = primary)",
    )
    ap.add_argument("--no-mouse", action="store_true", help="overlay only, do not move cursor")
    ap.add_argument("--scale", type=float, default=0.55, help="preview window scale")
    args = ap.parse_args()

    import mss

    cfg_path = os.path.join(ROOT, "config", "config_custom.yaml")
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    # Live preview: keep alerts quieter
    la_cfg = dict(cfg)
    la = LieAlert(la_cfg)
    la.alert_cooldown_s = 9999.0  # no beep spam during preview

    sct = mss.mss()
    mon_idx = int(args.monitor)
    if mon_idx < 0:
        mon_idx = 1
        for i, m in enumerate(sct.monitors):
            if i == 0:
                continue
            if m.get("is_primary"):
                mon_idx = i
                break
    if mon_idx < 0 or mon_idx >= len(sct.monitors):
        print("bad --monitor; available:")
        for i, m in enumerate(sct.monitors):
            print(f"  {i}: {m}")
        return 1
    for i, m in enumerate(sct.monitors):
        mark = " <=" if i == mon_idx else ""
        print(f"monitor[{i}] {m}{mark}")
    mon = sct.monitors[mon_idx]
    region = {
        "left": int(mon["left"]),
        "top": int(mon["top"]),
        "width": int(mon["width"]),
        "height": int(mon["height"]),
    }
    capture = _MonitorCapture(region)
    # title_bar_height would double-offset; force 0 for raw monitor grab
    if "game_window" not in la.cfg:
        la.cfg["game_window"] = {}
    la.cfg["game_window"]["title_bar_height"] = 0

    win = "LieAlert LIVE (Q quit | F force | R reset)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("=" * 56)
    print("LieAlert LIVE — play your video fullscreen/windowed now")
    print(f"monitor={args.monitor} region={region}")
    print("Keys: Q=quit  F=force-start follow  R=reset")
    if args.force:
        print("--force: follow session armed immediately")
    print("=" * 56)

    frame_count = 0
    trail = []
    t_start = time.time()
    forced = False

    if args.force:
        now = time.time()
        la.visible = True
        la._was_visible = True
        la._session_start_t = now
        la._hold_until = now + 999
        delay = max(0.0, la.follow_start_delay_s)
        la._follow_ready_t = now + delay
        la._intro_block_until = now + delay
        la._reset_bg_model()
        forced = True
        print(f"[force] session start — follow in {delay:.1f}s")

    try:
        while True:
            shot = np.array(sct.grab(region))
            frame = np.ascontiguousarray(shot[:, :, :3])  # BGRA→BGR
            frame_count += 1
            now = time.time()

            if forced:
                # Keep sticky hold alive; run follow path only
                la._hold_until = now + 30
                la.visible = True
                if now >= la._follow_ready_t:
                    if la._should_block_follow_for_intro(frame, now):
                        la.following = False
                        la.debug_pt = None
                    else:
                        if not args.no_mouse:
                            la._follow_shape(frame, capture, now)
                        else:
                            pt = la._locate_shape(frame, now)
                            if pt is not None:
                                la._locked = True
                                la._last_seen_t = now
                                la.debug_pt = (int(pt[0]), int(pt[1]))
                                la.following = True
                            else:
                                la.following = False
                                la.debug_pt = None
            else:
                # Full production path (OCR detect → delay → follow)
                la.maybe_check(frame, frame_count, kb=None, capture=capture)

            vis = frame.copy()
            pt = la.debug_pt
            if pt is not None:
                trail.append((int(pt[0]), int(pt[1])))
                if len(trail) > 50:
                    trail.pop(0)
                for i in range(1, len(trail)):
                    cv2.line(vis, trail[i - 1], trail[i], (0, 200, 255), 2)
                cv2.circle(vis, pt, 18, (0, 0, 255), 2)
                cv2.drawMarker(vis, pt, (0, 255, 0), cv2.MARKER_CROSS, 26, 2)

            name = getattr(la, "_last_shape_name", "") or "-"
            status = (
                f"vis={int(la.visible)} follow={int(la.following)} "
                f"locked={int(la._locked)} name={name} "
                f"pt={pt} t={now - t_start:.0f}s"
            )
            cv2.putText(
                vis, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4
            )
            cv2.putText(
                vis, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )
            if la._stale_ban_xy is not None and now < float(la._stale_ban_until):
                bx, by = int(la._stale_ban_xy[0]), int(la._stale_ban_xy[1])
                r = int(la.stale_ban_radius_px)
                cv2.circle(vis, (bx, by), r, (0, 128, 255), 1)
                cv2.putText(
                    vis, "BAN", (bx + r, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1
                )

            sc = max(0.25, min(1.0, float(args.scale)))
            if sc < 0.99:
                vis_s = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            else:
                vis_s = vis
            cv2.imshow(win, vis_s)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("f"), ord("F")):
                la._session_start_t = now
                la._hold_until = now + 999
                la.visible = True
                la._was_visible = True
                la._locked = False
                la._track_xy = None
                la._lock_anchor_xy = None
                la._stale_ban_xy = None
                delay = max(0.0, la.follow_start_delay_s)
                la._follow_ready_t = now + delay
                la._intro_block_until = now + delay
                la._reset_bg_model()
                forced = True
                trail.clear()
                print(f"[F] force session — follow in {delay:.1f}s")
            if key in (ord("r"), ord("R")):
                forced = False
                la.visible = False
                la._was_visible = False
                la._hold_until = 0.0
                la._session_start_t = 0.0
                la._locked = False
                la._track_xy = None
                la._lock_anchor_xy = None
                la._stale_ban_xy = None
                la.following = False
                la.debug_pt = None
                la._reset_bg_model()
                trail.clear()
                print("[R] reset")

            if frame_count % 60 == 0:
                print(
                    f"[{now - t_start:5.1f}s] vis={la.visible} follow={la.following} "
                    f"locked={la._locked} pt={pt} name={name}"
                )
    finally:
        cv2.destroyAllWindows()
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
