# -*- coding: utf-8 -*-
"""
Play a lie-detector clip fullscreen and move the real OS cursor onto the
tracked target each frame (same path as live bot).

Usage:
  .venv\\Scripts\\python.exe tools\\demo_lie_mouse_play.py
  .venv\\Scripts\\python.exe tools\\demo_lie_mouse_play.py --video _debug_out\\bilibili\\BV1Ueht6rEig_game.mp4
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


class _WinCapture:
    def __init__(self, left: int, top: int, title: str = "LieDemo"):
        self.region = {"left": left, "top": top, "width": 0, "height": 0}
        self.window_title = title


def _primary_size():
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def _letterbox(frame, sw: int, sh: int):
    """Fit frame into screen size with black bars; return canvas + offset/scale."""
    fh, fw = frame.shape[:2]
    scale = min(sw / fw, sh / fh)
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    ox = (sw - nw) // 2
    oy = (sh - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, ox, oy, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        default=os.path.join("_debug_out", "bilibili", "BV1Ueht6rEig_game.mp4"),
    )
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--no-mouse", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print("Video not found:", args.video)
        return 1

    cfg = yaml.safe_load(
        open(os.path.join(ROOT, "config", "config_custom.yaml"), "r", encoding="utf-8")
    )
    la = LieAlert(cfg)
    la.alert_cooldown_s = 9999.0
    la.follow_start_delay_s = 0.0
    la.intro_ocr_block_keywords = []
    if "game_window" not in la.cfg:
        la.cfg["game_window"] = {}
    la.cfg["game_window"]["title_bar_height"] = 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open", args.video)
        return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sw, sh = _primary_size()
    print(f"video {vw}x{vh} @ {fps:.1f}fps frames={n}  screen={sw}x{sh}")
    print("Fullscreen play + real mouse follow. Q/Esc = quit")

    win = "LieAlert DEMO mouse-follow"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    capture = _WinCapture(0, 0, win)
    capture.region = {"left": 0, "top": 0, "width": sw, "height": sh}
    t0 = time.time()
    la.visible = True
    la._was_visible = True
    la._session_start_t = t0
    la._follow_ready_t = t0
    la._intro_block_until = 0.0
    la._hold_until = t0 + 999
    la._mouse_yielded = False
    la._reset_bg_model()

    frame_i = 0
    delay = max(1, int(round(1000.0 / max(1.0, fps * max(0.2, float(args.speed))))))
    moved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            now = t0 + frame_i / max(1.0, fps)
            vis, ox, oy, scale = _letterbox(frame, sw, sh)

            if frame_i <= 12:
                la._learn_background(frame, locked=False)
                la.following = False
                la.debug_pt = None
            else:
                la._follow_ready_t = 0.0
                la._intro_block_until = 0.0
                pt = la._locate_shape(frame, now)
                if pt is not None:
                    la._last_seen_t = now
                    la._locked = True
                    la.debug_pt = (int(pt[0]), int(pt[1]))
                    la.following = True
                    la._hold_until = now + 8.0
                    if not args.no_mouse and not la._mouse_yielded:
                        sx = ox + float(pt[0]) * scale
                        sy = oy + float(pt[1]) * scale
                        la._move_mouse_to_frame_point((sx, sy), capture)
                        moved += 1
                elif la._locked and la._track_xy is not None:
                    pt = la._track_xy
                    la.debug_pt = (int(pt[0]), int(pt[1]))
                    la.following = True
                    if not args.no_mouse and not la._mouse_yielded:
                        sx = ox + float(pt[0]) * scale
                        sy = oy + float(pt[1]) * scale
                        la._move_mouse_to_frame_point((sx, sy), capture)
                        moved += 1
                else:
                    la.following = False
                    la.debug_pt = None

            if la.debug_pt is not None:
                dx = int(ox + la.debug_pt[0] * scale)
                dy = int(oy + la.debug_pt[1] * scale)
                cv2.circle(vis, (dx, dy), 22, (0, 0, 255), 3)
                cv2.drawMarker(vis, (dx, dy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
            if getattr(la, "debug_bbox", None) is not None:
                x, y, bw, bh = [int(v) for v in la.debug_bbox]
                x1, y1 = int(ox + x * scale), int(oy + y * scale)
                x2, y2 = int(ox + (x + bw) * scale), int(oy + (y + bh) * scale)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

            name = getattr(la, "_last_shape_name", "") or "-"
            status = (
                f"f={frame_i}/{n} locked={int(la._locked)} follow={int(la.following)} "
                f"name={name} mouse_moves={moved} yielded={int(la._mouse_yielded)}"
            )
            cv2.putText(vis, status, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(
                vis, status, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("quit by user")
                break
            if frame_i % 30 == 0:
                print(
                    f"f={frame_i} t={frame_i/max(1.0,fps):.1f}s locked={int(la._locked)} "
                    f"name={name} moved={moved} pt={la.debug_pt}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"done frames={frame_i} mouse_moves={moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
