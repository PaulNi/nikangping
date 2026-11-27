# -*- coding: utf-8 -*-
"""Offline live-sim for heatmap-only lie follow (no real mouse / no YOLO)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.LieAlert import LieAlert  # noqa: E402


class _NullCapture:
    region = {"left": 0, "top": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--sample-dir", default="")
    args = ap.parse_args()
    if not os.path.isfile(args.video):
        print("missing video", args.video)
        return 1

    cfg = yaml.safe_load(
        open(os.path.join(ROOT, "config", "config_custom.yaml"), "r", encoding="utf-8")
    )
    la = LieAlert(cfg)
    print(
        f"mode auto_follow={la.auto_follow} heatmap={la.heatmap_enable} "
        f"heatmap_only={la.heatmap_only} hm={la._heatmap is not None}"
    )
    if not la.heatmap_enable or la._heatmap is None:
        print("heatmap tracker not loaded — enable heatmap in config")
        return 1

    # Never move the real cursor during offline sim.
    moves: list[tuple] = []

    def _fake_move(pt, capture):
        if la._mouse_yielded:
            return
        moves.append((float(pt[0]), float(pt[1]), time.time()))
        la._mouse_placed_xy = (float(pt[0]), float(pt[1]))
        la._mouse_xy = (float(pt[0]), float(pt[1]))
        la._mouse_last_move_t = time.time()
        la._mouse_last_move_ok = True

    la._move_mouse_to_frame_point = _fake_move  # type: ignore[method-assign]
    la._set_cursor_screen = lambda *_a, **_k: True  # type: ignore[method-assign]
    la._warn_mouse_elevation_block = lambda *_a, **_k: None  # type: ignore

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("cannot open", args.video)
        return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = args.out or os.path.join(
        "_debug_out", f"heatmap_only_sim_{os.path.splitext(os.path.basename(args.video))[0]}.mp4"
    )
    sample_dir = args.sample_dir or os.path.join(
        "_debug_out", "heatmap_only_sim_samples"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"video {w}x{h} @ {fps:.1f}fps frames={n} -> {out}")

    t0 = time.time()
    la.visible = True
    la._was_visible = True
    la._session_start_t = t0
    la._follow_ready_t = t0
    la._hold_until = t0 + 999
    la._intro_block_until = 0.0
    la.follow_start_delay_s = 0.0
    la._reset_bg_model()
    if la._heatmap is not None:
        la._heatmap.reset(t0)

    capture = _NullCapture()
    sources: Counter = Counter()
    move_n = 0
    title_n = 0
    open_logged = False
    i = 0
    wall0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if args.max_frames and i > args.max_frames:
            break
        now = t0 + i / max(fps, 1.0)
        if i <= 12:
            la._learn_background(frame, locked=False)

        before_moves = len(moves)
        la._follow_shape(frame, capture, now)
        if len(moves) > before_moves:
            move_n += 1

        hm = la._heatmap
        est = getattr(hm, "_last_est", None) if hm is not None else None
        src = "-"
        if est is not None:
            src = str(est.source)
            sources[src] += 1
        if getattr(la, "_lie_title_tl", None) is not None:
            title_n += 1
        if hm is not None and hm.opening_confirmed and not open_logged:
            open_logged = True
            print(f"opening_confirmed @ f={i} t={i/max(fps,1):.2f}s")

        vis = frame.copy()
        title_tl = getattr(la, "_lie_title_tl", None)
        if title_tl is not None:
            cv2.circle(vis, (int(title_tl[0]), int(title_tl[1])), 6, (255, 128, 0), 2)
        if moves:
            mx, my, _ = moves[-1]
            color = (0, 255, 255) if src in ("heatmap", "inject") else (0, 165, 255)
            cv2.drawMarker(vis, (int(mx), int(my)), color, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(vis, (int(mx), int(my)), 10, color, 2)
        if est is not None:
            cv2.putText(
                vis,
                f"HM {est.source} c={est.confidence:.2f} s={est.secondary_ratio:.2f} "
                f"move={int(len(moves)>before_moves)} crop={getattr(hm,'_crop_mode','-')}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if src in ("heatmap", "inject") else (0, 165, 255),
                2,
            )
        cv2.putText(
            vis,
            f"f={i}/{n} open={int(bool(hm and hm.opening_confirmed))} "
            f"title={title_tl} streak={getattr(hm,'_predict_streak',0) if hm else '-'}",
            (12, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(vis)

        t_sec = i / max(fps, 1.0)
        if i in {30, 60, 90, 120, 150, 180, 210, 240, 300, 360} or (
            6.0 <= t_sec <= 12.0 and i % 15 == 0
        ):
            cv2.imwrite(os.path.join(sample_dir, f"f{i:05d}.jpg"), vis)
        if i % 60 == 0:
            last = moves[-1] if moves else None
            print(
                f"f={i} open={int(bool(hm and hm.opening_confirmed))} "
                f"title={title_tl} src={src} "
                f"est={None if est is None else (round(est.x,1), round(est.y,1), round(est.confidence,3))} "
                f"moved={last}"
            )

    cap.release()
    writer.release()
    elapsed = time.time() - wall0
    print("---")
    print(f"frames={i} wall={elapsed:.1f}s fps={i/max(elapsed,1e-6):.1f}")
    print(f"title_tl_frames={title_n}/{i} mouse_move_frames={move_n}")
    print(f"sources={dict(sources)}")
    if hm is not None:
        print(
            f"opening_confirmed={hm.opening_confirmed} "
            f"predict_streak_end={hm._predict_streak}"
        )
    # Phase lock: early / fade / late move ratios
    if moves and fps > 0:
        def _ratio(a, b):
            # fraction of frames in [a,b) seconds that had a move this tick — approx via move timestamps
            return sum(1 for _, _, mt in moves if a <= (mt - t0) < b)

        print(
            f"moves_by_phase 0-6s={_ratio(0,6)} 6-11s={_ratio(6,11)} 11-15s={_ratio(11,15)}"
        )
    print("out", out)
    print("samples", sample_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
