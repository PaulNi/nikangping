# -*- coding: utf-8 -*-
"""
Offline test: LieAlert white-acquire + smooth track.

RED box = center target pattern (tracker).
Same-shape decoys stay internal (flock velocity only) and are not drawn.

Usage:
  .venv\\Scripts\\python.exe tools\\test_lie_follow_video.py --video path\\to.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.LieAlert import LieAlert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        default=os.path.join("_debug_out", "bilibili", "BV1Ueht6rEig_game.mp4"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join("_debug_out", "lie_follow_bilibili_game.mp4"),
    )
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--skip-intro-block", action="store_true", default=True)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--scale", type=float, default=0.9)
    ap.add_argument("--delay-ms", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print("Video not found:", args.video)
        return 1

    cfg = yaml.safe_load(
        open(os.path.join(ROOT, "config", "config_custom.yaml"), "r", encoding="utf-8")
    )
    la = LieAlert(cfg)
    if args.skip_intro_block:
        la.follow_start_delay_s = 0.0
        la.intro_ocr_block_keywords = []
        la._follow_ready_t = 0.0
        la._intro_block_until = 0.0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open", args.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"video {w}x{h} @ {fps:.1f}fps frames={n}")
    print(
        "mode: RED=center white target"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sample_dir = os.path.join("_debug_out", "lie_follow_samples")
    os.makedirs(sample_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    t0 = time.time()
    la.visible = True
    la._was_visible = True
    la._session_start_t = t0
    la._follow_ready_t = t0
    la._hold_until = t0 + 999
    la._reset_bg_model()

    hit = 0
    locked_frames = 0
    names = {}
    frame_i = 0
    trail = []
    decoy_frames = 0
    win = "LieAlert (RED=center)"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if args.max_frames and frame_i > args.max_frames:
            break

        now = t0 + frame_i / max(1.0, fps)
        if frame_i <= 15:
            la._learn_background(frame, locked=False)
            pt = None
        else:
            la._follow_ready_t = 0.0
            la._intro_block_until = 0.0
            pt = la._locate_shape(frame, now)
            if pt is not None:
                la._last_seen_t = now
                la._locked = True
                hit += 1
            elif la._locked and la._track_xy is not None:
                pt = la._track_xy
            if la._locked:
                locked_frames += 1

        vis = frame.copy()
        name = getattr(la, "_last_shape_name", "") or "-"
        bbox = getattr(la, "debug_bbox", None)

        # Primary = track box if locked, else center-most pattern at open
        primary = bbox if (pt is not None and bbox is not None) else None
        primary, decoys = la.refresh_pattern_overlay(frame, primary_bbox=primary)

        if decoys:
            decoy_frames += 1

        if primary is not None:
            x, y, bw, bh = [int(v) for v in primary]
            # Keep engine bbox in sync when we picked center at open
            if bbox is None and pt is None:
                la.debug_bbox = (x, y, bw, bh)
            if pt is not None:
                names[name] = names.get(name, 0) + 1
                label = f"TARGET {name}"
            else:
                label = "TARGET center"
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            trail.append((x + bw // 2, y + bh // 2))
            if len(trail) > 40:
                trail.pop(0)
            for i in range(1, len(trail)):
                cv2.line(vis, trail[i - 1], trail[i], (0, 200, 255), 2)
            cv2.putText(
                vis,
                label,
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        status = (
            f"f={frame_i}/{n} locked={int(la._locked)} "
            f"decoys={len(decoys)} RED=white-target"
        )
        cv2.putText(vis, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(
            vis, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
        writer.write(vis)

        if n > 0:
            checkpoints = {max(1, n // 8), n // 4, n // 2, (3 * n) // 4}
        else:
            checkpoints = {30, 90, 150, 300}
        # Extra early samples + fade window (~7s white→tint)
        early = {20, 30, 45, 60, 90}
        fade_window = set(range(190, 280, 5)) | set(range(280, 400, 10)) | {
            200, 210, 215, 220, 230, 240, 250, 300, 320, 340, 360
        }
        if frame_i in checkpoints or frame_i in early or frame_i in fade_window or (
            pt is not None and frame_i % 60 == 0 and frame_i < 400
        ):
            cv2.imwrite(os.path.join(sample_dir, f"f{frame_i:05d}.jpg"), vis)

        if not args.no_preview:
            sc = max(0.3, min(1.0, float(args.scale)))
            show = (
                cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                if sc < 0.99
                else vis
            )
            cv2.imshow(win, show)
            key = cv2.waitKey(max(1, int(args.delay_ms))) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("quit by user")
                break

        if frame_i % 60 == 0 or (190 <= frame_i <= 260 and frame_i % 10 == 0):
            dv = getattr(la, "_decoy_vel", None)
            dvs = f"({dv[0]:.0f},{dv[1]:.0f})" if dv else "-"
            print(
                f"f={frame_i} t={frame_i/max(1.0,fps):.2f}s locked={int(la._locked)} "
                f"decoys={len(decoys)} name={name} decoy_vel={dvs}"
            )

    writer.release()
    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    usable = max(1, frame_i - 15)
    print("--- result ---")
    print("frames", frame_i, "detect_hits", hit, f"({100.0 * hit / usable:.1f}% after warmup)")
    print("locked_frames", locked_frames)
    print("by_name", names)
    print("frames_with_decoys", decoy_frames)
    print("out_video", os.path.abspath(args.out))
    print("samples", os.path.abspath(sample_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
