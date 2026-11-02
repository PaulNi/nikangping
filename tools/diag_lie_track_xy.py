"""Print track xy/name over time for a lie clip (no video encode)."""
import os
import sys
import time

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine.LieAlert import LieAlert  # noqa: E402


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "_debug_out/lie_clips/lie_20261101_200132.mp4"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    cfg = yaml.safe_load(
        open(os.path.join(ROOT, "config", "config_custom.yaml"), "r", encoding="utf-8")
    )
    # Faster diag: skip fade YOLO load
    if isinstance(cfg.get("lie_alert"), dict):
        cfg["lie_alert"]["fade_yolo_model"] = ""
        cfg["lie_alert"]["shape_peak_model"] = ""

    la = LieAlert(cfg)
    la.follow_start_delay_s = 0.0
    la.intro_ocr_block_keywords = []
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0 = time.time()
    la.visible = True
    la._session_start_t = t0
    la._follow_ready_t = t0
    la._hold_until = t0 + 999
    la._reset_bg_model()

    prev = None
    for i in range(1, max_frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        now = t0 + i / fps
        if i <= 12:
            la._learn_background(frame, locked=False)
            continue
        la._follow_ready_t = 0.0
        pt = la._locate_shape(frame, now)
        if pt is not None:
            la._locked = True
            la._last_seen_t = now
        name = getattr(la, "_last_shape_name", "") or "-"
        xy = la._track_xy
        jump = 0.0
        if xy is not None and prev is not None:
            jump = ((xy[0] - prev[0]) ** 2 + (xy[1] - prev[1]) ** 2) ** 0.5
        if xy is not None:
            prev = xy
        if i % 15 == 0 or jump >= 18:
            xs = f"({xy[0]:.0f},{xy[1]:.0f})" if xy else "-"
            print(
                f"f={i:4d} t={i/fps:5.2f}s locked={int(la._locked)} "
                f"name={name:8s} xy={xs} jump={jump:.1f}"
            )
    cap.release()
    print("done")


if __name__ == "__main__":
    main()
