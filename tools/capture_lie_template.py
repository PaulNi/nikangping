# -*- coding: utf-8 -*-
"""One-shot: capture screen center band -> assets/lie_detector/lie.png
Use when 测谎 is visible for a 盛趣-accurate title crop. Does NOT solve captcha.
Run:  .venv\Scripts\python.exe tools\capture_lie_template.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

OUT = os.path.join("assets", "lie_detector", "lie.png")

def grab():
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = np.array(sct.grab(mon))
        return np.ascontiguousarray(shot[:, :, :3])

def main():
    img = grab()
    h, w = img.shape[:2]
    x0, x1 = int(w * 0.25), int(w * 0.75)
    y0, y1 = int(h * 0.25), int(h * 0.55)
    band = img[y0:y1, x0:x1]
    preview = os.path.join("assets", "lie_detector", "_last_center_band.png")
    cv2.imwrite(preview, band)
    strip = band[0 : max(40, band.shape[0] // 5), :]
    cv2.imwrite(OUT, strip)
    print("Wrote", OUT, strip.shape[1], "x", strip.shape[0])
    print("Preview:", preview)
    print("If this is not the 测谎 title, open preview and manually crop 「测谎」 over lie.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
