# -*- coding: utf-8 -*-
"""Capture floating nameplate into character/name_XXXXX.png

Usage (game window restored, character standing still):
  .venv\Scripts\python.exe tools\capture_nameplate.py 12138
  .venv\Scripts\python.exe tools\capture_nameplate.py 12139
"""
import os, sys, cv2, numpy as np, win32gui, win32ui
from ctypes import windll

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def find_hwnd():
    found = []
    def cb(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            if t:
                found.append((h, t))
    win32gui.EnumWindows(cb, None)
    for h, t in found:
        if any(k in t for k in ("冒险", "Maple", "新枫", "Nexon")):
            return h, t
    return (found[0] if found else (None, None))

def grab_client(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        raise RuntimeError("Game window minimized/invalid client size — restore it first")
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(bmp)
    windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    bits = bmp.GetBitmapBits(True)
    img = np.frombuffer(bits, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].copy()
    win32gui.DeleteObject(bmp.GetHandle())
    saveDC.DeleteDC(); mfcDC.DeleteDC(); win32gui.ReleaseDC(hwnd, hwndDC)
    return img

def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "12138"
    out = os.path.join("character", f"name_{suffix}.png")
    # Prefer matching existing template region, then let user confirm
    hwnd, title = find_hwnd()
    print("window:", title)
    if not hwnd:
        raise SystemExit("No game window")
    bgr = grab_client(hwnd)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ref_path = "character/name.png"
    if not os.path.exists(ref_path):
        raise SystemExit("missing character/name.png as size reference")
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    th, tw = ref.shape[:2]
    res = cv2.matchTemplate(gray, ref, cv2.TM_SQDIFF_NORMED)
    minv, _, minl, _ = cv2.minMaxLoc(res)
    x, y = minl
    print(f"best match score={minv:.4f} at {(x,y)} size={tw}x{th}")
    # Expand a bit vertically/horizontally for safer crop of new name
    pad_x, pad_y = 8, 4
    x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
    x1 = min(bgr.shape[1], x + tw + pad_x); y1 = min(bgr.shape[0], y + th + pad_y)
    crop = bgr[y0:y1, x0:x1]
    os.makedirs("character", exist_ok=True)
    cv2.imwrite(out, crop)
    marked = bgr.copy()
    cv2.rectangle(marked, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(os.path.join("_debug_out", f"name_{suffix}_capture_marked.png"), marked)
    print("wrote", out, crop.shape)
    print("Also wrote _debug_out/name_%s_capture_marked.png — check the red box covers the nameplate." % suffix)

if __name__ == "__main__":
    main()