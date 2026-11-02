"""
Game window capturer for the "MapleStoryGo" game window on Windows.

Uses mss for screen capture + win32gui for window locating.
Pattern follows the macOS version in MapleStoryAutoLevelUp-main/src/input/GameWindowCapturorForMac.py
"""
import time
import threading

import mss
import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.common import load_image, get_game_window_title_by_token


class GameWindowCapturor:
    """
    Captures frames from the "MapleStoryGo" game window.

    Uses mss (cross-platform screen capture) + win32gui (window location).
    Capture runs in a daemon thread. Pattern matches the macOS reference.
    """

    def __init__(self, cfg: dict, test_image_path: str = ""):
        self.cfg = cfg
        self.frame = None
        self.lock = threading.Lock()
        self.is_terminated = False

        if test_image_path:
            # Offline test mode: load a static image
            logger.info(f"[GameWindowCapturor] Test mode: loading {test_image_path}")
            self.frame = load_image(test_image_path)
            logger.info(f"[GameWindowCapturor] Test image loaded: {self.frame.shape}")
            self.fps = 0
            self.window_title = ""
            return

        # Live capture mode
        import win32gui
        self._win32gui = win32gui

        self._title_token = str(cfg["game_window"]["title"])
        self.window_title = get_game_window_title_by_token(self._title_token)
        if not self.window_title:
            logger.error(
                f"[GameWindowCapturor] Unable to find game window for "
                f"'{self._title_token}' (skipped browser/IDE titles)"
            )
            self.frame = None
            return

        logger.info(f"[GameWindowCapturor] Found window: '{self.window_title}'")

        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0

        # Use mss to capture the screen region
        self.capture = mss.mss()

        # Get initial window region
        self.update_window_region()

        # Start capture in daemon thread (matches macOS reference exactly)
        threading.Thread(target=self.start_capture, daemon=True).start()

        # Busy-wait for first frame (matches macOS reference exactly)
        time.sleep(0.1)
        while self.frame is None:
            self.limit_fps()

    def start_capture(self):
        """Capture loop — runs in daemon thread."""
        while not self.is_terminated:
            self.update_window_region()
            self.capture_frame()
            self.limit_fps()

    def stop(self):
        """Stop the capture thread."""
        self.is_terminated = True
        logger.info("[GameWindowCapturor] Terminated")

    def rebind_window_title(self) -> bool:
        """Re-resolve game title if HWND vanished or was a bad match."""
        token = getattr(self, "_title_token", "") or str(
            (self.cfg.get("game_window") or {}).get("title", "")
        )
        title = get_game_window_title_by_token(token)
        if not title:
            return False
        if title != self.window_title:
            logger.info(
                f"[GameWindowCapturor] rebound window: '{self.window_title}' → '{title}'"
            )
            self.window_title = title
        return True

    def update_window_region(self):
        """Update the game window's screen region (client area in screen coords)."""
        try:
            import win32con
            hwnd = self._win32gui.FindWindow(None, self.window_title)
            if not hwnd:
                # Title may have been a stale Cursor/B站 match — try again.
                if self.rebind_window_title():
                    hwnd = self._win32gui.FindWindow(None, self.window_title)
            if not hwnd:
                self.region = None
            else:
                if self._win32gui.IsIconic(hwnd):
                    try:
                        self._win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        time.sleep(0.05)
                    except Exception:
                        pass
                # Re-check after restore attempt
                if self._win32gui.IsIconic(hwnd):
                    self.region = None
                else:
                    _cl, _ct, cr, cb = self._win32gui.GetClientRect(hwnd)
                    pt = self._win32gui.ClientToScreen(hwnd, (0, 0))
                    width = max(0, int(cr))
                    height = max(0, int(cb))
                    self.region = {
                        "left": int(pt[0]),
                        "top": int(pt[1]),
                        "width": width,
                        "height": height,
                    }
                    if width < 400 or height < 300:
                        wl, wt, wr, wb = self._win32gui.GetWindowRect(hwnd)
                        if (wr - wl) >= 400 and (wb - wt) >= 300 and wl > -10000:
                            self.region = {
                                "left": int(wl),
                                "top": int(wt),
                                "width": int(wr - wl),
                                "height": int(wb - wt),
                            }
                        else:
                            self.region = None
        except Exception as e:
            logger.debug(f"[GameWindowCapturor] region error: {e}")
            self.region = None

        if self.region is None:
            now = time.time()
            if now - getattr(self, "_t_last_region_err", 0) > 2.0:
                logger.error(f"Cannot find usable window region: {self.window_title} (minimized?)")
                self._t_last_region_err = now

    def capture_frame(self):
        """Capture the current game window region."""
        if self.region is None or self.region.get("width", 0) < 400 or self.region.get("height", 0) < 300:
            # Invalidate so hunt will not chase on a frozen old frame
            with self.lock:
                self.frame = None
            return

        try:
            img = self.capture.grab(self.region)
            frame = np.array(img)
            if frame is None or frame.size == 0 or frame.shape[0] < 300 or frame.shape[1] < 400:
                with self.lock:
                    self.frame = None
                return
            with self.lock:
                self.frame = frame
        except Exception as e:
            logger.debug(f"[GameWindowCapturor] grab error: {e}")
            with self.lock:
                self.frame = None

    def get_frame(self):
        """
        Thread-safe getter for the latest game frame.

        Returns:
            BGR numpy array, or None if no frame is available.
        """
        with self.lock:
            if self.frame is None:
                return None
            # Only grab reference inside lock — convert outside
            frame_ref = self.frame

        if frame_ref is None or getattr(frame_ref, "size", 0) == 0:
            return None
        if len(frame_ref.shape) < 3 or frame_ref.shape[0] <= 0 or frame_ref.shape[1] <= 0:
            return None

        # Convert BGRA (mss format) to BGR outside lock (avoids blocking capture)
        title_bar_height = self.cfg["game_window"].get("title_bar_height", 0)
        if title_bar_height > 0:
            if title_bar_height >= frame_ref.shape[0]:
                return None
            cropped = frame_ref[title_bar_height:, :, :]
            if cropped is None or cropped.size == 0:
                return None
            return cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(frame_ref, cv2.COLOR_BGRA2BGR)

    def limit_fps(self):
        """Limit FPS to save system resources."""
        target_duration = 1.0 / self.fps_limit
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        self.fps = round(1.0 / max(time.time() - self.t_last_run, 0.001))
        self.t_last_run = time.time()
