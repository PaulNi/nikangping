"""
MapleStoryGo — Monster Detection System

Entry point for the monster detection engine.
Processing runs in background thread, main thread handles display (no stutter).

Usage:
    python src/main.py                          # Live detection on MapleStoryGo
    python src/main.py --test_image test/xxx.png # Offline test on a screenshot
    python src/main.py --debug                   # Enable debug logging
    python src/main.py --disable_viz             # Disable visual debug window

Reference:
  MapleStoryAutoLevelUp-main/src/engine/MapleStoryAutoLevelUp.py
"""
import argparse
import logging
import os
import sys
import time

import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger
from src.utils.common import load_yaml, override_cfg, is_mac
from src.engine.MonsterDetector import MonsterDetector


def main():
    parser = argparse.ArgumentParser(
        description="MapleStoryGo — Monster Detection System"
    )
    parser.add_argument("--test_image", type=str, default="",
                        help="Path to a test screenshot for offline debugging")
    parser.add_argument("--cfg", type=str, default="custom",
                        help="Custom config suffix")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--disable_viz", action="store_true",
                        help="Disable the debug visualization window")
    args = parser.parse_args()

    if args.debug:
        logger.set_level(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("MapleStoryGo — Monster Detection System")
    logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------
    logger.info("Loading configuration...")
    cfg_path = "config/config_default.yaml"
    if not os.path.exists(cfg_path):
        logger.error(f"Default config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    logger.info(f"  Loaded: {cfg_path}")

    if is_mac():
        platform_cfg_path = "config/config_macOS.yaml"
        if os.path.exists(platform_cfg_path):
            cfg = override_cfg(cfg, load_yaml(platform_cfg_path))
            logger.info(f"  Applied: {platform_cfg_path}")

    custom_cfg_path = f"config/config_{args.cfg}.yaml"
    if os.path.exists(custom_cfg_path):
        cfg = override_cfg(cfg, load_yaml(custom_cfg_path))
        logger.info(f"  Applied: {custom_cfg_path}")
    else:
        logger.info(f"  No custom config found at {custom_cfg_path}")

    if args.disable_viz:
        cfg["disable_viz"] = True
        logger.info("  Debug visualization: disabled")

    # -------------------------------------------------------------------
    # Initialize the monster detector
    # -------------------------------------------------------------------
    logger.info("Initializing MonsterDetector...")
    detector = MonsterDetector(cfg, test_image=args.test_image)

    logger.info("Loading monster templates...")
    if not detector.load_monster_images():
        logger.error("Failed to load monster images. Exiting.")
        sys.exit(1)

    # Load map and routes if enabled
    if cfg["bot"].get("enable_route") and cfg["bot"].get("map"):
        logger.info(f"Loading map/route: {cfg['bot']['map']}...")
        detector.load_map_and_routes(cfg["bot"]["map"])

    # Create keyboard controller for movement + listener for F1/F2/F3 profiles
    from src.input.KeyBoardController import KeyBoardController
    from src.input.KeyBoardListener import KeyBoardListener
    kb = KeyBoardController(cfg)
    detector._kb = kb  # Wire to detector
    kb.set_detector(detector)
    kb_listener = KeyBoardListener(cfg, is_autobot=True)
    # F1=12139 小地图黄点边缘巡逻；F2=12138 中区；F3=定点；同键再按关闭
    kb_listener.register_func_key_handler('f1', lambda: kb.start_profile("f1"))
    kb_listener.register_func_key_handler('f2', lambda: kb.start_profile("f2"))
    kb_listener.register_func_key_handler('f3', lambda: kb.start_profile("f3"))
    logger.info(
        "Hotkeys: F1=12139黄点边缘巡逻(A 0.5-0.75s)  F2=12138中区(S+A)  "
        "F3=12139定点站桩  同键再按=停止  F11=detect/hunt"
    )

    logger.info("Starting game window capture...")
    detector.start_capture()

    # -------------------------------------------------------------------
    # Start processing in background thread (matches reference: thread_auto_bot)
    # -------------------------------------------------------------------
    detector.start()

    # HP/MP toggles: F6/F7 (F1–F3 are profiles)
    if detector.health_monitor is not None:
        kb_listener.register_func_key_handler(
            'f6', detector.health_monitor.toggle_hp
        )
        kb_listener.register_func_key_handler(
            'f7', detector.health_monitor.toggle_mp
        )
        logger.info("Hotkeys: F6=toggle HP  F7=toggle MP")

    # Register F11 to toggle bot mode (detect <-> hunt)
    kb_listener.register_func_key_handler('f11', detector.toggle_mode)

    # -------------------------------------------------------------------
    # Main thread: display loop (matches reference: main() while loop)
    # Processing runs in background, display never blocks.
    # -------------------------------------------------------------------
    win_name = "NewMaple"
    window_positioned = False
    last_frame_seq = -1
    t_last_display = time.time()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        # Keep monitor visible but avoid forcing it above the game forever.
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.request_title_selection_from_display(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            detector.clear_title_selection()

    cv2.setMouseCallback(win_name, _on_mouse)

    try:
        while not detector.is_terminated:
            # Show debug image — only when a new frame is ready
            # (exact same pattern as reference: MapleStoryAutoLevelUp.py:1851-1864)
            if detector.is_frame_done and detector.frame_seq != last_frame_seq:
                if detector.img_display is not None:
                    cv2.imshow(win_name, detector.img_display)
                    now = time.time()
                    detector.display_fps = round(1.0 / max(now - t_last_display, 0.001))
                    t_last_display = now
                    last_frame_seq = detector.frame_seq

                    if not window_positioned:
                        try:
                            import ctypes
                            user32 = ctypes.windll.user32
                            get_system_metrics = getattr(user32, "GetSystemMetrics")
                            screen_w = get_system_metrics(0)
                            screen_h = get_system_metrics(1)
                            rect = cv2.getWindowImageRect(win_name)
                            if rect:
                                _, _, win_w, win_h = rect
                                cv2.moveWindow(win_name,
                                               screen_w - win_w - 20,
                                               screen_h - win_h - 100)
                                window_positioned = True
                        except Exception:
                            pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Quit key pressed.")
                detector.is_terminated = True
            elif key == ord('s'):
                from src.utils.common import screenshot
                screenshot(detector.img_display, "detection")
                logger.info("Screenshot saved.")
            elif key == ord('c'):
                detector.clear_title_selection()

            # No extra fixed sleep: let waitKey pump events smoothly.

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()
        kb.stop()
        kb_listener.stop()

    logger.info("Done.")


if __name__ == "__main__":
    main()
