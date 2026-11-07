"""
Monster Detection Engine for MapleStoryGo.

Adapted from the monster detection logic in:
  MapleStoryAutoLevelUp-main/src/engine/MapleStoryAutoLevelUp.py

Core features:
  - Loads monster template images from monster/<name>/ directories
  - Captures live game frames from the "MapleStoryGo" window
  - Performs template matching with mask support
  - Applies Non-Maximum Suppression (NMS) to filter overlapping detections
  - Supports multiple detection modes: color, grayscale, contour_only, template_free
  - Provides real-time debug visualization
"""
import time
import glob
import os
import threading

import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.common import (
    load_yaml, load_image, get_mask, find_pattern_sqdiff,
    nms, draw_rectangle, screenshot, to_opencv_hsv,
    activate_game_window, is_mac, is_windows,
    get_minimap_loc_size, mask_route_colors
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.HealthMonitor import HealthMonitor
from src.engine.LieAlert import LieAlert


class MonsterDetector:
    """
    Detects monsters in the MapleStoryGo game window using template matching.

    Usage:
        detector = MonsterDetector(cfg)
        detector.load_monster_images()
        detector.start()
        # Frames are processed automatically in the main loop
    """

    def __init__(self, cfg: dict, test_image: str = ""):
        """
        Initialize the monster detector.

        Args:
            cfg: Configuration dictionary.
            test_image: Path to a test image for offline debugging.
        """
        self.cfg = cfg
        self.test_image = test_image

        # Monster data
        self.monsters_info: dict = {}   # {name: [(img, mask), ...]}
        self.monsters: list = []        # Detected monsters in current frame

        # Frame data
        self.frame = None               # Raw captured frame
        self.img_frame = None           # Working-size game frame (BGR)
        self.img_frame_gray = None      # Grayscale version
        self.img_frame_debug = None     # Debug visualization frame
        self.img_display = None         # Pre-scaled frame for display thread

        # Player location — detected via title template matching
        self.loc_player = (0, 0)
        self.loc_title = None  # Last known title position (for fast local search)
        self._title_last_seen_frame = -1
        self._title_state_lock = threading.Lock()
        self._title_candidates_snapshot = []
        self._manual_title_locked = False
        self._selected_title_loc = None
        self._pending_title_select_loc = None
        self._title_lock_miss_count = 0

        # Minimap
        self.loc_minimap = (0, 0)
        self.img_minimap = None
        self.loc_player_minimap = None

        # Route following
        self.img_map = None             # Global stitched map (map.png)
        self.img_routes = []            # Route images (route*.png)
        self.img_route = None           # Current route
        self.idx_routes = 0             # Current route index
        self.color_code = {}            # RGB→command mapping
        self.color_code_up_down = {}    # Up/down RGB→command mapping
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self.loc_player_global = (0.0, 0.0)
        self.loc_minimap_global = (0, 0)
        self._last_pmm = None   # Last player minimap position (for displacement tracking)
        self.is_on_ladder = False
        self._kb = None  # Keyboard controller (set externally)
        self.health_monitor = None  # Health monitor (initialized in start())
        self.lie_alert = LieAlert(cfg)  # 测谎 alert (+ optional mouse-follow if enabled)

        # Hunt-mode search state (used when no monsters are detected)
        self._hunt_no_monster_frames = 0
        self._hunt_search_dir = "right"
        self._hunt_search_flip_frame = 0
        self._hunt_search_sweeps = 0
        self._hunt_search_pause_until = 0
        self._hunt_search_anchor_x = None
        self._hunt_search_last_progress_x = None
        self._hunt_search_stall_frames = 0
        self._hunt_search_last_reason = "idle"
        self._hunt_dir_change_frame = 0
        self._hunt_smooth_x = None
        self._hunt_smooth_y = None
        self._hunt_progress_src = None
        self._hunt_dir_forced = False
        self._hunt_mob_steer_frame = 0
        self._hunt_mob_last_lr = (0, 0)
        # Instant bound-flip helpers (F2 mid-zone)
        self._hunt_edge_last_raw_x = None
        self._hunt_edge_stall_frames = 0
        self._hunt_edge_enter_t = None
        self._hunt_edge_enter_side = None
        # F2: after N mid-zone 往返, one pass uses full minimap width.
        self._patrol_round_trips = 0
        self._patrol_leg_count = 0
        self._patrol_full_map = False

        # In-place attack "walk about" state machine (prevents attack lockout)
        self._in_place_state = "attack"       # "attack" | "pre_walk" | "walk_away" | "walk_back"
        self._in_place_state_frame = 0        # frame counter for current state
        self._in_place_walk_dir = "left"      # fallback direction when no facing
        self._in_place_backward_dir = "left"  # backward dir (opposite of facing)

        # F3 station-hold: lock minimap yellow X at enable time, walk back if drifted
        self._station_center_x = None
        self._station_center_y = None
        self._station_lock_logged = False

        # Load title template for player detection (fallback)
        self.img_title = None
        self.img_title_gray = None
        if os.path.exists("character/title.png"):
            self.img_title = load_image("character/title.png")
            self.img_title_gray = load_image("character/title.png", cv2.IMREAD_GRAYSCALE)
            logger.info(f"  Loaded title template: {self.img_title.shape[1]}x{self.img_title.shape[0]}")
        else:
            logger.warning("  No character/title.png found")

        # Player floating nameplate (preferred localization)
        self.img_name = None
        self.img_name_gray = None
        self.loc_name = None
        self._name_last_seen_frame = -1
        self._name_last_score = 1.0
        self._player_src = "none"
        name_cfg = cfg.get("player_name", {})
        name_tmpl = str(name_cfg.get("template", "character/name.png"))
        if name_cfg.get("enable", True) and os.path.exists(name_tmpl):
            self.img_name = load_image(name_tmpl)
            self.img_name_gray = load_image(name_tmpl, cv2.IMREAD_GRAYSCALE)
            logger.info(
                f"  Loaded name template ({name_cfg.get('text', '?')}): "
                f"{self.img_name.shape[1]}x{self.img_name.shape[0]} from {name_tmpl}"
            )
        else:
            logger.warning(
                f"  No name template at {name_tmpl}; will fall back to title/center"
            )

        # Timers
        self.t_last_frame = time.time()
        self.frame_count = 0
        self.fps = 0
        self.display_fps = 0
        self.frame_seq = 0
        self.best_score = 1.0       # Best (lowest) match score this frame

        # Flags
        self.is_terminated = False
        self.is_frame_done = False
        self.is_show_debug = not cfg.get("disable_viz", False)

        # Capture thread
        self.capture = None

    # -------------------------------------------------------------------
    # Monster image loading
    # -------------------------------------------------------------------

    def load_monster_images(self) -> bool:
        """
        Load monster template images from monster/ directories.

        For each monster directory, loads all PNG images, creates
        green-background masks, and generates horizontally-flipped
        copies for detecting facing-left monsters.

        The expected directory layout:
            monster/
              OrangeMushroom/
                stand_0.png
                move_0.png
                ...

        Returns:
            True if at least one monster was loaded successfully.
        """
        # Load specified monsters from config, or scan all directories
        specified = self.cfg["monster_detect"].get("monsters", [])
        if specified:
            monster_dirs = [os.path.join("monster", m) for m in specified
                            if os.path.isdir(os.path.join("monster", m))]
            if not monster_dirs:
                logger.error(f"Specified monster dirs not found: {specified}")
                return False
        else:
            monster_dirs = [d for d in glob.glob("monster/*") if os.path.isdir(d)]
            if not monster_dirs:
                logger.error("No monster directories found in monster/")
                return False

        for monster_dir in monster_dirs:
            monster_name = os.path.basename(monster_dir)
            imgs = []

            # Load monster images: try <name>*.png first (reference pattern),
            # fall back to all *.png in the directory
            png_files = sorted(glob.glob(os.path.join(monster_dir, f"{monster_name}*.png")))
            if not png_files:
                png_files = sorted(glob.glob(os.path.join(monster_dir, "*.png")))
            if not png_files:
                logger.warning(f"No usable PNG images in {monster_dir}, skipping.")
                continue
            # Strict material-photo mode: keep only core frames (move/stand0-3/jump/hit).
            # Large stand_4+ snowball sprites falsely match snowy tree tops on ice maps.
            if bool(self.cfg.get("monster_detect", {}).get("strict_templates", True)):
                def _keep_tmpl(path: str) -> bool:
                    base = os.path.basename(path).lower()
                    stem = os.path.splitext(base)[0]
                    if stem.startswith(("move_", "jump_", "hit1_", "hit_")):
                        return True
                    if stem.startswith("stand_"):
                        try:
                            n = int(stem.split("_", 1)[1])
                            return 0 <= n <= 3
                        except Exception:
                            return False
                    # named like OrangeMushroom_0.png etc.
                    return True
                filtered = [f for f in png_files if _keep_tmpl(f)]
                if filtered:
                    png_files = filtered
                else:
                    logger.warning(f"strict_templates filtered all in {monster_dir}; keeping originals")

            # Load all images + horizontally flipped copies (for both directions)
            for file in png_files:
                try:
                    img, mask = self._load_monster_template(file)
                    imgs.append((img, mask))
                    # Flipped copy for right-facing monsters
                    img_flip = cv2.flip(img, 1)
                    mask_flip = cv2.flip(mask, 1) if mask is not None else None
                    imgs.append((img_flip, mask_flip))
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")

            if not imgs:
                continue

            if imgs:
                self.monsters_info[monster_name] = imgs
                logger.info(
                    f"  Loaded {len(png_files)} images for '{monster_name}' "
                    f"({len(imgs)} with flips)"
                )
            else:
                logger.warning(f"No valid images in {monster_dir}")

        if not self.monsters_info:
            logger.error("Failed to load any monster images.")
            return False

        logger.info(f"Total monsters loaded: {list(self.monsters_info.keys())}")
        return True

    def _load_monster_template(self, file: str) -> tuple:
        """
        Load a monster template image and create its mask.

        Supports three formats:
          - RGBA with alpha channel: alpha > 128 = monster, alpha == 0 = ignore
          - BGR with green (0,255,0) background: green pixels ignored
          - BGR without special background: no mask, match full image

        Returns:
            (img_bgr, mask) — mask is None if no masking needed
        """
        raw = load_image(file, cv2.IMREAD_UNCHANGED)

        if len(raw.shape) == 3 and raw.shape[2] == 4:
            # RGBA image — use alpha channel as mask
            alpha = raw[:, :, 3]
            mask = np.where(alpha > 128, 255, 0).astype(np.uint8)
            img = raw[:, :, :3].copy()
            img[mask == 0] = [0, 0, 0]
        else:
            img = raw[:, :, :3] if len(raw.shape) == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            # Check if green-background (reference project style)
            green_px = np.sum(np.all(img == [0, 255, 0], axis=2))
            total_px = img.shape[0] * img.shape[1]
            if green_px > total_px * 0.1:  # >10% green pixels
                mask = get_mask(img, (0, 255, 0))
            else:
                mask = None  # No mask — match full image

        return img, mask

    # -------------------------------------------------------------------
    # Frame capture
    # -------------------------------------------------------------------

    def start_capture(self):
        """Initialize and start the game window capturer."""
        self.capture = GameWindowCapturor(self.cfg, self.test_image)

    def get_img_frame(self) -> np.ndarray:
        """
        Get the latest processed game frame.

        Returns:
            BGR numpy array of the game frame (working size), or None.
        """
        if self.capture is None:
            self.start_capture()

        frame = self.capture.get_frame()
        if frame is None:
            return None

        self.frame = frame
        return frame

    # -------------------------------------------------------------------
    # Monster detection (core logic)
    # -------------------------------------------------------------------

    def get_monsters_in_range(
        self,
        top_left: tuple,
        bottom_right: tuple
    ) -> list:
        """
        Detect monsters within a specified region of the game frame.

        This is the core detection function. It iterates through all
        loaded monster templates and performs template matching within
        the region of interest (ROI).

        Supports multiple detection modes:
          - "color": Full-color template matching with mask
          - "grayscale": Grayscale template matching with mask
          - "contour_only": Black contour matching (blurred)
          - "template_free": Connected-component-based (no templates)

        Args:
            top_left: (x0, y0) top-left corner of the search region.
            bottom_right: (x1, y1) bottom-right corner.

        Returns:
            List of detected monster dicts, each with:
              - "name": monster name
              - "position": (x, y) top-left of bounding box
              - "size": (h, w) of bounding box
              - "score": match quality (lower = better for SQDIFF)
        """
        x0, y0 = top_left
        x1, y1 = bottom_right

        # Clamp to frame bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.img_frame.shape[1], x1)
        y1 = min(self.img_frame.shape[0], y1)

        if x1 <= x0 or y1 <= y0:
            return []

        img_roi = self.img_frame[y0:y1, x0:x1]

        # Shift player location into ROI coordinate system
        px, py = self.loc_player
        px_in_roi = px - x0
        py_in_roi = py - y0

        # Define player character bounding box in ROI coords (to exclude)
        char_half_w = self.cfg["character"]["width"] // 2
        char_half_h = self.cfg["character"]["height"] // 2
        char_x_min = max(0, px_in_roi - char_half_w)
        char_x_max = min(img_roi.shape[1], px_in_roi + char_half_w)
        char_y_min = max(0, py_in_roi - char_half_h)
        char_y_max = min(img_roi.shape[0], py_in_roi + char_half_h)

        mode = self.cfg["monster_detect"]["mode"]
        monsters = []
        self.best_score = 1.0  # Reset each frame

        if mode == "template_free":
            monsters = self._detect_template_free(
                img_roi, x0, y0, char_x_min, char_y_min,
                char_x_max, char_y_max
            )
        elif mode in ("color", "grayscale", "contour_only"):
            roi_h, roi_w = img_roi.shape[:2]

            for monster_name, monster_imgs in self.monsters_info.items():
                for img_monster, mask_monster in monster_imgs:
                    th, tw = img_monster.shape[:2]
                    if th > roi_h or tw > roi_w or tw < 10 or th < 10:
                        continue

                    found = self._match_template(
                        img_roi,
                        img_monster, mask_monster,
                        x0, y0, monster_name, mode,
                        char_x_min, char_y_min, char_x_max, char_y_max
                    )
                    monsters.extend(found)
        else:
            logger.error(f"Unsupported detection mode: {mode}")
            return []

        # Apply Non-Maximum Suppression
        monsters = nms(monsters, iou_threshold=0.4)

        # Drop obvious UI false positives (minimap / top chrome)
        monsters = self._filter_monsters_gameplay(monsters)

        # Also detect via enemy HP bar if enabled
        if self.cfg["monster_detect"].get("with_enemy_hp_bar", False):
            hp_bar_monsters = self._detect_by_hp_bar(img_roi, x0, y0)
            monsters.extend(hp_bar_monsters)

        return monsters

    def _match_template(
        self,
        img_roi: np.ndarray,
        img_monster: np.ndarray,
        mask_monster: np.ndarray,
        x0: int, y0: int,
        monster_name: str,
        mode: str,
        char_x_min: int, char_y_min: int,
        char_x_max: int, char_y_max: int
    ) -> list:
        """
        Match a single monster template against an ROI.
        Uses multi-scale matching and returns best matches.
        """
        monsters = []
        diff_thres = self.cfg["monster_detect"]["diff_thres"]
        h, w = img_monster.shape[:2]

        # Multi-scale matching (configured via monster_detect.scales)
        scales = self.cfg["monster_detect"].get("scales", [1.0])

        for scale in scales:
            if scale == 1.0:
                tmpl, msk, th, tw = img_monster, mask_monster, h, w
            else:
                tw = max(8, int(w * scale))
                th = max(8, int(h * scale))
                # AREA for downscale / LINEAR for upscale — NEAREST hurt soft pixie edges
                interp = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
                tmpl = cv2.resize(img_monster, (tw, th), interpolation=interp)
                if mask_monster is not None:
                    msk = cv2.resize(mask_monster, (tw, th), interpolation=cv2.INTER_NEAREST)
                else:
                    msk = None

            try:
                if mode == "contour_only":
                    mask_pat = np.all(tmpl == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi[char_y_min:char_y_max, char_x_min:char_x_max] = 0
                    blur = self.cfg["monster_detect"].get("contour_blur", 5)
                    res = cv2.matchTemplate(
                        cv2.GaussianBlur(mask_roi, (blur, blur), 0),
                        cv2.GaussianBlur(mask_pat, (blur, blur), 0),
                        cv2.TM_SQDIFF_NORMED
                    )
                elif mode == "grayscale":
                    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    if msk is not None:
                        res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_SQDIFF_NORMED, mask=msk)
                    else:
                        res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_SQDIFF_NORMED)
                elif mode == "color":
                    if msk is not None:
                        res = cv2.matchTemplate(img_roi, tmpl, cv2.TM_SQDIFF_NORMED, mask=msk)
                    else:
                        res = cv2.matchTemplate(img_roi, tmpl, cv2.TM_SQDIFF_NORMED)
                else:
                    return []
            except cv2.error:
                continue

            # Masked SQDIFF can yield NaN on some OpenCV builds — treat as miss.
            res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
            # Track best score
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < self.best_score:
                self.best_score = float(min_val)

            # Collect matches below threshold, ranked by score (lower=better).
            # Scanning order is top-to-bottom; taking the first N would bias toward
            # UI/tree canopy and drop real ground mobs when the threshold is loose.
            match_locations = np.where(res <= diff_thres)
            ys, xs = match_locations
            if len(xs) > 0:
                scores = res[ys, xs]
                order = np.argsort(scores)  # best first
                max_keep = int(self.cfg.get("monster_detect", {}).get("max_matches_per_template", 20))
                for idx in order[:max_keep]:
                    px = int(xs[idx]); py = int(ys[idx])
                    monsters.append({
                        "name": monster_name,
                        "position": (px + x0, py + y0),
                        "size": (th, tw),
                        "score": float(scores[idx]),
                    })

        return monsters

    def _detect_template_free(
        self,
        img_roi: np.ndarray,
        x0: int, y0: int,
        char_x_min: int, char_y_min: int,
        char_x_max: int, char_y_max: int
    ) -> list:
        """
        Template-free monster detection using connected components of
        black pixels (monster outlines in the game).

        This is useful when monster templates are not available or
        when you want to detect any dark shape as a potential monster.
        """
        # Generate mask where pixel is exactly black
        black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

        # Exclude player character region
        black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

        # Morphological close to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # Draw character bounding box for debug
        draw_rectangle(
            self.img_frame_debug,
            (char_x_min + x0, char_y_min + y0),
            (self.cfg["character"]["height"], self.cfg["character"]["width"]),
            (255, 0, 0), "Character Box"
        )

        # Find connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            closed_mask, connectivity=8
        )

        monsters = []
        min_area = 1000
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > min_area:
                monsters.append({
                    "name": "Unknown",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        return monsters


    def _filter_monsters_gameplay(self, monsters: list) -> list:
        """Remove UI / canopy FPs; optionally keep ONLY bottom-floor monsters.

        When monster_detect.bottom_floor_only (or bottom_y_min_ratio) is set,
        keep boxes whose center Y is in the bottom platform band of the game
        client. This ignores Luster Pixies on floating/upper platforms so L/R
        mob-steer counts only the ground strip. Absolute screen-Y is used
        (not player Y) because nameplate Y is often wrong/center-fallback.
        """
        if not monsters or self.img_frame is None:
            return monsters
        md = self.cfg.get("monster_detect", {}) or {}
        fh, fw = self.img_frame.shape[:2]
        top_cut = int(fh * float(md.get("ignore_top_ratio", 0.18)))
        # Optional cut of chat/HUD strip at very bottom
        hud_cut = int(fh * (1.0 - float(md.get("ignore_bottom_hud_ratio", 0.08))))

        bottom_only = bool(md.get("bottom_floor_only", False))
        y_min_ratio = md.get("bottom_y_min_ratio", None)
        if bottom_only and y_min_ratio is None:
            y_min_ratio = 0.50
        bottom_y_min = int(fh * float(y_min_ratio)) if y_min_ratio is not None else None

        max_dy = int(md.get("max_dy_from_player", 140))
        # With absolute bottom-band filter, skip player-relative max_dy — a
        # false nameplate Y (upper lock / center) used to drop real ground mobs.
        use_max_dy = (bottom_y_min is None) and max_dy > 0
        py = int(self.loc_player[1]) if self.loc_player else fh // 2
        src = str(getattr(self, "_player_src", ""))
        if use_max_dy and self.cfg.get("bot", {}).get("mode") == "hunt" and not src.startswith("name"):
            py = fh // 2

        mm = None
        if self.img_minimap is not None:
            mx, my = self.loc_minimap
            mh, mw = self.img_minimap.shape[:2]
            pad = 8
            mm = (mx - pad, my - pad, mx + mw + pad, my + mh + pad)

        kept = []
        for m in monsters:
            x, y = m.get("position", (0, 0))
            h, w = m.get("size", (0, 0))
            cx, cy = x + w // 2, y + h // 2
            if cy < top_cut:
                continue
            if cy > hud_cut:
                continue
            if bottom_y_min is not None and cy < bottom_y_min:
                continue  # upper / floating platform — ignore
            if use_max_dy and abs(cy - py) > max_dy:
                continue
            if mm is not None:
                x0, y0, x1, y1 = mm
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    continue
            kept.append(m)
        return kept

    def _detect_by_hp_bar(
        self,
        img_roi: np.ndarray,
        x0: int, y0: int
    ) -> list:
        """
        Detect monsters by finding enemy HP bar colors in the ROI.
        """
        hp_color = np.array(self.cfg["monster_detect"]["hp_bar_color"])
        tol = int(self.cfg["monster_detect"].get("hp_bar_tolerance", 30))
        lower = np.clip(hp_color - tol, 0, 255)
        upper = np.clip(hp_color + tol, 0, 255)
        mask = cv2.inRange(img_roi, lower, upper)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        monsters = []
        # Estimate monster height from loaded templates
        min_monster_h = min(
            img.shape[0]
            for imgs in self.monsters_info.values()
            for img, _ in imgs
        ) if self.monsters_info else 100

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 3:  # Noise filter
                continue

            # Guess a monster bounding box below the HP bar
            monster_y = y + 10
            monster_x = max(0, x)
            monster_w = 70
            monster_h = min_monster_h

            monsters.append({
                "name": "Health Bar",
                "position": (x0 + monster_x, y0 + monster_y),
                "size": (monster_h, monster_w),
                "score": 1.0,
            })

        return monsters

    # -------------------------------------------------------------------
    # Debug visualization
    # -------------------------------------------------------------------

    def _draw_detection_debug(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        monsters: list
    ):
        """Draw fresh monster detection boxes (called only on detection frames)."""
        if self.img_frame_debug is None:
            return

        for monster in monsters:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            draw_rectangle(
                self.img_frame_debug,
                monster["position"],
                monster["size"],
                color,
                f"{monster['name']}:{monster['score']:.2f}",
                thickness=2, text_height=0.5
            )

    def _draw_search_region(self, x0: int, y0: int, x1: int, y1: int):
        """Draw the search region box on the debug frame."""
        if self.img_frame_debug is None:
            return
        # Bright cyan border, thicker for visibility
        draw_rectangle(
            self.img_frame_debug,
            (x0, y0), (y1 - y0, x1 - x0),
            (255, 255, 0), "Search", thickness=2, text_height=0.5
        )

    def _fallback_player_location(self):
        """Keep last valid player position, or fall back to frame center."""
        if self.loc_player != (0, 0):
            return
        h, w = self.img_frame.shape[:2]
        self.loc_player = (w // 2, h // 2)

    def _collect_title_candidates(self, img_roi, img_pat, max_candidates: int, score_threshold: float):
        """Collect multiple non-overlapping title candidates from a full-frame template match."""
        h_img, w_img = img_roi.shape[:2]
        h_pat, w_pat = img_pat.shape[:2]
        if h_pat > h_img or w_pat > w_img:
            return []

        result = cv2.matchTemplate(img_roi, img_pat, cv2.TM_SQDIFF_NORMED)
        result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
        work = result.copy()
        suppress_x = max(12, w_pat // 2)
        suppress_y = max(8, h_pat // 2)

        candidates = []
        for _ in range(max_candidates):
            min_val, _, min_loc, _ = cv2.minMaxLoc(work)
            if min_val > score_threshold:
                break
            x, y = int(min_loc[0]), int(min_loc[1])
            candidates.append({
                "loc": (x, y),
                "score": float(min_val),
                "cost": float(min_val),
                "box": (x, y, w_pat, h_pat),
            })
            x0 = max(0, x - suppress_x)
            x1 = min(work.shape[1], x + suppress_x + 1)
            y0 = max(0, y - suppress_y)
            y1 = min(work.shape[0], y + suppress_y + 1)
            work[y0:y1, x0:x1] = 1.0

        return candidates

    def _update_title_candidates_snapshot(self, candidates, template_size):
        """Store the latest title candidates for debug rendering and click selection."""
        tw, th = template_size
        normalized = []
        with self._title_state_lock:
            selected_loc = self._selected_title_loc
            locked = self._manual_title_locked
        for idx, cand in enumerate(sorted(candidates, key=lambda c: c.get("score", 1.0))):
            x, y = cand["loc"]
            is_selected = False
            if locked and selected_loc is not None:
                is_selected = abs(x - selected_loc[0]) <= max(12, tw // 2) and abs(y - selected_loc[1]) <= max(8, th // 2)
            normalized.append({
                "idx": idx + 1,
                "loc": (int(x), int(y)),
                "box": (int(x), int(y), int(tw), int(th)),
                "score": float(cand.get("score", 1.0)),
                "src": cand.get("src", "cand"),
                "selected": is_selected,
            })
        with self._title_state_lock:
            self._title_candidates_snapshot = normalized

    def request_title_selection_from_display(self, display_x: int, display_y: int) -> bool:
        """Queue manual title selection by clicking a candidate box in the display window."""
        scale = float(self.cfg.get("monster_detect", {}).get("display_scale", 0.55))
        scale = max(scale, 0.01)
        raw_x = int(round(display_x / scale))
        raw_y = int(round(display_y / scale))

        with self._title_state_lock:
            hits = []
            for cand in self._title_candidates_snapshot:
                x, y, w, h = cand["box"]
                if x <= raw_x <= x + w and y <= raw_y <= y + h:
                    hits.append(cand)
            if not hits:
                return False
            chosen = min(hits, key=lambda c: c["score"])
            self._pending_title_select_loc = chosen["loc"]

        logger.info(
            f"[Title] Queued manual selection idx={chosen['idx']} loc={chosen['loc']} score={chosen['score']:.4f}"
        )
        return True

    def clear_title_selection(self):
        """Clear the manual title lock and return to automatic title selection."""
        with self._title_state_lock:
            self._manual_title_locked = False
            self._selected_title_loc = None
            self._pending_title_select_loc = None
            self._title_lock_miss_count = 0
        self.loc_title = None
        self._title_last_seen_frame = -1
        logger.info("[Title] Cleared manual title selection")

    def _draw_title_candidates(self):
        """Draw title candidates and manual selection state on the debug frame."""
        if self.img_frame_debug is None:
            return

        with self._title_state_lock:
            candidates = list(self._title_candidates_snapshot)
            locked = self._manual_title_locked
            miss_count = self._title_lock_miss_count

        for cand in candidates:
            x, y, w, h = cand["box"]
            color = (0, 255, 0) if cand.get("selected") else (0, 255, 255)
            label = f"T{cand['idx']}:{cand['score']:.2f}"
            cv2.rectangle(self.img_frame_debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                self.img_frame_debug,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        status = f"TitleSel: {'LOCKED' if locked else 'AUTO'} | LClick select | RClick/C clear | miss={miss_count}"
        cv2.putText(
            self.img_frame_debug,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if locked else (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


    def reload_player_name(self) -> bool:
        """Reload nameplate template from current cfg[player_name] (profile switch)."""
        from src.utils.common import load_image
        name_cfg = self.cfg.get("player_name", {}) or {}
        name_tmpl = str(name_cfg.get("template", "character/name.png"))
        self.loc_name = None
        self._name_last_seen_frame = -1
        self._name_last_score = 1.0
        self._player_src = "none"
        if name_cfg.get("enable", True) and os.path.exists(name_tmpl):
            self.img_name = load_image(name_tmpl)
            self.img_name_gray = load_image(name_tmpl, cv2.IMREAD_GRAYSCALE)
            logger.info(
                f"  Reloaded name template ({name_cfg.get('text', '?')}): "
                f"{self.img_name.shape[1]}x{self.img_name.shape[0]} from {name_tmpl}"
            )
            return True
        self.img_name = None
        self.img_name_gray = None
        logger.warning(f"  reload_player_name: missing template {name_tmpl}")
        return False


    def _detect_player_by_name(self) -> bool:
        """Locate player by matching floating nameplate template (甩锅侠12139).

        Returns True when loc_player was updated from a name hit.
        Prefer local search around previous hit for speed; full-frame periodically.
        """
        name_cfg = self.cfg.get("player_name", {})
        if not name_cfg.get("enable", True) or self.img_name_gray is None:
            return False
        if self.img_frame_gray is None:
            return False
        # Flash patrol mainly uses minimap yellow; name can run every N frames.
        name_every = max(1, int(name_cfg.get("name_every_n", 1)))
        if name_every > 1 and (self.frame_count % name_every) != 0:
            if self.loc_name is not None and self.loc_player not in (None, (0, 0)):
                return True
            return False

        match_threshold = float(name_cfg.get("match_threshold", 0.22))
        offset = name_cfg.get("offset", [0, -55])
        if not isinstance(offset, (list, tuple)) or len(offset) < 2:
            offset = [0, -55]
        prev_radius = int(name_cfg.get("prev_search_radius", 240))
        full_interval = max(1, int(name_cfg.get("full_search_interval", 12)))
        ignore_bottom = float(name_cfg.get("ignore_bottom_ratio", 0.16))
        ignore_top = float(name_cfg.get("ignore_top_ratio", 0.08))
        max_jump = int(name_cfg.get("max_jump_px", 280))

        img = self.img_frame_gray
        pat = self.img_name_gray
        fh, fw = img.shape[:2]
        th, tw = pat.shape[:2]
        if th >= fh or tw >= fw:
            return False

        y_min = int(fh * ignore_top)
        y_max = int(fh * (1.0 - ignore_bottom)) - th
        if y_max <= y_min:
            y_min, y_max = 0, fh - th

        def _search(roi, x_off, y_off):
            if roi.shape[0] < th or roi.shape[1] < tw:
                return None
            result = cv2.matchTemplate(roi, pat, cv2.TM_SQDIFF_NORMED)
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
            min_val, _, min_loc, _ = cv2.minMaxLoc(result)
            gx = int(min_loc[0] + x_off)
            gy = int(min_loc[1] + y_off)
            return float(min_val), (gx, gy)

        candidates = []

        # 1) Local around previous name hit
        if self.loc_name is not None:
            lx, ly = self.loc_name
            x0 = max(0, lx - prev_radius)
            y0 = max(y_min, ly - prev_radius)
            x1 = min(fw, lx + prev_radius + tw)
            y1 = min(fh, ly + prev_radius + th)
            if x1 - x0 >= tw and y1 - y0 >= th:
                hit = _search(img[y0:y1, x0:x1], x0, y0)
                if hit is not None:
                    score, loc = hit
                    candidates.append(("prev", score, loc))

        # 2) Full / center search periodically or on miss
        need_full = (
            self.loc_name is None
            or self.frame_count % full_interval == 0
            or not candidates
            or candidates[0][1] > match_threshold
        )
        if need_full:
            # Prefer mid-band (character usually mid-screen); still cover full width
            hit = _search(img[y_min:y_max + th, :], 0, y_min)
            if hit is not None:
                score, loc = hit
                candidates.append(("full", score, loc))

        if not candidates:
            return False

        # Pick best (lowest SQDIFF), reject HUD / illegal bands
        best = None
        for src, score, loc in candidates:
            gx, gy = loc
            if gy < y_min or gy > y_max:
                continue
            # reject minimap area
            if self.img_minimap is not None:
                mx, my = self.loc_minimap
                mh, mw = self.img_minimap.shape[:2]
                if mx - 4 <= gx <= mx + mw + 4 and my - 4 <= gy <= my + mh + 4:
                    continue
            if best is None or score < best[1]:
                best = (src, score, loc)

        if best is None:
            return False

        src, score, loc = best
        if score > match_threshold:
            if self.frame_count <= 30 or self.frame_count % 45 == 0:
                logger.info(
                    f"[Name] MISS score={score:.4f} thresh={match_threshold:.2f} src={src}"
                )
            return False

        # Anti-jump: keep previous if jump huge and score not excellent
        if self.loc_name is not None:
            jump = ((loc[0] - self.loc_name[0]) ** 2 + (loc[1] - self.loc_name[1]) ** 2) ** 0.5
            if jump > max_jump and score > max(0.05, match_threshold * 0.45):
                loc = self.loc_name
                src = "hold"

        self.loc_name = loc
        self._name_last_seen_frame = self.frame_count
        self._name_last_score = score
        self.loc_player = (
            loc[0] + tw // 2 + int(offset[0]),
            loc[1] + th // 2 + int(offset[1]),
        )
        self._player_src = f"name:{src}"

        if self.img_frame_debug is not None:
            draw_rectangle(
                self.img_frame_debug,
                loc, (th, tw), (0, 255, 0),
                f"NAME {score:.2f}", thickness=2, text_height=0.55
            )
            # Nameplate label
            label = str(name_cfg.get("text", "NAME"))
            cv2.putText(
                self.img_frame_debug, label,
                (loc[0], max(14, loc[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

        if self.frame_count <= 40 or self.frame_count % 30 == 0:
            logger.info(
                f"[Name] HIT score={score:.4f} src={src} "
                f"name={loc} player={self.loc_player} text={name_cfg.get('text','')}"
            )
        return True

    def _detect_player_by_title(self):
        """Detect player location by matching the title template (at character feet)."""
        if self.img_title is None:
            self._fallback_player_location()
            return

        title_cfg = self.cfg.get("title", {})
        if not title_cfg.get("enable", True):
            self._fallback_player_location()
            return

        title_offset = title_cfg.get("offset", [0, -60])
        if not isinstance(title_offset, (list, tuple)) or len(title_offset) < 2:
            title_offset = [0, -60]

        mode = str(title_cfg.get("mode", "white_mask")).strip().lower()
        if mode not in ("white_mask", "grayscale"):
            mode = "white_mask"

        lower_white = int(title_cfg.get("white_lower", 150))
        upper_white = int(title_cfg.get("white_upper", 255))
        lower_white = max(0, min(255, lower_white))
        upper_white = max(lower_white, min(255, upper_white))

        blur_k_raw = title_cfg.get("blur_kernel", 3)
        try:
            blur_k = int(blur_k_raw)
        except (TypeError, ValueError):
            blur_k = 3
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            blur_k += 1

        match_threshold = float(title_cfg.get("match_threshold", 0.3))
        cache_ttl = int(title_cfg.get("cache_ttl", 30))
        center_search_radius = int(title_cfg.get("center_search_radius", 420))
        center_accept_radius = int(title_cfg.get("center_accept_radius", min(center_search_radius, 260)))
        center_bias_weight = float(title_cfg.get("center_bias_weight", 0.08))
        max_jump_px = int(title_cfg.get("max_jump_px", 220))
        prev_search_radius = int(title_cfg.get("prev_search_radius", 220))
        stick_to_prev_margin = float(title_cfg.get("stick_to_prev_margin", 0.02))
        strong_match_threshold = float(title_cfg.get("strong_match_threshold", max(0.03, match_threshold * 0.35)))
        top_k = max(1, int(title_cfg.get("top_k", 12)))
        manual_select = bool(title_cfg.get("manual_select", True))
        max_candidates = max(1, int(title_cfg.get("max_candidates", 6)))
        candidate_threshold = float(
            title_cfg.get("candidate_threshold", min(0.35, max(match_threshold, match_threshold * 1.25)))
        )
        lock_miss_tolerance = max(1, int(title_cfg.get("lock_miss_tolerance", 15)))
        manual_track_radius = max(20, int(title_cfg.get("manual_track_radius", prev_search_radius)))
        manual_track_threshold = float(
            title_cfg.get("manual_track_threshold", min(0.4, max(match_threshold + 0.05, match_threshold * 1.25)))
        )

        # Periodically drop cached title lock so the tracker can recover from a wrong person.
        if (self.loc_title is not None and cache_ttl > 0
                and self._title_last_seen_frame >= 0
                and self.frame_count - self._title_last_seen_frame > cache_ttl):
            self.loc_title = None

        had_lock = self.loc_title is not None

        def _prepare_title_match_images():
            if mode == "grayscale":
                img_pat = self.img_title_gray
                img_roi = self.img_frame_gray
                if blur_k > 1:
                    img_pat = cv2.GaussianBlur(img_pat, (blur_k, blur_k), 0)
                    img_roi = cv2.GaussianBlur(img_roi, (blur_k, blur_k), 0)
                return img_roi, img_pat

            # white_mask mode (legacy default)
            img_pat = cv2.inRange(self.img_title_gray, lower_white, upper_white)
            img_roi = cv2.inRange(self.img_frame_gray, lower_white, upper_white)
            if blur_k > 1:
                img_pat = cv2.GaussianBlur(img_pat, (blur_k, blur_k), 0)
                img_roi = cv2.GaussianBlur(img_roi, (blur_k, blur_k), 0)
            return img_roi, img_pat

        def _biased_sqdiff_match(img_roi, img_pat, anchor, radius=None, bias_weight=0.08):
            """Pick among top-K SQDIFF matches using a distance-to-anchor bias."""
            h_img, w_img = img_roi.shape[:2]
            h_pat, w_pat = img_pat.shape[:2]
            if h_pat > h_img or w_pat > w_img:
                return None

            x_off = 0
            y_off = 0
            search_img = img_roi

            if radius is not None:
                x0 = max(0, int(anchor[0] - radius))
                y0 = max(0, int(anchor[1] - radius))
                x1 = min(w_img, int(anchor[0] + radius + w_pat))
                y1 = min(h_img, int(anchor[1] + radius + h_pat))
                if x1 - x0 >= w_pat and y1 - y0 >= h_pat:
                    search_img = img_roi[y0:y1, x0:x1]
                    x_off, y_off = x0, y0

            result = cv2.matchTemplate(search_img, img_pat, cv2.TM_SQDIFF_NORMED)
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)

            flat = result.reshape(-1)
            if flat.size == 0:
                return None

            k = min(top_k, flat.size)
            idxs = np.argpartition(flat, k - 1)[:k]
            out_h, out_w = result.shape[:2]
            diag = max((w_img ** 2 + h_img ** 2) ** 0.5, 1.0)

            best = None
            for idx in idxs:
                y, x = divmod(int(idx), out_w)
                score = float(result[y, x])
                gx = x + x_off
                gy = y + y_off
                dist = ((gx - anchor[0]) ** 2 + (gy - anchor[1]) ** 2) ** 0.5 / diag
                cost = score + bias_weight * dist
                if best is None or cost < best["cost"]:
                    best = {
                        "loc": (gx, gy),
                        "score": score,
                        "cost": cost,
                        "center_dist_px": ((gx - center_anchor[0]) ** 2 + (gy - center_anchor[1]) ** 2) ** 0.5,
                    }
            return best

        try:
            img_roi_proc, img_title_proc = _prepare_title_match_images()
            th, tw = img_title_proc.shape[:2]
            fh, fw = img_roi_proc.shape[:2]

            center_anchor = (max(0, fw // 2 - tw // 2), max(0, fh // 2 - th // 2))
            candidate_pool = self._collect_title_candidates(
                img_roi_proc, img_title_proc,
                max_candidates=max_candidates,
                score_threshold=candidate_threshold,
            )
            self._update_title_candidates_snapshot(candidate_pool, (tw, th))

            with self._title_state_lock:
                if self._pending_title_select_loc is not None:
                    self._selected_title_loc = self._pending_title_select_loc
                    self._manual_title_locked = True
                    self.loc_title = self._pending_title_select_loc
                    self._title_lock_miss_count = 0
                    self._pending_title_select_loc = None
                    logger.info(f"[Title] Manual target armed at {self.loc_title}")
                manual_locked = manual_select and self._manual_title_locked
                selected_loc = self._selected_title_loc

            if manual_locked and selected_loc is not None:
                tracked = []
                for cand in candidate_pool:
                    dist = ((cand["loc"][0] - selected_loc[0]) ** 2 + (cand["loc"][1] - selected_loc[1]) ** 2) ** 0.5
                    cand["manual_dist_px"] = dist
                    cand["src"] = cand.get("src", "manual_pool")
                    if dist <= manual_track_radius:
                        tracked.append(cand)

                manual_best = min(
                    tracked,
                    key=lambda c: (c["score"], c.get("manual_dist_px", 0.0)),
                    default=None,
                )

                if manual_best is not None and manual_best["score"] <= manual_track_threshold:
                    loc = manual_best["loc"]
                    score = manual_best["score"]
                    match_src = "manual"
                    rejected = []
                    with self._title_state_lock:
                        self._selected_title_loc = loc
                        self._title_lock_miss_count = 0
                    self._update_title_candidates_snapshot(candidate_pool, (tw, th))
                else:
                    with self._title_state_lock:
                        self._title_lock_miss_count += 1
                        miss_count = self._title_lock_miss_count
                    if self.loc_title is not None and miss_count <= lock_miss_tolerance:
                        loc = self.loc_title
                        score = 1.0
                        match_src = "manual_hold"
                        rejected = []
                    else:
                        with self._title_state_lock:
                            self._manual_title_locked = False
                            self._selected_title_loc = None
                            self._title_lock_miss_count = 0
                        manual_locked = False

            candidates = []

            # Candidate 1: around previous lock (if any), weak bias for temporal stability.
            if not manual_locked and self.loc_title is not None:
                prev_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=self.loc_title,
                    radius=prev_search_radius,
                    bias_weight=0.04,
                )
                if prev_cand is not None:
                    prev_cand["src"] = "prev"
                    prev_cand["jump_px"] = 0.0
                    candidates.append(prev_cand)

            # Candidate 2: around center (primary anti-confusion strategy).
            if not manual_locked:
                center_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=center_anchor,
                    radius=center_search_radius,
                    bias_weight=center_bias_weight,
                )
                if center_cand is not None:
                    center_cand["src"] = "center"
                    center_cand["jump_px"] = (
                        ((center_cand["loc"][0] - self.loc_title[0]) ** 2
                         + (center_cand["loc"][1] - self.loc_title[1]) ** 2) ** 0.5
                        if self.loc_title is not None else 0.0
                    )
                    candidates.append(center_cand)

            # Candidate 3: global fallback only for recovery when there is no active lock.
            best_local_score = min([c["score"] for c in candidates], default=1.0)
            if not manual_locked and not had_lock and best_local_score > match_threshold * 0.9:
                global_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=center_anchor,
                    radius=None,
                    bias_weight=center_bias_weight,
                )
                if global_cand is not None:
                    global_cand["src"] = "global"
                    global_cand["jump_px"] = 0.0
                    candidates.append(global_cand)

            if not manual_locked and not candidates:
                self._fallback_player_location()
                return

            if not manual_locked:
                filtered = []
                rejected = []
                for cand in candidates:
                    src = cand.get("src", "unknown")
                    center_dist_px = cand.get("center_dist_px", 0.0)
                    jump_px = cand.get("jump_px", 0.0)

                    if src != "prev" and center_dist_px > center_accept_radius:
                        rejected.append((src, "far_from_center", round(center_dist_px, 1), round(cand["score"], 4)))
                        continue
                    if had_lock and src != "prev" and jump_px > max_jump_px:
                        rejected.append((src, "jump_too_large", round(jump_px, 1), round(cand["score"], 4)))
                        continue
                    filtered.append(cand)

                if not filtered and had_lock:
                    loc = self.loc_title
                    score = 1.0
                    match_src = "prev_keep"
                else:
                    if not filtered:
                        self._fallback_player_location()
                        return

                    prev_good = next(
                        (c for c in filtered
                         if c.get("src") == "prev" and c["score"] < match_threshold),
                        None,
                    )
                    best_other = min(
                        [c for c in filtered if c.get("src") != "prev"],
                        key=lambda c: c["cost"],
                        default=None,
                    )

                    if prev_good is not None:
                        if (best_other is None
                                or best_other["score"] >= strong_match_threshold
                                or best_other["score"] + stick_to_prev_margin >= prev_good["score"]):
                            best = prev_good
                        else:
                            best = best_other
                    else:
                        best = min(filtered, key=lambda c: c["cost"])

                    loc = best["loc"]
                    score = best["score"]
                    match_src = best.get("src", "unknown")
        except cv2.error as e:
            logger.warning(
                f"[Title] OpenCV error, skip frame: {e}. "
                f"mode={mode}, blur_kernel={blur_k}, white=({lower_white},{upper_white})"
            )
            self._fallback_player_location()
            return
        except Exception as e:
            logger.warning(f"[Title] Detection error, skip frame: {e}")
            self._fallback_player_location()
            return

        # Log score every frame for first 100 frames, then every 30 frames
        if self.frame_count <= 100 or self.frame_count % 30 == 0:
            status = "HIT" if (score < match_threshold or match_src in ("manual", "manual_hold", "prev_keep")) else "MISS"
            logger.info(
                f"[Title] frame={self.frame_count} score={score:.4f} "
                f"thresh={match_threshold:.2f} {status} src={match_src} mode={mode}"
            )
            if 'rejected' in locals() and rejected:
                logger.info(f"[Title] Rejected candidates: {rejected[:3]}")

        if score < match_threshold or match_src in ("prev_keep", "manual", "manual_hold"):
            # Anti-jump guard: reject sudden large jumps unless score is very confident.
            if self.loc_title is not None and match_src not in ("prev_keep", "manual_hold"):
                jump = ((loc[0] - self.loc_title[0]) ** 2 + (loc[1] - self.loc_title[1]) ** 2) ** 0.5
                if jump > max_jump_px and score > max(0.04, match_threshold * 0.45):
                    loc = self.loc_title
            self.loc_title = loc
            self._title_last_seen_frame = self.frame_count
            if 'candidate_pool' in locals() and 'tw' in locals() and 'th' in locals():
                self._update_title_candidates_snapshot(candidate_pool, (tw, th))
            tw, th = self.img_title.shape[1], self.img_title.shape[0]
            self.loc_player = (
                loc[0] + tw // 2 + title_offset[0],
                loc[1] + th // 2 + title_offset[1]
            )
            if self.img_frame_debug is not None:
                draw_rectangle(
                    self.img_frame_debug,
                    loc, (th, tw), (255, 0, 0),
                    f"TITLE {score:.2f}", thickness=3, text_height=0.7
                )
        elif self.loc_title is None:
            self._fallback_player_location()


    def _find_player_dot_on_minimap(self, img_minimap):
        """Locate the yellow player diamond on the minimap ROI.

        Classic MapleStory uses a bright yellow/cyan diamond. Prefer compact
        blobs away from the ROI border (border pixels caused false x≈6 locks
        on 云彩公园 V and kept mid-zone forever force-right while the char
        looked stuck on the left).
        """
        import cv2
        import numpy as np

        if img_minimap is None or getattr(img_minimap, "size", 0) == 0:
            return None
        h, w = img_minimap.shape[:2]
        if h < 8 or w < 8:
            return None

        mm_cfg = self.cfg.get("minimap", {}) or {}
        colors = mm_cfg.get("player_colors") or [
            mm_cfg.get("player_color", (0, 255, 255)),
            (0, 255, 255),
            (136, 255, 255),
            (50, 220, 255),
        ]
        base_tol = int(mm_cfg.get("player_tolerance", 20))
        border = int(mm_cfg.get("player_border_reject_px", 5))
        min_area = int(mm_cfg.get("player_min_area", 3))
        max_area = int(mm_cfg.get("player_max_area", 30))
        max_jump = float(mm_cfg.get("player_max_jump_px", 35))

        candidates = []
        for color in colors:
            color = tuple(int(c) for c in color)
            for tol in (base_tol, base_tol + 15, base_tol + 30):
                lower = tuple(max(0, c - tol) for c in color)
                upper = tuple(min(255, c + tol) for c in color)
                mask = cv2.inRange(img_minimap, lower, upper)
                num_lbl, _lbl, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, num_lbl):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < min_area or area > max_area:
                        continue
                    cx = float(centroids[i][0])
                    cy = float(centroids[i][1])
                    on_border = (
                        cx < border or cy < border or cx > (w - border - 1) or cy > (h - border - 1)
                    )
                    candidates.append({
                        "x": cx, "y": cy, "area": area, "border": on_border, "tol": tol
                    })
                if candidates:
                    break
            if candidates:
                break

        if not candidates:
            # HSV yellow diamond fallback (high S/V, hue ~20-40)
            hsv = cv2.cvtColor(img_minimap, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (15, 140, 200), (45, 255, 255))
            num_lbl, _lbl, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_lbl):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                cx = float(centroids[i][0])
                cy = float(centroids[i][1])
                on_border = (
                    cx < border or cy < border or cx > (w - border - 1) or cy > (h - border - 1)
                )
                candidates.append({
                    "x": cx, "y": cy, "area": area, "border": on_border, "tol": -1
                })

        if not candidates:
            return None

        # F2: when walking into a minimap edge, the REAL yellow diamond sits on
        # the border and used to be rejected (player_border_reject_px). That left
        # a false interior blob (~x=30-45) locked for seconds while the char
        # pressed the physical wall — bound flip only fired after the false
        # lock vanished. Prefer edge-side border candidates matching patrol dir.
        dir_hint = str(getattr(self, "_hunt_search_dir", "") or "")
        def _is_soft_border(c):
            if not c["border"]:
                return False
            # Keep hard reject for top/bottom chrome; allow left/right edge dots.
            cy = float(c["y"])
            if cy < border or cy > (h - border - 1):
                return True
            cx = float(c["x"])
            if dir_hint == "left" and cx < border:
                return False
            if dir_hint == "right" and cx > (w - border - 1):
                return False
            return True

        for c in candidates:
            if c["border"] and not _is_soft_border(c):
                c["border"] = False
                c["edge_ok"] = True
            else:
                c["edge_ok"] = False

        interior = [c for c in candidates if not c["border"]]
        pool = interior if interior else candidates

        prev = getattr(self, "loc_player_minimap", None)
        def score(c):
            # Prefer larger compact dots; penalize huge jumps from last good fix
            s = float(c["area"])
            if prev is not None:
                dist = abs(c["x"] - float(prev[0])) + abs(c["y"] - float(prev[1]))
                if dist > max_jump:
                    s -= 50.0 + dist
                else:
                    s += 10.0 - 0.1 * dist
            if c["border"]:
                s -= 100.0
            if c.get("edge_ok"):
                s += 40.0  # strongly prefer real edge diamond while walking into wall
            return s

        best = max(pool, key=score)
        # If best is a huge jump and we have a recent fix, keep previous for this frame
        # UNLESS the jump is toward the patrol edge (false-lock -> real edge diamond).
        if prev is not None:
            dist = abs(best["x"] - float(prev[0])) + abs(best["y"] - float(prev[1]))
            toward_edge = (
                (dir_hint == "left" and best["x"] < float(prev[0]) - 3.0)
                or (dir_hint == "right" and best["x"] > float(prev[0]) + 3.0)
                or bool(best.get("edge_ok"))
            )
            if dist > max_jump and interior and not toward_edge:
                near = [c for c in interior if abs(c["x"] - float(prev[0])) + abs(c["y"] - float(prev[1])) <= max_jump]
                if near:
                    best = max(near, key=score)
                else:
                    # Keep stale one more frame; caller may still use previous
                    miss = int(getattr(self, "_pmm_miss_frames", 0)) + 1
                    self._pmm_miss_frames = miss
                    if miss < 8:
                        return (int(prev[0]), int(prev[1]))
            else:
                self._pmm_miss_frames = 0
        else:
            self._pmm_miss_frames = 0

        return (int(round(best["x"])), int(round(best["y"])))


    def _detect_minimap(self):
        """
        Prefer fixed config minimap (flash_patrol). Full-frame white-border
        auto-detect is expensive (~10–40ms) — only refresh on an interval.
        """
        mm_cfg = self.cfg.get("minimap", {}) or {}
        cfg_box = None
        if mm_cfg.get("width", 0) > 10 and mm_cfg.get("height", 0) > 10:
            cfg_box = (
                int(mm_cfg["x"]),
                int(mm_cfg["y"]),
                int(mm_cfg["width"]),
                int(mm_cfg["height"]),
            )

        auto_every = float(mm_cfg.get("auto_detect_interval_s", 3.0) or 0.0)
        now = time.time()
        cached = getattr(self, "_minimap_cached", None)
        last_t = float(getattr(self, "_minimap_auto_t", 0.0) or 0.0)

        # Fast path: trust config / cache between rare auto refreshes.
        if cfg_box is not None and (auto_every <= 0 or (now - last_t) < auto_every):
            return cached or cfg_box

        self._minimap_auto_t = now
        result = get_minimap_loc_size(self.img_frame)
        if result:
            x, y, w_m, h_m = result
            if self.frame_count == 1:
                logger.info(f"[Minimap] Auto-detected: ({x},{y}) {w_m}x{h_m}")
            x_i, y_i, w_i, h_i = (x + 1, y + 1, w_m - 2, h_m - 2)
            if w_i > 0 and h_i > 0:
                self._minimap_cached = (x_i, y_i, w_i, h_i)
                return self._minimap_cached

        if cfg_box is not None:
            self._minimap_cached = cfg_box
            return cfg_box
        return cached

    # -------------------------------------------------------------------
    # Route following
    # -------------------------------------------------------------------

    def load_map_and_routes(self, map_name):
        """Load map.png and route*.png for the given map."""
        import glob
        map_path = f"minimaps/{map_name}/map.png"
        if not os.path.exists(map_path):
            logger.warning(f"[Route] Map not found: {map_path}")
            return False

        self.img_map = load_image(map_path, cv2.IMREAD_COLOR)
        logger.info(f"[Route] Loaded map: {map_path} ({self.img_map.shape[1]}x{self.img_map.shape[0]})")

        # Load route images
        route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
        route_files = [p for p in route_files if "route_rest" not in p]
        self.img_routes = []
        for rf in route_files:
            img = cv2.cvtColor(load_image(rf), cv2.COLOR_BGR2RGB)
            # Strip map-resident colors matching route codes from route image
            img = mask_route_colors(self.img_map.copy(), img, self.cfg["route"]["color_code"])
            img = mask_route_colors(self.img_map.copy(), img, self.cfg["route"]["color_code_up_down"])
            self.img_routes.append(img)
            logger.info(f"[Route] Loaded: {rf}")

        if self.img_routes:
            self.img_route = self.img_routes[0]

        # Parse color→command dictionaries
        for k, v in self.cfg["route"]["color_code"].items():
            self.color_code[tuple(map(int, k.split(',')))] = v
        for k, v in self.cfg["route"]["color_code_up_down"].items():
            self.color_code_up_down[tuple(map(int, k.split(',')))] = v

        logger.info(f"[Route] Ready: {len(self.img_routes)} routes")
        return True

    def get_player_global_position(self):
        """Track global position: X=player dot, Y=manual offset from climbing."""
        if self.img_map is None or self.img_minimap is None:
            return
        if self.loc_player_minimap is None:
            return

        # Initial positioning: template match to find starting point
        if self.loc_player_global == (0, 0):
            loc, score, _ = find_pattern_sqdiff(self.img_map, self.img_minimap)
            self._last_match_score = score
            self.loc_minimap_global = loc
            self.loc_player_global = (
                loc[0] + self.loc_player_minimap[0],
                loc[1] + self.loc_player_minimap[1],
            )
            self._last_pmm = self.loc_player_minimap
            return

        # X + Y: track via player dot displacement
        dx = self.loc_player_minimap[0] - self._last_pmm[0]
        dy = self.loc_player_minimap[1] - self._last_pmm[1]
        if abs(dx) < 20 and abs(dy) < 20:
            self.loc_player_global = (
                self.loc_player_global[0] + dx,
                self.loc_player_global[1] + dy,
            )

        # Fallback: Y offset when climbing (supports 0.5 etc)
        offset = float(self.cfg["route"].get("climb_y_offset", 1))
        interval = int(self.cfg["route"].get("climb_y_interval", 15))
        if self.frame_count % interval == 0:
            if self.cmd_move_y == "up":
                self.loc_player_global = (self.loc_player_global[0], self.loc_player_global[1] - offset)
            elif self.cmd_move_y == "down":
                self.loc_player_global = (self.loc_player_global[0], self.loc_player_global[1] + offset)

        self._last_pmm = self.loc_player_minimap
        self.loc_minimap_global = (
            self.loc_player_global[0] - self.loc_player_minimap[0],
            self.loc_player_global[1] - self.loc_player_minimap[1],
        )

        # Periodic drift correction (every 60 frames)
        if self.frame_count % 60 == 0:
            loc, score, _ = find_pattern_sqdiff(self.img_map, self.img_minimap)
            self._last_match_score = score
            if score < 0.15:
                self.loc_minimap_global = loc
                self.loc_player_global = (
                    loc[0] + self.loc_player_minimap[0],
                    loc[1] + self.loc_player_minimap[1],
                )

    def get_nearest_route_color(self):
        """Scan route image around player for nearest color-coded pixel."""
        if self.img_route is None or self.loc_player_global == (0.0, 0.0):
            return None, None

        return self._get_nearest_route_color_from_image(self.img_route)

    def _get_nearest_route_color_from_image(self, route_img, search_range=None):
        """Scan a specific route image around player for nearest route pixels."""
        if route_img is None or self.loc_player_global == (0.0, 0.0):
            return None, None

        x0, y0 = int(self.loc_player_global[0]), int(self.loc_player_global[1])
        h, w = route_img.shape[:2]
        if search_range is None:
            search_range = self.cfg["route"].get("search_range", 10)

        x_min = max(0, x0 - search_range)
        x_max = min(w, x0 + search_range)
        y_min = max(0, y0 - search_range)
        y_max = min(h, y0 + search_range)

        nearest = None
        nearest_ud = None
        min_dist = float('inf')
        min_dist_ud = float('inf')

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(route_img[y, x])
                dist = abs(x - x0) + abs(y - y0)
                if pixel in self.color_code and dist < min_dist:
                    nearest = {"pixel": (x, y), "command": self.color_code[pixel], "distance": dist}
                    min_dist = dist
                if pixel in self.color_code_up_down and dist < min_dist_ud:
                    nearest_ud = {"pixel": (x, y), "command": self.color_code_up_down[pixel], "distance": dist}
                    min_dist_ud = dist

        return nearest, nearest_ud

    def _select_best_route(self):
        """Select the route image nearest to the current global player position."""
        if not self.img_routes or self.loc_player_global == (0.0, 0.0):
            return

        search_range = int(self.cfg.get("route", {}).get("search_range", 10))
        auto_select_range = int(self.cfg.get("route", {}).get("auto_select_range", max(30, search_range * 4)))

        best = None
        for idx, route_img in enumerate(self.img_routes):
            nearest, nearest_ud = self._get_nearest_route_color_from_image(route_img, auto_select_range)
            distances = []
            if nearest is not None:
                distances.append(nearest["distance"])
            if nearest_ud is not None:
                distances.append(nearest_ud["distance"])
            if not distances:
                continue
            dist = min(distances)
            if best is None or dist < best["dist"]:
                best = {"idx": idx, "dist": dist}

        if best is not None and best["idx"] != self.idx_routes:
            self.idx_routes = best["idx"]
            self.img_route = self.img_routes[self.idx_routes]
            logger.info(
                f"[Route] Auto-selected route {self.idx_routes + 1}/{len(self.img_routes)} "
                f"for global=({int(self.loc_player_global[0])},{int(self.loc_player_global[1])}) "
                f"dist={best['dist']}"
            )

    def update_cmd_by_route(self):
        """Convert nearest route color to movement commands."""
        self._select_best_route()
        nearest, nearest_ud = self.get_nearest_route_color()

        if nearest and nearest_ud:
            if nearest["distance"] < nearest_ud["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
                _, cmd, _ = nearest_ud["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
                cmd, _, _ = nearest["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd
        elif nearest:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
        elif nearest_ud:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
        else:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = "none", "none", "none"

        # Goal → cycle to next route
        if self.cmd_action == "goal" and self.img_routes:
            self.idx_routes = (self.idx_routes + 1) % len(self.img_routes)
            self.img_route = self.img_routes[self.idx_routes]
            logger.info(f"[Route] Goal reached → route {self.idx_routes + 1}/{len(self.img_routes)}")

    def _draw_player_marker(self):
        """Draw the player position marker on the debug frame."""
        if self.img_frame_debug is None:
            return
        cv2.circle(
            self.img_frame_debug,
            self.loc_player, radius=5,
            color=(0, 0, 255), thickness=-1
        )
        src = str(getattr(self, "_player_src", ""))
        label = f"Player[{src}]" if src else "Player"
        cv2.putText(
            self.img_frame_debug, label,
            (self.loc_player[0] - 40, self.loc_player[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    def _get_nearest_monster_dx(self):
        """Return horizontal offset (monster_center_x - player_x) to nearest monster, or None."""
        if not self.monsters:
            return None
        px, _ = self.loc_player
        nearest_dx = None
        for m in self.monsters:
            mx, _ = m.get("position", (0, 0))
            _mh, mw = m.get("size", (0, 0))
            cx = mx + (mw // 2)
            dx = cx - px
            if nearest_dx is None or abs(dx) < abs(nearest_dx):
                nearest_dx = dx
        return nearest_dx


    def _hunt_player_x(self):
        """Horizontal player anchor for hunt chase.

        Prefer name/title-based loc_player. Screen center only when explicitly enabled
        and name tracking is unavailable this frame.
        """
        if self.img_frame is None:
            return int(self.loc_player[0])
        use_center = bool(self.cfg.get("directional_attack", {}).get("chase_use_screen_center", False))
        if use_center and not str(getattr(self, "_player_src", "")).startswith("name"):
            return int(self.img_frame.shape[1] // 2)
        return int(self.loc_player[0])

    def _get_nearest_monster_dx_from(self, px: int):
        """Return horizontal offset to nearest monster relative to px, or None."""
        if not self.monsters:
            return None
        nearest_dx = None
        for m in self.monsters:
            mx, _ = m.get("position", (0, 0))
            _mh, mw = m.get("size", (0, 0))
            cx = mx + (mw // 2)
            dx = cx - px
            if nearest_dx is None or abs(dx) < abs(nearest_dx):
                nearest_dx = dx
        return nearest_dx

    def _get_attack_facing(self) -> str:
        """Return left/right facing toward the nearest detected monster."""
        nearest_dx = self._get_nearest_monster_dx()
        if nearest_dx is None:
            return "none"

        deadzone = int(self.cfg.get("directional_attack", {}).get("face_deadzone_px", 8))
        if abs(nearest_dx) <= deadzone:
            return "none"
        return "right" if nearest_dx > 0 else "left"


    def _count_monsters_lr(self):
        """Count BOTTOM-FLOOR monsters left/right of the player screen X.

        Uses monster box center vs player screen X (name/title anchor).
        self.monsters is already bottom-filtered when bottom_floor_only is on;
        still re-check bottom_y_min_ratio here so L/R steer never counts
        upper-platform pixies if a stale box slips through.
        Returns (left_count, right_count). Monsters inside deadzone are ignored.
        """
        if not self.monsters:
            self._hunt_mob_last_lr = (0, 0)
            return 0, 0
        da = self.cfg.get("directional_attack", {}) or {}
        md = self.cfg.get("monster_detect", {}) or {}
        deadzone = max(0, int(da.get("search_mob_deadzone_px", 24)))
        px = self._hunt_player_x()
        bottom_y_min = None
        if self.img_frame is not None:
            fh = int(self.img_frame.shape[0])
            y_min_ratio = md.get("bottom_y_min_ratio", None)
            if bool(md.get("bottom_floor_only", False)) and y_min_ratio is None:
                y_min_ratio = 0.50
            if y_min_ratio is not None:
                bottom_y_min = int(fh * float(y_min_ratio))
        left_n = 0
        right_n = 0
        for m in self.monsters:
            mx, my = m.get("position", (0, 0))
            mh, mw = m.get("size", (0, 0))
            cx = int(mx) + (int(mw) // 2)
            cy = int(my) + (int(mh) // 2)
            if bottom_y_min is not None and cy < bottom_y_min:
                continue
            if cx < px - deadzone:
                left_n += 1
            elif cx > px + deadzone:
                right_n += 1
        self._hunt_mob_last_lr = (left_n, right_n)
        return left_n, right_n

    def _mob_count_prefer_dir(self):
        """Prefer side with MORE monsters. None if equal/zero/disabled/no clear lead.

        Default steer OFF (pre-monster flash_patrol). Only when
        search_mob_steer is explicitly true do L/R mob counts steer.
        """
        da = self.cfg.get("directional_attack", {}) or {}
        if not bool(da.get("search_mob_steer", False)):
            return None, 0, 0
        if not self.monsters:
            self._hunt_mob_last_lr = (0, 0)
            return None, 0, 0
        left_n, right_n = self._count_monsters_lr()
        lead = max(1, int(da.get("search_mob_lead", 1)))
        if right_n >= left_n + lead:
            return "right", left_n, right_n
        if left_n >= right_n + lead:
            return "left", left_n, right_n
        return None, left_n, right_n

    def _reset_hunt_search_state(self):
        """Clear no-monster counter only; keep patrol direction while attacking."""
        self._hunt_no_monster_frames = 0
        self._hunt_search_last_reason = "target"

    def reset_patrol_cycle(self):
        """Reset mid↔full-map cycle (call when F2 profile enables)."""
        self._patrol_round_trips = 0
        self._patrol_leg_count = 0
        self._patrol_full_map = False

    def _patrol_effective_ratios(self, cfg):
        """Return (left_ratio, right_ratio); full-map override when cycle is active."""
        left = cfg.get("search_left_ratio", None)
        right = cfg.get("search_right_ratio", None)
        if left is None or right is None:
            return left, right
        every = int(cfg.get("search_full_map_every_round_trips", 0) or 0)
        if every > 0 and bool(getattr(self, "_patrol_full_map", False)):
            return (
                float(cfg.get("search_full_map_left_ratio", 0.0)),
                float(cfg.get("search_full_map_right_ratio", 1.0)),
            )
        return float(left), float(right)

    def _on_patrol_leg_flip(self, cfg):
        """Count mid-zone 往返; after N, run one full-map pass then resume mid."""
        every = int(cfg.get("search_full_map_every_round_trips", 0) or 0)
        if every <= 0:
            return
        self._patrol_leg_count = int(getattr(self, "_patrol_leg_count", 0) or 0) + 1
        if bool(getattr(self, "_patrol_full_map", False)):
            need = max(1, int(cfg.get("search_full_map_legs", 2) or 2))
            if self._patrol_leg_count >= need:
                self._patrol_full_map = False
                self._patrol_leg_count = 0
                self._patrol_round_trips = 0
                logger.info("[Patrol] full-map pass done — resume mid zone")
            return
        # Mid zone: 2 one-way legs = 1 往返
        if self._patrol_leg_count >= 2:
            self._patrol_leg_count = 0
            self._patrol_round_trips = int(getattr(self, "_patrol_round_trips", 0) or 0) + 1
            logger.info(
                f"[Patrol] mid 往返 {self._patrol_round_trips}/{every}"
            )
            if self._patrol_round_trips >= every:
                self._patrol_full_map = True
                self._patrol_leg_count = 0
                self._patrol_round_trips = 0
                logger.info(
                    f"[Patrol] after {every} mid 往返 — next pass FULL map"
                )

    def _smooth_hunt_x(self, raw_x, src):
        """EMA-smooth hunt X; reset when coordinate source/scale changes."""
        da = self.cfg.get("directional_attack", {}) or {}
        alpha = float(da.get("search_x_ema", 0.35))
        alpha = min(1.0, max(0.05, alpha))
        prev_src = getattr(self, "_hunt_progress_src", None)
        if self._hunt_smooth_x is None or prev_src != src:
            self._hunt_smooth_x = float(raw_x)
        else:
            self._hunt_smooth_x = (1.0 - alpha) * self._hunt_smooth_x + alpha * float(raw_x)
        self._hunt_progress_src = src
        return float(self._hunt_smooth_x)

    def _get_hunt_search_progress_x(self):
        """Return stable X progress for search / patrol logic.

        search_progress:
          - character / screen / name: nameplate/title screen X (F1 wall flip)
          - minimap: yellow diamond
          - auto (default): mid-zone → yellow then name; else route/minimap
        """
        da = self.cfg.get("directional_attack", {}) or {}
        prefer = str(da.get("search_progress", "auto") or "auto").strip().lower()
        mid_zone = (
            da.get("search_left_ratio") is not None
            and da.get("search_right_ratio") is not None
        )

        def _screen_progress():
            src = str(getattr(self, "_player_src", "") or "")
            use_screen = (
                self.img_frame is not None
                and self.loc_player is not None
                and self.loc_player != (0, 0)
                and (src.startswith("name") or src.startswith("title"))
            )
            if use_screen:
                return self._smooth_hunt_x(self.loc_player[0], "screen"), "screen"
            return None, "none"

        if prefer in ("character", "screen", "name", "player"):
            sx, ssrc = _screen_progress()
            if ssrc == "screen":
                return sx, ssrc
            # Nameplate miss this frame: fall back so patrol does not freeze.
            if self.loc_player_minimap is not None and self.img_minimap is not None:
                return self._smooth_hunt_x(self.loc_player_minimap[0], "minimap"), "minimap"
            return None, "none"

        if prefer in ("minimap", "yellow"):
            if self.loc_player_minimap is not None and self.img_minimap is not None:
                return self._smooth_hunt_x(self.loc_player_minimap[0], "minimap"), "minimap"
            return _screen_progress()

        if mid_zone:
            # Yellow first — reliable at physical map edges.
            if self.loc_player_minimap is not None and self.img_minimap is not None:
                return self._smooth_hunt_x(self.loc_player_minimap[0], "minimap"), "minimap"
            return _screen_progress()

        if self.cfg.get("bot", {}).get("enable_route") and self.loc_player_global != (0.0, 0.0):
            return float(self.loc_player_global[0]), "global"
        if self.loc_player_minimap is not None:
            return float(self.loc_player_minimap[0]), "minimap"
        return _screen_progress()

    def _midzone_bound_forces(self, left_ratio, right_ratio):
        """Multi-signal OR for F2 edge turns.

        Returns (force_left, force_right, detail).

        CRITICAL: when minimap yellow is available, bound forces use YELLOW
        ONLY for hard cuts. A frozen/false nameplate locked near screen-right
        (s≈0.91) used to fire mid_force_left as soon as yellow left the
        far-left band (yr>0.30), trapping patrol in the left half forever.

        Screen may only *agree* with yellow on the same side (wall stick),
        never override yellow alone toward the opposite half.
        """
        force_left = False
        force_right = False
        bits = []
        yr = None
        sr = None
        mm_cfg = self.cfg.get("directional_attack", {}) or {}
        edge_margin = float(mm_cfg.get("search_edge_margin", 0.02))
        # Extra inward band while still walking toward that wall (jump overshoot).
        approach = float(mm_cfg.get("search_approach_margin", 0.04))
        hunt_dir = str(getattr(self, "_hunt_search_dir", "") or "")

        if self.loc_player_minimap is not None and self.img_minimap is not None:
            mm_w = float(self.img_minimap.shape[1])
            if mm_w > 1:
                raw_x = float(self.loc_player_minimap[0])
                yr = raw_x / mm_w
                # edge_margin pulls force band inward so we flip before the
                # physical wall. approach flips even earlier when still heading
                # into that wall (jump_attack overshoot).
                left_cut = float(left_ratio) + max(0.0, edge_margin)
                right_cut = float(right_ratio) - max(0.0, edge_margin)
                if right_cut <= left_cut:
                    left_cut = float(left_ratio)
                    right_cut = float(right_ratio)
                left_approach = left_cut + max(0.0, approach)
                right_approach = right_cut - max(0.0, approach)
                if right_approach <= left_approach:
                    left_approach, right_approach = left_cut, right_cut

                if yr >= right_cut or raw_x >= (mm_w * right_cut):
                    force_left = True
                    bits.append(f"yellow_right:{yr:.2f}")
                elif hunt_dir == "right" and (
                    yr >= right_approach or raw_x >= (mm_w * right_approach)
                ):
                    force_left = True
                    bits.append(f"yellow_right_approach:{yr:.2f}")

                if yr <= left_cut or raw_x <= max(3.0, mm_w * left_cut):
                    force_right = True
                    bits.append(f"yellow_left:{yr:.2f}")
                elif hunt_dir == "left" and (
                    yr <= left_approach or raw_x <= max(3.0, mm_w * left_approach)
                ):
                    force_right = True
                    bits.append(f"yellow_left_approach:{yr:.2f}")

        src = str(getattr(self, "_player_src", "") or "")
        name_ok = src.startswith("name") or src.startswith("title")
        have_name = (
            name_ok
            and self.img_frame is not None
            and self.loc_player is not None
            and self.loc_player != (0, 0)
        )
        if have_name:
            fw = float(self.img_frame.shape[1])
            if fw > 1:
                sr = float(self.loc_player[0]) / fw

        screen_edge = float(mm_cfg.get("search_wall_screen_ratio", 0.86))
        screen_edge = min(0.95, max(0.70, screen_edge))
        agree = bool(mm_cfg.get("search_wall_screen_agree", True))
        # When still walking into a side, screen edge alone can force a turn —
        # yellow often freezes mid-map then snaps to the physical wall (logs:
        # raw 87→114, flip only at yr=0.92 past the 0.80 soft bound).
        dir_screen = bool(mm_cfg.get("search_wall_dir_screen", True))

        if yr is None and have_name and sr is not None:
            # Yellow missing: screen fallback (symmetric).
            if sr >= float(right_ratio):
                force_left = True
                bits.append(f"screen_right:{sr:.2f}")
            elif sr <= float(left_ratio):
                force_right = True
                bits.append(f"screen_left:{sr:.2f}")
        elif have_name and sr is not None:
            if dir_screen:
                # NEVER let a stuck nameplate (s≈0.90 while yellow is left)
                # fight yellow_left_approach every frame — that was the F2
                # left/right thrash on 2026-11-03 21:43.
                if hunt_dir == "right" and sr >= screen_edge:
                    if yr is None or yr >= max(0.50, float(right_ratio) - 0.22):
                        force_left = True
                        bits.append(f"dir_screen_right:s={sr:.2f}")
                    else:
                        bits.append(
                            f"dir_screen_ignore(s={sr:.2f},y={yr:.2f})"
                        )
                if hunt_dir == "left" and sr <= (1.0 - screen_edge):
                    if yr is None or yr <= min(0.50, float(left_ratio) + 0.22):
                        force_right = True
                        bits.append(f"dir_screen_left:s={sr:.2f}")
                    else:
                        bits.append(
                            f"dir_screen_ignore(s={sr:.2f},y={yr:.2f})"
                        )
            if yr is not None and agree:
                # Extra: yellow already on that half + screen jammed.
                if (
                    sr >= screen_edge
                    and yr >= 0.55
                    and (hunt_dir == "right" or yr >= float(right_ratio) - 0.10)
                ):
                    force_left = True
                    bits.append(f"screen_agree_right:s={sr:.2f},y={yr:.2f}")
                if (
                    sr <= (1.0 - screen_edge)
                    and yr <= 0.45
                    and (hunt_dir == "left" or yr <= float(left_ratio) + 0.10)
                ):
                    force_right = True
                    bits.append(f"screen_agree_left:s={sr:.2f},y={yr:.2f}")
            if yr is not None and (
                (sr >= float(right_ratio) and yr < 0.55 and hunt_dir != "right")
                or (sr <= float(left_ratio) and yr > 0.45 and hunt_dir != "left")
            ):
                bits.append(f"name_ignore_vs_yellow(s={sr:.2f},y={yr:.2f})")

        # If both fire, trust yellow side; prefer force_right on far-left lock.
        if force_left and force_right:
            if yr is not None and yr >= 0.5:
                force_right = False
                bits.append("resolve_yellow_right")
            elif yr is not None:
                force_left = False
                bits.append("resolve_yellow_left")
            else:
                force_left = False
                bits.append("resolve_prefer_right")

        return force_left, force_right, ",".join(bits) if bits else "none"

    def _get_hunt_search_progress_y(self):
        """Yellow-dot Y on minimap (downward = larger). Used for bottom-floor band."""
        if self.loc_player_minimap is None:
            return None
        raw_y = float(self.loc_player_minimap[1])
        da = self.cfg.get("directional_attack", {}) or {}
        alpha = float(da.get("search_x_ema", 0.35))
        alpha = min(1.0, max(0.05, alpha))
        if self._hunt_smooth_y is None:
            self._hunt_smooth_y = raw_y
        else:
            self._hunt_smooth_y = (1.0 - alpha) * self._hunt_smooth_y + alpha * raw_y
        return float(self._hunt_smooth_y)

    def _get_hunt_search_command(self):
        """Return (move_x, move_y, action) for left/right sweeping while searching monsters."""
        cfg = self.cfg.get("directional_attack", {})
        if not bool(cfg.get("search_enabled", True)):
            return "none", "none", "none"

        # Keep ladder handling to route logic; avoid lateral sweep while climbing.
        if self.is_on_ladder:
            return "none", "none", "none"

        trigger_frames = max(1, int(cfg.get("search_trigger_frames", 4)))
        if self._hunt_no_monster_frames < trigger_frames:
            return "none", "none", "none"

        if self.frame_count < self._hunt_search_pause_until:
            return "none", "none", "none"

        # search_flip_frames<=0 disables timer flip (edge-only patrol).
        flip_frames_cfg = int(cfg.get("search_flip_frames", 18))
        flip_timer_enabled = flip_frames_cfg > 0
        flip_frames = max(4, flip_frames_cfg) if flip_timer_enabled else 0
        progress_x, progress_src = self._get_hunt_search_progress_x()
        if progress_x is not None and self._hunt_search_anchor_x is None:
            self._hunt_search_anchor_x = progress_x
        if progress_x is not None and self._hunt_search_last_progress_x is None:
            self._hunt_search_last_progress_x = progress_x

        flip_reason = None
        self._hunt_dir_forced = False
        # Minimap/global edge flip: reach left/right end (or mid-zone bounds) -> turn around
        # search_left_ratio / search_right_ratio: fraction of span width (e.g. 0.10/0.90)
        # After N mid 往返, one pass uses search_full_map_* (default 0%~100%).
        edge_px = float(cfg.get("search_edge_px", 18))
        left_ratio, right_ratio = self._patrol_effective_ratios(cfg)
        span_w = 0.0
        mid_zone = (left_ratio is not None and right_ratio is not None)
        sticky_frames = max(0, int(cfg.get("search_dir_sticky_frames", 60 if mid_zone else 0)))
        can_flip_sticky = (
            sticky_frames <= 0
            or (self.frame_count - int(getattr(self, "_hunt_dir_change_frame", 0))) >= sticky_frames
        )
        if (
            flip_reason is None
            and progress_x is not None
            and progress_src in ("minimap", "global", "screen")
            and (progress_src != "minimap" or self.img_minimap is not None)
            and (progress_src != "screen" or self.img_frame is not None)
        ):
            if progress_src == "screen" and self.img_frame is not None:
                # F2: ratios are fractions of game client width (nameplate screen X)
                span_w = float(self.img_frame.shape[1])
            elif progress_src == "minimap" and self.img_minimap is not None:
                span_w = float(self.img_minimap.shape[1])
            else:
                # global route coords: treat configured map width if present, else soft edge by absolute
                span_w = float(cfg.get("search_global_width", 0) or 0)
            if span_w > 0:
                if mid_zone:
                    left_bound = float(left_ratio) * span_w
                    right_bound = float(right_ratio) * span_w
                    if right_bound < left_bound:
                        left_bound, right_bound = right_bound, left_bound
                    # Multi-signal OR: yellow near edge OR screen near edge → instant flip.
                    # Sticky must NEVER block wall turns (only dampens mid-band thrash).
                    # Scale hysteresis to span: edge_px=28 is huge on a ~140px minimap.
                    hysteresis = max(3.0, min(edge_px * 0.45, span_w * 0.04))
                    force_left, force_right, force_detail = self._midzone_bound_forces(
                        left_ratio, right_ratio
                    )
                    # RAW yellow X (no EMA) — EMA progress_x can lag so mid_in_* never
                    # fires while the char is already past the physical edge.
                    raw_mm_x = None
                    if self.loc_player_minimap is not None:
                        try:
                            raw_mm_x = float(self.loc_player_minimap[0])
                        except Exception:
                            raw_mm_x = None
                    # EDGE_ENTER log: first frame raw yellow crosses configured band.
                    import time as _time
                    if raw_mm_x is not None and span_w > 0:
                        _yr = raw_mm_x / span_w
                        if _yr <= float(left_ratio) or _yr >= float(right_ratio):
                            side = "left" if _yr <= float(left_ratio) else "right"
                            if getattr(self, "_hunt_edge_enter_t", None) is None or self._hunt_edge_enter_side != side:
                                self._hunt_edge_enter_t = _time.time()
                                self._hunt_edge_enter_side = side
                                logger.info(
                                    f"[Patrol] EDGE_ENTER side={side} raw={raw_mm_x:.1f} "
                                    f"yr={_yr:.2f} bounds=[{left_bound:.0f},{right_bound:.0f}]/"
                                    f"{span_w:.0f} dir={self._hunt_search_dir}"
                                )
                        else:
                            self._hunt_edge_enter_t = None
                            self._hunt_edge_enter_side = None

                    # Edge-stall: raw yellow not progressing toward current dir.
                    # Also fire past zone mid (false yellow often freezes ~0.5–0.65
                    # while the char is already on the physical wall).
                    edge_stall_n = max(0, int(cfg.get("search_edge_stall_frames", 10)))
                    edge_stall_third = float(cfg.get("search_edge_stall_third", 0.40))
                    edge_stall_min_delta = float(cfg.get("search_edge_stall_min_delta_px", 1.0))
                    stall_band = float(cfg.get("search_edge_stall_bound_band", 0.08))
                    edge_stall_hit = False
                    if raw_mm_x is not None and edge_stall_n > 0 and span_w > 0:
                        last_raw = getattr(self, "_hunt_edge_last_raw_x", None)
                        if last_raw is not None:
                            if self._hunt_search_dir == "left":
                                progressed = (float(last_raw) - raw_mm_x) >= edge_stall_min_delta
                            elif self._hunt_search_dir == "right":
                                progressed = (raw_mm_x - float(last_raw)) >= edge_stall_min_delta
                            else:
                                progressed = abs(raw_mm_x - float(last_raw)) >= edge_stall_min_delta
                            if progressed:
                                self._hunt_edge_stall_frames = 0
                            else:
                                self._hunt_edge_stall_frames = int(
                                    getattr(self, "_hunt_edge_stall_frames", 0)
                                ) + 1
                        else:
                            self._hunt_edge_stall_frames = 0
                        self._hunt_edge_last_raw_x = raw_mm_x
                        _yr2 = raw_mm_x / span_w
                        # Stall ONLY near the configured soft bounds — NOT past
                        # zone mid (that made the whole half-map a stall zone and
                        # flipped every ~5 frames while cruising).
                        left_stall_cut = min(
                            float(left_ratio) + max(0.0, stall_band),
                            max(0.05, float(edge_stall_third)),
                        )
                        right_stall_cut = max(
                            float(right_ratio) - max(0.0, stall_band),
                            min(0.95, 1.0 - float(edge_stall_third)),
                        )
                        near_left = _yr2 <= left_stall_cut
                        near_right = _yr2 >= right_stall_cut
                        if self._hunt_edge_stall_frames >= edge_stall_n:
                            if self._hunt_search_dir == "left" and near_left:
                                edge_stall_hit = True
                            elif self._hunt_search_dir == "right" and near_right:
                                edge_stall_hit = True
                    elif raw_mm_x is not None:
                        self._hunt_edge_last_raw_x = raw_mm_x

                    desired = None
                    reason = None
                    at_bound = False
                    if force_right:
                        # At left wall / left signal -> walk right
                        desired, reason, at_bound = "right", "mid_force_right", True
                    elif force_left:
                        # At right wall / right signal -> walk left
                        desired, reason, at_bound = "left", "mid_force_left", True
                    elif edge_stall_hit and self._hunt_search_dir == "left":
                        desired, reason, at_bound = "right", "mid_edge_stall_left", True
                    elif edge_stall_hit and self._hunt_search_dir == "right":
                        desired, reason, at_bound = "left", "mid_edge_stall_right", True
                    elif (progress_x <= left_bound) or (
                        raw_mm_x is not None and raw_mm_x <= left_bound
                    ):
                        desired, reason, at_bound = "right", "mid_in_left", True
                    elif (progress_x >= right_bound) or (
                        raw_mm_x is not None and raw_mm_x >= right_bound
                    ):
                        desired, reason, at_bound = "left", "mid_in_right", True
                    elif self._hunt_search_dir == "left" and (
                        progress_x <= (left_bound + hysteresis)
                        or (raw_mm_x is not None and raw_mm_x <= (left_bound + hysteresis))
                    ):
                        desired, reason, at_bound = "right", "mid_left", True
                    elif self._hunt_search_dir == "right" and (
                        progress_x >= (right_bound - hysteresis)
                        or (raw_mm_x is not None and raw_mm_x >= (right_bound - hysteresis))
                    ):
                        desired, reason, at_bound = "left", "mid_right", True
                    if desired is not None:
                        def _patrol_force_dir(new_dir):
                            """Invalidate + instantly re-hold; abort in-flight S+A combo."""
                            kb = getattr(self, "_kb", None)
                            if kb is None:
                                return
                            if not bool(getattr(kb, "is_enable", False)):
                                return
                            try:
                                if hasattr(kb, "request_dir_override"):
                                    kb.request_dir_override(new_dir)
                                elif hasattr(kb, "_invalidate_held_dirs"):
                                    kb._invalidate_held_dirs()
                            except Exception:
                                pass

                        # Hard yellow wall (past cut) may flip immediately.
                        # Approach / dir_screen / stall need a gap or they
                        # oscillate when nameplate and yellow disagree.
                        hard_yellow = (
                            "yellow_left:" in force_detail
                            or "yellow_right:" in force_detail
                            or reason in ("mid_in_left", "mid_in_right")
                        )
                        flip_gap = max(0, int(cfg.get("search_flip_min_frames", 12)))
                        since_flip = self.frame_count - int(
                            getattr(self, "_hunt_dir_change_frame", 0) or 0
                        )
                        allow_flip = hard_yellow or since_flip >= flip_gap
                        if self._hunt_search_dir != desired:
                            if allow_flip:
                                # Bound/force flips after gap (hard yellow: immediate).
                                prev_dir = self._hunt_search_dir
                                self._hunt_search_dir = desired
                                flip_reason = reason
                                self._hunt_dir_forced = True
                                self._hunt_dir_change_frame = self.frame_count
                                stall_now = int(getattr(self, "_hunt_edge_stall_frames", 0))
                                self._hunt_edge_stall_frames = 0
                                # Snap EMA to raw so next frames don't lag behind the wall.
                                if raw_mm_x is not None:
                                    self._hunt_smooth_x = float(raw_mm_x)
                                _patrol_force_dir(desired)
                                # Brief attack pause so jump_attack does not immediately
                                # leap back into the wall; walk away one step first.
                                try:
                                    kb = getattr(self, "_kb", None)
                                    pause_s = float(cfg.get("search_flip_attack_pause_s", 0.25))
                                    if kb is not None and pause_s > 0:
                                        kb._attack_pause_until = max(
                                            float(getattr(kb, "_attack_pause_until", 0.0)),
                                            _time.time() + pause_s,
                                        )
                                except Exception:
                                    pass
                                raw_s = f"{raw_mm_x:.1f}" if raw_mm_x is not None else "na"
                                enter_t = getattr(self, "_hunt_edge_enter_t", None)
                                dt_ms = (
                                    (_time.time() - enter_t) * 1000.0 if enter_t else -1.0
                                )
                                logger.info(
                                    f"[Patrol] FLIP {prev_dir}->{desired} reason={reason} "
                                    f"x={progress_x:.1f}/{progress_src} raw={raw_s} "
                                    f"bounds=[{left_bound:.0f},{right_bound:.0f}]/"
                                    f"{span_w:.0f} force={force_detail} "
                                    f"stall={stall_now} flip_dt_ms={dt_ms:.0f}"
                                )
                                self._hunt_edge_enter_t = None
                                self._hunt_edge_enter_side = None
                            else:
                                # Cooldown: keep current dir (blocks nameplate↔yellow thrash).
                                self._hunt_dir_forced = False
                        else:
                            # Already facing the turn dir. Only rarely re-hold while
                            # pinned on a HARD wall — every-3-frame re-press was
                            # spamming key_up/key_down and made F2 feel very laggy.
                            self._hunt_search_last_reason = "mid_hold_in"
                            self._hunt_dir_forced = False
                            hard_wall = bool(force_left or force_right or edge_stall_hit)
                            rehold_n = max(20, int(cfg.get("search_wall_rehold_frames", 45)))
                            if hard_wall and at_bound and (self.frame_count % rehold_n == 0):
                                _patrol_force_dir(desired)
                            if hard_wall and self.frame_count % 30 == 0:
                                raw_s = f"{raw_mm_x:.1f}" if raw_mm_x is not None else "na"
                                logger.info(
                                    f"[Patrol] HOLD {desired} reason={reason} "
                                    f"x={progress_x:.1f}/{progress_src} raw={raw_s} "
                                    f"force={force_detail}"
                                )
                    else:
                        self._hunt_dir_forced = False
                        # Inside mid band: steer toward side with MORE monsters.
                        # Bound forces above always win; equal/zero keeps sticky dir.
                        prefer, left_n, right_n = self._mob_count_prefer_dir()
                        if prefer is not None:
                            mob_sticky = max(
                                0,
                                int(cfg.get("search_mob_sticky_frames", sticky_frames or 45)),
                            )
                            steer_frame = int(getattr(self, "_hunt_mob_steer_frame", 0))
                            never_steered = steer_frame <= 0
                            lead_gap = abs(int(left_n) - int(right_n))
                            lead_need = max(1, int(cfg.get("search_mob_lead", 1)))
                            # Strong majority OR first steer: sticky must not block
                            # (logs showed L/R=5/0 while dir stayed right / last=idle).
                            can_mob_flip = (
                                never_steered
                                or mob_sticky <= 0
                                or lead_gap >= max(2, lead_need + 1)
                                or (self.frame_count - steer_frame) >= mob_sticky
                            )
                            if prefer != self._hunt_search_dir and can_mob_flip:
                                prev_dir = self._hunt_search_dir
                                self._hunt_search_dir = prefer
                                flip_reason = f"mid_mob_{prefer}"
                                self._hunt_dir_forced = True
                                self._hunt_dir_change_frame = self.frame_count
                                self._hunt_mob_steer_frame = self.frame_count
                                kb = getattr(self, "_kb", None)
                                if kb is not None and hasattr(kb, "_invalidate_held_dirs"):
                                    try:
                                        kb._invalidate_held_dirs()
                                    except Exception:
                                        pass
                                logger.info(
                                    f"[Patrol] MOB {prev_dir}->{prefer} "
                                    f"L={left_n} R={right_n} x={progress_x:.1f}/{progress_src}"
                                )
                            elif prefer == self._hunt_search_dir:
                                self._hunt_search_last_reason = (
                                    f"mid_mob_hold_L{left_n}R{right_n}"
                                )
                            else:
                                self._hunt_search_last_reason = (
                                    f"mid_mob_sticky_L{left_n}R{right_n}"
                                )
                                if self.frame_count % 15 == 0:
                                    logger.info(
                                        f"[Patrol] MOB blocked sticky want={prefer} "
                                        f"dir={self._hunt_search_dir} "
                                        f"L={left_n} R={right_n} "
                                        f"age={self.frame_count - steer_frame}/{mob_sticky}"
                                    )
                    # Brief per-turn debug (throttled)
                    log_n = max(15, int((self.cfg.get("system") or {}).get("hunt_log_every_n", 45)))
                    if self.frame_count % log_n == 0:
                        lr = getattr(self, "_hunt_mob_last_lr", (0, 0))
                        raw_s = f"{raw_mm_x:.1f}" if raw_mm_x is not None else "na"
                        logger.info(
                            f"[Patrol] src={progress_src} x={progress_x:.1f} raw={raw_s} "
                            f"bounds=[{left_bound:.0f},{right_bound:.0f}]/{span_w:.0f} "
                            f"dir={self._hunt_search_dir} reason={self._hunt_search_last_reason} "
                            f"force={force_detail} mobLR={lr[0]}/{lr[1]} "
                            f"steer={bool((self.cfg.get('directional_attack') or {}).get('search_mob_steer', False))}"
                        )
                else:
                    # Full-map edge patrol (F1 character / minimap). Prefer
                    # search_edge_ratio on screen width; else search_edge_px.
                    self._hunt_dir_forced = False
                    edge_ratio = cfg.get("search_edge_ratio", None)
                    if edge_ratio is not None and span_w > 0:
                        er = min(0.45, max(0.05, float(edge_ratio)))
                        left_lim = er * span_w
                        right_lim = (1.0 - er) * span_w
                    else:
                        left_lim = float(edge_px)
                        right_lim = max(float(edge_px), span_w - float(edge_px))

                    # Stall flip ONLY when already inside the edge band.
                    # Using a wide "third" (0.40) flipped mid-map and never covered
                    # the full platform (logs: stall at x≈70/130 while lim=[25,181]).
                    edge_stall_n = max(0, int(cfg.get("search_edge_stall_frames", 0)))
                    edge_stall_min_delta = float(
                        cfg.get("search_edge_stall_min_delta_px", 1.5)
                    )
                    stall_margin = float(cfg.get("search_edge_stall_margin", 1.35))
                    near_left = progress_x <= max(left_lim * stall_margin, left_lim + 4.0)
                    near_right = progress_x >= min(
                        span_w - (span_w - right_lim) * stall_margin,
                        right_lim - 4.0,
                    )
                    # Prefer raw yellow X for wall decisions when available.
                    edge_x = progress_x
                    if (
                        progress_src == "minimap"
                        and self.loc_player_minimap is not None
                    ):
                        try:
                            edge_x = float(self.loc_player_minimap[0])
                            near_left = edge_x <= max(
                                left_lim * stall_margin, left_lim + 4.0
                            )
                            near_right = edge_x >= min(
                                span_w - (span_w - right_lim) * stall_margin,
                                right_lim - 4.0,
                            )
                        except Exception:
                            edge_x = progress_x
                    edge_stall_hit = False
                    if edge_stall_n > 0 and progress_x is not None and span_w > 0:
                        last_raw = getattr(self, "_hunt_edge_last_raw_x", None)
                        if last_raw is not None:
                            if self._hunt_search_dir == "left":
                                progressed = (float(last_raw) - edge_x) >= edge_stall_min_delta
                            elif self._hunt_search_dir == "right":
                                progressed = (edge_x - float(last_raw)) >= edge_stall_min_delta
                            else:
                                progressed = abs(edge_x - float(last_raw)) >= edge_stall_min_delta
                            if progressed:
                                self._hunt_edge_stall_frames = 0
                            else:
                                self._hunt_edge_stall_frames = int(
                                    getattr(self, "_hunt_edge_stall_frames", 0)
                                ) + 1
                        else:
                            self._hunt_edge_stall_frames = 0
                        self._hunt_edge_last_raw_x = edge_x
                        if self._hunt_edge_stall_frames >= edge_stall_n:
                            if self._hunt_search_dir == "left" and near_left:
                                edge_stall_hit = True
                            elif self._hunt_search_dir == "right" and near_right:
                                edge_stall_hit = True
                    else:
                        self._hunt_edge_last_raw_x = edge_x

                    if self._hunt_search_dir == "left" and edge_x <= left_lim:
                        flip_reason = "edge_left"
                    elif self._hunt_search_dir == "right" and edge_x >= right_lim:
                        flip_reason = "edge_right"
                    elif edge_stall_hit and self._hunt_search_dir == "left":
                        flip_reason = "edge_stall_left"
                    elif edge_stall_hit and self._hunt_search_dir == "right":
                        flip_reason = "edge_stall_right"

                    if flip_reason is not None:
                        desired = (
                            "right"
                            if flip_reason in ("edge_left", "edge_stall_left")
                            else "left"
                        )
                        prev_dir = self._hunt_search_dir
                        self._hunt_search_dir = desired
                        self._hunt_dir_forced = True
                        self._hunt_edge_stall_frames = 0
                        kb = getattr(self, "_kb", None)
                        if kb is not None and hasattr(kb, "request_dir_override"):
                            try:
                                kb.request_dir_override(desired)
                            except Exception:
                                pass
                        if self.frame_count % 3 == 0 or "stall" in flip_reason:
                            logger.info(
                                f"[Patrol] {flip_reason} {prev_dir}->{desired} "
                                f"x={progress_x:.1f}/{progress_src} "
                                f"lim=[{left_lim:.0f},{right_lim:.0f}]/{span_w:.0f}"
                            )

        # Stall recovery:
        # - Mid-zone (F2 yellow-dot 80% patrol): NEVER stall-flip / stall-nudge inside the
        #   band. Jump+attack (S then A) barely moves X for many frames; the old
        #   stall_in <-> stall_nudge pair caused left/right thrashing every ~1.5s.
        # - Edge flash_patrol (F1): stall disabled (edge-only).
        # - Non-flash legacy: keep stall toggle.
        flash_patrol = bool(self.cfg.get("bot", {}).get("flash_patrol", False))
        if mid_zone:
            # Only recover when stuck OUTSIDE the mid band (false lock near wall).
            stall_frames_limit = int(cfg.get("search_stall_frames", 0))
        else:
            stall_frames_limit = 0 if flash_patrol else int(cfg.get("search_stall_frames", 8))
        if stall_frames_limit > 0:
            stall_min_delta_px = float(cfg.get("search_stall_min_delta_px", 1.0))
            if progress_x is not None and self._hunt_search_last_progress_x is not None:
                progress_delta = abs(progress_x - self._hunt_search_last_progress_x)
                if progress_delta <= stall_min_delta_px:
                    self._hunt_search_stall_frames += 1
                else:
                    self._hunt_search_stall_frames = 0
                self._hunt_search_last_progress_x = progress_x
                if progress_src in ("global", "minimap") and self._hunt_search_stall_frames >= stall_frames_limit:
                    if mid_zone and progress_x is not None and span_w > 0:
                        left_bound = float(left_ratio) * span_w
                        right_bound = float(right_ratio) * span_w
                        if right_bound < left_bound:
                            left_bound, right_bound = right_bound, left_bound
                        # Only force inward when outside band; never nudge opposite inside.
                        if progress_x < left_bound or progress_x > right_bound:
                            desired = "right" if progress_x < left_bound else "left"
                            if self._hunt_search_dir != desired:
                                self._hunt_search_dir = desired
                                self._hunt_dir_forced = True
                                flip_reason = "stall_in"
                                self._hunt_dir_change_frame = self.frame_count
                        # inside band + stalled: keep current sticky direction (no flip)
                    else:
                        flip_reason = "stall"

        max_offset_px = 0.0 if flash_patrol else float(cfg.get("search_max_offset_px", 0))
        if (
            flip_reason is None
            and max_offset_px > 0
            and progress_src in ("global", "minimap")
            and progress_x is not None
            and self._hunt_search_anchor_x is not None
        ):
            offset_x = progress_x - self._hunt_search_anchor_x
            if self._hunt_search_dir == "left" and offset_x <= -max_offset_px:
                flip_reason = "bound"
            elif self._hunt_search_dir == "right" and offset_x >= max_offset_px:
                flip_reason = "bound"

        if flip_timer_enabled and not flash_patrol:
            if self._hunt_search_flip_frame <= 0:
                self._hunt_search_flip_frame = self.frame_count + flip_frames
            elif flip_reason is None and self.frame_count >= self._hunt_search_flip_frame:
                flip_reason = "timer"
        else:
            # Edge-only: never schedule or fire timer flips.
            self._hunt_search_flip_frame = 0

        if flip_reason is not None:
            # Mid-zone already set _hunt_search_dir; other reasons still toggle.
            if not getattr(self, "_hunt_dir_forced", False):
                self._hunt_search_dir = "right" if self._hunt_search_dir == "left" else "left"
            self._hunt_dir_forced = False
            self._hunt_dir_change_frame = self.frame_count
            self._hunt_search_flip_frame = (
                self.frame_count + flip_frames if flip_timer_enabled and not flash_patrol else 0
            )
            self._hunt_search_sweeps += 1
            self._hunt_search_stall_frames = 0
            self._hunt_search_last_reason = flip_reason
            if progress_x is not None:
                self._hunt_search_anchor_x = progress_x
                self._hunt_search_last_progress_x = progress_x
            if mid_zone and flash_patrol:
                self._on_patrol_leg_flip(cfg)

        max_sweeps = int(cfg.get("search_max_sweeps", 0))
        if max_sweeps > 0 and self._hunt_search_sweeps >= max_sweeps:
            pause_frames = max(0, int(cfg.get("search_pause_frames", 12)))
            self._hunt_search_pause_until = self.frame_count + pause_frames
            self._hunt_search_sweeps = 0
            self._hunt_search_last_reason = "pause"
            return "none", "none", "none"

        action = "attack" if bool(cfg.get("search_attack_while_move", True)) else "none"

        # Random jump during search (looks more human-like)
        jump_chance = float(cfg.get("search_random_jump_chance", 0.0))
        if jump_chance > 0:
            jump_cd = max(1, int(cfg.get("search_random_jump_cd", 30)))
            if not hasattr(self, '_hunt_next_jump_frame'):
                self._hunt_next_jump_frame = 0
            if self.frame_count >= self._hunt_next_jump_frame:
                import random
                if random.random() < jump_chance:
                    action = "jump"
                    self._hunt_next_jump_frame = self.frame_count + jump_cd

        return self._hunt_search_dir, "none", action

    def reset_station_hold(self):
        """Clear F3 center lock so the next frame re-captures current position."""
        self._station_center_x = None
        self._station_center_y = None
        self._station_lock_logged = False

    def _get_station_hold_command(self):
        """
        Stand at the enable-time minimap position and AoE on cooldown.

        If yellow-dot X drifts beyond station_deadzone_px, walk back toward
        the locked center. Y is optional (station_hold_y).
        Returns (move_x, move_y, action).
        """
        da = self.cfg.get("directional_attack", {}) or {}
        deadzone = float(da.get("station_deadzone_px", 6))
        hold_y = bool(da.get("station_hold_y", False))
        deadzone_y = float(da.get("station_deadzone_y_px", deadzone))

        px = py = None
        if self.loc_player_minimap is not None:
            px = float(self.loc_player_minimap[0])
            py = float(self.loc_player_minimap[1])

        if px is None:
            # No yellow yet — still AoE in place
            return "none", "none", "aoe"

        if self._station_center_x is None:
            self._station_center_x = px
            self._station_center_y = py
            if not self._station_lock_logged:
                self._station_lock_logged = True
                logger.info(
                    f"[Station] F3 center locked minimap=({px:.1f},{py if py is not None else '?'}) "
                    f"deadzone={deadzone:.0f}px"
                )

        move_x = "none"
        dx = px - float(self._station_center_x)
        if dx > deadzone:
            move_x = "left"
        elif dx < -deadzone:
            move_x = "right"

        move_y = "none"
        if hold_y and py is not None and self._station_center_y is not None:
            dy = py - float(self._station_center_y)
            if dy > deadzone_y:
                move_y = "up"
            elif dy < -deadzone_y:
                move_y = "down"

        return move_x, move_y, "aoe"

    def _get_in_place_command(self):
        """
        In-place attack with periodic small back-and-forth, to avoid MapleStory's
        attack lockout when standing still too long.

        Walks backward (opposite of facing) first, then forward (facing dir)
        to return to roughly the original spot.

        State machine:
          attack -> (after interval) -> walk_away(backward) ->
          walk_back(forward) -> (after walk_frames) -> attack
        Returns (move_x, move_y, action).
        """
        cfg = self.cfg.get("directional_attack", {})
        interval = max(30, int(cfg.get("in_place_walk_interval", 300)))
        walk_frames = max(1, int(cfg.get("in_place_walk_frames", 20)))
        stop_frames = max(1, int(cfg.get("in_place_stop_attack_frames", 8)))  # ~0.5s @15fps

        # Advance state machine
        if self._in_place_state == "attack":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= interval:
                # Decide backward direction: opposite of current facing.
                facing = self._get_attack_facing()
                if facing in ("left", "right"):
                    self._in_place_backward_dir = "right" if facing == "left" else "left"
                else:
                    # No facing info (no monster) — alternate as fallback.
                    self._in_place_backward_dir = (
                        "right" if self._in_place_walk_dir == "left" else "left"
                    )
                self._in_place_walk_dir = self._in_place_backward_dir
                self._in_place_state = "pre_walk"
                self._in_place_state_frame = 0
        elif self._in_place_state == "pre_walk":
            # Stop attacking before moving (avoids attacking while walking).
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= stop_frames:
                self._in_place_state = "walk_away"
                self._in_place_state_frame = 0
        elif self._in_place_state == "walk_away":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= walk_frames:
                self._in_place_state = "walk_back"
                self._in_place_state_frame = 0
        elif self._in_place_state == "walk_back":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= walk_frames:
                self._in_place_state = "attack"
                self._in_place_state_frame = 0

        # Produce command per state
        if self._in_place_state == "walk_away":
            return self._in_place_backward_dir, "none", "none"   # 往后走
        elif self._in_place_state == "walk_back":
            forward_dir = "right" if self._in_place_backward_dir == "left" else "left"
            return forward_dir, "none", "none"                    # 往前走

        # "attack" / "pre_walk": stand still, no movement.
        # Attack only during "attack" state (not during pre_walk pause).
        if self._in_place_state == "attack":
            if self.monsters:
                aoe_threshold = int(
                    self.cfg.get("directional_attack", {}).get("aoe_monster_count", 3)
                )
                action = "aoe" if len(self.monsters) >= aoe_threshold else "attack"
                # Turn to face the nearest monster before attacking.
                facing = self._get_attack_facing()
                if facing in ("left", "right"):
                    return f"face_{facing}", "none", action
                return "none", "none", action
            return "none", "none", "none"
        return "none", "none", "none"

    def _update_debug_overlay(self):
        """Add FPS, HP/MP/EXP, and status info to the debug frame."""

        # --- HP / MP / EXP display (top-right) ---
        if self.health_monitor is not None:
            hm = self.health_monitor
            bar_count = self.cfg.get("health_monitor", {}).get("bar_count", 3)
            if bar_count == 2:
                bar_labels = [("HP", hm.hp_percent, (0, 0, 255), hm.enable_hp),
                              ("MP", hm.mp_percent, (255, 0, 0), hm.enable_mp)]
            else:
                bar_labels = [("HP", hm.hp_percent, (0, 0, 255), hm.enable_hp),
                              ("MP", hm.mp_percent, (255, 0, 0), hm.enable_mp),
                              ("EXP", hm.exp_percent, (0, 255, 0), True)]
            fw = self.img_frame.shape[1]
            x_s = fw - 220  # Right-aligned
            y_s = 30
            for i, (label, pct, color, enabled) in enumerate(bar_labels):
                status = "ON" if enabled else "OFF"
                text = f"{label}: {pct:.1f}% [{status}]"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(
                    self.img_frame_debug,
                    text,
                    (x_s, y_s + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
                # Bar preview to the left of text
                if i < len(hm.loc_size_bars):
                    bx, by, bw, bh = hm.loc_size_bars[i]
                    if bw > 0 and bh > 0:
                        ui_y = self.cfg.get("ui_coords", {}).get("ui_y_start", 610)
                        bar_src = self.img_frame[ui_y:, :]
                        h_bar, w_bar = bar_src.shape[:2]
                        if by + bh <= h_bar and bx + bw <= w_bar:
                            preview = bar_src[by:by + bh, bx:bx + bw].copy()
                            preview = cv2.resize(preview, (bw * 2, bh * 2),
                                                 interpolation=cv2.INTER_NEAREST)
                            px = x_s - bw * 2 - 10  # Preview to left of text
                            py = y_s + 30 * i - bh
                            ph, pw = preview.shape[:2]
                            if (py >= 0 and px >= 0
                                    and py + ph <= self.img_frame_debug.shape[0]
                                    and px + pw <= self.img_frame_debug.shape[1]):
                                self.img_frame_debug[py:py + ph, px:px + pw] = preview

        # --- Bot mode indicator (top-right, below HP/MP) ---
        bot_mode = self.cfg.get("bot", {}).get("mode", "detect")
        mode_color = (0, 255, 0) if bot_mode == "hunt" else (0, 200, 255)
        mode_text = f"Bot: {bot_mode.upper()} [F11]"
        (mw, mh), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        mode_x = self.img_frame.shape[1] - mw - 12
        mode_y = 30 + 30 * 3  # below HP/MP/EXP rows
        cv2.rectangle(self.img_frame_debug,
                      (mode_x - 4, mode_y - mh - 4),
                      (mode_x + mw + 4, mode_y + 4),
                      (0, 0, 0), -1)
        cv2.putText(self.img_frame_debug, mode_text,
                    (mode_x, mode_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA)

        # --- Status texts (bottom-left) ---
        diff_thres = self.cfg["monster_detect"]["diff_thres"]
        route_on = self.cfg["bot"].get("enable_route", False)
        texts = [
            f"FPS(P/D): {self.fps}/{self.display_fps} | KB: {'ON' if self._kb and self._kb.is_enable else 'OFF'}",
            f"Monsters: {len(self.monsters)} | Best: {self.best_score:.4f}/{diff_thres}",
            f"Mode: {self.cfg['monster_detect']['mode']} | Scale: {self.cfg.get('monster_detect',{}).get('display_scale',1)}",
            f"Frame: {self.img_frame.shape[1]}x{self.img_frame.shape[0]}",
        ]

        if self.cfg.get("bot", {}).get("mode", "detect") == "hunt":
            texts.append(
                f"Search: noMob={self._hunt_no_monster_frames} dir={self._hunt_search_dir} sweeps={self._hunt_search_sweeps} stall={self._hunt_search_stall_frames} last={self._hunt_search_last_reason}"
            )

        if route_on:
            gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
            mx, my = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])
            texts += [
                f"Route: {self.idx_routes+1}/{len(self.img_routes)} Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                f"Global: ({gx},{gy}) MM:({mx},{my}) PMM:{self.loc_player_minimap or '?'} Match:{getattr(self,'_last_match_score',1):.3f}",
                f"Ladder: {self.is_on_ladder} | Map: {self.img_map.shape[1]}x{self.img_map.shape[0]}" if self.img_map is not None else "No map loaded",
            ]
        prof = getattr(self._kb, "_active_profile", None) if self._kb else None
        kb_on = bool(self._kb and self._kb.is_enable)
        texts += [
            f"S=shot Q=quit | F1=12139 F2=12138 F3=station | "
            f"KB:{'ON' if kb_on else 'IDLE'} prof:{prof or '-'}"
        ]
        if prof == "f3" and self._station_center_x is not None:
            cur = self.loc_player_minimap
            texts.append(
                f"Station: center={self._station_center_x:.0f} "
                f"cur={cur[0] if cur else '?'} dz="
                f"{float((self.cfg.get('directional_attack') or {}).get('station_deadzone_px', 6)):.0f}"
            )

        # Bottom-left positioning (matches reference project)
        text_y_interval = 25
        text_y_start = self.img_frame.shape[0] - len(texts) * text_y_interval - 10
        font_scale = 0.8  # +2 sizes from 0.6

        for i, text in enumerate(texts):
            x, y = 10, text_y_start + text_y_interval * i
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(
                    self.img_frame_debug, text,
                    (x + dx, y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA
                )
            cv2.putText(
                self.img_frame_debug, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA
            )

        # Monster count in top-right corner (big text, eye-catching)
        if self.monsters:
            count_text = f"Monsters: {len(self.monsters)}"
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(
                self.img_frame_debug, count_text,
                (self.img_frame.shape[1] - tw - 15, th + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
            )

        self._draw_title_candidates()

    # -------------------------------------------------------------------
    # Main processing loop
    # -------------------------------------------------------------------

    def run_once(self) -> int:
        """
        Process a single game frame.

        Returns:
            0 on success, -1 if frame capture failed.
        """
        self.frame_count += 1
        self.is_frame_done = False

        # Capture frame
        self.img_frame = self.get_img_frame()
        if self.img_frame is None or self.img_frame.shape[0] < 300 or self.img_frame.shape[1] < 400:
            logger.warning("Failed to capture game frame (missing/minimized?). Waiting...")
            if self._kb is not None:
                self._kb._chase_active = False
                self._kb.set_command("none none none")
                # Only release script-held keys while AFK is ON; when OFF, leave
                # the user's keyboard alone.
                if bool(getattr(self._kb, "is_enable", False)):
                    self._kb.release_all()
            if is_windows() and self.capture is not None:
                activate_game_window(self.capture.window_title)
            return -1

        # Convert to grayscale (for title detection + grayscale matching)
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # 测谎 dialog: alert + pause AFK (mouse-follow only when auto_follow=true)
        if getattr(self, "lie_alert", None) is not None:
            try:
                self.lie_alert.maybe_check(
                    self.img_frame,
                    self.frame_count,
                    kb=self._kb,
                    capture=self.capture,
                )
            except Exception as _lie_e:
                if self.frame_count % 120 == 0:
                    logger.debug(f"[LieAlert] check error: {_lie_e}")

        # Update health monitor with bottom UI strip
        if self.health_monitor is not None:
            ui_y = self.cfg.get("ui_coords", {}).get("ui_y_start", 610)
            frame_h = self.img_frame.shape[0]
            if 0 <= ui_y < frame_h:
                self.health_monitor.update_frame(self.img_frame[ui_y:, :])
            else:
                self.health_monitor.update_frame(None)
                if self.frame_count % 90 == 0:
                    logger.warning(
                        f"[HealthMonitor] ui_y_start out of frame: ui_y_start={ui_y}, frame_h={frame_h}"
                    )

        # Detect minimap (adapted for MapleStoryGo: bright fill + dark borders)
        minimap_result = self._detect_minimap()
        self.img_minimap = None
        if minimap_result:
            x, y, w, h = minimap_result
            fh, fw = self.img_frame.shape[:2]
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(fw, int(x + w))
            y1 = min(fh, int(y + h))
            self.loc_minimap = (x0, y0)
            if x1 > x0 and y1 > y0:
                self.img_minimap = self.img_frame[y0:y1, x0:x1]
            elif self.frame_count % 60 == 0:
                logger.warning(
                    f"[Minimap] Empty ROI after clamp: raw=({x},{y},{w},{h}) "
                    f"frame=({fw}x{fh})"
                )

            # Find player yellow diamond on minimap (reject border false locks)
            if self.img_minimap is not None and self.img_minimap.size > 0:
                found = self._find_player_dot_on_minimap(self.img_minimap)
                if found is not None:
                    self.loc_player_minimap = found
                elif self.frame_count % 60 == 0:
                    logger.warning("[Minimap] Player yellow diamond not found this frame")

            # Global positioning (layer 3)
            if self.cfg["bot"].get("enable_route") and self.img_map is not None:
                self.get_player_global_position()

        # Route following
        route_action = "none"
        if self.cfg["bot"].get("enable_route") and self.img_route is not None:
            # Ladder detection
            prev = getattr(self, '_prev_loc', None)
            if prev and self.loc_player != (0, 0):
                dx = abs(self.loc_player[0] - prev[0])
                dy = abs(self.loc_player[1] - prev[1])
                if self.is_on_ladder:
                    if dx > 3:
                        self.is_on_ladder = False
                else:
                    if dx < 3 and dy != 0:
                        self.is_on_ladder = True
            self._prev_loc = self.loc_player

            self.update_cmd_by_route()
            route_action = self.cmd_action

        # Prepare debug frame before player detect so NAME/TITLE boxes can draw.
        # Only every N frames — full-res .copy()+overlay was a major stutter source.
        viz_every = max(1, int(
            (self.cfg.get("monster_detect") or {}).get("display_every_n", 2)
        ))
        self._do_viz = bool(self.is_show_debug) and (self.frame_count % viz_every == 0)
        if self._do_viz:
            self.img_frame_debug = self.img_frame.copy()
        else:
            self.img_frame_debug = None

        # Detect player: prefer floating nameplate. On name miss in hunt, use screen
        # center (not sticky wrong title) so monster ROI / max_dy stay valid.
        # During 测谎 follow, skip nameplate — AFK is paused and CPU goes to mouse track.
        lie_busy = False
        la = getattr(self, "lie_alert", None)
        if la is not None:
            lie_busy = bool(getattr(la, "visible", False) or getattr(la, "following", False))
        found_name = False if lie_busy else self._detect_player_by_name()
        if not found_name:
            name_enabled = bool(self.cfg.get("player_name", {}).get("enable", True)) and self.img_name_gray is not None
            hunt_mode = self.cfg.get("bot", {}).get("mode") == "hunt"
            use_center_fallback = bool(
                self.cfg.get("directional_attack", {}).get("chase_use_screen_center", False)
            )
            # Name is preferred; if it misses while hunting, pin center so we still see mobs.
            if self.img_frame is not None and (hunt_mode and name_enabled or use_center_fallback):
                h, w = self.img_frame.shape[:2]
                self.loc_player = (w // 2, h // 2)
                self._player_src = "center"
                if self.frame_count <= 3 or self.frame_count % 60 == 0:
                    logger.info(f"[Name] miss -> center fallback player=({w//2},{h//2})")
            else:
                self._detect_player_by_title()
                if self._player_src.startswith("name"):
                    pass
                elif self.loc_title is not None:
                    self._player_src = "title"
                else:
                    self._player_src = "fallback"

        # Compute search region (same formula for detection & display)
        margin = self.cfg["monster_detect"].get("search_box_margin", 50)
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
        else:
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin

        # Prefer name/title player anchor; optional screen-center only if explicitly enabled
        anchor_x, anchor_y = self.loc_player
        force_center = bool(self.cfg.get("directional_attack", {}).get("chase_use_screen_center", False))
        name_ok = self._player_src.startswith("name")
        if (
            self.cfg.get("bot", {}).get("mode") == "hunt"
            and force_center
            and not name_ok
        ):
            anchor_x = self.img_frame.shape[1] // 2
            anchor_y = self.img_frame.shape[0] // 2
            dx = max(dx, self.img_frame.shape[1] // 2 - 20)
        # 垂直至少覆盖 max_dy 过滤带 + 模板高度余量，避免脚下真怪被 ROI 裁掉
        max_dy = int(self.cfg.get("monster_detect", {}).get("max_dy_from_player", 140))
        dy = max(dy, max_dy + 80)
        x0 = max(0, anchor_x - dx)
        x1 = min(self.img_frame.shape[1], anchor_x + dx)
        y0 = max(0, anchor_y - dy)
        y1 = min(self.img_frame.shape[0], anchor_y + dy)

        # ---------------------------------------------------------------
        # Monster detection — skip entirely when enable=false (F2 pre-monster
        # flash_patrol: yellow bounds only; saves CPU / no Best overlay noise).
        # ---------------------------------------------------------------
        flash_patrol = bool(self.cfg.get("bot", {}).get("flash_patrol", False))
        md_enabled = bool(self.cfg.get("monster_detect", {}).get("enable", False))
        if not md_enabled:
            self.monsters = []
            self._hunt_mob_last_lr = (0, 0)
        else:
            # When detect+steer on: widen ROI for L/R floor counts.
            if flash_patrol and self.img_frame is not None:
                fw = int(self.img_frame.shape[1])
                fh = int(self.img_frame.shape[0])
                pad = int(self.cfg.get("monster_detect", {}).get("full_width_pad", 8))
                x0 = max(0, pad)
                x1 = max(x0 + 1, fw - pad)
                # keep vertical band around player / bottom half
                y0 = max(0, min(y0, int(fh * 0.25)))
                y1 = min(fh, max(y1, int(fh * 0.95)))
            interval = self.cfg["monster_detect"].get("detection_interval", 2)
            if self.frame_count % interval == 0:
                self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))
                if self.monsters:
                    names = set(m["name"] for m in self.monsters)
                    logger.debug(
                        f"Detected {len(self.monsters)} monsters: {names}"
                    )

        # Dispatch movement/action command after detection so attack can react this frame.
        if self._kb:
            move_x, move_y = "none", "none"
            action = "none"
            route_move_x, route_move_y = "none", "none"

            # Stopped via F1/F2: never dispatch hunt cmds.
            # Do NOT call release_all() here — that fights the user's held arrows
            # and makes the character unable to move after toggle-off.
            # disable() already released keys once.
            if not bool(getattr(self._kb, "is_enable", False)):
                try:
                    self._kb._chase_active = False
                    self._kb.set_command("none none none")
                except Exception:
                    pass
            elif getattr(self, "lie_alert", None) is not None and getattr(self.lie_alert, "visible", False):
                # Keep standing while 测谎 is up; mouse-follow handles the mini-game
                try:
                    self._kb._chase_active = False
                    self._kb.set_command("none none none")
                    self._kb.pause_for_lie_alert(
                        float((self.cfg.get("lie_alert") or {}).get("freeze_seconds", 3.0))
                    )
                except Exception:
                    pass
            elif self.cfg["bot"].get("enable_route") and self.img_route is not None:
                route_move_x, route_move_y = self.cmd_move_x, self.cmd_move_y
                move_x, move_y = route_move_x, route_move_y
                action = route_action

            # Attack switch: in hunt mode, combat takes priority over route actions.
            mode = self.cfg.get("bot", {}).get("mode", "detect")
            attack_style = self.cfg.get("bot", {}).get("attack", "directional")
            attack_in_place = bool(
                self.cfg.get("directional_attack", {}).get("attack_in_place", False)
            )

            lie_up = bool(getattr(self, "lie_alert", None) is not None and getattr(self.lie_alert, "visible", False))
            station_hold = bool(
                self.cfg.get("directional_attack", {}).get("station_hold", False)
            )
            if (not lie_up) and bool(getattr(self._kb, "is_enable", False)) and mode == "hunt" and station_hold:
                # F3: lock enable-time yellow X, walk back if drifted, AoE on CD
                self._reset_hunt_search_state()
                move_x, move_y, action = self._get_station_hold_command()
                self._kb._chase_active = False
            elif (not lie_up) and bool(getattr(self._kb, "is_enable", False)) and mode == "hunt" and attack_in_place:
                # 原地攻击 + 定期左右走动（防攻击失效）
                self._reset_hunt_search_state()
                move_x, move_y, action = self._get_in_place_command()
            elif (not lie_up) and bool(getattr(self._kb, "is_enable", False)) and mode == "hunt":
                flash_patrol = bool(self.cfg.get("bot", {}).get("flash_patrol", False))
                if flash_patrol:
                    # F2 mid patrol: yellow bounds force turn; keep simple L/R mid
                    # patrol + AoE. Monster L/R steering only if search_mob_steer.
                    da_fp = self.cfg.get("directional_attack", {}) or {}
                    md_on = bool(self.cfg.get("monster_detect", {}).get("enable", False))
                    steer_on = bool(da_fp.get("search_mob_steer", False))
                    if md_on and steer_on and self.monsters:
                        try:
                            self._count_monsters_lr()
                        except Exception:
                            pass
                    else:
                        self._hunt_mob_last_lr = (0, 0)
                    self._hunt_no_monster_frames = max(
                        self._hunt_no_monster_frames + 1,
                        int(self.cfg.get("directional_attack", {}).get("search_trigger_frames", 1)),
                    )
                    patrol_x, patrol_y, _patrol_action = self._get_hunt_search_command()
                    move_x = patrol_x if patrol_x in ("left", "right") else self._hunt_search_dir
                    move_y = "none"
                    # Bottom-floor band (yellow-dot Y): if above bottom platform, hold Down
                    # so jump (S) drops through pads back to the ground strip.
                    da = self.cfg.get("directional_attack", {}) or {}
                    bot_ymin = da.get("search_bottom_y_min_ratio", None)
                    if bot_ymin is not None and self.img_minimap is not None:
                        py = self._get_hunt_search_progress_y()
                        mh = float(self.img_minimap.shape[0])
                        if py is not None and mh > 0 and py < float(bot_ymin) * mh:
                            move_y = "down"
                    action = "aoe"
                    # Mark chase_active so walk_teleport flashes Z while holding left/right
                    self._kb._chase_active = True if move_x in ("left", "right") else False
                else:
                    # Legacy: chase nearest monster then AoE
                    if self.monsters:
                        self._reset_hunt_search_state()
                    else:
                        self._hunt_no_monster_frames += 1

                    saved_no = self._hunt_no_monster_frames
                    self._hunt_no_monster_frames = max(
                        saved_no,
                        int(self.cfg.get("directional_attack", {}).get("search_trigger_frames", 1)),
                    )
                    patrol_x, patrol_y, patrol_action = self._get_hunt_search_command()
                    self._hunt_no_monster_frames = saved_no

                    if self.monsters:
                        aoe_threshold = int(
                            self.cfg.get("directional_attack", {}).get("aoe_monster_count", 3)
                        )
                        if attack_style == "aoe_skill":
                            action = "aoe"
                        else:
                            action = "aoe" if len(self.monsters) >= aoe_threshold else "attack"

                        hunt_px = self._hunt_player_x()
                        nearest_dx = self._get_nearest_monster_dx_from(hunt_px)
                        engage_px = int(
                            self.cfg.get("directional_attack", {}).get("engage_px", 140)
                        )
                        deadzone = int(
                            self.cfg.get("directional_attack", {}).get("face_deadzone_px", 8)
                        )
                        aoe_half = int(self.cfg.get("aoe_skill", {}).get("range_x", 400)) // 2
                        in_aoe = nearest_dx is not None and abs(nearest_dx) <= max(engage_px, aoe_half)

                        if nearest_dx is None:
                            move_x, move_y = "none", "none"
                            self._kb._chase_active = False
                        elif not in_aoe:
                            move_x = "right" if nearest_dx > 0 else "left"
                            move_y = "none"
                            self._hunt_search_dir = move_x
                            self._kb._chase_active = True
                            if bool(self.cfg.get("directional_attack", {}).get("chase_no_attack", False)):
                                action = "none"
                        else:
                            face = "right" if nearest_dx > 0 else "left"
                            if abs(nearest_dx) <= deadzone:
                                move_x = "none"
                            else:
                                move_x = f"face_{face}"
                            move_y = "none"
                            self._hunt_search_dir = face
                            self._kb._chase_active = False
                    else:
                        self._kb._chase_active = False
                        if patrol_x != "none" or patrol_y != "none" or patrol_action != "none":
                            move_x, move_y, action = patrol_x, patrol_y, patrol_action


            log_n = max(15, int((self.cfg.get("system") or {}).get("hunt_log_every_n", 45)))
            if mode == "hunt" and self.frame_count % log_n == 0:
                dx_log = None
                if mode == "hunt":
                    try:
                        dx_log = self._get_nearest_monster_dx_from(self._hunt_player_x())
                    except Exception:
                        dx_log = None
                px_mm, px_src = self._get_hunt_search_progress_x()
                py_mm = self._get_hunt_search_progress_y()
                da = self.cfg.get("directional_attack", {}) or {}
                lr = da.get("search_left_ratio")
                rr = da.get("search_right_ratio")
                if px_src == "screen" and self.img_frame is not None:
                    span = float(self.img_frame.shape[1])
                elif self.img_minimap is not None:
                    span = float(self.img_minimap.shape[1])
                else:
                    span = 0.0
                span_h = float(self.img_minimap.shape[0]) if self.img_minimap is not None else 0.0
                zone = ""
                if lr is not None and rr is not None and span > 0:
                    zone = f" mid=[{float(lr)*span:.0f},{float(rr)*span:.0f}]/{span:.0f}"
                if py_mm is not None and span_h > 0:
                    zone += f" y={py_mm:.0f}/{span_h:.0f}"
                en = bool(getattr(self._kb, "is_enable", False))
                blocked = bool(getattr(self._kb, "_startup_combat_blocked", False))
                lr = getattr(self, "_hunt_mob_last_lr", (0, 0))
                logger.info(
                    f"[Hunt] en={en} blocked={blocked} flash={bool(self.cfg.get('bot',{}).get('flash_patrol',False))} "
                    f"monsters={len(self.monsters)} L/R={lr[0]}/{lr[1]} dx={dx_log} chase={getattr(self._kb,'_chase_active',False)} "
                    f"dispatch=({move_x},{move_y},{action}) "
                    f"search=(noMob:{self._hunt_no_monster_frames},dir:{self._hunt_search_dir},"
                    f"last:{self._hunt_search_last_reason},x={px_mm}/{px_src}{zone}) "
                    f"fps={self.fps}"
                )

            # Buff casting / initial buff wait: force stand still, no attack.
            # Otherwise left/right + AoE keep landing and cancel the cast.
            try:
                if hasattr(self._kb, "is_buff_frozen") and self._kb.is_buff_frozen():
                    move_x, move_y, action = "none", "none", "none"
                    self._kb._chase_active = False
            except Exception:
                pass
            if lie_up:
                move_x, move_y, action = "none", "none", "none"
                try:
                    self._kb._chase_active = False
                except Exception:
                    pass
            self._kb.set_command(f"{move_x} {move_y} {action}")

        # Draw decorations only on viz ticks (img_frame_debug is None otherwise).
        if self.is_show_debug and self.img_frame_debug is not None:
            self._update_debug_overlay()        # Text first (bottom layer)
            self._draw_search_region(x0, y0, x1, y1)  # Search border
            # 测谎 star follow reticle (same idea as the demo video overlay)
            try:
                la = getattr(self, "lie_alert", None)
                pt = getattr(la, "debug_pt", None) if la is not None else None
                if pt is not None and self.img_frame_debug is not None:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(self.img_frame_debug, (px, py), 14, (0, 0, 255), 2)
                    cv2.drawMarker(
                        self.img_frame_debug,
                        (px, py),
                        (0, 255, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )
                    shape_name = getattr(la, "_last_shape_name", "") or ""
                    if getattr(la, "following", False):
                        tag = f"LIE:{shape_name}" if shape_name else "LIE-FOLLOW"
                    else:
                        tag = "LIE"
                    cv2.putText(
                        self.img_frame_debug,
                        tag,
                        (px + 12, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
            except Exception:
                pass
            # Draw minimap rectangle (matches reference project)
            if self.img_minimap is not None:
                draw_rectangle(
                    self.img_frame_debug,
                    self.loc_minimap,
                    self.img_minimap.shape[:2],
                    (0, 0, 255), "minimap", thickness=2
                )
                # Draw player dot position on minimap
                if self.loc_player_minimap:
                    mx, my = self.loc_player_minimap
                    px = self.loc_minimap[0] + mx
                    py = self.loc_minimap[1] + my
                    # Bright green dot + label (BGR: 0,255,0 = green, not yellow)
                    cv2.circle(self.img_frame_debug, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(self.img_frame_debug, "P", (px+8, py+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Route minimap preview in top-right corner (matches reference project)
            if self.img_route is not None and self.loc_player_global != (0.0, 0.0):
                route_bgr = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)
                gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
                crop_w, crop_h = 80, 80
                # Draw player dot on route preview (yellow)
                cv2.circle(route_bgr, (gx, gy), 2, (0, 255, 255), -1)
                # Draw minimap matching rectangle
                if self.loc_minimap_global != (0, 0):
                    mmx, mmy = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])
                    mmw = self.img_minimap.shape[1] if self.img_minimap is not None else 0
                    mmh = self.img_minimap.shape[0] if self.img_minimap is not None else 0
                    if mmw > 0 and mmh > 0:
                        cv2.rectangle(route_bgr, (mmx, mmy), (mmx + mmw, mmy + mmh), (0, 255, 255), 1)
                x0 = max(0, gx - crop_w // 2)
                y0 = max(0, gy - crop_h // 2)
                x1 = min(route_bgr.shape[1], x0 + crop_w)
                y1 = min(route_bgr.shape[0], y0 + crop_h)
                if x1 > x0 and y1 > y0:
                    crop = route_bgr[y0:y1, x0:x1]
                    crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                      interpolation=cv2.INTER_NEAREST)
                    ch, cw = crop.shape[:2]
                    fh, fw = self.img_frame_debug.shape[:2]
                    px_paste = fw - cw - 10
                    py_paste = 10
                    self.img_frame_debug[py_paste:py_paste + ch, px_paste:px_paste + cw] = crop
                    cv2.rectangle(self.img_frame_debug, (px_paste, py_paste),
                                  (px_paste + cw, py_paste + ch), (255, 255, 255), 2)

            self._draw_player_marker()          # Player dot
            for monster in self.monsters:       # Monster boxes (top layer)
                color = (0, 255, 255) if monster["name"] == "Health Bar" else (0, 255, 0)
                draw_rectangle(
                    self.img_frame_debug,
                    monster["position"], monster["size"],
                    color,
                    f"{monster['name']}:{monster['score']:.2f}",
                    thickness=2, text_height=0.5
                )

            # Pre-scale for UI thread (only when we built a debug frame).
            scale = float(
                (self.cfg.get("monster_detect") or {}).get("display_scale", 0.55)
            )
            if scale != 1.0:
                self.img_display = cv2.resize(
                    self.img_frame_debug, None,
                    fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                self.img_display = self.img_frame_debug
            self.frame_seq += 1

        self.is_frame_done = True
        return 0

    def toggle_mode(self):
        """Toggle bot mode between "detect" (only visualize) and "hunt" (attack)."""
        current = self.cfg.get("bot", {}).get("mode", "detect")
        new_mode = "hunt" if current == "detect" else "detect"
        self.cfg["bot"]["mode"] = new_mode
        logger.info(f"[Bot] Mode switched to: {new_mode}")
        return new_mode

    def start(self):
        """
        Start the background processing thread and health monitor.
        Matches reference: MapleStoryAutoBot.start() → thread_auto_bot
        """
        # Start health monitor if enabled
        if (self.cfg.get("health_monitor", {}).get("enable", False)
                and self._kb is not None):
            self.health_monitor = HealthMonitor(self.cfg, self._kb)
            self.health_monitor.start()

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("[MonsterDetector] Background processing started.")

    def _process_loop(self):
        """
        Background processing loop. Captures frames and runs detection.
        Display is handled by the main thread in main.py.
        Matches reference: MapleStoryAutoBot.loop()
        """
        if is_windows() and self.capture and not self.test_image:
            activate_game_window(self.capture.window_title)

        logger.info("[MonsterDetector] Processing loop started...")

        while not self.is_terminated:
            t_start = time.time()

            # Process one frame (capture + optional detection)
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"[MonsterDetector] Frame processing error: {e}")
                continue

            # Cap FPS (matches reference project pattern)
            frame_duration = time.time() - t_start
            try:
                fps_limit = float(self.cfg["system"]["fps_limit_main"])
            except (ValueError, TypeError, KeyError):
                fps_limit = 15.0
            target_duration = 1.0 / max(fps_limit, 0.001)
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

            # FPS shown on overlay should reflect full loop time (including cap sleep).
            loop_duration = time.time() - t_start
            self.fps = round(1.0 / max(loop_duration, 0.001))
            # Periodic loop health — makes 2fps freezes obvious in the log.
            now_log = time.time()
            if now_log - float(getattr(self, "_fps_log_t", 0.0) or 0.0) >= 5.0:
                self._fps_log_t = now_log
                logger.info(
                    f"[Loop] fps={self.fps} frame_ms={frame_duration*1000:.0f} "
                    f"viz={int(bool(self.is_show_debug))} "
                    f"lie={int(bool(getattr(getattr(self,'lie_alert',None),'visible',False)))}"
                )

    def cleanup(self):
        """Clean up resources."""
        logger.info("[MonsterDetector] Cleaning up...")
        if self.health_monitor:
            self.health_monitor.stop()
        if self.capture:
            self.capture.stop()
        cv2.destroyAllWindows()
        logger.info("[MonsterDetector] Terminated.")

    def save_screenshot(self):
        """Save the current debug frame to disk."""
        if self.img_frame_debug is not None:
            screenshot(self.img_frame_debug, "detection")
        else:
            screenshot(self.img_frame, "raw_frame")
