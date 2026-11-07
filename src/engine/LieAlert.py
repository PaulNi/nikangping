# -*- coding: utf-8 -*-
"""
Lie-detector (测谎 / 谎言探测仪) alert + optional mouse-follow mini-game.

Detection:
  1) OpenCV TM_SQDIFF on UI chrome templates (CN title / EN banner)
  2) Optional RapidOCR for keywords (测谎 / 谎言探测仪 / LIE DETECTOR)

On detect: loud alert + pause AFK keys.

Auto-follow (盛趣「移动鼠标放在逐渐透明且移动的图形上」):
  Shapes are random (star / heart / polygon / …). After intro countdown we track
  any pale translucent moving silhouette via:
    - optional multi-shape templates under assets/lie_detector/shapes/
    - pale residual vs blurred background
    - frame-difference motion
  then move the real mouse onto it every frame (no clicks).
  If the user moves the cursor during the session, mouse-follow stops
  until this lie-detector dialog ends.
"""
from __future__ import annotations

import glob
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from src.engine.shape_peak_model import ShapePeakModel
from src.utils.logger import logger


class LieAlert:
    """Periodic template (+ optional OCR) match, alert, and optional shape follow."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        la = cfg.get("lie_alert") or {}
        self.enable = bool(la.get("enable", True))
        self.template_path = str(la.get("template", "assets/lie_detector/lie.png"))
        extras = la.get("templates") or []
        if isinstance(extras, str):
            extras = [extras]
        self.extra_templates = [str(p) for p in extras if str(p).strip()]
        self.match_threshold = float(la.get("match_threshold", 0.18))
        # Tiny/CN phone crops false-match easily — require near-perfect SQDIFF
        self.cn_title_match_threshold = float(la.get("cn_title_match_threshold", 0.035))
        self.check_interval_frames = max(1, int(la.get("check_interval_frames", 15)))
        self.alert_cooldown_s = float(la.get("alert_cooldown_s", 2.0))
        self.freeze_seconds = float(la.get("freeze_seconds", 8.0))
        self.hold_seconds = float(la.get("hold_seconds", 15.0))
        self.center_roi_ratio = float(la.get("center_roi_ratio", 0.85))
        # After a session times out, ignore new hits briefly (stops false-alarm loops)
        self.rearm_cooldown_s = float(la.get("rearm_cooldown_s", 8.0))
        # CN title template alone is unreliable; require OCR keyword confirm
        self.cn_title_require_ocr = bool(la.get("cn_title_require_ocr", True))
        self.ocr_enable = bool(la.get("ocr_enable", True))
        self.ocr_interval_frames = max(
            self.check_interval_frames, int(la.get("ocr_interval_frames", 90))
        )
        # RapidOCR is ~2s/call on 1366x768 — NEVER run it on every template miss.
        # Wall-clock gate keeps AFK loop ~20fps; templates catch intro banner.
        self.ocr_min_interval_s = float(la.get("ocr_min_interval_s", 2.5))
        self.ocr_during_follow = bool(la.get("ocr_during_follow", False))
        self._last_ocr_t = 0.0
        kws = la.get("ocr_keywords") or [
            "测谎",
            "测谎仪",
            "谎言探测仪",
            "谎言探测器",
            "测谎小游戏",
            "LIE DETECTOR",
            "Lie Detector",
        ]
        if isinstance(kws, str):
            kws = [kws]
        self.ocr_keywords = [str(k).strip() for k in kws if str(k).strip()]

        # Mouse-follow mini-game (CN 盛趣) — any translucent moving shape
        self.auto_follow = bool(la.get("auto_follow", True))
        self.follow_start_delay_s = float(la.get("follow_start_delay_s", 2.3))
        self.follow_timeout_s = float(la.get("follow_timeout_s", 50.0))
        self.follow_interval_frames = max(1, int(la.get("follow_interval_frames", 1)))
        # Prefer shape_*; keep star_* aliases for older configs
        self.shape_match_threshold = float(
            la.get(
                "shape_match_threshold",
                la.get("star_match_threshold", 0.50),
            )
        )
        # Separate thresholds: harder to LOCK, easier to KEEP once locked
        self.acquire_min_score = float(la.get("acquire_min_score", 0.58))
        self.track_min_score = float(la.get("track_min_score", 0.38))
        self.template_min_score = float(la.get("template_min_score", 0.70))
        self.max_track_jump_px = float(la.get("max_track_jump_px", 22))
        self.max_frame_step_px = float(la.get("max_frame_step_px", 22))
        self.max_speed_px_s = float(la.get("max_speed_px_s", 360.0))
        self.lock_search_radius = int(la.get("lock_search_radius", 56))
        self.vel_smooth = float(la.get("vel_smooth", 0.85))
        self.shape_blob_min_area = int(
            la.get("shape_blob_min_area", la.get("star_blob_min_area", 120))
        )
        self.shape_blob_max_area = int(
            la.get("shape_blob_max_area", la.get("star_blob_max_area", 4500))
        )
        self.white_v_min = int(la.get("white_v_min", 185))
        self.white_s_max = int(la.get("white_s_max", 55))
        self.local_contrast_min = float(la.get("local_contrast_min", 28.0))
        self.exclude_bottom_ratio = float(la.get("exclude_bottom_ratio", 0.14))
        self.exclude_top_ratio = float(la.get("exclude_top_ratio", 0.08))
        self.mouse_smooth = float(la.get("mouse_smooth", 0.55))
        self.mouse_takeover_px = float(la.get("mouse_takeover_px", 30.0))
        # Short settle only; never use this to ignore a large user yank.
        self.mouse_move_grace_s = float(la.get("mouse_move_grace_s", 0.08))
        self.lose_target_grace_s = float(la.get("lose_target_grace_s", 1.2))
        # White can flicker 1–2 frames at low FPS; do not flip to YOLO immediately.
        self.white_miss_grace_s = float(la.get("white_miss_grace_s", 0.45))
        # Wall-clock YOLO throttle (CPU predict was freezing the AFK loop ~2fps).
        self.fade_yolo_min_interval_s = float(la.get("fade_yolo_min_interval_s", 0.12))
        self.motion_weight = float(la.get("motion_weight", 0.25))
        self.template_weight = float(la.get("template_weight", 0.55))
        scales = (
            la.get("shape_scales")
            or la.get("star_scales")
            or [0.7, 0.85, 1.0, 1.15, 1.3]
        )
        self.shape_scales = [float(s) for s in scales]
        # Intro-only OCR gate. Never include「移动鼠标」/「逐渐变透明」— those stay
        # on the instruction bar for the whole mini-game and would block follow.
        intro_kws = la.get("intro_ocr_block_keywords") or [
            "2秒后",
            "3秒后",
            "1秒后",
            "测谎小游戏开始",
            "请前往安全地点",
            "放置在",
        ]
        if isinstance(intro_kws, str):
            intro_kws = [intro_kws]
        self.intro_ocr_block_keywords = [str(k).strip() for k in intro_kws if str(k).strip()]
        self.shapes_dir = str(
            la.get("shapes_dir", "assets/lie_detector/shapes")
        ).strip()
        shape_paths = []
        for key in ("shape_template", "star_template"):
            sp = la.get(key)
            if sp:
                shape_paths.append(str(sp))
        extras_s = la.get("shape_templates") or la.get("star_templates") or []
        if isinstance(extras_s, str):
            extras_s = [extras_s]
        shape_paths.extend(str(p) for p in extras_s if str(p).strip())
        # Legacy single files only if shapes/ folder is missing/empty
        shapes_dir_has = False
        sd = str(la.get("shapes_dir", "assets/lie_detector/shapes")).strip()
        if sd and os.path.isdir(sd):
            shapes_dir_has = any(
                glob.glob(os.path.join(sd, ext))
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            )
        if not shapes_dir_has:
            for legacy in (
                "assets/lie_detector/star.png",
                "assets/lie_detector/star_faded.png",
            ):
                shape_paths.append(legacy)

        self._tmpl_list = []  # list of (path, gray ndarray)
        self._shape_list = []  # list of (path, bgr ndarray) — any silhouette
        self._shape_paths = shape_paths
        self._tmpl_missing_logged = False
        self._was_visible = False
        self._last_alert_t = 0.0
        self._session_alert_played = False
        self._last_score = 1.0
        self._last_hit = ""
        self._hold_until = 0.0
        self._dialog_miss = 0  # consecutive detect ticks without dialog confirm
        self.visible = False
        self.following = False
        self.debug_pt = None  # (x, y) in frame coords for viz
        self._sound_lock = threading.Lock()
        self._sound_thread = None
        self._ocr = None
        self._ocr_init_tried = False
        self._ocr_fail_logged = False
        self._ocr_lock = threading.Lock()
        self._ocr_busy = False
        self._pending_ocr_hit = ""
        self._ocr_submit_t = 0.0
        self._model_load_lock = threading.Lock()
        self._track_models_warming = False
        self._follow_tick_t = 0.0

        self._session_start_t = 0.0
        self._follow_ready_t = 0.0
        self._track_xy = None  # float (x, y) frame
        self._track_vel = (0.0, 0.0)
        self._last_track_t = 0.0
        self._last_seen_t = 0.0
        self._mouse_xy = None  # float screen, last position we commanded
        self._mouse_placed_xy = None  # commanded screen xy after our last move
        self._mouse_watch_xy = None  # cursor while waiting to start follow
        self._mouse_last_move_t = 0.0  # time of last SetCursorPos we issued
        self._mouse_last_move_ok = False
        self._mouse_stuck_log_t = 0.0
        self._mouse_fail_log_t = 0.0
        self._mouse_elev_warned = False
        self._mouse_yielded = False  # user took the mouse; stop follow this session
        self._white_miss_since = 0.0  # wall clock of first consecutive white miss
        self._yolo_last_infer_t = 0.0
        self._yolo_last_hit = None  # cached (conf,cx,cy,bbox) between throttled runs
        self._follow_log_t = 0.0
        self._prev_gray = None  # for motion tracking
        self._last_shape_name = ""
        self._rearm_after_t = 0.0
        self._timeout_logged = False
        self._locked = False
        self._intro_block_until = 0.0
        self._last_intro_check_t = 0.0
        # Stale-lock: real target moves; static UI icons must not pin forever
        self._lock_anchor_xy = None
        self._lock_moved_t = 0.0
        self._stale_ban_xy = None  # after stale unlock, ignore this spot briefly
        self._stale_ban_until = 0.0
        self.stale_lock_s = float(la.get("stale_lock_s", 1.2))
        self.stale_lock_min_move_px = float(la.get("stale_lock_min_move_px", 10.0))
        self.stale_ban_s = float(la.get("stale_ban_s", 2.5))
        self.stale_ban_radius_px = float(la.get("stale_ban_radius_px", 55.0))
        self.exclude_side_ratio = float(la.get("exclude_side_ratio", 0.18))
        # After acquire, shape fades — track with flow/motion/coast (NOT green)
        self._flow_pts = None
        self._patch_gray = None
        self._patch_hw = (0, 0)
        self.debug_bbox = None
        self.debug_decoy_bboxes = []  # same-shape camouflage decoys (viz only)
        self._shape_sil = None  # binary silhouette from white acquire (for decoy match)
        self._shape_edge = None  # Canny edge of silhouette
        self._fill_sil = None  # native-size white mask; smoothness handoff after tint
        self._fill_rot = {}
        self._fill_hold = 0  # frames to stay put when a fill match misses
        self._shape_nat_wh = (64, 64)  # natural acquire size (w,h) for decoy scale
        # Decoys share one path; target moves randomly — track flock velocity to avoid them
        self._decoy_centers_prev = []
        self._decoy_vel = None  # (vx, vy) px/s consensus
        self._decoy_vel_t = 0.0
        self._lock_age = 0
        self._coast_streak = 0
        self._white_gone_t = 0.0  # when white target disappeared (fade start)
        # Last velocity that disagreed with decoy flock (random target motion)
        self._target_indep_vel = (0.0, 0.0)
        # Only the real target spins continuously — track orientation (deg) + spin rate
        self._target_angle = None  # degrees
        self._target_angle_t = 0.0
        self._target_spin_dps = 0.0  # deg/s EMA of wrapped delta
        self._spin_hits = 0  # frames with a real same-blob turn
        # Red on a non-rotating blob → snap back to the blob that is turning.
        self._still_hits = 0
        self._spin_watch = []  # same-blob orientation memory
        self._last_spin_xy = None
        self._spinback_cool = 0
        self._rot_sil_cache = {}  # angle_deg -> rotated silhouette (uint8)
        # Vision model proposes same-shape boxes only. It is not the red target.
        self.shape_peak_model_path = str(
            la.get("shape_peak_model", "assets/lie_detector/shape_peak.onnx")
        )
        self._shape_peak_model = None
        self._shape_peak_failed = False
        self.fade_yolo_path = str(
            la.get("fade_yolo_model", "assets/lie_detector/fade_filled.pt")
        )
        self.fade_yolo_conf = float(la.get("fade_yolo_conf", 0.20))
        self.fade_yolo_step_scale = float(la.get("fade_yolo_step_scale", 2.2))
        self.fade_yolo_hold_frames = max(0, int(la.get("fade_yolo_hold_frames", 4)))
        self.fade_yolo_search_scale = float(la.get("fade_yolo_search_scale", 6.0))
        # auto | cpu | cuda | cuda:0 …
        self.fade_yolo_device = str(la.get("fade_yolo_device", "auto") or "auto").strip()
        self._fade_yolo = None
        self._fade_yolo_failed = False
        self._fade_yolo_device_resolved = None
        self._yolo_hold = 0
        # All same-shape instances (target + decoys) for motion-outlier pick
        self._all_shape_prev = []  # [(cx,cy,ang), ...]
        self._all_shape_prev_t = 0.0
        self._stuck_frames = 0  # nearly-static track while flock moves
        # Background-subtraction tracker (SpotTheDiffMS-style absdiff / MOG2)
        self._fgbg = None
        self._bg_gray = None  # float32 running average fallback
        self.use_bg_subtractor = bool(la.get("use_bg_subtractor", True))
        self.bg_history = int(la.get("bg_history", 45))
        self.bg_var_threshold = float(la.get("bg_var_threshold", 14))
        self.bg_learn_rate = float(la.get("bg_learn_rate", 0.05))
        self.bg_learn_rate_locked = float(la.get("bg_learn_rate_locked", 0.008))
        self.bg_diff_threshold = int(la.get("bg_diff_threshold", 18))
        self.bg_min_area = int(la.get("bg_min_area", 400))
        self.bg_max_area = int(la.get("bg_max_area", 32000))

        # Auto-save a raw game clip when 测谎 triggers (for YOLO / offline review).
        self.record_clip = bool(la.get("record_clip", True))
        self.record_dir = str(la.get("record_dir", "_debug_out/lie_clips"))
        self.record_fps = float(la.get("record_fps", 30.0))
        # Duplicate-pad to wall clock made clips look like freeze→jump (one slow
        # YOLO tick could write up to ~2.5s of identical frames). Off by default.
        self.record_pad_wallclock = bool(la.get("record_pad_wallclock", False))
        self.record_pad_max_frames = max(1, int(la.get("record_pad_max_frames", 2)))
        self.record_preroll_s = float(la.get("record_preroll_s", 2.0))
        self.record_postroll_s = float(la.get("record_postroll_s", 4.0))
        self.record_max_s = float(la.get("record_max_s", 120.0))
        self.record_draw_track = bool(la.get("record_draw_track", False))
        # Idle preroll sample rate. Full-frame .copy() every capture tick
        # GC-thrashes after long AFK and makes screenshots stutter.
        self.record_preroll_fps = float(la.get("record_preroll_fps", 10.0))
        # Don't kill sticky mid-game on a couple of OCR/template misses.
        self.dialog_miss_limit = max(2, int(la.get("dialog_miss_limit", 5)))
        self.dialog_min_session_s = float(la.get("dialog_min_session_s", 20.0))
        self._clip_writer = None
        self._clip_path = None
        self._clip_size = None  # (w, h)
        self._clip_frames = 0
        self._clip_start_t = 0.0
        self._clip_post_until = 0.0  # keep writing after session for postroll
        self._clip_last_write_t = 0.0
        self._clip_last_frame = None  # last written BGR (for wall-clock pad)
        self._preroll = []  # recent BGR frames before trigger
        self._preroll_last_t = 0.0
        pre_fps = max(4.0, min(float(self.record_fps), float(self.record_preroll_fps)))
        if float(self.record_preroll_s) <= 0:
            self._preroll_max = 0
        else:
            self._preroll_max = max(1, int(round(pre_fps * self.record_preroll_s)))

        self._load_templates()
        self._load_shape_templates()
        # Do NOT warm RapidOCR here — loading + first infer stalls startup and
        # idle AFK. It loads lazily on the rare OCR fallback path.
        # Preload peak/YOLO off-thread so the first lie session does not freeze
        # the main loop (~28s YOLO load on 2026-11-05 00:17).
        if self.enable and self.auto_follow:
            self._warm_track_models_async()
        # Prebuild alert WAV so the first 测谎 hit is not delayed by synth.
        if self.enable:
            try:
                self._alert_wav_path()
            except Exception:
                pass

    def reload_cfg(self, cfg: dict):
        """Refresh settings after profile switch (paths usually stay global)."""
        self.cfg = cfg
        la = cfg.get("lie_alert") or {}
        self.enable = bool(la.get("enable", True))
        new_path = str(la.get("template", self.template_path))
        extras = la.get("templates") or []
        if isinstance(extras, str):
            extras = [extras]
        new_extras = [str(p) for p in extras if str(p).strip()]
        self.match_threshold = float(la.get("match_threshold", self.match_threshold))
        self.cn_title_match_threshold = float(
            la.get("cn_title_match_threshold", self.cn_title_match_threshold)
        )
        self.check_interval_frames = max(
            1, int(la.get("check_interval_frames", self.check_interval_frames))
        )
        self.alert_cooldown_s = float(la.get("alert_cooldown_s", self.alert_cooldown_s))
        self.freeze_seconds = float(la.get("freeze_seconds", self.freeze_seconds))
        self.hold_seconds = float(la.get("hold_seconds", self.hold_seconds))
        self.center_roi_ratio = float(la.get("center_roi_ratio", self.center_roi_ratio))
        self.rearm_cooldown_s = float(la.get("rearm_cooldown_s", self.rearm_cooldown_s))
        self.cn_title_require_ocr = bool(
            la.get("cn_title_require_ocr", self.cn_title_require_ocr)
        )
        self.ocr_enable = bool(la.get("ocr_enable", self.ocr_enable))
        self.ocr_interval_frames = max(
            self.check_interval_frames,
            int(la.get("ocr_interval_frames", self.ocr_interval_frames)),
        )
        self.ocr_min_interval_s = float(
            la.get("ocr_min_interval_s", getattr(self, "ocr_min_interval_s", 2.5))
        )
        self.ocr_during_follow = bool(
            la.get("ocr_during_follow", getattr(self, "ocr_during_follow", False))
        )
        kws = la.get("ocr_keywords") or self.ocr_keywords
        if isinstance(kws, str):
            kws = [kws]
        self.ocr_keywords = [str(k).strip() for k in kws if str(k).strip()]
        self.auto_follow = bool(la.get("auto_follow", self.auto_follow))
        self.follow_start_delay_s = float(
            la.get("follow_start_delay_s", self.follow_start_delay_s)
        )
        self.follow_timeout_s = float(la.get("follow_timeout_s", self.follow_timeout_s))
        self.follow_interval_frames = max(
            1, int(la.get("follow_interval_frames", self.follow_interval_frames))
        )
        self.shape_match_threshold = float(
            la.get(
                "shape_match_threshold",
                la.get("star_match_threshold", self.shape_match_threshold),
            )
        )
        self.mouse_smooth = float(la.get("mouse_smooth", self.mouse_smooth))
        self.mouse_takeover_px = float(
            la.get("mouse_takeover_px", self.mouse_takeover_px)
        )
        self.mouse_move_grace_s = float(
            la.get("mouse_move_grace_s", getattr(self, "mouse_move_grace_s", 0.08))
        )
        self.white_miss_grace_s = float(
            la.get("white_miss_grace_s", getattr(self, "white_miss_grace_s", 0.45))
        )
        self.fade_yolo_min_interval_s = float(
            la.get(
                "fade_yolo_min_interval_s",
                getattr(self, "fade_yolo_min_interval_s", 0.12),
            )
        )
        self.lose_target_grace_s = float(
            la.get("lose_target_grace_s", self.lose_target_grace_s)
        )
        scales = (
            la.get("shape_scales")
            or la.get("star_scales")
            or self.shape_scales
        )
        self.shape_scales = [float(s) for s in scales]
        if new_path != self.template_path or new_extras != self.extra_templates:
            self.template_path = new_path
            self.extra_templates = new_extras
            self._tmpl_missing_logged = False
            self._load_templates()
        self.record_clip = bool(la.get("record_clip", self.record_clip))
        self.record_dir = str(la.get("record_dir", self.record_dir))
        self.record_fps = float(la.get("record_fps", self.record_fps))
        self.record_pad_wallclock = bool(
            la.get("record_pad_wallclock", getattr(self, "record_pad_wallclock", False))
        )
        self.record_pad_max_frames = max(
            1,
            int(la.get("record_pad_max_frames", getattr(self, "record_pad_max_frames", 2))),
        )
        self.record_preroll_s = float(la.get("record_preroll_s", self.record_preroll_s))
        self.record_postroll_s = float(
            la.get("record_postroll_s", self.record_postroll_s)
        )
        self.record_max_s = float(la.get("record_max_s", self.record_max_s))
        self.record_draw_track = bool(la.get("record_draw_track", self.record_draw_track))
        self.record_preroll_fps = float(
            la.get("record_preroll_fps", getattr(self, "record_preroll_fps", 10.0))
        )
        self.fade_yolo_path = str(
            la.get("fade_yolo_model", self.fade_yolo_path)
        )
        self.fade_yolo_conf = float(la.get("fade_yolo_conf", self.fade_yolo_conf))
        self.fade_yolo_step_scale = float(
            la.get("fade_yolo_step_scale", self.fade_yolo_step_scale)
        )
        self.fade_yolo_hold_frames = max(
            0, int(la.get("fade_yolo_hold_frames", self.fade_yolo_hold_frames))
        )
        self.fade_yolo_search_scale = float(
            la.get("fade_yolo_search_scale", self.fade_yolo_search_scale)
        )
        self.fade_yolo_device = str(
            la.get("fade_yolo_device", getattr(self, "fade_yolo_device", "auto"))
            or "auto"
        ).strip()
        self.dialog_miss_limit = max(
            2, int(la.get("dialog_miss_limit", self.dialog_miss_limit))
        )
        self.dialog_min_session_s = float(
            la.get("dialog_min_session_s", self.dialog_min_session_s)
        )
        pre_fps = max(4.0, min(float(self.record_fps), float(self.record_preroll_fps)))
        if float(self.record_preroll_s) <= 0:
            self._preroll_max = 0
            self._preroll = []
        else:
            self._preroll_max = max(1, int(round(pre_fps * self.record_preroll_s)))

    def _all_template_paths(self):
        paths = [self.template_path] + list(self.extra_templates)
        seen = set()
        out = []
        for p in paths:
            p = str(p).strip()
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _load_templates(self):
        self._tmpl_list = []
        if not self.enable:
            logger.info("[LieAlert] disabled in config")
            return
        loaded = 0
        for path in self._all_template_paths():
            if not os.path.exists(path):
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logger.warning(f"[LieAlert] Failed to read template: {path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            self._tmpl_list.append((path, gray))
            loaded += 1
            logger.info(f"[LieAlert] Loaded template {w}x{h} from {path}")
        if loaded == 0 and not self._tmpl_missing_logged:
            logger.warning(
                f"[LieAlert] No template at {self.template_path} — "
                "drop a crop of the 测谎 dialog title/banner there, or rely on OCR. "
                "See assets/lie_detector/README.txt"
            )
            self._tmpl_missing_logged = True

    def _load_shape_templates(self):
        """Load any silhouette crops (star/heart/…); folder + explicit paths."""
        self._shape_list = []
        if not self.auto_follow:
            return
        seen_paths = set()
        seen_names = set()
        paths = list(self._shape_paths)
        if self.shapes_dir and os.path.isdir(self.shapes_dir):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                paths.extend(glob.glob(os.path.join(self.shapes_dir, ext)))
        for path in paths:
            path = os.path.normpath(str(path).strip())
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            if not os.path.exists(path):
                continue
            base = os.path.basename(path).lower()
            if base.startswith("lie") or base.startswith("ref_") or "title" in base:
                continue
            name = os.path.splitext(base)[0]
            if name in seen_names:
                continue
            seen_names.add(name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            self._shape_list.append((path, img))
            logger.info(
                f"[LieAlert] Shape template {img.shape[1]}x{img.shape[0]} from {path}"
            )
        if not self._shape_list:
            logger.info(
                "[LieAlert] No shape templates — will track by pale residual + motion "
                f"(drop crops into {self.shapes_dir or 'assets/lie_detector/shapes'})"
            )
        else:
            logger.info(
                f"[LieAlert] Loaded {len(self._shape_list)} shape template(s) "
                "(any silhouette OK; not star-only)"
            )

    def _ensure_ocr(self):
        if self._ocr is not None or self._ocr_init_tried:
            return
        self._ocr_init_tried = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
            logger.info(
                f"[LieAlert] OCR fallback ready (RapidOCR) keywords={self.ocr_keywords}"
            )
        except Exception as e:
            self._ocr = None
            if not self._ocr_fail_logged:
                logger.warning(
                    f"[LieAlert] OCR unavailable ({e}). Install: pip install rapidocr-onnxruntime. "
                    "Template matching still works if lie.png matches your client."
                )
                self._ocr_fail_logged = True

    def _submit_ocr_async(self, img_bgr, reason: str = ""):
        """Run RapidOCR off the detect thread — it is ~2s and must not block AFK/follow."""
        if not self.ocr_enable:
            return
        now = time.time()
        if (now - float(getattr(self, "_ocr_submit_t", 0.0) or 0.0)) < max(
            0.5, float(getattr(self, "ocr_min_interval_s", 2.5) or 2.5)
        ):
            return
        with self._ocr_lock:
            if self._ocr_busy:
                return
            self._ocr_busy = True
            self._ocr_submit_t = now
            self._last_ocr_t = now
        try:
            frame = img_bgr.copy()
        except Exception:
            with self._ocr_lock:
                self._ocr_busy = False
            return

        def _worker():
            hit = ""
            try:
                self._ensure_ocr()
                if self._ocr is not None:
                    hit = self._match_ocr(frame) or ""
            except Exception as e:
                logger.debug(f"[LieAlert] async OCR error: {e}")
            finally:
                with self._ocr_lock:
                    if hit:
                        self._pending_ocr_hit = hit
                        logger.info(
                            f"[LieAlert] async OCR hit={hit}"
                            + (f" ({reason})" if reason else "")
                        )
                    self._ocr_busy = False

        threading.Thread(target=_worker, daemon=True, name="LieOCR").start()

    def _consume_pending_ocr(self) -> str:
        with self._ocr_lock:
            hit = str(getattr(self, "_pending_ocr_hit", "") or "")
            self._pending_ocr_hit = ""
            return hit

    def _preroll_push(self, img_bgr):
        if not self.record_clip or img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return
        if float(self.record_preroll_s) <= 0 or int(self._preroll_max) <= 0:
            return
        # Idle AFK path: copying a full BGR every detector tick causes memory
        # churn and capture FPS collapse after tens of minutes. Sample at
        # record_preroll_fps only.
        now = time.time()
        pre_fps = max(4.0, min(float(self.record_fps), float(self.record_preroll_fps)))
        min_gap = 1.0 / pre_fps
        last_t = float(getattr(self, "_preroll_last_t", 0.0) or 0.0)
        if last_t > 0.0 and (now - last_t) < min_gap:
            return
        self._preroll_last_t = now
        self._preroll.append(img_bgr.copy())
        overflow = len(self._preroll) - self._preroll_max
        if overflow > 0:
            del self._preroll[:overflow]

    def _clip_frame(self, img_bgr):
        """Optionally stamp the red track box onto a copy for review clips."""
        out = img_bgr
        if not self.record_draw_track:
            return out
        out = img_bgr.copy()
        box = getattr(self, "debug_bbox", None)
        if box is not None and len(box) >= 4:
            x, y, w, h = [int(v) for v in box[:4]]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        pt = self.debug_pt
        if pt is not None:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        return out

    def _clip_start(self, img_bgr):
        self._clip_finish("restart")
        if not self.record_clip or img_bgr is None:
            return
        try:
            os.makedirs(self.record_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"[LieAlert] cannot create clip dir {self.record_dir}: {e}")
            return
        h, w = img_bgr.shape[:2]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.record_dir, f"lie_{stamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = max(5.0, float(self.record_fps))
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            logger.warning(f"[LieAlert] VideoWriter failed for {path}")
            return
        self._clip_writer = writer
        self._clip_path = path
        self._clip_size = (w, h)
        self._clip_frames = 0
        self._clip_start_t = time.time()
        self._clip_last_write_t = 0.0
        self._clip_last_frame = None
        # Flush preroll so the dialog open is not cut off.
        for fr in self._preroll:
            self._clip_write(fr, count_toward_max=False, pad_wallclock=False)
        self._preroll.clear()
        self._clip_last_write_t = time.time()
        logger.info(f"[LieAlert] recording clip → {path}")

    def _clip_write(
        self, img_bgr, count_toward_max: bool = True, pad_wallclock: bool | None = None
    ):
        """Write one capture frame.

        Optional wall-clock pad (off by default): when the bot loop is slower
        than record_fps, duplicate the previous frame a little so file duration
        stays closer to real time. Heavy pad used to freeze 1–2s then jump,
        which made clips look like unstable FPS (~85% near-dup on last take).
        """
        wr = self._clip_writer
        if wr is None or img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return
        h, w = img_bgr.shape[:2]
        tw, th = self._clip_size or (w, h)
        frame = self._clip_frame(img_bgr)
        if (w, h) != (tw, th):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        now = time.time()
        fps = max(5.0, float(self.record_fps))
        n_write = 1
        do_pad = (
            bool(self.record_pad_wallclock)
            if pad_wallclock is None
            else bool(pad_wallclock)
        )
        if do_pad and self._clip_last_write_t > 0:
            gap = now - float(self._clip_last_write_t)
            n_write = max(1, int(round(gap * fps)))
            # Tiny cap only — never dump seconds of identical frames.
            n_write = min(n_write, max(1, int(self.record_pad_max_frames)))
        try:
            for i in range(n_write):
                # Hold the previous picture across the gap; last slot is fresh.
                out = (
                    frame
                    if (i == n_write - 1 or self._clip_last_frame is None)
                    else self._clip_last_frame
                )
                wr.write(out)
                self._clip_frames += 1
        except Exception as e:
            logger.debug(f"[LieAlert] clip write failed: {e}")
            self._clip_finish("write_error")
            return
        self._clip_last_frame = frame
        self._clip_last_write_t = now
        if count_toward_max and self.record_max_s > 0:
            if (now - self._clip_start_t) >= self.record_max_s:
                self._clip_finish("max_seconds")

    def _clip_finish(self, reason: str = ""):
        wr = self._clip_writer
        if wr is None:
            return
        path = self._clip_path
        n = self._clip_frames
        started = float(getattr(self, "_clip_start_t", 0.0) or 0.0)
        self._clip_writer = None
        self._clip_path = None
        self._clip_size = None
        self._clip_frames = 0
        self._clip_start_t = 0.0
        self._clip_post_until = 0.0
        self._clip_last_write_t = 0.0
        self._clip_last_frame = None
        try:
            wr.release()
        except Exception:
            pass
        if path and n > 0 and os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            why = f" ({reason})" if reason else ""
            decl = max(5.0, float(self.record_fps))
            play_s = n / decl
            wall_s = (time.time() - started) if started > 0 else play_s
            capt_fps = (n / wall_s) if wall_s > 0.05 else decl
            logger.info(
                f"[LieAlert] clip saved {path} frames={n} size={mb:.1f}MB "
                f"play~{play_s:.1f}s@{decl:.0f}fps wall={wall_s:.1f}s "
                f"capt~{capt_fps:.1f}fps pad={int(self.record_pad_wallclock)}"
                f"{why}"
            )
        elif path and os.path.exists(path) and n <= 0:
            try:
                os.remove(path)
            except Exception:
                pass

    def maybe_check(self, img_frame_bgr, frame_count: int, kb=None, capture=None) -> bool:
        """
        Run detection every N frames. Returns True if 测谎 is currently visible
        (or still inside the sticky hold window after the last hit).

        When auto_follow is on and past the intro delay, track the moving
        translucent shape (any silhouette) and move the mouse onto it.
        """
        if not self.enable:
            self._clip_finish("disabled")
            self.visible = False
            self.following = False
            self.debug_pt = None
            self._was_visible = False
            self._hold_until = 0.0
            self._dialog_miss = 0
            self._prev_gray = None
            return False
        if not self._tmpl_list and frame_count % 90 == 1:
            self._load_templates()
        if img_frame_bgr is None or getattr(img_frame_bgr, "size", 0) == 0:
            return self.visible

        # Always keep a short preroll so a fresh trigger has dialog-open context.
        if self.record_clip and self._clip_writer is None:
            self._preroll_push(img_frame_bgr)

        now = time.time()
        # Detection only on cadence — NOT every frame while visible (that caused
        # false-positive spam re-arming hold forever). Follow still runs below.
        do_detect = frame_count % self.check_interval_frames == 0
        ocr_due = (now - float(getattr(self, "_last_ocr_t", 0.0) or 0.0)) >= max(
            0.5, float(getattr(self, "ocr_min_interval_s", 2.5) or 2.5)
        )
        # Sticky reconfirm: templates preferred; OCR only on rare wall-clock ticks
        # and never while actively following (OCR ~2s freezes mouse track).
        do_ocr_reconfirm = (
            self.visible
            and not self.following
            and ocr_due
            and (frame_count % self.ocr_interval_frames == 0)
        )

        found = False
        hit = ""
        # Async OCR result from a previous tick (never blocks this frame).
        pending_ocr = self._consume_pending_ocr() if self.ocr_enable else ""
        if do_detect and now >= float(self._rearm_after_t):
            tmpl_found, tmpl_hit, tmpl_is_cn = self._match_templates(img_frame_bgr)
            ocr_hit = pending_ocr
            pending_ocr = ""
            # OCR is async + rare. Template miss used to sync-call RapidOCR every
            # check_interval and froze the bot at ~2–3 fps (log gaps 4–5s).
            need_ocr = self.ocr_enable and ocr_due and (
                (tmpl_is_cn and self.cn_title_require_ocr and tmpl_found)
                or (not tmpl_found and not self.visible)
                or do_ocr_reconfirm
            )
            if need_ocr:
                why = "cn_confirm" if (tmpl_is_cn and tmpl_found) else "fallback"
                self._submit_ocr_async(img_frame_bgr, reason=why)

            if tmpl_found and tmpl_is_cn and self.cn_title_require_ocr:
                # Phone-crop CN title alone is a known false-positive source
                if ocr_hit:
                    found = True
                    hit = f"{tmpl_hit}+ocr:{ocr_hit}"
                else:
                    # Kick async confirm; accept on a later tick when OCR returns.
                    logger.debug(
                        f"[LieAlert] ignore weak CN title hit {tmpl_hit} "
                        f"score={self._last_score:.3f} (await OCR confirm)"
                    )
            elif tmpl_found:
                found = True
                hit = tmpl_hit
            elif ocr_hit:
                found = True
                hit = f"ocr:{ocr_hit}"

            if found:
                self._on_dialog_found(img_frame_bgr, hit, now)
            elif self.visible:
                # Detect tick with no dialog confirm → count toward ending sticky
                self._dialog_miss += 1
        elif pending_ocr and now >= float(self._rearm_after_t):
            # OCR finished in background while we were not on a detect tick.
            found = True
            hit = f"ocr:{pending_ocr}"
            self._on_dialog_found(img_frame_bgr, hit, now)
        elif (
            do_ocr_reconfirm
            and now >= float(self._rearm_after_t)
            and self.ocr_enable
            and not self.following
        ):
            # Soft reconfirm during sticky hold — async; never during mouse follow.
            self._submit_ocr_async(img_frame_bgr, reason="reconfirm")

        sticky = now < float(self._hold_until)
        session_age = (
            (now - float(self._session_start_t)) if self._session_start_t > 0 else 0.0
        )
        # Actively tracking the mini-game ⇒ dialog is still up; ignore OCR misses.
        if self._locked and (now - float(self._last_seen_t)) < 1.5:
            self._dialog_miss = 0
        # Dialog chrome gone for several detect ticks → end sticky early, but only
        # after the mini-game has had time to run (earlier miss was cutting clips).
        if (
            sticky
            and not found
            and self._dialog_miss >= int(self.dialog_miss_limit)
            and session_age >= float(self.dialog_min_session_s)
        ):
            logger.info(
                f"[LieAlert] dialog confirm missed x{self._dialog_miss} "
                f"after {session_age:.1f}s — ending session"
            )
            sticky = False
            self._hold_until = 0.0
            self._dialog_miss = 0
            self._rearm_after_t = now + max(1.0, self.rearm_cooldown_s)

        # Timeout a follow/alert session (log once; cool down before re-arm).
        # Alert-only: never sync OCR here (blocks ~2s). Follow mode may extend.
        if (sticky or found) and self._session_start_t > 0:
            if (now - self._session_start_t) > self.follow_timeout_s:
                still_up = False
                if self.auto_follow:
                    try:
                        still_up = bool(self._match_ocr(img_frame_bgr))
                    except Exception:
                        still_up = False
                if still_up or found:
                    self._session_start_t = now - self.follow_timeout_s * 0.5
                    self._hold_until = max(self._hold_until, now + 8.0)
                    self._dialog_miss = 0
                    if not self._timeout_logged:
                        logger.info("[LieAlert] follow timeout extended (dialog still up)")
                        self._timeout_logged = True
                else:
                    if not self._timeout_logged:
                        logger.info("[LieAlert] follow session timeout — clearing")
                        self._timeout_logged = True
                    sticky = False
                    found = False
                    self._hold_until = 0.0
                    self._dialog_miss = 0
                    self._rearm_after_t = now + max(1.0, self.rearm_cooldown_s)
                    self._session_start_t = 0.0

        self.visible = bool(found or sticky)

        if self.visible:
            if self._clip_writer is None and self.record_clip and self._session_start_t > 0:
                # Session already open but writer missing (e.g. prior write error).
                self._clip_start(img_frame_bgr)
            # Keep recording for the whole sticky window (not tied to AFK alone).
            if self._clip_writer is not None:
                self._clip_post_until = 0.0
                self._clip_write(img_frame_bgr)
            rising = not self._was_visible
            # Alarm fires in _on_dialog_found (first). Rising here is a fallback
            # if sticky hold re-arms without a fresh find path.
            if rising and not bool(getattr(self, "_session_alert_played", False)):
                self._last_alert_t = now
                self._session_alert_played = True
                via = hit or self._last_hit or "hold"
                logger.warning(
                    f"[ALERT] 测谎 detected via {via} (score={self._last_score:.3f}, "
                    f"hold={max(0.0, self._hold_until - now):.1f}s) — pause AFK"
                    + (
                        ", auto-follow shape"
                        if self.auto_follow
                        else ", answer manually"
                    )
                )
                self._play_loud_alert()
            if kb is not None:
                try:
                    kb.pause_for_lie_alert(self.freeze_seconds)
                except Exception as e:
                    logger.debug(f"[LieAlert] pause failed: {e}")

            # Warm BG only before first lock — learning while the white shape is
            # visible teaches MOG2 that the target is background.
            if (
                self.auto_follow
                and self.use_bg_subtractor
                and not self._locked
                and now < self._follow_ready_t + 1.0
            ):
                self._learn_background(img_frame_bgr, locked=False)

            # Mouse follow mini-game. Real takeover only counts AFTER our first
            # snap — bumps during the intro delay must not abort AFK follow.
            if self.auto_follow:
                self._watch_user_mouse()
            if self._mouse_yielded:
                self.following = False
                self.debug_pt = None
            elif self.auto_follow and now >= self._follow_ready_t:
                if self._should_block_follow_for_intro(img_frame_bgr, now):
                    # Do NOT wipe an existing lock — intro OCR can false-hit mid-game
                    self.following = False
                    self.debug_pt = None
                    if now - float(getattr(self, "_intro_hold_log_t", 0.0) or 0.0) > 1.0:
                        self._intro_hold_log_t = now
                        left = max(
                            0.0,
                            float(getattr(self, "_intro_block_until", 0.0) or 0.0) - now,
                        )
                        logger.info(
                            f"[LieAlert] intro still up — hold mouse follow "
                            f"(block_left={left:.2f}s, "
                            f"+{now - float(self._session_start_t):.1f}s)"
                        )
                elif frame_count % self.follow_interval_frames == 0:
                    if not getattr(self, "_follow_started_logged", False):
                        self._follow_started_logged = True
                        logger.info(
                            f"[LieAlert] mouse follow armed "
                            f"(+{now - float(self._session_start_t):.1f}s)"
                        )
                    self._follow_shape(img_frame_bgr, capture, now)
            else:
                self.following = False
                self.debug_pt = None
                if (
                    self.auto_follow
                    and self._session_start_t > 0
                    and now < self._follow_ready_t
                    and now - float(getattr(self, "_delay_hold_log_t", 0.0) or 0.0) > 1.0
                ):
                    self._delay_hold_log_t = now
                    left = max(0.0, self._follow_ready_t - now)
                    logger.info(f"[LieAlert] follow delay — mouse idle {left:.1f}s left")
        else:
            if self._was_visible:
                # AFK can resume, but keep rolling the clip briefly past close.
                post = max(0.0, float(self.record_postroll_s))
                if self._clip_writer is not None and post > 0:
                    self._clip_post_until = now + post
                    self._clip_write(img_frame_bgr)
                elif self._clip_writer is not None:
                    self._clip_finish("session_end")
                if kb is not None:
                    try:
                        kb.clear_lie_alert_freeze()
                    except Exception as e:
                        logger.debug(f"[LieAlert] clear freeze failed: {e}")
                logger.info("[LieAlert] 测谎 gone — AFK resume")
            elif self._clip_writer is not None and now < float(self._clip_post_until):
                self._clip_write(img_frame_bgr)
            elif self._clip_writer is not None and float(self._clip_post_until) > 0:
                self._clip_finish("session_end")
            self.following = False
            self.debug_pt = None
            self._mouse_yielded = False
            self._mouse_placed_xy = None
            self._mouse_watch_xy = None
            self._mouse_last_move_t = 0.0
            self._mouse_last_move_ok = False
            self._white_miss_since = 0.0
            self._yolo_last_infer_t = 0.0
            self._yolo_last_hit = None
            self._session_alert_played = False
            self._track_xy = None
            self._track_vel = (0.0, 0.0)
            self._locked = False
            self._lock_anchor_xy = None
            self._stale_ban_xy = None
            self._dialog_miss = 0
            if self._session_start_t and now >= float(self._rearm_after_t):
                self._session_start_t = 0.0
            self._prev_gray = None
            if self.auto_follow:
                self._reset_bg_model()

        self._was_visible = self.visible
        return self.visible

    def _center_roi(self, gray):
        fh, fw = gray.shape[:2]
        ratio = min(max(self.center_roi_ratio, 0.4), 1.0)
        rw = int(fw * ratio)
        rh = int(fh * ratio)
        x0 = max(0, (fw - rw) // 2)
        y0 = max(0, (fh - rh) // 2)
        x1 = min(fw, x0 + rw)
        y1 = min(fh, y0 + rh)
        roi = gray[y0:y1, x0:x1]
        return roi, (x0, y0, x1, y1)

    @staticmethod
    def _is_cn_title_template(path: str) -> bool:
        base = os.path.basename(path).lower()
        return (
            "lie_cn" in base
            or "cn_title" in base
            or "谎言" in base
            or ("title" in base and "en" not in base and "lie.png" != base)
        )

    def _threshold_for_template(self, path: str) -> float:
        if self._is_cn_title_template(path):
            return float(self.cn_title_match_threshold)
        base = os.path.basename(path).lower()
        # Intro instruction panel (「3秒后…」): very tight — sky/cloud false
        # matches sit around 0.04–0.08 on this client.
        if "intro_body" in base or "intro_banner" in base:
            return float(
                (self.cfg.get("lie_alert") or {}).get(
                    "intro_match_threshold", 0.025
                )
            )
        return float(self.match_threshold)

    @staticmethod
    def _is_intro_template(path: str) -> bool:
        base = os.path.basename(path).lower()
        return "intro_body" in base or "intro_banner" in base

    def _dialog_band(self, gray):
        """Mid-screen band where the 测谎说明窗 / title sits."""
        fh, fw = gray.shape[:2]
        y0, y1 = int(fh * 0.18), int(fh * 0.58)
        x0, x1 = int(fw * 0.28), int(fw * 0.72)
        return gray[y0:y1, x0:x1], (x0, y0)

    def _intro_panel_visible(self, img_bgr) -> bool:
        """True while the white intro instruction panel (banner/body) is on screen."""
        if img_bgr is None or not self._tmpl_list:
            return False
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        band, _ = self._dialog_band(gray)
        for path, tmpl in self._tmpl_list:
            if not self._is_intro_template(path):
                continue
            th, tw = tmpl.shape[:2]
            if tw >= fw or th >= fh:
                continue
            search = band
            if search.shape[0] <= th or search.shape[1] <= tw:
                search = gray
            try:
                res = cv2.matchTemplate(search, tmpl, cv2.TM_SQDIFF_NORMED)
                min_val = float(np.nanmin(res))
            except Exception:
                continue
            if np.isfinite(min_val) and min_val <= self._threshold_for_template(path):
                return True
        return False

    def _on_dialog_found(self, img_frame_bgr, hit: str, now: float):
        """Apply sticky hold + one-shot session init when dialog is first seen."""
        self._last_hit = hit
        self._hold_until = now + max(1.0, self.hold_seconds)
        self._dialog_miss = 0
        if self._was_visible:
            return
        self._session_start_t = now
        # Fire siren FIRST on rising edge — before clip/BG/model warm so the
        # laptop alert-only path has zero follow-related delay.
        self._last_alert_t = now
        self._session_alert_played = True
        logger.warning(
            f"[ALERT] 测谎 detected via {hit} (score={self._last_score:.3f}, "
            f"hold={max(0.0, self._hold_until - now):.1f}s) — pause AFK"
            + (", auto-follow shape" if self.auto_follow else ", answer manually")
        )
        self._play_loud_alert()
        self._track_xy = None
        self._track_vel = (0.0, 0.0)
        self._mouse_xy = None
        self._mouse_placed_xy = None
        self._mouse_watch_xy = None
        self._mouse_last_move_t = 0.0
        self._mouse_last_move_ok = False
        self._mouse_elev_warned = False
        self._mouse_yielded = False
        self._white_miss_since = 0.0
        self._yolo_last_infer_t = 0.0
        self._yolo_last_hit = None
        self._prev_gray = None
        self._last_shape_name = ""
        self._timeout_logged = False
        self._locked = False
        self._lock_anchor_xy = None
        self._lock_moved_t = 0.0
        self._stale_ban_xy = None
        self._stale_ban_until = 0.0
        self._follow_started_logged = False
        self._delay_hold_log_t = 0.0
        self._intro_hold_log_t = 0.0
        if not self.auto_follow:
            # Alert-only: no follow delay, no BG model, no YOLO warm.
            self._follow_ready_t = now
            self._intro_block_until = 0.0
            if self.record_clip:
                self._clip_start(img_frame_bgr)
            logger.info(
                f"[LieAlert] session start — alert-only hit={hit} "
                f"(auto_follow=False, record={int(self.record_clip)})"
            )
            return
        # Intro still up → wait out countdown. Late OCR (title only,
        # panel already gone) → arm immediately so we do not sit idle
        # after the game countdown already finished.
        delay = self._compute_session_follow_delay(img_frame_bgr, hit)
        self._follow_ready_t = now + delay
        # Only a short OCR intro latch — do NOT pin intro_block to the
        # full delay (bad keywords could keep extending through the game).
        self._intro_block_until = (
            now + min(delay, 1.0) if delay > 0 else 0.0
        )
        self._reset_bg_model()
        self._clip_start(img_frame_bgr)
        # Warm heavy models off the follow hot path.
        self._warm_track_models_async()
        logger.info(
            f"[LieAlert] session start — follow in "
            f"{delay:.1f}s hit={hit} "
            f"(auto_follow={self.auto_follow}, "
            f"shapes={len(self._shape_list)}, bg_sub={self.use_bg_subtractor}, "
            f"record={int(self.record_clip)})"
        )

    def _compute_session_follow_delay(self, img_bgr, hit: str) -> float:
        """
        How long to wait before first mouse snap.

        Early hit (intro banner / 「3秒后」): use configured delay.
        Late hit (OCR title after panel already gone): 0 — countdown already
        over from the player's view; waiting again leaves the cursor idle.
        """
        base = max(0.0, float(self.follow_start_delay_s))
        hit_s = str(hit or "")
        hit_l = hit_s.lower()
        early_markers = (
            "intro_banner",
            "intro_body",
            "3秒后",
            "2秒后",
            "请前往安全",
        )
        if any(m.lower() in hit_l if m.isascii() else m in hit_s for m in early_markers):
            return base
        if self._intro_panel_visible(img_bgr):
            # Detected via title OCR while panel still up — short wait only.
            return min(base, 0.7)
        logger.info(
            "[LieAlert] intro panel already gone at detect — arm follow now "
            f"(hit={hit_s})"
        )
        return 0.0

    def _intro_dialog_still_up(self, img_bgr) -> bool:
        """True while countdown/intro chrome is still a strong match."""
        if self._intro_panel_visible(img_bgr):
            return True
        if not self._tmpl_list:
            return False
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        roi, _ = self._center_roi(gray)
        for path, tmpl in self._tmpl_list:
            if not self._is_cn_title_template(path):
                continue
            th, tw = tmpl.shape[:2]
            if tw >= fw or th >= fh:
                continue
            search = roi
            if search.shape[0] <= th or search.shape[1] <= tw:
                search = gray
            try:
                res = cv2.matchTemplate(search, tmpl, cv2.TM_SQDIFF_NORMED)
                min_val = float(np.nanmin(res))
            except Exception:
                continue
            if np.isfinite(min_val) and min_val <= self._threshold_for_template(path):
                return True
        return False

    def _match_templates(self, img_bgr):
        """Return (found, hit_str, is_cn_title).

        Cheap title templates first; large intro banner/body only if those miss.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        roi, _ = self._center_roi(gray)
        band, _ = self._dialog_band(gray)
        best_pass_score = 1.0
        best_pass_path = ""
        best_any_score = 1.0

        def _scan(items):
            nonlocal best_pass_score, best_pass_path, best_any_score
            for path, tmpl in items:
                th, tw = tmpl.shape[:2]
                if tw >= fw or th >= fh:
                    continue
                # Intro 说明窗: only search the mid dialog band with a tight score
                # so sky/cloud patches cannot trip the alarm.
                if self._is_intro_template(path):
                    search = band
                else:
                    search = roi
                if search.shape[0] <= th or search.shape[1] <= tw:
                    search = gray
                try:
                    res = cv2.matchTemplate(search, tmpl, cv2.TM_SQDIFF_NORMED)
                    min_val = float(np.nanmin(res))
                except Exception:
                    continue
                if not np.isfinite(min_val):
                    continue
                if min_val < best_any_score:
                    best_any_score = min_val
                thr = self._threshold_for_template(path)
                if min_val <= thr and min_val < best_pass_score:
                    best_pass_score = min_val
                    best_pass_path = path

        cheap = [(p, t) for p, t in self._tmpl_list if not self._is_intro_template(p)]
        intro = [(p, t) for p, t in self._tmpl_list if self._is_intro_template(p)]
        _scan(cheap)
        if not best_pass_path:
            _scan(intro)
        if best_pass_path:
            self._last_score = best_pass_score
            return (
                True,
                f"tmpl:{os.path.basename(best_pass_path)}",
                self._is_cn_title_template(best_pass_path),
            )
        self._last_score = best_any_score
        return False, "", False
    @staticmethod
    def _normalize_ocr_text(s: str) -> str:
        return "".join(ch for ch in str(s).upper() if not ch.isspace())

    def _match_ocr(self, img_bgr) -> str:
        """Return matched keyword if OCR finds it; try center + dialog band."""
        if self._ocr is None or not self.ocr_keywords:
            return ""
        fh, fw = img_bgr.shape[:2]
        ratio = min(max(self.center_roi_ratio, 0.4), 1.0)
        rw = int(fw * ratio)
        rh = int(fh * ratio)
        x0 = max(0, (fw - rw) // 2)
        y0 = max(0, (fh - rh) // 2)
        # Full center ROI, plus a tighter mid band where the intro panel sits.
        # Wide ROI alone often missed「3秒后…」while reading chat/nameplates.
        rois = [
            img_bgr[y0 : min(fh, y0 + rh), x0 : min(fw, x0 + rw)],
            img_bgr[
                int(fh * 0.22) : int(fh * 0.72),
                int(fw * 0.22) : int(fw * 0.78),
            ],
        ]
        for roi in rois:
            if roi is None or getattr(roi, "size", 0) == 0:
                continue
            h, w = roi.shape[:2]
            if max(h, w) < 480:
                scale = 480.0 / max(h, w)
                roi = cv2.resize(
                    roi,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            try:
                result, _ = self._ocr(roi)
            except Exception as e:
                logger.debug(f"[LieAlert] OCR error: {e}")
                continue
            if not result:
                continue
            blob = " ".join(str(item[1]) for item in result if len(item) >= 2)
            blob_n = self._normalize_ocr_text(blob)
            for kw in self.ocr_keywords:
                if not kw:
                    continue
                if self._normalize_ocr_text(kw) in blob_n or kw in blob:
                    self._last_score = 0.0
                    return kw
        return ""

    def _should_block_follow_for_intro(self, img_bgr, now: float) -> bool:
        """Block mouse-follow while countdown/instruction text is still visible.

        Hard cap from session start so OCR/template cannot stall follow for
        seconds after follow_start_delay (last session armed at +3.8s).
        Past follow_ready_t we never run OCR on this path — RapidOCR on a
        large ROI blocks the capture thread and delays the first mouse snap.
        """
        session_t = float(getattr(self, "_session_start_t", 0.0) or 0.0)
        delay = max(0.0, float(self.follow_start_delay_s))
        # Absolute deadline: delay + small slack. Never hold past this.
        intro_deadline = session_t + delay + 0.35 if session_t > 0 else 0.0
        if intro_deadline > 0 and now >= intro_deadline:
            self._intro_block_until = 0.0
            return False

        if now < float(self._intro_block_until):
            return True

        # Title-template intro gate (cheap). Cap extension inside deadline.
        if self._intro_dialog_still_up(img_bgr):
            until = now + 0.45
            if intro_deadline > 0:
                until = min(until, intro_deadline)
            self._intro_block_until = until
            return True

        # OCR only BEFORE follow_ready_t. After that the delay gate already
        # waited out the countdown; blocking OCR here only adds latency.
        ready_t = float(getattr(self, "_follow_ready_t", 0.0) or 0.0)
        if ready_t > 0 and now >= ready_t:
            return False
        if now - float(self._last_intro_check_t) < 0.45:
            return False
        self._last_intro_check_t = now
        if not self.intro_ocr_block_keywords:
            return False
        self._ensure_ocr()
        if self._ocr is None:
            return False
        # Scan upper-mid band where instruction dialog sits
        fh, fw = img_bgr.shape[:2]
        y0, y1 = int(fh * 0.12), int(fh * 0.62)
        x0, x1 = int(fw * 0.15), int(fw * 0.85)
        roi = img_bgr[y0:y1, x0:x1]
        try:
            result, _ = self._ocr(roi)
        except Exception:
            return False
        if not result:
            return False
        blob = " ".join(str(item[1]) for item in result if len(item) >= 2)
        for kw in self.intro_ocr_block_keywords:
            if kw and kw in blob:
                until = now + 0.5
                if intro_deadline > 0:
                    until = min(until, intro_deadline)
                self._intro_block_until = until
                logger.info(f"[LieAlert] intro UI still up ({kw}) — hold follow")
                return True
        return False

    def _reset_bg_model(self):
        self._fgbg = None
        self._bg_gray = None

    def _ensure_bg_subtractor(self):
        if self._fgbg is not None:
            return
        try:
            self._fgbg = cv2.createBackgroundSubtractorMOG2(
                history=max(20, self.bg_history),
                varThreshold=float(self.bg_var_threshold),
                detectShadows=False,
            )
            logger.info(
                f"[LieAlert] MOG2 bg-subtractor ready "
                f"(history={self.bg_history}, varThr={self.bg_var_threshold})"
            )
        except Exception as e:
            self._fgbg = None
            logger.warning(f"[LieAlert] MOG2 unavailable ({e}); using running-avg absdiff")

    def _learn_background(self, img_bgr, locked: bool):
        """Warm / update background so translucent overlays stand out via absdiff."""
        if not self.use_bg_subtractor or img_bgr is None:
            return
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lr = self.bg_learn_rate_locked if locked else self.bg_learn_rate
        self._ensure_bg_subtractor()
        if self._fgbg is not None:
            try:
                self._fgbg.apply(img_bgr, learningRate=float(lr))
            except Exception:
                pass
        if self._bg_gray is None or self._bg_gray.shape[:2] != gray.shape[:2]:
            self._bg_gray = gray.astype(np.float32)
        else:
            cv2.accumulateWeighted(gray, self._bg_gray, float(lr))

    def _fg_mask(self, img_bgr):
        """Foreground mask of moving/changed pixels (shape overlay)."""
        if not self.use_bg_subtractor:
            return None
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = None
        self._ensure_bg_subtractor()
        if self._fgbg is not None:
            try:
                lr = self.bg_learn_rate_locked if self._locked else self.bg_learn_rate
                mask = self._fgbg.apply(img_bgr, learningRate=float(lr))
            except Exception:
                mask = None
        if mask is None:
            if self._bg_gray is None or self._bg_gray.shape[:2] != gray.shape[:2]:
                self._bg_gray = gray.astype(np.float32)
                return None
            bg = cv2.convertScaleAbs(self._bg_gray)
            diff = cv2.absdiff(gray, bg)
            _, mask = cv2.threshold(
                diff, int(self.bg_diff_threshold), 255, cv2.THRESH_BINARY
            )
        # SpotTheDiffMS: dilate then contours
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        return mask

    def _bgdiff_candidates(self, fg_mask, img_bgr, x0, y0, x1, y1, frame_h, frame_w):
        """Contours on fg mask — only keep blobs that are also pale/bright (real target)."""
        out = []
        if fg_mask is None or img_bgr is None:
            return out
        roi = fg_mask[y0:y1, x0:x1]
        if roi.size == 0:
            return out
        crop = img_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in cnts:
            area = float(cv2.contourArea(ct))
            if area < self.bg_min_area or area > self.bg_max_area:
                continue
            x, y, w, h = cv2.boundingRect(ct)
            if w < 18 or h < 18 or w > 220 or h > 220:
                continue
            aspect = w / float(h)
            if aspect < 0.55 or aspect > 1.85:
                continue
            fill = area / float(max(1, w * h))
            if fill < 0.15:
                continue
            M = cv2.moments(ct)
            if M["m00"] <= 1e-3:
                cx = x + w / 2.0
                cy = y + h / 2.0
            else:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            abs_x = x0 + cx
            abs_y = y0 + cy
            if frame_h > 0:
                if abs_y < frame_h * self.exclude_top_ratio:
                    continue
                if abs_y > frame_h * (1.0 - self.exclude_bottom_ratio):
                    continue
            # Skip watermark / UI corners / side chrome
            if frame_w > 0 and frame_h > 0:
                side = float(getattr(self, "exclude_side_ratio", 0.10))
                if abs_x < frame_w * side or abs_x > frame_w * (1.0 - side):
                    continue
                if abs_x > frame_w * 0.78 and abs_y > frame_h * 0.72:
                    continue
                if abs_x < frame_w * 0.12 and abs_y < frame_h * 0.18:
                    continue
                if abs_y < frame_h * 0.32:
                    continue

            ix, iy = int(cx), int(cy)
            if not (0 <= iy < hsv.shape[0] and 0 <= ix < hsv.shape[1]):
                continue
            # CRITICAL: bgdiff alone matches rock edges — require pale/bright core
            vv = float(hsv[iy, ix, 2])
            ss = float(hsv[iy, ix, 1])
            if vv < 160 or ss > 90:
                continue
            mean_v = float(cv2.mean(gray, mask=self._contour_mask(gray.shape, ct))[0])
            if mean_v < 150:
                continue

            score = (
                0.30
                + 0.18 * min(1.0, vv / 255.0)
                + 0.14 * min(1.0, fill)
                + 0.22 * min(1.0, area / 8000.0)
                + 0.10 * (1.0 - abs(aspect - 1.0))
            )
            out.append((score, x0 + cx, y0 + cy, "bgdiff"))
        out.sort(key=lambda t: -t[0])
        return out[:5]

    @staticmethod
    def _contour_mask(shape_hw, contour):
        m = np.zeros(shape_hw[:2], dtype=np.uint8)
        cv2.drawContours(m, [contour], -1, 255, -1)
        return m

    def _clear_lock(self, reason: str = "", now: float = 0.0):
        last = None if self._track_xy is None else (float(self._track_xy[0]), float(self._track_xy[1]))
        if self._locked and reason:
            logger.info(
                f"[LieAlert] unlock ({reason}) "
                f"last={None if last is None else (int(last[0]), int(last[1]))}"
            )
        if reason == "stale_static" and last is not None:
            self._stale_ban_xy = last
            self._stale_ban_until = float(now) + max(0.5, self.stale_ban_s)
        self.following = False
        self.debug_pt = None
        self.debug_bbox = None
        self.debug_decoy_bboxes = []
        self._shape_sil = None
        self._shape_edge = None
        self._fill_sil = None
        self._fill_rot = {}
        self._fill_hold = 0
        self._yolo_hold = 0
        self._shape_nat_wh = (64, 64)
        self._decoy_centers_prev = []
        self._decoy_vel = None
        self._decoy_vel_t = 0.0
        self._locked = False
        self._track_xy = None
        self._track_vel = (0.0, 0.0)
        self._lock_anchor_xy = None
        self._lock_moved_t = float(now) if now else 0.0
        self._flow_pts = None
        self._patch_gray = None
        self._patch_hw = (0, 0)
        self._lock_age = 0
        self._coast_streak = 0
        self._white_gone_t = 0.0
        self._white_miss_since = 0.0
        self._yolo_last_infer_t = 0.0
        self._yolo_last_hit = None
        self._target_indep_vel = (0.0, 0.0)
        self._target_angle = None
        self._target_angle_t = 0.0
        self._target_spin_dps = 0.0
        self._spin_hits = 0
        self._still_hits = 0
        self._spin_watch = []
        self._last_spin_xy = None
        self._spinback_cool = 0
        self._rot_sil_cache = {}
        self._shape_peak_model = None
        self._all_shape_prev = []
        self._all_shape_prev_t = 0.0
        self._stuck_frames = 0

    def _is_stale_banned(self, x: float, y: float, now: float) -> bool:
        if self._stale_ban_xy is None or now >= float(self._stale_ban_until):
            return False
        bx, by = self._stale_ban_xy
        r = float(self.stale_ban_radius_px)
        return ((x - bx) ** 2 + (y - by) ** 2) <= (r * r)

    def _follow_shape(self, img_bgr, capture, now: float):
        # If the main loop stalled (YOLO load etc.), wall-clock stale timers lie.
        prev_tick = float(getattr(self, "_follow_tick_t", 0.0) or 0.0)
        gap = (now - prev_tick) if prev_tick > 0 else 0.0
        self._follow_tick_t = now
        pt = self._locate_shape(img_bgr, now)
        if pt is not None:
            # White can park on a static false highlight. Keep stale-unlock
            # for the whole white phase (not just the first ~12 frames).
            name = getattr(self, "_last_shape_name", "") or ""
            white_phase = name == "white" and float(getattr(self, "_white_gone_t", 0.0) or 0.0) <= 0.0
            if white_phase:
                if gap >= max(0.5, float(self.stale_lock_s)):
                    # Loop was frozen — reset stale clock; do not unlock.
                    self._lock_anchor_xy = (float(pt[0]), float(pt[1]))
                    self._lock_moved_t = now
                elif self._lock_anchor_xy is None:
                    self._lock_anchor_xy = (float(pt[0]), float(pt[1]))
                    self._lock_moved_t = now
                else:
                    ax, ay = self._lock_anchor_xy
                    moved = ((pt[0] - ax) ** 2 + (pt[1] - ay) ** 2) ** 0.5
                    if moved >= self.stale_lock_min_move_px:
                        self._lock_anchor_xy = (float(pt[0]), float(pt[1]))
                        self._lock_moved_t = now
                    elif (now - self._lock_moved_t) >= self.stale_lock_s:
                        self._clear_lock("stale_static", now)
                        return

            self._last_seen_t = now
            self._locked = True
            self.debug_pt = (int(pt[0]), int(pt[1]))
            self.following = True
            # Top-up sticky while following so a 35s hold does not end mid-game
            # after a long stall (00:16:48 + 35s == session end).
            self._hold_until = max(float(self._hold_until), now + 8.0)
            self._move_mouse_to_frame_point(pt, capture)
            if now - self._follow_log_t > 1.0:
                self._follow_log_t = now
                name = self._last_shape_name or "white"
                logger.info(
                    f"[LieAlert] following shape={name} locked={self._locked} @ "
                    f"frame=({pt[0]:.0f},{pt[1]:.0f})"
                )
            return

        # Brief coast on velocity — do NOT jump to unrelated blobs
        if self._locked and self._track_xy is not None and (
            now - self._last_seen_t
        ) < self.lose_target_grace_s:
            vx, vy = self._track_vel
            # Dampen prediction so we don't fly off
            pred = (
                self._track_xy[0] + vx * 0.03,
                self._track_xy[1] + vy * 0.03,
            )
            self.debug_pt = (int(pred[0]), int(pred[1]))
            self.following = True
            self._move_mouse_to_frame_point(pred, capture)
            return

        self._clear_lock("lost", now)

    def _seed_flow_and_patch(self, gray, cx: float, cy: float, bbox=None):
        """Init LK points + grayscale patch around an acquire hit."""
        fh, fw = gray.shape[:2]
        if bbox is not None:
            x, y, bw, bh = [int(v) for v in bbox]
        else:
            s = 36
            x, y, bw, bh = int(cx - s), int(cy - s), 2 * s, 2 * s
        x = max(0, min(fw - 2, x))
        y = max(0, min(fh - 2, y))
        bw = max(12, min(fw - x, bw))
        bh = max(12, min(fh - y, bh))
        self.debug_bbox = (x, y, bw, bh)
        roi = gray[y : y + bh, x : x + bw]
        if roi.size == 0:
            return
        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=24, qualityLevel=0.01, minDistance=4, blockSize=5
        )
        if pts is not None and len(pts) >= 3:
            pts = pts.copy()
            pts[:, 0, 0] += x
            pts[:, 0, 1] += y
            self._flow_pts = pts
        else:
            grid = []
            for gy in (0.3, 0.5, 0.7):
                for gx in (0.3, 0.5, 0.7):
                    grid.append([[x + bw * gx, y + bh * gy]])
            self._flow_pts = np.array(grid, dtype=np.float32)
        ph = min(48, bh)
        pw = min(48, bw)
        x2 = max(0, int(cx - pw / 2))
        y2 = max(0, int(cy - ph / 2))
        x2 = min(fw - pw, x2)
        y2 = min(fh - ph, y2)
        patch = gray[y2 : y2 + ph, x2 : x2 + pw]
        if patch.size > 0 and patch.shape[0] >= 12 and patch.shape[1] >= 12:
            self._patch_gray = patch.copy()
            self._patch_hw = (ph, pw)

    def _capture_shape_silhouette(self, img_bgr, bbox):
        """
        From the bright white acquire box, store silhouette for decoy matching.
        Decoys share this shape but blend into the background color.
        """
        if img_bgr is None or bbox is None:
            return
        fh, fw = img_bgr.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        x = max(0, min(fw - 2, x))
        y = max(0, min(fh - 2, y))
        bw = max(12, min(fw - x, bw))
        bh = max(12, min(fh - y, bh))
        crop = img_bgr[y : y + bh, x : x + bw]
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Bright core of the starting white pattern
        mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 90, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # Fallback: high-V pixels
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        ct = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(ct) < 80:
            return
        # Tight crop around the white contour (ignore oversized track box)
        sx, sy, sw, sh = cv2.boundingRect(ct)
        pad = 4
        sx = max(0, sx - pad)
        sy = max(0, sy - pad)
        sw = min(mask.shape[1] - sx, sw + 2 * pad)
        sh = min(mask.shape[0] - sy, sh + 2 * pad)
        sil = np.zeros((sh, sw), dtype=np.uint8)
        ct_local = ct - np.array([[sx, sy]])
        cv2.drawContours(sil, [ct_local], -1, 255, -1)
        # Natural on-screen size of the white pattern
        nat_w = max(28, min(180, sw))
        nat_h = max(28, min(180, sh))
        self._shape_nat_wh = (int(nat_w), int(nat_h))
        self._fill_sil = sil.copy()
        self._fill_rot = {}
        # Normalize template for Hu / multi-scale match
        th = max(32, min(96, nat_h))
        tw = max(32, min(96, nat_w))
        sil = cv2.resize(sil, (tw, th), interpolation=cv2.INTER_NEAREST)
        _, sil = cv2.threshold(sil, 127, 255, cv2.THRESH_BINARY)
        self._shape_sil = sil
        self._shape_edge = cv2.dilate(
            cv2.Canny(sil, 40, 120), np.ones((2, 2), np.uint8), iterations=1
        )
        self._rot_sil_cache = {}  # invalidate rotated cache after new acquire sil

    @staticmethod
    def _wrap_angle_delta(a: float, b: float) -> float:
        """Smallest signed delta a-b in degrees, range (-180, 180]."""
        return float((a - b + 180.0) % 360.0 - 180.0)

    def _rotated_silhouette(self, angle_deg: float):
        """Cached rotation of acquire silhouette (deg)."""
        if self._shape_sil is None:
            return None
        key = int(round(float(angle_deg))) % 180  # 180° enough for undirected sil match
        cached = self._rot_sil_cache.get(key)
        if cached is not None:
            return cached
        sil = self._shape_sil
        h, w = sil.shape[:2]
        # Pad so corners are not clipped
        pad = int(0.35 * max(h, w)) + 2
        canvas = cv2.copyMakeBorder(sil, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ch, cw = canvas.shape[:2]
        M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), float(key), 1.0)
        rot = cv2.warpAffine(
            canvas, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0
        )
        # Tight crop
        ys, xs = np.where(rot > 0)
        if len(xs) < 20:
            self._rot_sil_cache[key] = rot
            return rot
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        rot = rot[y0:y1, x0:x1]
        self._rot_sil_cache[key] = rot
        return rot

    def _note_target_angle(self, angle_deg: float, now: float):
        """Update signed spin-rate EMA from a newly measured target orientation."""
        if angle_deg is None or not np.isfinite(angle_deg):
            return
        ang = float(angle_deg)
        if self._target_angle is not None and self._target_angle_t > 0:
            dt = max(1e-3, min(0.12, now - self._target_angle_t))
            delta = self._wrap_angle_delta(ang, self._target_angle)
            dps = delta / dt
            # Ignore tiny jitter; clip insane spikes
            if abs(dps) < 15.0:
                dps = 0.0
            dps = float(np.clip(dps, -720.0, 720.0))
            a = 0.45
            self._target_spin_dps = a * dps + (1.0 - a) * float(self._target_spin_dps)
        self._target_angle = ang
        self._target_angle_t = float(now)

    def _best_angle_at(
        self, dog, cx_local: float, cy_local: float, tw: int, th: int, angles=None
    ):
        """
        Best silhouette rotation at an approximate local center on a DoG map.
        Returns (corr, angle_deg, cx_local, cy_local) or None.
        """
        if self._shape_sil is None or dog is None:
            return None
        best = None
        # Coarse 20° sweep — need coverage because only the target spins fast.
        # Callers measuring frame-to-frame spin pass a finer `angles` list.
        if angles is None:
            if self._target_angle is not None:
                b = int(round(float(self._target_angle))) % 180
                angles = []
                for off in range(-40, 41, 20):
                    angles.append((b + off) % 180)
                angles += list(range(0, 180, 45))
                seen = set()
                angles = [a for a in angles if not (a in seen or seen.add(a))]
            else:
                angles = list(range(0, 180, 20))
        pad = 8
        for ang in angles:
            rs = self._rotated_silhouette(ang)
            if rs is None:
                continue
            tmpl = cv2.resize(rs, (tw, th), interpolation=cv2.INTER_AREA)
            if tmpl.shape[0] >= dog.shape[0] or tmpl.shape[1] >= dog.shape[1]:
                continue
            x0 = int(cx_local - tw / 2)
            y0 = int(cy_local - th / 2)
            sx0 = max(0, x0 - pad)
            sy0 = max(0, y0 - pad)
            sx1 = min(dog.shape[1], x0 + tw + pad)
            sy1 = min(dog.shape[0], y0 + th + pad)
            if sx1 - sx0 <= tw + 1 or sy1 - sy0 <= th + 1:
                continue
            roi = dog[sy0:sy1, sx0:sx1]
            try:
                res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            _mn, mx, _mnloc, mxloc = cv2.minMaxLoc(res)
            if not np.isfinite(mx):
                continue
            if best is None or mx > best[0]:
                best = (
                    float(mx),
                    float(ang),
                    sx0 + mxloc[0] + tw / 2.0,
                    sy0 + mxloc[1] + th / 2.0,
                )
        return best

    def _ensure_shape_silhouette(self, img_bgr=None):
        """Prefer acquire silhouette; else fall back to loaded shape templates."""
        if self._shape_sil is not None and self._shape_edge is not None:
            return True
        # Build from first loaded shape template (star etc.)
        for _path, tmpl in getattr(self, "_shape_list", []) or []:
            if tmpl is None or tmpl.size == 0:
                continue
            if len(tmpl.shape) == 3 and tmpl.shape[2] == 4:
                alpha = tmpl[:, :, 3]
                sil = np.where(alpha > 32, 255, 0).astype(np.uint8)
            else:
                gray = (
                    cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                    if len(tmpl.shape) == 3
                    else tmpl
                )
                _, sil = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            if int(cv2.countNonZero(sil)) < 40:
                continue
            th, tw = sil.shape[:2]
            scale = 64.0 / max(th, tw)
            nh, nw = max(24, int(th * scale)), max(24, int(tw * scale))
            sil = cv2.resize(sil, (nw, nh), interpolation=cv2.INTER_NEAREST)
            self._shape_sil = sil
            self._shape_edge = cv2.dilate(
                cv2.Canny(sil, 40, 120), np.ones((2, 2), np.uint8), iterations=1
            )
            self._shape_nat_wh = (max(40, nw), max(40, nh))
            return True
        return False

    def _track_lk(self, gray):
        """Optical-flow step -> (cx, cy) or None."""
        if self._flow_pts is None or self._prev_gray is None:
            return None
        if self._prev_gray.shape != gray.shape:
            return None
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._flow_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if nxt is None or st is None:
            return None
        good = nxt[st.flatten() == 1].reshape(-1, 2)
        if good is None or len(good) < 2:
            return None
        self._flow_pts = good.reshape(-1, 1, 2).astype(np.float32)
        cx = float(np.median(good[:, 0]))
        cy = float(np.median(good[:, 1]))
        xs, ys = good[:, 0], good[:, 1]
        x0, y0 = int(xs.min() - 8), int(ys.min() - 8)
        x1, y1 = int(xs.max() + 8), int(ys.max() + 8)
        self.debug_bbox = (x0, y0, max(12, x1 - x0), max(12, y1 - y0))
        return (cx, cy)

    def _match_patch(self, gray, search_box):
        """NCC of stored patch inside search window -> (score, cx, cy) or None."""
        if self._patch_gray is None:
            return None
        x0, y0, x1, y1 = search_box
        crop = gray[y0:y1, x0:x1]
        th, tw = self._patch_gray.shape[:2]
        if crop.shape[0] <= th + 2 or crop.shape[1] <= tw + 2:
            return None
        try:
            res = cv2.matchTemplate(crop, self._patch_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except Exception:
            return None
        max_val = float(max_val)
        if not np.isfinite(max_val) or max_val < 0.35:
            return None
        cx = x0 + max_loc[0] + tw / 2.0
        cy = y0 + max_loc[1] + th / 2.0
        self.debug_bbox = (int(cx - tw / 2), int(cy - th / 2), tw, th)
        return (max_val, cx, cy)

    def _dialog_roi(self, fw, fh):
        """Tight ROI over the lie-detector dialog (avoid Bilibili / HUD chrome)."""
        side = float(getattr(self, "exclude_side_ratio", 0.18))
        top = float(self.exclude_top_ratio)
        bot = float(self.exclude_bottom_ratio)
        x0 = int(fw * max(side, 0.22))
        x1 = int(fw * min(1.0 - side, 0.78))
        y0 = int(fh * max(top, 0.22))
        y1 = int(fh * min(1.0 - bot, 0.78))
        if x1 - x0 < 80 or y1 - y0 < 80:
            x0, x1 = int(fw * 0.25), int(fw * 0.75)
            y0, y1 = int(fh * 0.25), int(fh * 0.75)
        return (x0, y0, x1, y1)

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax, ay, aw, ah = [int(v) for v in a]
        bx, by, bw, bh = [int(v) for v in b]
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter) / float(max(1, union))

    def refresh_pattern_overlay(self, img_bgr, primary_bbox=None):
        """
        Viz helper:
          RED primary = tracked / center white target bbox (caller)
          GREEN decoys = SAME silhouette as start white, but camouflaged
                         (color close to local background — not bright white)

        Decoys never overlap primary (IoU filter + inset).
        """
        fh, fw = img_bgr.shape[:2]
        if primary_bbox is None:
            # Open-game: still pick brightest center white as red target hint
            dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)
            crop = img_bgr[dy0:dy1, dx0:dx1]
            whites = self._white_target_candidates(
                crop, None, frame_h=fh, frame_y0=dy0, frame_w=fw, frame_x0=dx0
            )
            if whites:
                best = max(
                    whites,
                    key=lambda t: t[0]
                    + 0.3
                    * (
                        1.0
                        - (
                            ((dx0 + t[1] - fw * 0.5) / max(1.0, fw * 0.5)) ** 2
                            + ((dy0 + t[2] - fh * 0.5) / max(1.0, fh * 0.5)) ** 2
                        )
                        ** 0.5
                    ),
                )
                primary_bbox = best[4]
                if self._shape_sil is None:
                    self._capture_shape_silhouette(img_bgr, primary_bbox)

        decoys = self._find_camouflage_decoys(img_bgr, primary_bbox)
        self.debug_decoy_bboxes = decoys
        return primary_bbox, decoys

    def _update_decoy_consensus(self, decoys, now: float):
        """
        Interference items share one motion path. Estimate that consensus velocity
        so the randomly-moving target is not mistaken for a decoy.
        """
        centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in decoys]
        if (
            self._decoy_centers_prev
            and len(centers) >= 2
            and self._decoy_vel_t > 0
        ):
            dt = max(1e-3, min(0.12, now - self._decoy_vel_t))
            vels = []
            used = set()
            for cx, cy in centers:
                best_j, best_d = -1, 1e9
                for j, (px, py) in enumerate(self._decoy_centers_prev):
                    if j in used:
                        continue
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_j = d, j
                if best_j >= 0 and best_d <= 90:
                    used.add(best_j)
                    px, py = self._decoy_centers_prev[best_j]
                    vels.append(((cx - px) / dt, (cy - py) / dt))
            if len(vels) >= 2:
                vx = float(np.median([v[0] for v in vels]))
                vy = float(np.median([v[1] for v in vels]))
                # Clamp insane spikes
                spd = (vx * vx + vy * vy) ** 0.5
                if spd > 900:
                    s = 900.0 / spd
                    vx, vy = vx * s, vy * s
                if self._decoy_vel is None:
                    self._decoy_vel = (vx, vy)
                else:
                    a = 0.55
                    ovx, ovy = self._decoy_vel
                    self._decoy_vel = (a * vx + (1 - a) * ovx, a * vy + (1 - a) * ovy)
        self._decoy_centers_prev = centers
        self._decoy_vel_t = float(now)
        self.debug_decoy_bboxes = list(decoys)

    def _near_decoy(self, cx: float, cy: float, pad: float = 10.0) -> bool:
        for x, y, w, h in getattr(self, "debug_decoy_bboxes", []) or []:
            if (x - pad) <= cx <= (x + w + pad) and (y - pad) <= cy <= (y + h + pad):
                return True
        return False

    def _step_follows_decoy_path(self, cx: float, cy: float, now: float) -> bool:
        """True if displacement from last track looks like the shared decoy path."""
        if self._decoy_vel is None or self._track_xy is None:
            return False
        dt = max(1e-3, min(0.10, now - self._last_track_t))
        dx = cx - self._track_xy[0]
        dy = cy - self._track_xy[1]
        mag = (dx * dx + dy * dy) ** 0.5
        if mag < 3.5:
            return False
        edx = self._decoy_vel[0] * dt
        edy = self._decoy_vel[1] * dt
        emag = (edx * edx + edy * edy) ** 0.5
        if emag < 2.0:
            return False
        cos = (dx * edx + dy * edy) / (mag * emag + 1e-6)
        err = ((dx - edx) ** 2 + (dy - edy) ** 2) ** 0.5
        return bool(cos > 0.82 and err < max(9.0, 0.6 * emag + 5.0))

    def _rejects_decoy_trap(self, cx: float, cy: float, now: float) -> bool:
        """
        Reject poses that latch onto the synchronized decoy flock.
        Path agreement is the primary signal (all greens share one path).
        Landing exactly on a known decoy box is secondary.
        """
        if self._step_follows_decoy_path(cx, cy, now):
            return True
        # Jumping onto a decoy far from prediction = flock latch
        if self._near_decoy(cx, cy, pad=8.0) and self._track_xy is not None:
            pred = self._pred_xy(now) or self._track_xy
            dpred = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
            if dpred > max(14.0, float(self.max_frame_step_px) * 0.55):
                return True
        return False

    def _dog_roi(self, gray_roi):
        dog = cv2.absdiff(
            cv2.GaussianBlur(gray_roi, (0, 0), 1.0),
            cv2.GaussianBlur(gray_roi, (0, 0), 3.2),
        )
        return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _local_spin_deg(
        self,
        dog_curr,
        dog_prev,
        cx_l,
        cy_l,
        tw,
        th,
        prev_cx_l=None,
        prev_cy_l=None,
        fine: bool = False,
    ):
        """
        Orientation change of the same moving blob across consecutive frames.
        Back-warp previous sample by track velocity. Target spins; decoys
        only translate → spin≈0. Returns (abs_deg, angle_now, corr_now) or None.
        """
        if dog_prev is None or dog_curr is None:
            return None
        if dog_prev.shape != dog_curr.shape:
            return None
        if prev_cx_l is None:
            prev_cx_l = cx_l
        if prev_cy_l is None:
            prev_cy_l = cy_l
        a_now = self._best_angle_at(dog_curr, cx_l, cy_l, tw, th)
        a_prv = self._best_angle_at(dog_prev, prev_cx_l, prev_cy_l, tw, th)
        if a_now is None or a_prv is None:
            return None
        if a_now[0] < 0.20 or a_prv[0] < 0.20:
            return None
        if fine:
            # 5° only for the leave-spin check. Faded ranking stays on the
            # coarse sweep — a fine pass was matching static neighbors as spin.
            fine_now = [(float(a_now[1]) + off) % 180.0 for off in range(-15, 16, 5)]
            fine_prv = [(float(a_prv[1]) + off) % 180.0 for off in range(-15, 16, 5)]
            a_now = (
                self._best_angle_at(dog_curr, cx_l, cy_l, tw, th, angles=fine_now) or a_now
            )
            a_prv = (
                self._best_angle_at(
                    dog_prev, prev_cx_l, prev_cy_l, tw, th, angles=fine_prv
                )
                or a_prv
            )
            if a_now[0] < 0.20 or a_prv[0] < 0.20:
                return None
        err = abs(self._wrap_angle_delta(a_now[1], a_prv[1]))
        err = min(err, 180.0 - err)
        return float(err), float(a_now[1]), float(a_now[0])

    def _spin_at_xy(
        self, img_bgr, cx: float, cy: float, fine: bool = False, prev_xy=None
    ):
        """Frame-to-frame spin of the acquire silhouette at one frame point.

        prev_xy, when set, is that same blob's center last frame. Otherwise the
        track velocity is used to back-warp the previous sample.
        """
        if img_bgr is None or self._shape_sil is None or self._prev_gray is None:
            return None
        fh, fw = img_bgr.shape[:2]
        if self._prev_gray.shape[:2] != (fh, fw):
            return None
        nat_w, nat_h = [int(v) for v in getattr(self, "_shape_nat_wh", (64, 64))]
        tw = max(28, min(nat_w, 80))
        th = max(28, min(nat_h, 80))
        if prev_xy is None:
            anchor_x, anchor_y = float(cx), float(cy)
            extra = 0.0
        else:
            anchor_x = (float(cx) + float(prev_xy[0])) * 0.5
            anchor_y = (float(cy) + float(prev_xy[1])) * 0.5
            extra = (
                (float(cx) - float(prev_xy[0])) ** 2
                + (float(cy) - float(prev_xy[1])) ** 2
            ) ** 0.5
            if extra > 70.0:
                return None
        rad = int(max(tw, th) * 0.75) + 6 + int(extra)
        x0 = max(0, int(anchor_x - rad))
        y0 = max(0, int(anchor_y - rad))
        x1 = min(fw, int(anchor_x + rad))
        y1 = min(fh, int(anchor_y + rad))
        if x1 - x0 <= tw + 8 or y1 - y0 <= th + 8:
            return None
        gray = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        dog = self._dog_roi(gray)
        dog_prev = self._dog_roi(self._prev_gray[y0:y1, x0:x1])
        if prev_xy is None:
            dt = 0.033
            pvx, pvy = self._track_vel if self._track_xy is not None else (0.0, 0.0)
            prev_cx_l = (cx - pvx * dt) - x0
            prev_cy_l = (cy - pvy * dt) - y0
        else:
            prev_cx_l = float(prev_xy[0]) - x0
            prev_cy_l = float(prev_xy[1]) - y0
        return self._local_spin_deg(
            dog,
            dog_prev,
            cx - x0,
            cy - y0,
            tw,
            th,
            prev_cx_l=prev_cx_l,
            prev_cy_l=prev_cy_l,
            fine=fine,
        )

    def _leaves_spinning_blob(self, img_bgr, cx: float, cy: float) -> bool:
        """
        After fade, a step onto a non-rotating spot leaves the real target.
        Decoys translate; only the target keeps turning.
        """
        if self._white_gone_t <= 0 or self._track_xy is None:
            return False
        # One noisy angle bin is not enough. Arm only after several
        # confirmed turns, then refuse a step onto a blob that is not turning.
        if int(getattr(self, "_spin_hits", 0)) < 3:
            return False
        jump = (
            (cx - self._track_xy[0]) ** 2 + (cy - self._track_xy[1]) ** 2
        ) ** 0.5
        if jump <= 8.0:
            return False
        there = self._spin_at_xy(img_bgr, cx, cy, fine=False)
        return there is None or there[0] < 15.0

    def _collect_shape_centers(self, img_bgr, max_n: int = 6):
        """Same-shape centers in the dialog. Not the red target."""
        if not self._ensure_shape_silhouette(img_bgr) or self._shape_sil is None:
            return []
        fh, fw = img_bgr.shape[:2]
        x0, y0, x1, y1 = self._dialog_roi(fw, fh)
        roi = img_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        dog = self._dog_roi(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        nat_w, nat_h = [int(v) for v in getattr(self, "_shape_nat_wh", (64, 64))]
        if self._target_angle is not None:
            base = int(round(float(self._target_angle))) % 180
            angles = [
                base,
                (base + 30) % 180,
                (base - 30) % 180,
                (base + 90) % 180,
            ]
        else:
            angles = [0, 45, 90, 135]
        raw = []
        for scale in (1.0, 1.25):
            tw = max(28, min(int(nat_w * scale), dog.shape[1] - 2))
            th = max(28, min(int(nat_h * scale), dog.shape[0] - 2))
            if tw >= dog.shape[1] - 2 or th >= dog.shape[0] - 2:
                continue
            for ang in angles:
                rs = self._rotated_silhouette(ang)
                if rs is None:
                    continue
                tmpl = cv2.resize(rs, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(dog, tmpl, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                work = res
                for _peak in range(2):
                    _mn, mx, _mnloc, mxloc = cv2.minMaxLoc(work)
                    if not np.isfinite(mx) or mx < 0.24:
                        break
                    px, py = int(mxloc[0]), int(mxloc[1])
                    if work is res:
                        work = res.copy()
                    cv2.circle(work, (px, py), max(tw, th) // 3, 0, -1)
                    raw.append(
                        (
                            float(mx),
                            x0 + px + tw / 2.0,
                            y0 + py + th / 2.0,
                            float(ang),
                            float(tw),
                            float(th),
                        )
                    )
        try:
            for corr, cx, cy, ang, tw, th in self._find_all_shape_peaks(
                img_bgr, max_peaks=8
            ):
                raw.append((float(corr), float(cx), float(cy), float(ang), float(tw), float(th)))
        except Exception:
            pass
        raw.sort(key=lambda t: -t[0])
        kept = []
        for item in raw:
            cx, cy = item[1], item[2]
            if any(((cx - p[1]) ** 2 + (cy - p[2]) ** 2) ** 0.5 < 48.0 for p in kept):
                continue
            kept.append(item)
            if len(kept) >= max_n:
                break
        return kept

    @staticmethod
    def _fold_angle_delta(delta: float) -> float:
        """Signed orientation delta for an undirected silhouette, in (-90, 90]."""
        d = float(delta)
        if d > 90.0:
            d -= 180.0
        elif d < -90.0:
            d += 180.0
        return d

    def _observe_spin_watch(self, img_bgr, centers):
        """
        Remember each blob's own orientation. A turn is a same-blob
        frame-to-frame change, not a one-frame angle-bin flip.
        Returns the strongest blob that is turning now, or None.
        """
        prev = list(getattr(self, "_spin_watch", []) or [])
        used = set()
        fresh = []
        best = None
        best_key = -1.0
        for corr, cx, cy, _ang, tw, th in centers:
            j_best, d_best = -1, 1e9
            for j, item in enumerate(prev):
                if j in used:
                    continue
                d = ((cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2) ** 0.5
                if d < d_best:
                    d_best, j_best = d, j
            linked = j_best >= 0 and d_best <= 48.0
            spin = None
            if linked:
                used.add(j_best)
                spin = self._spin_at_xy(
                    img_bgr,
                    cx,
                    cy,
                    fine=False,
                    prev_xy=(prev[j_best]["cx"], prev[j_best]["cy"]),
                )
            streak = 0
            sign = 0
            ang = None
            turn = 0.0
            if linked and spin is None:
                streak = max(0, int(prev[j_best]["streak"]) - 1)
                sign = int(prev[j_best]["sign"]) if streak > 0 else 0
                ang = prev[j_best].get("ang")
            elif spin is not None and linked:
                turn, ang, spin_corr = spin
                signed = self._fold_angle_delta(
                    self._wrap_angle_delta(ang, prev[j_best]["ang"])
                    if prev[j_best].get("ang") is not None
                    else 0.0
                )
                ad = abs(signed)
                # 12–45° is a real step. Larger jumps are coarse-bin noise.
                if spin_corr >= 0.26 and 12.0 <= ad <= 45.0 and ad >= 12.0:
                    sgn = 1 if signed >= 0.0 else -1
                    old = int(prev[j_best]["streak"])
                    old_sgn = int(prev[j_best]["sign"])
                    if old <= 0 or old_sgn == sgn:
                        streak = old + 1
                    else:
                        streak = 1
                    sign = sgn
                elif ad > 45.0:
                    streak = 0
                    sign = 0
                else:
                    streak = max(0, int(prev[j_best]["streak"]) - 1)
                    sign = int(prev[j_best]["sign"]) if streak > 0 else 0
            fresh.append(
                {
                    "cx": float(cx),
                    "cy": float(cy),
                    "ang": None if ang is None else float(ang),
                    "streak": int(streak),
                    "sign": int(sign),
                    "tw": float(tw),
                    "th": float(th),
                }
            )
            if streak >= 2 and turn >= 12.0 and self._track_xy is not None:
                away = (
                    (cx - self._track_xy[0]) ** 2 + (cy - self._track_xy[1]) ** 2
                ) ** 0.5
                if away >= 36.0:
                    key = streak * 10.0 + turn
                    if key > best_key:
                        best_key = key
                        dt = 0.033
                        if self._last_track_t:
                            dt = max(1e-3, min(0.08, 0.033))
                        pvx = (cx - prev[j_best]["cx"]) / dt
                        pvy = (cy - prev[j_best]["cy"]) / dt
                        spd = (pvx * pvx + pvy * pvy) ** 0.5
                        if spd > 400.0:
                            pvx *= 400.0 / spd
                            pvy *= 400.0 / spd
                        best = (
                            float(cx),
                            float(cy),
                            float(ang if ang is not None else 0.0),
                            float(tw),
                            float(th),
                            float(pvx),
                            float(pvy),
                        )
        self._spin_watch = fresh
        return best

    def _repair_spin_jump(self, img_bgr, now: float, held_spin):
        """
        Red center is not rotating → jump onto the shape that is.
        Runs ahead of flow, coast, outlier snap, and the step limit.
        Does not run while the white shape is still visible.
        """
        if self._white_gone_t <= 0 or self._track_xy is None:
            return None
        cool = int(getattr(self, "_spinback_cool", 0))
        if cool > 0:
            self._spinback_cool = cool - 1
        held_turn = held_spin is not None and held_spin[0] >= 12.0
        if held_turn:
            self._still_hits = 0
            self._last_spin_xy = (
                float(self._track_xy[0]),
                float(self._track_xy[1]),
            )
        else:
            self._still_hits = int(getattr(self, "_still_hits", 0)) + 1
        best = self._observe_spin_watch(
            img_bgr, self._collect_shape_centers(img_bgr)
        )
        if cool > 0 or best is None or int(self._still_hits) < 2:
            return None
        # Only jump back after the red center itself was seen turning,
        # then drifted off that blob. A quiet reading on the same spot stays.
        if self._last_spin_xy is None:
            return None
        drifted = (
            (self._track_xy[0] - self._last_spin_xy[0]) ** 2
            + (self._track_xy[1] - self._last_spin_xy[1]) ** 2
        ) ** 0.5
        if drifted < 36.0:
            return None
        cx, cy, ang, tw, th, pvx, pvy = best
        self._spinback_cool = 3
        self._still_hits = 0
        self._last_spin_xy = (float(cx), float(cy))
        bbox = (int(cx - tw / 2.0), int(cy - th / 2.0), int(tw), int(th))
        return (cx, cy, bbox, ang, pvx, pvy)

    def _find_faded_target_near(self, img_bgr, pred, rad: int, now: float):
        """
        After white fades: find same silhouette near prediction.
        Decoys share one path and do NOT spin; only the target rotates.
        Score = shape match + proximity + local frame-to-frame spin.
        Returns (score, cx, cy, bbox, angle_deg) or None.
        """
        if not self._ensure_shape_silhouette(img_bgr) or pred is None:
            return None
        sil = self._shape_sil
        fh, fw = img_bgr.shape[:2]
        dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)
        cx0, cy0 = float(pred[0]), float(pred[1])
        x0 = max(dx0, int(cx0 - rad))
        y0 = max(dy0, int(cy0 - rad))
        x1 = min(dx1, int(cx0 + rad))
        y1 = min(dy1, int(cy0 + rad))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None
        gray = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        dog = self._dog_roi(gray)
        dog_prev = None
        if self._prev_gray is not None and self._prev_gray.shape[:2] == (fh, fw):
            dog_prev = self._dog_roi(self._prev_gray[y0:y1, x0:x1])

        nat_w, nat_h = [int(v) for v in getattr(self, "_shape_nat_wh", (64, 64))]
        max_jump = float(self.max_frame_step_px) * 2.5
        # A few rotations so a spinning star still matches the acquire sil
        if self._target_angle is not None:
            base = int(round(float(self._target_angle))) % 180
            seed_angles = [base, (base + 20) % 180, (base - 20) % 180, (base + 40) % 180]
        else:
            seed_angles = [0, 20, 40, 60, 90]

        # Pass 1: gather shape peaks (cheap). Pass 2: spin-rank top few.
        cands = []
        for scale in (0.85, 1.05, 1.25, 1.45):
            tw = max(28, min(int(nat_w * scale), dog.shape[1] - 2))
            th = max(28, min(int(nat_h * scale), dog.shape[0] - 2))
            if tw >= dog.shape[1] - 2 or th >= dog.shape[0] - 2:
                continue
            for ang in seed_angles:
                rs = self._rotated_silhouette(ang)
                if rs is None:
                    continue
                tmpl = cv2.resize(rs, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(dog, tmpl, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                work = res.copy()
                for _peak in range(2):
                    _mn, mx, _mnloc, mxloc = cv2.minMaxLoc(work)
                    if not np.isfinite(mx) or mx < 0.23:
                        break
                    px, py = int(mxloc[0]), int(mxloc[1])
                    cv2.circle(work, (px, py), max(tw, th) // 3, 0, -1)
                    cx = x0 + px + tw / 2.0
                    cy = y0 + py + th / 2.0
                    dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                    if dist > rad:
                        continue
                    if self._track_xy is not None:
                        jump = (
                            (cx - self._track_xy[0]) ** 2
                            + (cy - self._track_xy[1]) ** 2
                        ) ** 0.5
                        if jump > max_jump:
                            continue
                    if self._step_follows_decoy_path(cx, cy, now):
                        continue
                    if self._near_decoy(cx, cy, pad=6.0) and dist > max(
                        16.0, rad * 0.30
                    ):
                        continue
                    prox = max(0.0, 0.40 - dist / max(1.0, rad))
                    indep = 0.25 if self._decoy_vel is not None else 0.05
                    pre = float(mx) + prox + indep
                    cands.append(
                        (pre, float(mx), cx, cy, px, py, tw, th, float(ang), dist)
                    )

        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        # De-dupe spatially, keep top 5 for spin scoring
        picked = []
        for item in cands:
            cx, cy = item[2], item[3]
            if any(((cx - p[2]) ** 2 + (cy - p[3]) ** 2) ** 0.5 < 28.0 for p in picked):
                continue
            picked.append(item)
            if len(picked) >= 5:
                break

        dt = max(1e-3, min(0.08, now - self._last_track_t)) if self._last_track_t else 0.033
        pvx, pvy = self._track_vel if self._track_xy is not None else (0.0, 0.0)
        best = None
        best_key = -1.0
        for pre, mx, cx, cy, px, py, tw, th, ang, dist in picked:
            ang_use = float(ang)
            spin_bonus = 0.0
            cx_l = px + tw / 2.0
            cy_l = py + th / 2.0
            spin_info = self._local_spin_deg(
                dog,
                dog_prev,
                cx_l,
                cy_l,
                tw,
                th,
                prev_cx_l=cx_l - pvx * dt,
                prev_cy_l=cy_l - pvy * dt,
            )
            if spin_info is not None:
                spin_deg, ang_use, _corr_a = spin_info
                # 5° measurement. Ignore sub-15° chatter; a real turn is larger
                # and outranks a static neighbor that is only a bit closer.
                if spin_deg >= 15.0:
                    spin_bonus = 0.95 * min(1.0, (spin_deg - 8.0) / 28.0)
                elif spin_deg <= 5.0:
                    spin_bonus = -0.50 if dist > 18.0 else -0.12
            key = pre + spin_bonus
            if key > best_key:
                best_key = key
                bbox = (int(x0 + px), int(y0 + py), int(tw), int(th))
                best = (float(mx), float(cx), float(cy), bbox, float(ang_use))
        return best

    def _ensure_shape_peak_model(self, allow_block: bool = True):
        if self._shape_peak_model is not None:
            return self._shape_peak_model
        if bool(getattr(self, "_shape_peak_failed", False)):
            return None
        if not allow_block:
            # Follow hot path: never sync-load; kick background warm.
            self._warm_track_models_async()
            return None
        path = str(getattr(self, "shape_peak_model_path", "") or "").strip()
        if not path or not os.path.exists(path):
            self._shape_peak_failed = True
            return None
        with self._model_load_lock:
            if self._shape_peak_model is not None:
                return self._shape_peak_model
            try:
                self._shape_peak_model = ShapePeakModel(path)
                self._shape_peak_model._ensure_session()
                logger.info(f"[LieAlert] Shape-peak model ready ({path})")
            except Exception as exc:
                logger.warning(f"[LieAlert] Shape-peak model unavailable ({exc})")
                self._shape_peak_model = None
                self._shape_peak_failed = True
        return self._shape_peak_model

    def _find_all_shape_peaks(self, img_bgr, max_peaks: int = 10):
        """
        Same-shape instances from the vision model (target + decoys).
        Returns (score, cx, cy, ang, tw, th). These boxes are not the red center.
        White sticky, spin, and decoy-path rejection still pick the target.
        """
        if not self._ensure_shape_silhouette(img_bgr):
            return []
        # Never block the detect/follow thread on ONNX load.
        model = self._ensure_shape_peak_model(allow_block=False)
        if model is None:
            return []
        fh, fw = img_bgr.shape[:2]
        try:
            return model.find_peaks(
                img_bgr,
                silhouette=self._shape_sil,
                nat_wh=getattr(self, "_shape_nat_wh", (64, 64)),
                roi=self._dialog_roi(fw, fh),
                dog_fn=self._dog_roi,
                max_peaks=max_peaks,
                hint_angle=self._target_angle,
            )
        except Exception as exc:
            logger.debug(f"[LieAlert] shape-peak inference failed: {exc}")
            return []

    def _pick_motion_outlier_target(self, img_bgr, now: float, prefer_near=None):
        """
        Decoys share one velocity; the real target is the motion outlier
        (and usually the one that is also spinning).
        Returns (corr, cx, cy, bbox, ang) or None.
        """
        peaks = self._find_all_shape_peaks(img_bgr, max_peaks=10)
        if len(peaks) < 2:
            self._all_shape_prev = [(p[1], p[2], p[3]) for p in peaks]
            self._all_shape_prev_t = float(now)
            return None

        dt = (
            max(1e-3, min(0.12, now - self._all_shape_prev_t))
            if self._all_shape_prev_t > 0
            else 0.033
        )
        # Associate to previous centers
        used = set()
        measured = []  # (cx,cy,ang,vx,vy,corr,tw,th)
        for corr, cx, cy, ang, tw, th in peaks:
            best_j, best_d = -1, 1e9
            for j, (px, py, _pa) in enumerate(self._all_shape_prev):
                if j in used:
                    continue
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= 100:
                used.add(best_j)
                px, py, pang = self._all_shape_prev[best_j]
                vx = (cx - px) / dt
                vy = (cy - py) / dt
                dang = abs(self._wrap_angle_delta(ang, pang))
                dang = min(dang, 180.0 - dang)
                measured.append((corr, cx, cy, ang, vx, vy, dang, tw, th))
            else:
                measured.append((corr, cx, cy, ang, 0.0, 0.0, 0.0, tw, th))

        self._all_shape_prev = [(p[1], p[2], p[3]) for p in peaks]
        self._all_shape_prev_t = float(now)

        moving = [m for m in measured if (m[4] * m[4] + m[5] * m[5]) ** 0.5 > 30.0]
        if len(moving) < 2:
            moving = measured
        # Consensus = median velocity of the flock (decoys dominate)
        cvs = [m[4] for m in moving]
        cys = [m[5] for m in moving]
        cvx = float(np.median(cvs))
        cvy = float(np.median(cys))
        self._decoy_vel = (cvx, cvy)
        self._decoy_vel_t = float(now)

        best = None
        best_key = -1.0
        for corr, cx, cy, ang, vx, vy, dang, tw, th in measured:
            mag = (vx * vx + vy * vy) ** 0.5
            cmag = (cvx * cvx + cvy * cvy) ** 0.5 + 1e-6
            if mag > 20.0 and cmag > 20.0:
                cos = (vx * cvx + vy * cvy) / (mag * cmag)
            else:
                cos = 1.0  # unknown → don't treat as outlier yet
            # Outlier score: disagree with flock.
            # Spin is the same blob's frame-to-frame turn, not a template-angle label.
            outlier = max(0.0, 1.0 - cos) * min(1.5, mag / 120.0)
            spin_s = 0.0
            if prefer_near is None or (
                ((cx - prefer_near[0]) ** 2 + (cy - prefer_near[1]) ** 2) ** 0.5 < 180.0
            ):
                spin_info = self._spin_at_xy(img_bgr, cx, cy)
                if spin_info is not None and spin_info[0] >= 15.0:
                    spin_s = min(1.0, spin_info[0] / 30.0)
                    ang = float(spin_info[1])
            near = 0.0
            if prefer_near is not None:
                dist = ((cx - prefer_near[0]) ** 2 + (cy - prefer_near[1]) ** 2) ** 0.5
                near = max(0.0, 0.35 - dist / 220.0)
            # Strongly prefer flock disagreement + spin
            key = outlier * 1.2 + spin_s * 1.45 + corr * 0.35 + near
            # Hard preference: clear anti-flock motion
            if cos < 0.35 and mag > 40.0:
                key += 0.55
            if key > best_key:
                best_key = key
                bbox = (int(cx - tw / 2), int(cy - th / 2), int(tw), int(th))
                best = (
                    float(corr),
                    float(cx),
                    float(cy),
                    bbox,
                    float(ang),
                    float(cos),
                    float(mag),
                    float(spin_s),
                )
        if best is None:
            return None
        # Require either strong outlier or a blob that is actually spinning.
        corr, cx, cy, bbox, ang, cos, mag, spin_s = best
        if cos > 0.72 and mag > 40.0 and spin_s < 0.40:
            # Still flock-like and not rotating — not a reliable target pick
            return None
        return (corr, cx, cy, bbox, ang)

    def _find_camouflage_decoys(self, img_bgr, primary_bbox):
        """
        Interference patterns = SAME silhouette as the starting white shape,
        but filled with background-like rock color (not bright white).

        Detect by matching the acquire silhouette on a DoG (structure) map,
        then keep only low-contrast / non-white hits away from the red target.
        """
        if not self._ensure_shape_silhouette(img_bgr):
            return []
        sil = self._shape_sil
        if sil is None:
            return []

        fh, fw = img_bgr.shape[:2]
        dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)
        roi = img_bgr[dy0:dy1, dx0:dx1]
        if roi.size == 0:
            return []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Difference-of-Gaussians: reveals shape structure independent of mean color
        dog = cv2.absdiff(
            cv2.GaussianBlur(gray, (0, 0), 1.0),
            cv2.GaussianBlur(gray, (0, 0), 3.2),
        )
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        nat_w, nat_h = [int(v) for v in getattr(self, "_shape_nat_wh", (64, 64))]
        candidates = []
        for scale in (0.75, 0.90, 1.0, 1.15, 1.30, 1.45):
            tw = max(28, min(int(nat_w * scale), dog.shape[1] - 2))
            th = max(28, min(int(nat_h * scale), dog.shape[0] - 2))
            if tw >= dog.shape[1] - 2 or th >= dog.shape[0] - 2:
                continue
            tmpl = cv2.resize(sil, (tw, th), interpolation=cv2.INTER_AREA)
            try:
                res = cv2.matchTemplate(dog, tmpl, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            for _ in range(5):
                _mn, mx, _mnloc, mxloc = cv2.minMaxLoc(res)
                if not np.isfinite(mx) or mx < 0.28:
                    break
                px, py = int(mxloc[0]), int(mxloc[1])
                abs_bbox = (dx0 + px, dy0 + py, tw, th)
                cv2.circle(res, (px, py), max(tw, th) // 3, 0, -1)

                if primary_bbox is not None:
                    if self._bbox_iou(abs_bbox, primary_bbox) >= 0.25:
                        continue
                    pcx = abs_bbox[0] + tw / 2.0
                    pcy = abs_bbox[1] + th / 2.0
                    rx, ry, rw, rh = [int(v) for v in primary_bbox]
                    if rx <= pcx <= rx + rw and ry <= pcy <= ry + rh:
                        continue

                # Must NOT be bright white (that's the real target)
                patch_hsv = hsv[py : py + th, px : px + tw]
                patch_g = gray[py : py + th, px : px + tw]
                st = cv2.resize(sil, (tw, th), interpolation=cv2.INTER_NEAREST)
                inside = (st > 0).astype(np.uint8) * 255
                if int(cv2.countNonZero(inside)) < 40:
                    continue
                white = cv2.inRange(
                    patch_hsv, np.array([0, 0, 175]), np.array([180, 70, 255])
                )
                white = cv2.bitwise_and(white, inside)
                wfrac = float(cv2.countNonZero(white)) / float(
                    max(1, int(cv2.countNonZero(inside)))
                )
                if wfrac > 0.10:
                    continue

                ring = cv2.dilate(inside, np.ones((11, 11), np.uint8))
                ring = cv2.subtract(ring, inside)
                mean_in = float(cv2.mean(patch_g, mask=inside)[0])
                mean_ring = (
                    float(cv2.mean(patch_g, mask=ring)[0])
                    if int(cv2.countNonZero(ring)) > 25
                    else mean_in
                )
                contrast = abs(mean_in - mean_ring)
                # Camouflage: blends into rock background
                if contrast > 36:
                    continue
                mean_v = float(cv2.mean(patch_hsv[:, :, 2], mask=inside)[0])
                if mean_v >= 190:
                    continue

                # Prefer stronger shape match + more camouflaged
                score = float(mx) * 0.75 + (1.0 - min(1.0, contrast / 36.0)) * 0.25
                candidates.append((score, abs_bbox))

        if not candidates:
            return []

        candidates.sort(key=lambda t: -t[0])
        kept = []
        for score, bbox in candidates:
            if any(self._bbox_iou(bbox, kb) >= 0.40 for kb in kept):
                continue
            # Prefer spatially spread decoys (left/right/above/below of target)
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            if any(
                ((cx - (kb[0] + kb[2] / 2.0)) ** 2 + (cy - (kb[1] + kb[3] / 2.0)) ** 2)
                ** 0.5
                < 70.0
                for kb in kept
            ):
                continue
            x, y, bw, bh = [int(v) for v in bbox]
            pad = 4
            if bw > 2 * pad + 12 and bh > 2 * pad + 12:
                x, y = x + pad, y + pad
                bw, bh = bw - 2 * pad, bh - 2 * pad
            kept.append((x, y, bw, bh))
            if len(kept) >= 6:
                break
        return kept

    def _pred_xy(self, now: float):
        if self._track_xy is None:
            return None
        dt = max(0.0, min(0.08, now - self._last_track_t))
        vx, vy = self._track_vel
        return (self._track_xy[0] + vx * dt, self._track_xy[1] + vy * dt)

    def _step_ok(self, cx: float, cy: float, now: float) -> bool:
        if self._track_xy is None:
            return True
        dt = max(1e-3, min(0.08, now - self._last_track_t))
        dist = ((cx - self._track_xy[0]) ** 2 + (cy - self._track_xy[1]) ** 2) ** 0.5
        max_step = max(8.0, float(self.max_frame_step_px))
        if int(getattr(self, "_lock_age", 0)) < 5:
            max_step *= 1.6
        max_step = max_step * max(1.0, dt * 30.0)
        if dist > max_step:
            return False
        if int(getattr(self, "_lock_age", 0)) >= 5:
            pred = self._pred_xy(now)
            if pred is not None:
                dpred = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
                if dpred > max_step * 1.2:
                    return False
        return True

    def _local_motion_peak(self, gray, pred, rad: int):
        """Small-window frame-diff peak near prediction (fallback)."""
        if self._prev_gray is None or pred is None:
            return None
        if self._prev_gray.shape != gray.shape:
            return None
        fh, fw = gray.shape[:2]
        cx0, cy0 = float(pred[0]), float(pred[1])
        x0 = max(0, int(cx0 - rad))
        y0 = max(0, int(cy0 - rad))
        x1 = min(fw, int(cx0 + rad))
        y1 = min(fh, int(cy0 + rad))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        diff = cv2.absdiff(gray[y0:y1, x0:x1], self._prev_gray[y0:y1, x0:x1])
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, m = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(cv2.countNonZero(m)) < 12:
            return None
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for ct in cnts:
            area = float(cv2.contourArea(ct))
            if area < 20 or area > 8000:
                continue
            x, y, w, h = cv2.boundingRect(ct)
            cx = x0 + x + w / 2.0
            cy = y0 + y + h / 2.0
            dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
            if dist > rad:
                continue
            score = area * (1.0 - dist / max(1.0, rad))
            if score > best_score:
                best_score = score
                best = (cx, cy)
        return best

    def _vel_aligns_decoy(self, vx: float, vy: float) -> bool:
        """True if a velocity vector matches the shared decoy flock direction."""
        if self._decoy_vel is None:
            return False
        dvx, dvy = self._decoy_vel
        mag = (vx * vx + vy * vy) ** 0.5
        dmag = (dvx * dvx + dvy * dvy) ** 0.5
        if mag < 25.0 or dmag < 25.0:
            return False
        cos = (vx * dvx + vy * dvy) / (mag * dmag + 1e-6)
        return bool(cos > 0.78)

    def _coast(self, now: float, name: str = "coast", frame_wh=None):
        """Constant-velocity coast; clamp to dialog; brake late.
        After white fade: if track vel locked onto decoy flock, coast with
        last independent (random-target) velocity instead.
        """
        if self._track_xy is None:
            return None
        self._coast_streak = int(getattr(self, "_coast_streak", 0)) + 1
        # Prefer random-target vel over flock-aligned vel after fade
        if self._white_gone_t > 0 and self._vel_aligns_decoy(*self._track_vel):
            ivx, ivy = self._target_indep_vel
            if (ivx * ivx + ivy * ivy) ** 0.5 > 20.0:
                self._track_vel = (ivx, ivy)
        if self._coast_streak >= 35:
            vx, vy = self._track_vel
            self._track_vel = (vx * 0.92, vy * 0.92)
        if self._coast_streak >= 60:
            self._track_vel = (0.0, 0.0)
        pred = self._pred_xy(now)
        if pred is None:
            return None
        cx, cy = pred
        if frame_wh is not None:
            fw, fh = frame_wh
            dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)
            hit_edge = cx < dx0 or cx > dx1 or cy < dy0 or cy > dy1
            cx = min(max(cx, dx0 + 2), dx1 - 2)
            cy = min(max(cy, dy0 + 2), dy1 - 2)
            if hit_edge:
                vx, vy = self._track_vel
                if cx <= dx0 + 2 or cx >= dx1 - 2:
                    vx = -0.3 * vx
                if cy <= dy0 + 2 or cy >= dy1 - 2:
                    vy = -0.3 * vy
                self._track_vel = (vx, vy)
        # After fade, stale velocity from the white phase must not carry the
        # box onto a neighboring silhouette. Real motion comes from faded/flow.
        if self._white_gone_t > 0 and self._track_xy is not None:
            dx = cx - self._track_xy[0]
            dy = cy - self._track_xy[1]
            dist = (dx * dx + dy * dy) ** 0.5
            cap = max(4.0, float(self.max_frame_step_px) * 0.28)
            if dist > cap:
                cx = self._track_xy[0] + dx / dist * cap
                cy = self._track_xy[1] + dy / dist * cap
                vx, vy = self._track_vel
                self._track_vel = (vx * 0.45, vy * 0.45)
        self._recenter_bbox(cx, cy)
        self._commit_track(cx, cy, now, name, allow_jump=True)
        return self._track_xy

    def _recenter_bbox(self, cx: float, cy: float):
        if self.debug_bbox is not None:
            _, _, bw, bh = self.debug_bbox
        else:
            bw = bh = 96
        bw = max(40, min(160, int(bw)))
        bh = max(40, min(160, int(bh)))
        self.debug_bbox = (int(cx - bw / 2), int(cy - bh / 2), bw, bh)

    def _best_white_near(self, img_bgr, center_xy, rad: int, fh: int, fw: int):
        """
        Best bright-white pattern near center_xy (still-white phase).
        Returns (score, cx, cy, bbox) or None.
        """
        if center_xy is None:
            return None
        cx0, cy0 = float(center_xy[0]), float(center_xy[1])
        dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)
        x0 = max(dx0, int(cx0 - rad))
        y0 = max(dy0, int(cy0 - rad))
        x1 = min(dx1, int(cx0 + rad))
        y1 = min(dy1, int(cy0 + rad))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        # Motion bias: moving white beats a static leftover highlight.
        motion = None
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if (
                self._prev_gray is not None
                and self._prev_gray.shape[0] >= y1
                and self._prev_gray.shape[1] >= x1
            ):
                prev = self._prev_gray[y0:y1, x0:x1]
                if prev.shape == gray.shape:
                    diff = cv2.absdiff(gray, prev)
                    _, motion = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        except Exception:
            motion = None
        whites = self._white_target_candidates(
            crop,
            motion,
            frame_h=fh,
            frame_y0=y0,
            frame_w=fw,
            frame_x0=x0,
        )
        # If we've been glued to one spot, strongly prefer a moving white
        # elsewhere (center UI glow is a common trap at ~dialog mid).
        stuck_s = 0.0
        lt = float(getattr(self, "_lock_moved_t", 0.0) or 0.0)
        ref = float(getattr(self, "_last_track_t", 0.0) or 0.0)
        if lt > 0.0 and ref > lt:
            stuck_s = ref - lt
        best = None
        best_key = -1.0
        for score, lx, ly, _name, bbox in whites:
            ax, ay = x0 + lx, y0 + ly
            dist = ((ax - cx0) ** 2 + (ay - cy0) ** 2) ** 0.5
            if dist > rad:
                continue
            area = 0.0
            if bbox is not None:
                area = float(max(1, int(bbox[2]) * int(bbox[3])))
            moving = 0.0
            if motion is not None:
                try:
                    iy, ix = int(ly), int(lx)
                    if 0 <= iy < motion.shape[0] and 0 <= ix < motion.shape[1]:
                        if int(motion[iy, ix]) > 0:
                            moving = 1.0
                except Exception:
                    moving = 0.0
            # Prefer strong / large / moving white; light distance preference.
            key = (
                float(score)
                + 0.08 * min(1.0, area / 6000.0)
                + max(0.0, 0.22 - dist / max(1.0, rad))
                + (0.55 * moving if stuck_s >= 0.7 else 0.18 * moving)
            )
            # When stuck, de-weight the exact parked spot so a mover can win.
            if stuck_s >= 0.7 and dist < 18.0 and moving < 0.5:
                key -= 0.35
            if key > best_key:
                best_key = key
                best = (float(score), ax, ay, bbox)
        return best

    def _rotate_fill(self, sil, angle_deg: float):
        """Cached rotation of the native white mask. 180° is enough."""
        key = int(round(float(angle_deg))) % 180
        cached = self._fill_rot.get(key)
        if cached is not None:
            return cached
        h, w = sil.shape[:2]
        pad = int(0.2 * max(h, w)) + 2
        canvas = cv2.copyMakeBorder(sil, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        ch, cw = canvas.shape[:2]
        M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), float(key), 1.0)
        rot = cv2.warpAffine(
            canvas, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0
        )
        ys, xs = np.where(rot > 0)
        if len(xs) >= 20:
            rot = rot[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        self._fill_rot[key] = rot
        return rot

    def _match_fade_fill(self, img_bgr, now: float, reach: bool = False):
        """
        Handoff after the white fill tints. The same blob stays a smooth patch
        of the acquired shape; background stone does not. Search only around
        the current center, and only within one step (a short reach on the
        first tint frames, so a leftover bright corner can return to the body).
        Returns (score, cx, cy, bbox, angle) or None.
        """
        sil = self._fill_sil
        if sil is None or self._track_xy is None:
            return None
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (15, 15))
        mean2 = cv2.boxFilter(g * g, -1, (15, 15))
        std = np.sqrt(np.maximum(0.0, mean2 - mean * mean))
        smooth = np.clip(255.0 - std * 10.0, 0, 255).astype(np.uint8)

        cx0, cy0 = float(self._track_xy[0]), float(self._track_xy[1])
        nat = max(int(sil.shape[0]), int(sil.shape[1]))
        max_step = max(8.0, float(self.max_frame_step_px))
        max_jump = max_step * 1.15
        if reach:
            max_jump = max(max_jump, 0.55 * float(nat))
        margin = nat // 2 + 8
        rad = int(max_jump + margin)
        x0 = max(0, int(cx0 - rad))
        y0 = max(0, int(cy0 - rad))
        x1 = min(fw, int(cx0 + rad))
        y1 = min(fh, int(cy0 + rad))
        crop = smooth[y0:y1, x0:x1]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return None

        best = None
        for ang in range(0, 180, 20):
            tmpl = self._rotate_fill(sil, ang)
            th, tw = tmpl.shape[:2]
            if th + 4 >= crop.shape[0] or tw + 4 >= crop.shape[1]:
                continue
            try:
                res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            work = res.copy()
            suppress = max(8, max(tw, th) // 3)
            for _peak in range(3):
                _mn, mx, _ml, loc = cv2.minMaxLoc(work)
                if not np.isfinite(mx) or float(mx) < 0.30:
                    break
                px, py = int(loc[0]), int(loc[1])
                cx = x0 + px + tw / 2.0
                cy = y0 + py + th / 2.0
                jump = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
                ya, yb = max(0, py - suppress), min(work.shape[0], py + suppress)
                xa, xb = max(0, px - suppress), min(work.shape[1], px + suppress)
                work[ya:yb, xa:xb] = -1.0
                if jump > max_jump:
                    continue
                # Greens often sit on this same blob. Reject only a step that
                # follows the flock path, not mere overlap with a green box.
                if self._step_follows_decoy_path(cx, cy, now):
                    continue
                # Prefer the nearer peak when scores are close, so a flat
                # stone one step away does not outrank the blob we are on.
                key = float(mx) - 0.15 * (jump / max(1.0, max_jump))
                if best is None or key > best[5]:
                    bbox = (int(x0 + px), int(y0 + py), int(tw), int(th))
                    best = (float(mx), float(cx), float(cy), bbox, float(ang), key)
        if best is None:
            return None
        return best[:5]

    def _white_blob_shrunken(self, bbox) -> bool:
        """True when the bright core is much smaller than the acquired shape.

        A leftover highlight on one corner is not the body. Following it is
        how the center walks off at the tint.
        """
        if bbox is None or self._fill_sil is None:
            return False
        nw, nh = [int(v) for v in self._shape_nat_wh]
        if nw < 28 or nh < 28:
            return False
        _x, _y, bw, bh = [int(v) for v in bbox]
        return (bw * bh) < 0.60 * float(nw * nh)

    def _resolve_yolo_device(self) -> str:
        """Pick ultralytics device: prefers CUDA when fade_yolo_device=auto."""
        pref = str(getattr(self, "fade_yolo_device", "auto") or "auto").strip().lower()
        if pref in ("cpu", "cuda") or pref.startswith("cuda:"):
            if pref != "cpu":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        logger.warning(
                            "[LieAlert] fade_yolo_device=%s but CUDA unavailable — using cpu",
                            pref,
                        )
                        return "cpu"
                except Exception:
                    return "cpu"
            return pref
        # auto
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"[LieAlert] YOLO device=cuda:0 ({name})")
                return "cuda:0"
        except Exception:
            pass
        logger.info("[LieAlert] YOLO device=cpu (no CUDA torch)")
        return "cpu"

    def _warm_track_models_async(self):
        """Load peak/YOLO in a daemon so the first follow frames stay responsive."""
        if bool(getattr(self, "_track_models_warming", False)):
            return
        if self._shape_peak_model is not None and self._fade_yolo is not None:
            return
        self._track_models_warming = True

        def _job():
            try:
                self._ensure_shape_peak_model(allow_block=True)
                yolo = self._ensure_fade_yolo(allow_block=True)
                # First ultralytics predict is slow — burn it off-thread.
                if yolo is not None:
                    try:
                        import numpy as np

                        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                        device = self._fade_yolo_device_resolved or "cpu"
                        yolo.predict(
                            dummy,
                            conf=0.25,
                            verbose=False,
                            imgsz=640,
                            device=device,
                        )
                        logger.info(
                            f"[LieAlert] Fade YOLO warm-predict done (device={device})"
                        )
                    except Exception as exc:
                        logger.debug(f"[LieAlert] YOLO warm-predict: {exc}")
            except Exception as exc:
                logger.debug(f"[LieAlert] warm models: {exc}")
            finally:
                self._track_models_warming = False

        threading.Thread(target=_job, daemon=True, name="lie-warm-models").start()

    def _ensure_fade_yolo(self, allow_block: bool = True):
        if self._fade_yolo_failed:
            return None
        if self._fade_yolo is not None:
            return self._fade_yolo
        if not allow_block:
            # Critical: follow thread must NOT load YOLO (was ~28s freeze).
            self._warm_track_models_async()
            return None
        path = str(getattr(self, "fade_yolo_path", "") or "")
        if not path or not os.path.exists(path):
            self._fade_yolo_failed = True
            return None
        with self._model_load_lock:
            if self._fade_yolo is not None:
                return self._fade_yolo
            try:
                from ultralytics import YOLO

                device = self._resolve_yolo_device()
                self._fade_yolo_device_resolved = device
                self._fade_yolo = YOLO(path)
                try:
                    self._fade_yolo.to(device)
                except Exception:
                    # predict(device=...) still works if .to fails on some builds
                    pass
                logger.info(f"[LieAlert] Fade YOLO ready ({path}) device={device}")
            except Exception as exc:
                logger.warning(f"[LieAlert] Fade YOLO unavailable ({exc})")
                self._fade_yolo = None
                self._fade_yolo_failed = True
        return self._fade_yolo

    def _pick_fade_yolo(self, img_bgr):
        """
        Filled shape after the white tint is gone.
        One clear box may reacquire. Several boxes: the one nearest the
        current center, not the highest score.
        Returns (conf, cx, cy, bbox) or None.
        """
        # Never sync-load on the follow path.
        model = self._ensure_fade_yolo(allow_block=False)
        if model is None:
            return None
        conf_min = max(0.10, float(getattr(self, "fade_yolo_conf", 0.20)))
        device = self._fade_yolo_device_resolved or "cpu"
        now = time.time()
        # CPU YOLO every frame dropped the loop to ~2fps and lost the white
        # target (2026-11-06). Throttle + reuse last hit between runs.
        min_gap = float(getattr(self, "fade_yolo_min_interval_s", 0.12) or 0.12)
        if str(device).startswith("cpu"):
            min_gap = max(min_gap, 0.28)
        last_t = float(getattr(self, "_yolo_last_infer_t", 0.0) or 0.0)
        if last_t > 0.0 and (now - last_t) < min_gap:
            cached = getattr(self, "_yolo_last_hit", None)
            return cached
        self._yolo_last_infer_t = now
        try:
            pred = model.predict(
                img_bgr,
                conf=conf_min,
                verbose=False,
                imgsz=640,
                device=device,
            )
        except Exception:
            self._yolo_last_hit = None
            return None
        if not pred or pred[0].boxes is None or len(pred[0].boxes) == 0:
            self._yolo_last_hit = None
            return None
        boxes = []
        for b in pred[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            boxes.append(
                (
                    conf,
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    (int(x1), int(y1), int(max(8.0, x2 - x1)), int(max(8.0, y2 - y1))),
                )
            )
        boxes.sort(key=lambda t: -t[0])
        hit = None
        # Prefer accepting YOLO more often after fade (still nearest-to-center).
        if len(boxes) == 1 and boxes[0][0] >= conf_min:
            hit = boxes[0]
        elif self._track_xy is None:
            hit = boxes[0] if boxes[0][0] >= conf_min + 0.05 else None
        else:
            tx, ty = float(self._track_xy[0]), float(self._track_xy[1])
            search = max(4.0, float(getattr(self, "fade_yolo_search_scale", 6.0)))
            limit = float(self.max_frame_step_px) * search
            near = []
            for item in boxes:
                if item[0] < conf_min:
                    continue
                dist = ((item[1] - tx) ** 2 + (item[2] - ty) ** 2) ** 0.5
                if dist <= limit:
                    near.append((dist, item))
            if near:
                near.sort(key=lambda t: t[0])
                hit = near[0][1]
            elif boxes[0][0] >= max(0.45, conf_min + 0.18) and (
                len(boxes) == 1 or boxes[0][0] >= boxes[1][0] + 0.12
            ):
                # Strong lone lead may reacquire even farther away.
                hit = boxes[0]
        self._yolo_last_hit = hit
        return hit

    def _locate_shape(self, img_bgr, now: float):
        """
        Locked target, in order. Green boxes are never the track center.

        1. Decoy consensus (viz / flock velocity only).
        2. White sticky while the bright shape is still visible. A bright
           core much smaller than the acquired shape is a leftover corner,
           not a white lock.
        3. After fade, ``spinback`` via ``_repair_spin_jump`` when the red
           center has left a blob that was turning and another same-shape
           blob is turning (two frames, 12–45°, same direction). This
           outranks the steps below and the step limit. A one-frame
           angle-bin flip is not rotation. Quiet reading on the same spot
           does not jump.
        4. ``yolo``: one filled-shape box from the fade YOLO. A single clear
           box may reacquire. Several boxes use the one nearest the
           current center, not the highest score.
        5. ``fill``: the acquired silhouette on the local smoothness map.
           This is the tint handoff. It stays inside one step, except a
           short reach on the first tint frames from a corner highlight
           back onto the body. It does not use green boxes or the model score.
        6. Motion outlier when stuck, coasting, or flock-aligned.
        7. Faded silhouette near the prediction (same-blob spin; no flock path).
        8. LK flow, local motion, patch — drop a step that follows the flock
           or leaves a spinning blob.
        9. Coast. After fade each step is capped to about
           ``0.28 * max_frame_step_px`` so a stale white-phase velocity
           cannot walk onto the next silhouette.

        ``_find_all_shape_peaks`` only proposes same-shape centers.
        The model score is not the red center.
        """
        fh, fw = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        locked = bool(
            self._locked
            and self._track_xy is not None
            and (now - self._last_seen_t) < self.lose_target_grace_s + 0.05
        )

        dx0, dy0, dx1, dy1 = self._dialog_roi(fw, fh)

        # --- LOCKED: white -> spinback -> fill -> outlier -> faded -> flow/motion/patch -> coast ---
        if locked and self._track_xy is not None and self._prev_gray is not None:
            pred = self._pred_xy(now) or self._track_xy
            rad = max(28, min(72, int(self.lock_search_radius) + 20))

            # Keep flock velocity current so we can reject shared-path traps
            try:
                decoys = self._find_camouflage_decoys(img_bgr, self.debug_bbox)
                self._update_decoy_consensus(decoys, now)
            except Exception:
                pass
            # CRITICAL: before the shape tints, keep RED on the white blob.
            # LK alone drifts onto background texture while the star is still white.
            white_rad = max(110, rad * 2)
            whit = self._best_white_near(img_bgr, pred, white_rad, fh, fw)
            if whit is None:
                # Widen once more around last pose / dialog center early on
                whit = self._best_white_near(
                    img_bgr, self._track_xy, max(160, white_rad), fh, fw
                )
            if whit is not None:
                wscore, wcx, wcy, wbbox = whit
                if wscore >= max(0.28, self.track_min_score * 0.85) and not self._white_blob_shrunken(wbbox):
                    self._white_gone_t = 0.0
                    self._white_miss_since = 0.0
                    dx = wcx - self._track_xy[0]
                    dy = wcy - self._track_xy[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    # Detector ticks often run <30fps; scale step by dt so a
                    # real white jump is not permanently lagged by 22px/frame.
                    dt = max(1e-3, min(0.12, now - float(self._last_track_t or now)))
                    max_step = max(14.0, float(self.max_frame_step_px) * 1.35)
                    max_step = max_step * max(1.0, dt * 30.0)
                    # Strong white inside search → hard snap (mouse must keep up).
                    hard_snap = wscore >= max(0.40, self.track_min_score) and dist <= max(
                        max_step * 2.8, float(white_rad) * 0.85
                    )
                    if hard_snap or dist <= max_step * 1.35:
                        cx, cy = wcx, wcy
                    else:
                        cx = self._track_xy[0] + dx / max(1e-6, dist) * max_step
                        cy = self._track_xy[1] + dy / max(1e-6, dist) * max_step
                    self._coast_streak = 0
                    if wbbox is not None and (hard_snap or dist <= max_step * 2.5):
                        self.debug_bbox = tuple(int(v) for v in wbbox)
                    else:
                        self._recenter_bbox(cx, cy)
                    self._commit_track(
                        cx, cy, now, "white", allow_jump=bool(hard_snap)
                    )
                    if self.debug_bbox is not None and (
                        self._fill_sil is None or int(self._lock_age) % 8 == 0
                    ):
                        self._capture_shape_silhouette(img_bgr, self.debug_bbox)
                    # Learn spin while white (every 3rd frame) for post-fade seeds
                    if int(getattr(self, "_lock_age", 0)) % 3 == 0 and self.debug_bbox is not None:
                        try:
                            bx, by, bw, bh = [int(v) for v in self.debug_bbox]
                            x1b, y1b = max(0, bx), max(0, by)
                            patch = img_bgr[y1b : y1b + max(1, bh), x1b : x1b + max(1, bw)]
                            if patch.size > 100:
                                dog_w = self._dog_roi(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
                                hit_a = self._best_angle_at(
                                    dog_w,
                                    bw / 2.0,
                                    bh / 2.0,
                                    max(24, min(bw - 4, 64)),
                                    max(24, min(bh - 4, 64)),
                                )
                                if hit_a is not None and hit_a[0] >= 0.25:
                                    self._note_target_angle(hit_a[1], now)
                        except Exception:
                            pass
                    # Keep LK features on the real white target
                    if hard_snap or dist <= max_step * 1.8:
                        self._seed_flow_and_patch(gray, cx, cy, self.debug_bbox)
                    self._prev_gray = gray
                    return self._track_xy

            # White miss: hold last white pose for a short grace so a 1–2 frame
            # flicker at ~2fps does not flip straight into YOLO/fill (12:22 fail).
            if float(getattr(self, "_white_miss_since", 0.0) or 0.0) <= 0.0:
                self._white_miss_since = float(now)
            miss_age = now - float(self._white_miss_since)
            white_grace = max(0.15, float(getattr(self, "white_miss_grace_s", 0.45)))
            if self._white_gone_t <= 0.0 and miss_age < white_grace:
                self._prev_gray = gray
                return self._track_xy

            # White just disappeared (~7s): mark fade and hunt faded silhouette.
            # Decoys share one path; only the target moves randomly AND spins.
            if self._white_gone_t <= 0:
                self._white_gone_t = float(now)
            since_fade = now - self._white_gone_t
            # Count real turns at the current center (20° bins). A static
            # decoy does not keep producing them.
            if self._track_xy is not None:
                held_spin = self._spin_at_xy(
                    img_bgr, self._track_xy[0], self._track_xy[1], fine=False
                )
                if held_spin is not None and held_spin[0] >= 15.0:
                    self._spin_hits = min(8, int(self._spin_hits) + 1)
                else:
                    self._spin_hits = max(0, int(self._spin_hits) - 1)
                # Not rotating here → snap onto the blob that is. This
                # outranks flow, coast, outlier snap, and the step limit.
                repaired = self._repair_spin_jump(img_bgr, now, held_spin)
                if repaired is not None:
                    rcx, rcy, rbbox, rang, rvx, rvy = repaired
                    self._coast_streak = 0
                    self._stuck_frames = 0
                    if rbbox is not None:
                        self.debug_bbox = tuple(int(v) for v in rbbox)
                    else:
                        self._recenter_bbox(rcx, rcy)
                    self._commit_track(
                        rcx,
                        rcy,
                        now,
                        "spinback",
                        allow_jump=True,
                        vel_override=(rvx, rvy),
                    )
                    self._note_target_angle(rang, now)
                    self._seed_flow_and_patch(gray, rcx, rcy, self.debug_bbox)
                    self._prev_gray = gray
                    return self._track_xy
            yolo_hit = self._pick_fade_yolo(img_bgr)
            if yolo_hit is not None:
                _conf, ycx, ycy, ybbox = yolo_hit
                self._coast_streak = 0
                self._stuck_frames = 0
                self._fill_hold = 0
                self._yolo_hold = max(
                    1, int(getattr(self, "fade_yolo_hold_frames", 4))
                )
                if ybbox is not None:
                    self.debug_bbox = tuple(int(v) for v in ybbox)
                else:
                    self._recenter_bbox(ycx, ycy)
                step_scale = max(
                    1.0, float(getattr(self, "fade_yolo_step_scale", 2.2))
                )
                step = max(8.0, float(self.max_frame_step_px) * step_scale)
                if self._track_xy is not None:
                    dx = ycx - float(self._track_xy[0])
                    dy = ycy - float(self._track_xy[1])
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > step:
                        dx, dy = dx / max(1e-6, dist) * step, dy / max(1e-6, dist) * step
                        ycx = float(self._track_xy[0]) + dx
                        ycy = float(self._track_xy[1]) + dy
                    vel = (dx / 0.033, dy / 0.033)
                else:
                    vel = (0.0, 0.0)
                self._commit_track(
                    ycx, ycy, now, "yolo", allow_jump=True, vel_override=vel
                )
                self._seed_flow_and_patch(gray, ycx, ycy, self.debug_bbox)
                self._prev_gray = gray
                return self._track_xy
            # Brief YOLO miss: hold last track instead of falling to fill/flow.
            if int(getattr(self, "_yolo_hold", 0)) > 0 and self._track_xy is not None:
                self._yolo_hold = int(self._yolo_hold) - 1
                self._prev_gray = gray
                return self._track_xy
            # Tint handoff: same shape, now a smooth fill instead of white.
            # Runs before outlier and flow so a corner highlight cannot walk off.
            filled = self._match_fade_fill(
                img_bgr, now, reach=(since_fade < 0.45)
            )
            if filled is not None:
                _fscore, fcx, fcy, fbbox, _fang = filled
                self._coast_streak = 0
                self._stuck_frames = 0
                if fbbox is not None:
                    self.debug_bbox = tuple(int(v) for v in fbbox)
                else:
                    self._recenter_bbox(fcx, fcy)
                self._commit_track(fcx, fcy, now, "fill")
                self._fill_hold = 2  # shorter — let YOLO reclaim sooner
                self._seed_flow_and_patch(gray, fcx, fcy, self.debug_bbox)
                self._prev_gray = gray
                return self._track_xy
            # One missed fill frame must not hand the center to flow.
            if int(self._fill_hold) > 0:
                self._fill_hold = int(self._fill_hold) - 1
                self._prev_gray = gray
                return self._track_xy
            # Detect "stuck on wrong blob": little translation while flock still moves
            if self._track_xy is not None and self._last_track_t > 0:
                dt_s = max(1e-3, now - self._last_track_t)
                spd = (self._track_vel[0] ** 2 + self._track_vel[1] ** 2) ** 0.5
                flock_spd = (
                    (self._decoy_vel[0] ** 2 + self._decoy_vel[1] ** 2) ** 0.5
                    if self._decoy_vel is not None
                    else 0.0
                )
                if spd < 35.0 and flock_spd > 50.0:
                    self._stuck_frames = int(getattr(self, "_stuck_frames", 0)) + 1
                elif self._vel_aligns_decoy(*self._track_vel):
                    self._stuck_frames = int(getattr(self, "_stuck_frames", 0)) + 1
                else:
                    self._stuck_frames = max(0, int(getattr(self, "_stuck_frames", 0)) - 1)

            need_outlier = (
                int(getattr(self, "_stuck_frames", 0)) >= 4
                or int(self._coast_streak) >= 2
                or self._vel_aligns_decoy(*self._track_vel)
            )
            # Periodic flock-outlier scan keeps us on the randomly-moving target
            if need_outlier or (int(getattr(self, "_lock_age", 0)) % 5 == 0):
                outlier = self._pick_motion_outlier_target(
                    img_bgr, now, prefer_near=pred
                )
                if outlier is not None:
                    oscore, ocx, ocy, obbox, oang = outlier
                    jump = (
                        ((ocx - self._track_xy[0]) ** 2 + (ocy - self._track_xy[1]) ** 2)
                        ** 0.5
                        if self._track_xy is not None
                        else 0.0
                    )
                    # Allow a larger snap when recovering from stuck/coast
                    max_snap = float(self.max_frame_step_px) * (
                        3.5 if need_outlier else 2.2
                    )
                    if oscore >= 0.22 and jump <= max_snap:
                        self._coast_streak = 0
                        self._stuck_frames = 0
                        if obbox is not None:
                            self.debug_bbox = tuple(int(v) for v in obbox)
                        else:
                            self._recenter_bbox(ocx, ocy)
                        self._commit_track(ocx, ocy, now, "outlier")
                        self._note_target_angle(oang, now)
                        self._seed_flow_and_patch(gray, ocx, ocy, self.debug_bbox)
                        self._prev_gray = gray
                        return self._track_xy

            fade_rad = max(rad + 24, int(60 + min(70.0, since_fade * 35.0)))
            if (
                abs(float(self._target_spin_dps)) < 40.0
                or int(self._coast_streak) >= 2
                or int(getattr(self, "_stuck_frames", 0)) >= 3
            ):
                fade_rad = max(fade_rad, rad + 70)
            faded = self._find_faded_target_near(img_bgr, pred, fade_rad, now)
            if faded is not None:
                fscore, fcx, fcy, fbbox, fang = faded
                # Angle continues predicted spin → same rotating target
                spin_cont = False
                if self._target_angle is not None and fang is not None:
                    dt_a = max(
                        1e-3,
                        min(
                            0.08,
                            now - self._target_angle_t if self._target_angle_t else 0.033,
                        ),
                    )
                    expected = (
                        float(self._target_angle) + float(self._target_spin_dps) * dt_a
                    ) % 180.0
                    err = abs(self._wrap_angle_delta(fang, expected))
                    err = min(err, 180.0 - err)
                    spin_cont = err <= 20.0 and abs(float(self._target_spin_dps)) > 25.0
                step_ok = self._step_ok(fcx, fcy, now)
                # Slightly larger snap only when orientation continues the spin
                if not step_ok and spin_cont and self._track_xy is not None:
                    jump = (
                        (fcx - self._track_xy[0]) ** 2
                        + (fcy - self._track_xy[1]) ** 2
                    ) ** 0.5
                    step_ok = jump <= float(self.max_frame_step_px) * 2.8
                if (
                    fscore >= 0.24
                    and step_ok
                    and not self._rejects_decoy_trap(fcx, fcy, now)
                ):
                    self._coast_streak = 0
                    self._stuck_frames = 0
                    if fbbox is not None:
                        self.debug_bbox = tuple(int(v) for v in fbbox)
                    else:
                        self._recenter_bbox(fcx, fcy)
                    self._commit_track(fcx, fcy, now, "faded")
                    self._note_target_angle(fang, now)
                    self._seed_flow_and_patch(gray, fcx, fcy, self.debug_bbox)
                    self._prev_gray = gray
                    return self._track_xy

            if self._flow_pts is not None:
                # Snapshot — rejected LK must not poison next-frame features
                flow_pts_bak = self._flow_pts.copy()
                bbox_bak = self.debug_bbox
                flow_pt = self._track_lk(gray)
                if flow_pt is not None:
                    cx, cy = flow_pt
                    # After fade: refuse flow that follows the decoy flock path
                    flow_ok = (
                        dx0 <= cx <= dx1
                        and dy0 <= cy <= dy1
                        and self._step_ok(cx, cy, now)
                        and not self._rejects_decoy_trap(cx, cy, now)
                    )
                    if flow_ok and self._track_xy is not None:
                        fvx = (cx - self._track_xy[0]) / max(
                            1e-3, now - self._last_track_t
                        )
                        fvy = (cy - self._track_xy[1]) / max(
                            1e-3, now - self._last_track_t
                        )
                        if self._vel_aligns_decoy(fvx, fvy):
                            flow_ok = False
                    if flow_ok and self._leaves_spinning_blob(img_bgr, cx, cy):
                        flow_ok = False
                    if flow_ok and int(getattr(self, "_stuck_frames", 0)) < 6:
                        self._coast_streak = 0
                        self._recenter_bbox(cx, cy)
                        self._commit_track(cx, cy, now, "flow")
                        if self._patch_gray is not None:
                            ph, pw = self._patch_hw
                            if ph > 0 and pw > 0:
                                x2 = max(0, min(fw - pw, int(cx - pw / 2)))
                                y2 = max(0, min(fh - ph, int(cy - ph / 2)))
                                patch = gray[y2 : y2 + ph, x2 : x2 + pw]
                                if patch.shape[:2] == (ph, pw):
                                    self._patch_gray = cv2.addWeighted(
                                        self._patch_gray, 0.75, patch, 0.25, 0
                                    )
                        self._prev_gray = gray
                        return self._track_xy
                self._flow_pts = flow_pts_bak
                self.debug_bbox = bbox_bak

            mot = self._local_motion_peak(gray, pred, rad)
            if mot is not None:
                cx, cy = mot
                if (
                    self._step_ok(cx, cy, now)
                    and not self._rejects_decoy_trap(cx, cy, now)
                    and int(getattr(self, "_stuck_frames", 0)) < 6
                    and not self._leaves_spinning_blob(img_bgr, cx, cy)
                ):
                    self._coast_streak = 0
                    self._recenter_bbox(cx, cy)
                    self._commit_track(cx, cy, now, "motion")
                    # Re-seed LK around accepted motion so flow can recover
                    self._seed_flow_and_patch(gray, cx, cy, self.debug_bbox)
                    self._prev_gray = gray
                    return self._track_xy

            x0 = max(dx0, int(pred[0] - rad))
            y0 = max(dy0, int(pred[1] - rad))
            x1 = min(dx1, int(pred[0] + rad))
            y1 = min(dy1, int(pred[1] + rad))
            hit = self._match_patch(gray, (x0, y0, x1, y1))
            if hit is not None:
                corr, cx, cy = hit
                if (
                    corr >= 0.45
                    and self._step_ok(cx, cy, now)
                    and not self._rejects_decoy_trap(cx, cy, now)
                    and int(getattr(self, "_stuck_frames", 0)) < 6
                    and not self._leaves_spinning_blob(img_bgr, cx, cy)
                ):
                    self._coast_streak = 0
                    self._recenter_bbox(cx, cy)
                    self._commit_track(cx, cy, now, "patch")
                    self._seed_flow_and_patch(gray, cx, cy, self.debug_bbox)
                    self._prev_gray = gray
                    return self._track_xy

            self._prev_gray = gray
            coasted = self._coast(now, "coast", frame_wh=(fw, fh))
            if coasted is not None:
                return coasted

        search_boxes = []
        if locked and self._track_xy is not None:
            pred = self._pred_xy(now) or self._track_xy
            rad = max(24, min(int(self.lock_search_radius), int(self.max_frame_step_px) + 20))
            x0 = max(dx0, int(pred[0] - rad))
            y0 = max(dy0, int(pred[1] - rad))
            x1 = min(dx1, int(pred[0] + rad))
            y1 = min(dy1, int(pred[1] + rad))
            if x1 - x0 > 20 and y1 - y0 > 20:
                search_boxes.append((x0, y0, x1, y1))
        else:
            search_boxes.append((dx0, dy0, dx1, dy1))

        motion_full = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_full = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
            motion_full = cv2.morphologyEx(
                motion_full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
        if not locked or self._prev_gray is None:
            self._prev_gray = gray
        elif locked and self._flow_pts is None:
            self._prev_gray = gray

        cands = []
        for x0, y0, x1, y1 in search_boxes:
            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            motion = None if motion_full is None else motion_full[y0:y1, x0:x1]

            if not locked:
                for item in self._white_target_candidates(
                    crop,
                    motion,
                    frame_h=fh,
                    frame_y0=y0,
                    frame_w=fw,
                    frame_x0=x0,
                ):
                    score, lx, ly, name, bbox = item
                    ax, ay = x0 + lx, y0 + ly
                    cands.append((score + 0.25, ax, ay, name, bbox))
            else:
                hit = self._match_patch(gray, (x0, y0, x1, y1))
                if hit is not None:
                    corr, cx, cy = hit
                    if self._step_ok(cx, cy, now):
                        cands.append(
                            (0.45 + 0.50 * corr, cx, cy, "patch", self.debug_bbox)
                        )

        if locked and not cands:
            return self._coast(now, "coast", frame_wh=(fw, fh))

        if not cands:
            return None

        if self._stale_ban_xy is not None and now < float(self._stale_ban_until):
            filtered = [
                c for c in cands if not self._is_stale_banned(c[1], c[2], now)
            ]
            if filtered:
                cands = filtered
            elif not locked:
                return None

        if locked and self._track_xy is not None:
            pred = self._pred_xy(now) or self._track_xy
            jump = max(8.0, float(self.max_frame_step_px))
            near = []
            for score, cx, cy, name, bbox in cands:
                if not self._step_ok(cx, cy, now):
                    continue
                dist = ((cx - pred[0]) ** 2 + (cy - pred[1]) ** 2) ** 0.5
                if dist <= jump:
                    near.append((score + max(0.0, 0.25 - dist / 80.0), cx, cy, name, bbox))
            if not near:
                return self._coast(now, "coast", frame_wh=(fw, fh))
            cands = near

        cands.sort(key=lambda t: -t[0])
        score, cx, cy, name, bbox = cands[0]
        min_need = self.track_min_score if locked else self.acquire_min_score
        if score < min_need:
            if locked:
                return self._coast(now, "coast", frame_wh=(fw, fh))
            return None

        if bbox is not None and not locked:
            self.debug_bbox = tuple(int(v) for v in bbox)
        else:
            self._recenter_bbox(cx, cy)

        was_locked = locked
        self._coast_streak = 0
        self._commit_track(cx, cy, now, name)
        if not was_locked:
            self._lock_age = 0
            self._seed_flow_and_patch(gray, cx, cy, self.debug_bbox)
            # Remember white silhouette so camouflage same-shape decoys can be marked
            if name == "white" or self._shape_sil is None:
                self._capture_shape_silhouette(img_bgr, self.debug_bbox)
        elif self._flow_pts is None:
            self._seed_flow_and_patch(gray, cx, cy, self.debug_bbox)
        else:
            self._lock_age = int(getattr(self, "_lock_age", 0)) + 1
        return self._track_xy

    def _commit_track(
        self,
        cx: float,
        cy: float,
        now: float,
        name: str,
        allow_jump: bool = False,
        vel_override=None,
    ):
        self._last_shape_name = name
        if name in ("flow", "patch", "motion", "white", "faded", "outlier", "spinback", "fill", "yolo"):
            self._lock_age = int(getattr(self, "_lock_age", 0)) + 1
            self._coast_streak = 0
        elif name == "coast":
            self._lock_age = int(getattr(self, "_lock_age", 0)) + 1
        if vel_override is not None:
            self._track_vel = (float(vel_override[0]), float(vel_override[1]))
            self._track_xy = (float(cx), float(cy))
            self._last_track_t = now
            self.debug_pt = (int(cx), int(cy))
            return
        if self._track_xy is not None and self._last_track_t > 0:
            dt = max(1e-3, now - self._last_track_t)
            if not allow_jump and not self._step_ok(cx, cy, now):
                pred = self._pred_xy(now)
                if pred is not None:
                    cx, cy = pred
            nvx = (cx - self._track_xy[0]) / dt
            nvy = (cy - self._track_xy[1]) / dt
            spd = (nvx * nvx + nvy * nvy) ** 0.5
            max_spd = max(60.0, float(self.max_speed_px_s))
            if spd > max_spd:
                s = max_spd / spd
                nvx *= s
                nvy *= s
            a = min(max(float(self.vel_smooth), 0.5), 0.95)
            ovx, ovy = self._track_vel
            self._track_vel = ((1.0 - a) * nvx + a * ovx, (1.0 - a) * nvy + a * ovy)
            # Remember last non-flock velocity (random target motion)
            if name != "coast" and not self._vel_aligns_decoy(nvx, nvy):
                if spd > 20.0:
                    self._target_indep_vel = (float(nvx), float(nvy))
        else:
            self._track_vel = (0.0, 0.0)
        self._track_xy = (float(cx), float(cy))
        self._last_track_t = now
        self.debug_pt = (int(cx), int(cy))

    @staticmethod
    def _safe_motion(motion, lx, ly) -> bool:
        iy, ix = int(ly), int(lx)
        if iy < 0 or ix < 0 or iy >= motion.shape[0] or ix >= motion.shape[1]:
            return False
        return bool(motion[iy, ix] > 0)

    def _match_shape_templates(self, crop_bgr):
        """Best multi-shape template → (corr, cx, cy, name) or None."""
        if not self._shape_list:
            return None
        ch, cw = crop_bgr.shape[:2]
        best = None
        for path, tmpl in self._shape_list:
            th0, tw0 = tmpl.shape[:2]
            name = os.path.splitext(os.path.basename(path))[0]
            for scale in self.shape_scales:
                tw = max(10, int(tw0 * scale))
                th = max(10, int(th0 * scale))
                if tw >= cw or th >= ch:
                    continue
                try:
                    scaled = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(crop_bgr, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                except Exception:
                    continue
                max_val = float(max_val)
                if not np.isfinite(max_val):
                    continue
                cx = max_loc[0] + tw / 2.0
                cy = max_loc[1] + th / 2.0
                if best is None or max_val > best[0]:
                    best = (max_val, cx, cy, name)
        return best

    def _white_target_candidates(
        self, crop_bgr, motion_mask=None, frame_h=0, frame_y0=0, frame_w=0, frame_x0=0
    ):
        """
        Detect bright white geometric overlays (star/diamond/heart/…).

        Requires local contrast vs surrounding ring so rocky highlights are ignored.
        """
        out = []
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape[:2]

        # Multi-pass: solid white first, then faded (higher S / lower V)
        passes = [
            (max(200, self.white_v_min), min(40, self.white_s_max), 1.00),
            (self.white_v_min, self.white_s_max, 0.92),
            (max(155, self.white_v_min - 30), min(85, self.white_s_max + 25), 0.80),
        ]
        seen = set()
        for v_min, s_max, pass_w in passes:
            mask = cv2.inRange(
                hsv, np.array([0, 0, int(v_min)]), np.array([180, int(s_max), 255])
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for ct in cnts:
                area = float(cv2.contourArea(ct))
                if area < self.shape_blob_min_area or area > self.shape_blob_max_area:
                    continue
                x, y, w, h = cv2.boundingRect(ct)
                # Real white targets are often 80–180px; countdown digits are smaller
                if w < 18 or h < 18 or w > 220 or h > 220:
                    continue
                aspect = w / float(h)
                # Stars/diamonds ~square; tall pale blobs are often countdown digits
                if aspect < 0.65 or aspect > 1.55:
                    continue
                fill = area / float(max(1, w * h))
                if fill < 0.22:
                    continue

                M = cv2.moments(ct)
                if M["m00"] <= 1e-3:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                else:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                # Absolute frame filters (countdown / HUD / watermarks)
                abs_x = frame_x0 + cx
                abs_y = frame_y0 + cy
                if frame_h > 0:
                    if abs_y < frame_h * self.exclude_top_ratio:
                        continue
                    if abs_y > frame_h * (1.0 - self.exclude_bottom_ratio):
                        continue
                if frame_w > 0 and frame_h > 0:
                    side = float(getattr(self, "exclude_side_ratio", 0.10))
                    # Reject left/right chrome (log failure locked to x=15)
                    if abs_x < frame_w * side or abs_x > frame_w * (1.0 - side):
                        continue
                    if abs_x > frame_w * 0.78 and abs_y > frame_h * 0.72:
                        continue
                    if abs_x < frame_w * 0.12 and abs_y < frame_h * 0.18:
                        continue
                    # Digits sit in upper dialog; real target is mid-band
                    if abs_y < frame_h * 0.32:
                        continue

                key = (int(cx) // 8, int(cy) // 8)
                if key in seen:
                    continue

                # Local contrast: inside mean vs dilated ring mean
                inside = np.zeros_like(gray)
                cv2.drawContours(inside, [ct], -1, 255, -1)
                ring = cv2.dilate(inside, np.ones((15, 15), np.uint8))
                ring = cv2.subtract(ring, inside)
                if int(cv2.countNonZero(inside)) < 20 or int(cv2.countNonZero(ring)) < 20:
                    continue
                mean_in = float(cv2.mean(gray, mask=inside)[0])
                mean_ring = float(cv2.mean(gray, mask=ring)[0])
                contrast = mean_in - mean_ring
                if contrast < self.local_contrast_min:
                    continue

                motion_hit = 0.0
                if motion_mask is not None and self._safe_motion(motion_mask, cx, cy):
                    motion_hit = 1.0

                # Prefer near-square mid-screen solid whites (star/diamond) over
                # countdown digits (taller, higher in dialog) / START letters.
                aspect_pref = 1.0 - min(1.0, abs(aspect - 1.0) / 0.55)
                size_pref = min(1.0, area / 8000.0)
                center_pref = 0.0
                mid_band_pref = 0.0
                if frame_w > 0 and frame_h > 0:
                    dx = (abs_x - frame_w * 0.5) / max(1.0, frame_w * 0.5)
                    # Bias slightly below geometric center (digit above, star below)
                    dy = (abs_y - frame_h * 0.52) / max(1.0, frame_h * 0.5)
                    center_pref = max(0.0, 1.0 - (dx * dx + dy * dy) ** 0.5)
                    # Soft mid-band: y in ~38%–68% of frame
                    yr = abs_y / float(frame_h)
                    mid_band_pref = max(0.0, 1.0 - abs(yr - 0.52) / 0.22)

                score = (
                    pass_w
                    * (
                        0.24 * min(1.0, contrast / 60.0)
                        + 0.18 * min(1.0, mean_in / 255.0)
                        + 0.12 * min(1.0, fill)
                        + self.motion_weight * motion_hit
                        + 0.16 * size_pref
                        + 0.16 * aspect_pref
                        + 0.12 * center_pref
                        + 0.12 * mid_band_pref
                    )
                )
                seen.add(key)
                bbox = (int(frame_x0 + x), int(frame_y0 + y), int(w), int(h))
                out.append((score, cx, cy, "white", bbox))

        out.sort(key=lambda t: -t[0])
        return out[:6]

    @staticmethod
    def _is_process_elevated() -> bool:
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False

    @staticmethod
    def _hwnd_is_elevated(hwnd) -> bool:
        """True if the process owning hwnd is running elevated (admin)."""
        if not hwnd:
            return False
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            advapi32 = ctypes.windll.advapi32
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
            if not pid.value:
                return False
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid.value)
            )
            if not handle:
                return False
            try:
                TOKEN_QUERY = 0x0008
                token = wintypes.HANDLE()
                if not advapi32.OpenProcessToken(handle, TOKEN_QUERY, ctypes.byref(token)):
                    return False
                try:

                    class TOKEN_ELEVATION(ctypes.Structure):
                        _fields_ = [("TokenIsElevated", wintypes.DWORD)]

                    elev = TOKEN_ELEVATION()
                    retlen = wintypes.DWORD()
                    # TokenElevation = 20
                    ok = advapi32.GetTokenInformation(
                        token,
                        20,
                        ctypes.byref(elev),
                        ctypes.sizeof(elev),
                        ctypes.byref(retlen),
                    )
                    return bool(ok and elev.TokenIsElevated)
                finally:
                    kernel32.CloseHandle(token)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    def _warn_mouse_elevation_block(self, capture=None):
        """Log once if game is admin while bot is not — UIPI blocks mouse inject."""
        if getattr(self, "_mouse_elev_warned", False):
            return
        if self._is_process_elevated():
            return
        hwnd = None
        try:
            import win32gui

            title = ""
            if capture is not None:
                title = str(getattr(capture, "window_title", "") or "")
            if title:
                hwnd = win32gui.FindWindow(None, title)
        except Exception:
            hwnd = None
        game_elev = self._hwnd_is_elevated(hwnd) if hwnd else True
        if not game_elev:
            return
        self._mouse_elev_warned = True
        logger.error(
            "[LieAlert] 鼠标跟形被系统拦住：游戏以管理员运行，但 bot 不是管理员。"
            "Windows UIPI 会屏蔽 SetCursorPos/SendInput。"
            "请右键 start.bat / Python →「以管理员身份运行」后重试。"
        )

    def _cursor_pos(self):
        """Current cursor in screen pixels, or None if it cannot be read."""
        try:
            import ctypes

            class _POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            pt = _POINT()
            if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return None
            return float(pt.x), float(pt.y)
        except Exception:
            return None

    def _yield_to_user(self, reason: str):
        self._mouse_yielded = True
        self.following = False
        self.debug_pt = None
        logger.info(f"[LieAlert] {reason}")

    def _mouse_button_down(self) -> bool:
        """True if L/R mouse button is held — clear player intervene signal."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            # VK_LBUTTON=0x01, VK_RBUTTON=0x02; high bit set while pressed
            return bool(user32.GetAsyncKeyState(0x01) & 0x8000) or bool(
                user32.GetAsyncKeyState(0x02) & 0x8000
            )
        except Exception:
            return False

    def _watch_user_mouse(self):
        """
        Stop mouse-follow when the player takes over AFTER our first snap.

        Pre-follow (intro delay): only refresh the baseline. A bump, dialog
        focus jump, or accidental nudge must NOT abort AFK follow — that was
        the 2026-11-03 21:19 failure (70px → permanent yield, zero snaps).

        After we command the cursor: grace only ignores tiny settle jitter
        near the commanded point. A large yank or mouse-button press yields
        immediately — previously every-frame SetCursorPos kept us inside
        grace forever so player intervene never fired (2026-11-06 12:22).
        """
        if self._mouse_yielded:
            return
        # After first snap: any click = player intervene → stop fighting.
        if self._mouse_placed_xy is not None and self._mouse_button_down():
            self._yield_to_user(
                "user mouse button — stop auto-follow (player intervene)"
            )
            return
        pos = self._cursor_pos()
        if pos is None:
            return
        # Before first snap: keep baseline current; never permanent-yield.
        if self._mouse_placed_xy is None:
            prev = self._mouse_watch_xy
            if prev is not None:
                thr = max(8.0, float(self.mouse_takeover_px))
                dist = ((pos[0] - prev[0]) ** 2 + (pos[1] - prev[1]) ** 2) ** 0.5
                if dist >= thr:
                    now = time.time()
                    if now - float(getattr(self, "_pre_follow_move_log_t", 0.0) or 0.0) > 1.0:
                        self._pre_follow_move_log_t = now
                        logger.info(
                            f"[LieAlert] cursor moved {dist:.0f}px during delay "
                            f"— still auto-follow (baseline refresh)"
                        )
            self._mouse_watch_xy = pos
            return
        thr = max(8.0, float(self.mouse_takeover_px))
        anchor = self._mouse_placed_xy
        dist_cmd = ((pos[0] - anchor[0]) ** 2 + (pos[1] - anchor[1]) ** 2) ** 0.5
        if dist_cmd < thr:
            # Cursor is where we put it — refresh watch for next comparison.
            self._mouse_watch_xy = pos
            return
        # Settle grace ONLY when still near commanded (OS lag / smooth).
        # Far from commanded → fall through even if we just SetCursorPos.
        last_move_t = float(getattr(self, "_mouse_last_move_t", 0.0) or 0.0)
        grace = float(getattr(self, "mouse_move_grace_s", 0.08) or 0.08)
        landed = bool(getattr(self, "_mouse_last_move_ok", False))
        if (
            landed
            and last_move_t > 0.0
            and (time.time() - last_move_t) < grace
            and dist_cmd < thr * 1.5
        ):
            return
        watch = self._mouse_watch_xy
        if watch is not None:
            dist_watch = ((pos[0] - watch[0]) ** 2 + (pos[1] - watch[1]) ** 2) ** 0.5
            if dist_watch < thr:
                # Still at the old spot: our move did not stick. Do not yield.
                now = time.time()
                if now - float(getattr(self, "_mouse_stuck_log_t", 0.0) or 0.0) > 1.0:
                    self._mouse_stuck_log_t = now
                    logger.warning(
                        f"[LieAlert] cursor ignore our move "
                        f"(still@({pos[0]:.0f},{pos[1]:.0f}) "
                        f"cmd=({anchor[0]:.0f},{anchor[1]:.0f}) "
                        f"dist={dist_cmd:.0f}px) — keep trying"
                    )
                    self._warn_mouse_elevation_block()
                return
        # Cursor left both commanded and pre-move positions → real user.
        self._yield_to_user(
            f"user moved the mouse — stop auto-follow "
            f"(dist={dist_cmd:.0f}px from commanded)"
        )

    def _frame_to_screen(self, pt, capture):
        """Map frame (x,y) → absolute screen coords using capturor region."""
        fx, fy = float(pt[0]), float(pt[1])
        left = top = 0
        title_bar = int((self.cfg.get("game_window") or {}).get("title_bar_height", 0) or 0)
        if capture is not None:
            region = getattr(capture, "region", None) or {}
            left = int(region.get("left", 0) or 0)
            top = int(region.get("top", 0) or 0)
        return int(left + fx), int(top + title_bar + fy)

    def _set_cursor_screen(self, ix: int, iy: int) -> bool:
        """Move OS cursor; SetCursorPos + absolute SendInput (virtual desktop)."""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            ix, iy = int(ix), int(iy)
            # Always try SetCursorPos first — it updates the system cursor
            # even when the game later ignores injected relative moves.
            user32.SetCursorPos(ix, iy)

            # Absolute SendInput uses 0..65535 over the virtual desktop.
            vx = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
            vy = int(user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
            vw = max(1, int(user32.GetSystemMetrics(78)))  # SM_CXVIRTUALSCREEN
            vh = max(1, int(user32.GetSystemMetrics(79)))  # SM_CYVIRTUALSCREEN
            abs_x = int((ix - vx) * 65535 / max(1, vw - 1))
            abs_y = int((iy - vy) * 65535 / max(1, vh - 1))
            abs_x = max(0, min(65535, abs_x))
            abs_y = max(0, min(65535, abs_y))

            ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ULONG_PTR),
                ]

            class INPUT_I(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT)]

            class INPUT(ctypes.Structure):
                _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_I)]

            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_ABSOLUTE = 0x8000
            MOUSEEVENTF_VIRTUALDESK = 0x4000
            inp = INPUT()
            inp.type = 0  # INPUT_MOUSE
            inp.ii.mi = MOUSEINPUT(
                abs_x,
                abs_y,
                0,
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                0,
                ULONG_PTR(0),
            )
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            # Relative nudge helps some DirectInput paths when absolute is ignored.
            before = self._cursor_pos()
            if before is not None:
                rdx = int(ix - before[0])
                rdy = int(iy - before[1])
                if abs(rdx) > 0 or abs(rdy) > 0:
                    rel = INPUT()
                    rel.type = 0
                    rel.ii.mi = MOUSEINPUT(rdx, rdy, 0, MOUSEEVENTF_MOVE, 0, ULONG_PTR(0))
                    user32.SendInput(1, ctypes.byref(rel), ctypes.sizeof(INPUT))
            return True
        except Exception:
            try:
                import ctypes

                ctypes.windll.user32.SetCursorPos(int(ix), int(iy))
                return True
            except Exception:
                try:
                    import pyautogui

                    pyautogui.moveTo(int(ix), int(iy), duration=0)
                    return True
                except Exception as e:
                    logger.debug(f"[LieAlert] mouse move failed: {e}")
                    return False

    def _move_mouse_to_frame_point(self, pt, capture):
        if self._mouse_yielded:
            return
        # Last chance: if player already yanked/clicked, do not fight this frame.
        if self._mouse_placed_xy is not None and self._mouse_button_down():
            self._yield_to_user(
                "user mouse button — stop auto-follow (player intervene)"
            )
            return
        self._warn_mouse_elevation_block(capture)
        sx, sy = self._frame_to_screen(pt, capture)
        if self._mouse_xy is None:
            mx, my = float(sx), float(sy)
        else:
            # Large track jumps: snap immediately so OS cursor does not lag
            # the white shape (user then "takes over" and we falsely yield).
            gap = (
                (sx - self._mouse_xy[0]) ** 2 + (sy - self._mouse_xy[1]) ** 2
            ) ** 0.5
            a = min(max(self.mouse_smooth, 0.15), 1.0)
            if gap >= max(28.0, float(self.mouse_takeover_px)):
                a = 1.0
            mx = self._mouse_xy[0] * (1.0 - a) + sx * a
            my = self._mouse_xy[1] * (1.0 - a) + sy * a
        self._mouse_xy = (mx, my)
        ix, iy = int(round(mx)), int(round(my))
        # Remember where the cursor was before we try — used to detect
        # "move did not stick" vs real user takeover.
        before = self._cursor_pos()
        if before is not None and self._mouse_watch_xy is None:
            self._mouse_watch_xy = before
        self._mouse_placed_xy = (float(ix), float(iy))
        self._mouse_last_move_t = time.time()
        if not self._set_cursor_screen(ix, iy):
            self._mouse_last_move_ok = False
            return
        after = self._cursor_pos()
        thr = max(8.0, float(self.mouse_takeover_px) * 0.5)
        ok = False
        if after is not None:
            dist = ((after[0] - ix) ** 2 + (after[1] - iy) ** 2) ** 0.5
            if dist > thr:
                # Snap hard to the target once — smooth lag may have left us
                # far from where GetCursorPos can verify.
                self._mouse_xy = (float(sx), float(sy))
                self._mouse_placed_xy = (float(sx), float(sy))
                self._set_cursor_screen(int(sx), int(sy))
                self._mouse_last_move_t = time.time()
                after2 = self._cursor_pos()
                if after2 is not None:
                    dist2 = (
                        (after2[0] - sx) ** 2 + (after2[1] - sy) ** 2
                    ) ** 0.5
                    ok = dist2 <= thr
                if not ok:
                    now = time.time()
                    if now - float(getattr(self, "_mouse_fail_log_t", 0.0) or 0.0) > 1.0:
                        self._mouse_fail_log_t = now
                        logger.warning(
                            f"[LieAlert] mouse move did not land "
                            f"(before={before} cmd=({sx:.0f},{sy:.0f}) "
                            f"after={after2 or after})"
                        )
                        self._warn_mouse_elevation_block(capture)
            else:
                ok = True
        self._mouse_last_move_ok = ok
        if ok and before is not None:
            # Successful move — watch tracks the new resting spot.
            self._mouse_watch_xy = after if after is not None else (float(ix), float(iy))

    def _play_loud_alert(self):
        """Non-blocking loud alert (WAV siren + system beeps)."""
        with self._sound_lock:
            if self._sound_thread is not None and self._sound_thread.is_alive():
                return
            self._sound_thread = threading.Thread(
                target=self._sound_worker, daemon=True, name="LieAlertSound"
            )
            self._sound_thread.start()

    @classmethod
    def _alert_wav_path(cls) -> str:
        """Cached loud stereo siren WAV under temp (regenerated if missing)."""
        import math
        import struct
        import tempfile
        import wave

        path = os.path.join(tempfile.gettempdir(), "lie_alert_siren_v2.wav")
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            return path
        rate = 22050
        secs = 1.6
        n = int(rate * secs)
        buf = bytearray()
        for i in range(n):
            t = i / rate
            # Two-tone siren, near-full amplitude so mixer can hear over the game.
            f = 1760.0 if (int(t * 5) % 2 == 0) else 880.0
            env = 0.55 + 0.45 * abs(math.sin(2 * math.pi * 2.5 * t))
            v = int(32000 * env * math.sin(2 * math.pi * f * t))
            v = max(-32767, min(32767, v))
            buf.extend(struct.pack("<h", v))
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(buf))
        return path

    @classmethod
    def _sound_worker(cls):
        ok = False
        try:
            print("\a\a\a *** 测谎 ALERT ***", flush=True)
        except Exception:
            pass
        try:
            import winsound
        except ImportError:
            logger.warning("[LieAlert] winsound missing — no alert sound")
            return
        # 1) Loud WAV through the normal audio device (survives muted system-schemes).
        try:
            wav = cls._alert_wav_path()
            for _ in range(2):
                winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                ok = True
        except Exception as e:
            logger.warning(f"[LieAlert] alert WAV failed: {e}")
        # 2) System beeps as backup (some setups only hear one path).
        try:
            for _ in range(3):
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception:
                    pass
                try:
                    winsound.Beep(2000, 250)
                    winsound.Beep(1100, 200)
                    winsound.Beep(2000, 300)
                    ok = True
                except Exception:
                    try:
                        winsound.PlaySound(
                            "SystemHand",
                            winsound.SND_ALIAS | winsound.SND_NODEFAULT,
                        )
                        ok = True
                    except Exception:
                        pass
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"[LieAlert] alert beep failed: {e}")
        if ok:
            logger.info("[LieAlert] alert sound played")
        else:
            logger.warning(
                "[LieAlert] alert sound FAILED — check Windows volume mixer "
                "for python.exe / system sounds"
            )