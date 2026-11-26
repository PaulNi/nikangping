# -*- coding: utf-8 -*-
"""
Causal heatmap tracker driven by gostart's lie-detector-observation.onnx.

Pipeline (aligned with tracker.pyd LieDetectorMLTrackingSession):
  1) take 752×501 BGR puzzle crop (from dialog ROI)
  2) green wipe → gray → INTER_AREA work frame (144×96)
  3) `_build_features` replica → uint8 (10,96,144) → float32 / 255
  4) ONNX → already-normalized probability [1,96,144]
  5) belief filter + calibrated confidence → frame-space point
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from src.engine.gostart_lie_features import (
    WORK_H,
    WORK_W,
    build_features as gostart_build_features,
    prepare_gray,
    registered_previous,
)

CROP_W = 752
CROP_H = 501
# gostart tracker.PUZZLE_IMAGE_CROP_OFFSETS relative to title top-left
PUZZLE_FROM_TITLE = (3, 27, 755, 528)  # left, top, right, bottom deltas
DEFAULT_MODEL = "assets/lie_detector/lie-detector-observation.onnx"
DEFAULT_META = "assets/lie_detector/lie-detector-observation.json"


@dataclass
class HeatmapEstimate:
    x: float  # frame coords
    y: float
    confidence: float
    secondary_ratio: float
    source: str  # heatmap | inject | predict
    work_xy: Tuple[float, float]


def _remove_green_cursor(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (40, 80, 80), (95, 255, 255))
    m2 = cv2.inRange(bgr, (0, 180, 0), (120, 255, 120))
    mask = cv2.bitwise_or(m1, m2)
    if int(cv2.countNonZero(mask)) < 8:
        return bgr
    out = bgr.copy()
    out[mask > 0] = cv2.blur(bgr, (7, 7))[mask > 0]
    return out


def puzzle_rect_from_title(
    title_tl: Tuple[int, int],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Exact gostart 752×501 puzzle rect from dialog title top-left."""
    tx, ty = int(title_tl[0]), int(title_tl[1])
    dl, dt, dr, db = PUZZLE_FROM_TITLE
    x0, y0, x1, y1 = tx + dl, ty + dt, tx + dr, ty + db
    if x1 - x0 != CROP_W or y1 - y0 != CROP_H:
        return None
    if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
        return None
    return (x0, y0, x1, y1)


def extract_puzzle_crop(
    frame_bgr: np.ndarray,
    dialog_xyxy: Optional[Tuple[int, int, int, int]] = None,
    title_tl: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Return (crop752x501, ox, oy, sx, sy) mapping crop-local → frame:
      frame_x = ox + local_x * sx
      frame_y = oy + local_y * sy

    Prefer gostart title→puzzle offsets when `title_tl` is known.
    """
    fh, fw = frame_bgr.shape[:2]
    if title_tl is not None:
        rect = puzzle_rect_from_title(title_tl, fw, fh)
        if rect is not None:
            x0, y0, x1, y1 = rect
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.shape[0] == CROP_H and crop.shape[1] == CROP_W:
                return crop, float(x0), float(y0), 1.0, 1.0

    if dialog_xyxy is None:
        x0, y0 = int(fw * 0.22), int(fh * 0.22)
        x1, y1 = int(fw * 0.78), int(fh * 0.78)
    else:
        x0, y0, x1, y1 = [int(v) for v in dialog_xyxy]
    x0 = max(0, min(fw - 1, x0))
    y0 = max(0, min(fh - 1, y0))
    x1 = max(x0 + 1, min(fw, x1))
    y1 = max(y0 + 1, min(fh, y1))
    roi = frame_bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    if rw >= CROP_W and rh >= CROP_H:
        cx, cy = rw // 2, rh // 2
        lx = max(0, min(rw - CROP_W, cx - CROP_W // 2))
        ly = max(0, min(rh - CROP_H, cy - CROP_H // 2))
        crop = roi[ly : ly + CROP_H, lx : lx + CROP_W]
        return crop, float(x0 + lx), float(y0 + ly), 1.0, 1.0
    # Fallback: stretch ROI to expected puzzle size.
    crop = cv2.resize(roi, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    sx = float(rw) / float(CROP_W)
    sy = float(rh) / float(CROP_H)
    return crop, float(x0), float(y0), sx, sy


def work_to_local(work_xy: Tuple[float, float]) -> Tuple[float, float]:
    """Match gostart _work_to_local on 752×501."""
    wx, wy = float(work_xy[0]), float(work_xy[1])
    return wx * (CROP_W - 1) / max(1.0, WORK_W - 1), wy * (CROP_H - 1) / max(
        1.0, WORK_H - 1
    )


def local_to_work(local_xy: Tuple[float, float]) -> Tuple[float, float]:
    lx, ly = float(local_xy[0]), float(local_xy[1])
    return lx * (WORK_W - 1) / max(1.0, CROP_W - 1), ly * (WORK_H - 1) / max(
        1.0, WORK_H - 1
    )


def _secondary_peak_ratio(prob: np.ndarray, exclusion_radius: int = 4) -> float:
    flat = prob.reshape(-1)
    i = int(np.argmax(flat))
    primary = float(flat[i])
    if primary <= 1e-12:
        return 1.0
    wy, wx = divmod(i, WORK_W)
    mask = np.ones_like(prob, dtype=bool)
    y0, y1 = max(0, wy - exclusion_radius), min(WORK_H, wy + exclusion_radius + 1)
    x0, x1 = max(0, wx - exclusion_radius), min(WORK_W, wx + exclusion_radius + 1)
    mask[y0:y1, x0:x1] = False
    secondary = float(prob[mask].max()) if mask.any() else 0.0
    return secondary / (primary + 1e-12)


def _normalized_concentration(prob: np.ndarray) -> float:
    """Inverse entropy proxy used by gostart calibration (0..1-ish)."""
    p = prob.astype(np.float64).reshape(-1)
    s = float(p.sum())
    if s <= 1e-12:
        return 0.0
    p = p / s
    p = p[p > 1e-12]
    entropy = float(-(p * np.log(p)).sum())
    max_entropy = math.log(float(WORK_H * WORK_W))
    return float(max(0.0, 1.0 - entropy / max(1e-12, max_entropy)))


def _local_centroid(prob: np.ndarray, radius: int = 3) -> Tuple[float, float]:
    flat = prob.reshape(-1)
    i = int(np.argmax(flat))
    wy, wx = divmod(i, WORK_W)
    y0, y1 = max(0, wy - radius), min(WORK_H, wy + radius + 1)
    x0, x1 = max(0, wx - radius), min(WORK_W, wx + radius + 1)
    patch = prob[y0:y1, x0:x1]
    total = float(patch.sum())
    if total <= 1e-12:
        return float(wx) + 0.5, float(wy) + 0.5
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((patch * xx).sum() / total), float((patch * yy).sum() / total)


def _calibrated_confidence(
    prob: np.ndarray,
    secondary_ratio: float,
    *,
    active_phase: bool,
    coeffs: dict,
    ambiguity_cap: float,
    boundary_cap: float,
    boundary_clamped: bool = False,
) -> float:
    concentration = _normalized_concentration(prob)
    dominance = 1.0 / (1.0 + max(0.0, float(secondary_ratio)))
    # remaining mass outside primary peak as soft dominance alt
    primary = float(prob.max())
    dominance = primary / (primary + primary * float(secondary_ratio) + 1e-12)
    logit = (
        float(coeffs.get("intercept", -0.1727907))
        + float(coeffs.get("normalized_heatmap_concentration", 1.84598363))
        * concentration
        + float(coeffs.get("primary_peak_dominance", 3.60162816)) * dominance
        + float(coeffs.get("learning_phase_indicator", 0.00876922))
        * (0.0 if active_phase else 1.0)
    )
    conf = 1.0 / (1.0 + math.exp(-logit))
    if secondary_ratio >= ambiguity_cap:
        # fail-closed soft cap (gostart may_move rejects; keep conf visible)
        conf = min(conf, 0.35)
    if boundary_clamped:
        conf = min(conf, boundary_cap)
    return float(conf)


class LieHeatmapTracker:
    def __init__(self, cfg: Optional[dict] = None):
        la = (cfg or {}).get("lie_alert") or {}
        self.enable = bool(la.get("heatmap_enable", True))
        self.model_path = str(
            la.get("heatmap_model", DEFAULT_MODEL)
        ).strip()
        meta_path = str(la.get("heatmap_meta", DEFAULT_META)).strip()
        self.control_min_confidence = float(
            la.get("control_min_confidence", 0.60)
        )
        self.ambiguity_cap = float(
            la.get("heatmap_ambiguity_secondary_ratio", 0.78)
        )
        self.opening_confirm_frames = max(1, int(la.get("opening_confirm_frames", 4)))
        self.opening_confirm_timeout_s = float(
            la.get("opening_confirm_timeout_s", 8.0)
        )
        self.predict_coast_s = float(la.get("predict_coast_s", 0.30))
        self.secondary_excl = 4

        # Belief params from gostart JSON / tracker constants
        self.learn_sigma = 5.0
        self.learn_power = 2.0
        self.learn_floor = 0.01
        self.learn_recovery = 0.1
        self.active_sigma = 3.0
        self.active_power = 2.5
        self.active_floor = 0.01
        self.active_recovery = 0.025
        self.velocity_gain = 1.0
        self.velocity_momentum = 0.5
        self.velocity_max_work_px = 10.0
        # gostart tracking_safety (crop-local px)
        self.innovation_gate_base_px = 20.0
        self.innovation_gate_speed_px_s = 180.0
        self.innovation_gate_max_dt_s = 0.35
        self.observation_min_confidence = 0.60
        self.calib_coeffs = {
            "intercept": -0.1727907,
            "normalized_heatmap_concentration": 1.84598363,
            "primary_peak_dominance": 3.60162816,
            "learning_phase_indicator": 0.00876922,
        }
        self.boundary_cap = 0.25
        self._load_meta(meta_path)

        self._sess = None
        self._input_name = "features"
        self._output_name = "probability"
        self._history: Deque[np.ndarray] = deque(maxlen=16)
        self._belief = None
        self._active = False
        self._vel = np.zeros(2, dtype=np.float32)
        self._last_work_xy = None
        self._last_local_xy = None  # crop-local accepted track
        self._last_push_t = 0.0
        self._opening_hits = 0
        self._opening_confirmed = False
        self._opening_start_t = 0.0
        self._last_est: Optional[HeatmapEstimate] = None
        self._crop_origin = (0.0, 0.0, 1.0, 1.0)  # ox, oy, sx, sy
        self._crop_mode = "roi"  # title | roi

        if self.enable:
            self._ensure_session()

    def _load_meta(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return
        tf = meta.get("temporal_filter") or {}
        learn = tf.get("learning") or {}
        active = tf.get("active") or {}
        self.learn_sigma = float(learn.get("sigma", self.learn_sigma))
        self.learn_power = float(learn.get("measurement_power", self.learn_power))
        self.learn_floor = float(learn.get("uniform_floor", self.learn_floor))
        self.learn_recovery = float(learn.get("recovery", self.learn_recovery))
        self.active_sigma = float(active.get("sigma", self.active_sigma))
        self.active_power = float(active.get("measurement_power", self.active_power))
        self.active_floor = float(active.get("uniform_floor", self.active_floor))
        self.active_recovery = float(active.get("recovery", self.active_recovery))
        self.velocity_gain = float(tf.get("velocity_gain", self.velocity_gain))
        self.velocity_momentum = float(
            tf.get("velocity_momentum", self.velocity_momentum)
        )
        self.velocity_max_work_px = float(
            tf.get("velocity_max_work_px", self.velocity_max_work_px)
        )
        cal = meta.get("confidence_calibration") or {}
        coeffs = cal.get("coefficients") or {}
        if coeffs:
            self.calib_coeffs = {
                "intercept": float(coeffs.get("intercept", -0.1727907)),
                "normalized_heatmap_concentration": float(
                    coeffs.get("normalized_heatmap_concentration", 1.84598363)
                ),
                "primary_peak_dominance": float(
                    coeffs.get("primary_peak_dominance", 3.60162816)
                ),
                "learning_phase_indicator": float(
                    coeffs.get("learning_phase_indicator", 0.00876922)
                ),
            }
        self.ambiguity_cap = float(
            cal.get(
                "ambiguity_cap_starts_at_secondary_peak_ratio",
                self.ambiguity_cap,
            )
        )
        self.boundary_cap = float(cal.get("boundary_confidence_cap", self.boundary_cap))
        safety = meta.get("tracking_safety") or {}
        # JSON uses a string formula; keep numeric defaults unless overridden in lie_alert.
        if "held_out_normal_motion_coverage" in safety:
            pass  # informational only

    def _ensure_session(self) -> bool:
        if self._sess is not None:
            return True
        path = self.model_path
        if not os.path.isfile(path):
            return False
        try:
            import onnxruntime as ort

            self._sess = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            ins = self._sess.get_inputs()
            outs = self._sess.get_outputs()
            if ins:
                self._input_name = ins[0].name
            if outs:
                self._output_name = outs[0].name
            return True
        except Exception:
            self._sess = None
            return False

    def reset(self, now: float = 0.0):
        self._history.clear()
        self._belief = None
        self._active = False
        self._vel[:] = 0
        self._last_work_xy = None
        self._last_local_xy = None
        self._last_push_t = 0.0
        self._opening_hits = 0
        self._opening_confirmed = False
        self._opening_start_t = float(now or 0.0)
        self._last_est = None
        self._crop_mode = "roi"

    @property
    def opening_confirmed(self) -> bool:
        return bool(self._opening_confirmed)

    def note_opening_white(self, hit: bool, now: float) -> bool:
        if self._opening_confirmed:
            return True
        if self._opening_start_t <= 0:
            self._opening_start_t = float(now)
        if hit:
            self._opening_hits += 1
        else:
            self._opening_hits = max(0, self._opening_hits - 1)
        if self._opening_hits >= self.opening_confirm_frames:
            self._opening_confirmed = True
            self._active = True
            return True
        return False

    def opening_timed_out(self, now: float) -> bool:
        if self._opening_confirmed:
            return False
        t0 = float(self._opening_start_t or now)
        return (float(now) - t0) >= self.opening_confirm_timeout_s

    def _update_belief(self, measurement: np.ndarray) -> np.ndarray:
        m = measurement.astype(np.float32)
        m = np.clip(m, 0.0, None)
        s = float(m.sum())
        if s > 1e-8:
            m = m / s
        else:
            m = np.full((WORK_H, WORK_W), 1.0 / (WORK_H * WORK_W), dtype=np.float32)

        sigma = self.active_sigma if self._active else self.learn_sigma
        power = self.active_power if self._active else self.learn_power
        floor = self.active_floor if self._active else self.learn_floor
        recovery = self.active_recovery if self._active else self.learn_recovery

        if self._belief is None:
            belief = m.copy()
        else:
            dx, dy = float(self._vel[0]), float(self._vel[1])
            prior = self._belief
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                prior = cv2.warpAffine(
                    prior,
                    M,
                    (WORK_W, WORK_H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=float(sigma))
            uniform = 1.0 / float(WORK_H * WORK_W)
            prior = prior + float(floor) * uniform
            prior = prior / (float(prior.sum()) + 1e-12)
            meas = np.power(m + 1e-12, float(power))
            meas = meas / (float(meas.sum()) + 1e-12)
            belief = (1.0 - recovery) * prior * meas + recovery * meas
            belief = belief / (float(belief.sum()) + 1e-12)
        self._belief = belief
        return belief

    def _innovation_gate_px(self, delta_s: float) -> float:
        dt = min(max(0.0, float(delta_s)), float(self.innovation_gate_max_dt_s))
        return float(self.innovation_gate_base_px) + float(
            self.innovation_gate_speed_px_s
        ) * dt

    def push(
        self,
        frame_bgr: np.ndarray,
        now: float,
        inject_xy: Optional[Tuple[float, float]] = None,
        inject_weight: float = 0.35,
        dialog_xyxy: Optional[Tuple[int, int, int, int]] = None,
        title_tl: Optional[Tuple[int, int]] = None,
    ) -> Optional[HeatmapEstimate]:
        if not self.enable or not self._ensure_session():
            return None

        crop, ox, oy, sx, sy = extract_puzzle_crop(
            frame_bgr, dialog_xyxy=dialog_xyxy, title_tl=title_tl
        )
        self._crop_origin = (ox, oy, sx, sy)
        self._crop_mode = "title" if title_tl is not None else "roi"
        clean = _remove_green_cursor(crop)
        gray = prepare_gray(clean)
        feat_u8 = gostart_build_features(
            self._history, gray, registered_previous=registered_previous
        )
        self._history.append(gray)
        feat = feat_u8.astype(np.float32)[None, ...] / 255.0

        try:
            out = self._sess.run(
                [self._output_name], {self._input_name: feat}
            )[0]
        except Exception:
            return None
        meas = np.asarray(out, dtype=np.float32)
        if meas.ndim == 3:
            meas = meas[0]
        elif meas.ndim == 4:
            meas = meas[0, 0]
        s = float(meas.sum())
        if s > 1e-8:
            meas = meas / s
        else:
            return None

        if inject_xy is not None:
            fx, fy = float(inject_xy[0]), float(inject_xy[1])
            local_x = (fx - ox) / max(1e-6, sx)
            local_y = (fy - oy) / max(1e-6, sy)
            wx, wy = local_to_work((local_x, local_y))
            yy, xx = np.mgrid[0:WORK_H, 0:WORK_W]
            gauss = np.exp(
                -0.5 * (((xx - wx) / 3.0) ** 2 + ((yy - wy) / 3.0) ** 2)
            )
            gauss = gauss / (float(gauss.sum()) + 1e-12)
            w = float(np.clip(inject_weight, 0.0, 0.9))
            meas = (1.0 - w) * meas + w * gauss
            meas = meas / (float(meas.sum()) + 1e-12)

        # Peak of raw measurement (before belief) for innovation gate
        meas_wx, meas_wy = _local_centroid(meas, radius=3)
        meas_lx, meas_ly = work_to_local((meas_wx, meas_wy))
        dt = 0.05
        if self._last_push_t > 0:
            dt = max(0.0, float(now) - float(self._last_push_t))
        accept_obs = True
        if self._last_local_xy is not None:
            prev_lx, prev_ly = self._last_local_xy
            innov = float(
                ((meas_lx - prev_lx) ** 2 + (meas_ly - prev_ly) ** 2) ** 0.5
            )
            gate = self._innovation_gate_px(dt)
            raw_sec = _secondary_peak_ratio(meas, self.secondary_excl)
            raw_conf = _calibrated_confidence(
                meas,
                raw_sec,
                active_phase=self._active,
                coeffs=self.calib_coeffs,
                ambiguity_cap=self.ambiguity_cap,
                boundary_cap=self.boundary_cap,
            )
            if innov > gate or raw_conf < float(self.observation_min_confidence):
                accept_obs = False
        if accept_obs or self._belief is None:
            belief = self._update_belief(meas)
        else:
            # Reject observation: coast belief with velocity only.
            prior = self._belief
            if abs(float(self._vel[0])) > 0.05 or abs(float(self._vel[1])) > 0.05:
                M = np.float32(
                    [[1, 0, float(self._vel[0])], [0, 1, float(self._vel[1])]]
                )
                prior = cv2.warpAffine(
                    prior,
                    M,
                    (WORK_W, WORK_H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                prior = prior / (float(prior.sum()) + 1e-12)
                self._belief = prior
            belief = self._belief

        wx, wy = _local_centroid(belief, radius=3)
        sec_ratio = _secondary_peak_ratio(belief, self.secondary_excl)
        conf = _calibrated_confidence(
            belief,
            sec_ratio,
            active_phase=self._active,
            coeffs=self.calib_coeffs,
            ambiguity_cap=self.ambiguity_cap,
            boundary_cap=self.boundary_cap,
        )

        if self._last_work_xy is not None:
            raw = np.array(
                [wx - self._last_work_xy[0], wy - self._last_work_xy[1]],
                dtype=np.float32,
            )
            span = float(np.linalg.norm(raw))
            if span > self.velocity_max_work_px:
                raw *= self.velocity_max_work_px / (span + 1e-6)
            self._vel = (
                self.velocity_momentum * self._vel
                + (1.0 - self.velocity_momentum) * raw * self.velocity_gain
            )
        self._last_work_xy = (wx, wy)

        lx, ly = work_to_local((wx, wy))
        if accept_obs:
            self._last_local_xy = (lx, ly)
        self._last_push_t = float(now)

        fx = ox + lx * sx
        fy = oy + ly * sy
        src = "heatmap" if accept_obs else "predict"
        if accept_obs and inject_xy is not None and inject_weight >= 0.5:
            src = "inject"
        est = HeatmapEstimate(
            x=fx,
            y=fy,
            confidence=conf if accept_obs else max(0.0, conf * 0.85),
            secondary_ratio=sec_ratio,
            source=src,
            work_xy=(wx, wy),
        )
        self._last_est = est
        return est

    def predict_coast(self) -> Optional[HeatmapEstimate]:
        if self._last_est is None or self._last_work_xy is None:
            return None
        wx = self._last_work_xy[0] + float(self._vel[0]) * 0.5
        wy = self._last_work_xy[1] + float(self._vel[1]) * 0.5
        ox, oy, sx, sy = self._crop_origin
        lx, ly = work_to_local((wx, wy))
        fx = ox + lx * sx
        fy = oy + ly * sy
        return HeatmapEstimate(
            x=fx,
            y=fy,
            confidence=max(0.0, float(self._last_est.confidence) * 0.7),
            secondary_ratio=1.0,
            source="predict",
            work_xy=(wx, wy),
        )

    def may_move(self, est: Optional[HeatmapEstimate]) -> bool:
        if est is None:
            return False
        if not self._opening_confirmed:
            return False
        # Rejected observations must not keep ownership of the cursor.
        if est.source == "predict":
            return False
        if est.secondary_ratio >= self.ambiguity_cap:
            return False
        return float(est.confidence) >= self.control_min_confidence
