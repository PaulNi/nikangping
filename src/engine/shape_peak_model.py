# -*- coding: utf-8 -*-
"""
Same-shape peak model for the lie-detector dialog.

The network only proposes boxes and centers of silhouettes that match the
acquired shape. It does not choose the red target. White-sticky, frame-to-frame
spin, and decoy-path rejection stay in LieAlert.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_IN_HW = 48


def _gabor_bank() -> np.ndarray:
    """Eight oriented edge filters, shape (8, 1, 7, 7)."""
    kernels = []
    for sigma in (1.2, 2.2):
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            kern = cv2.getGaborKernel(
                (7, 7), sigma, theta, lambd=4.0, gamma=0.55, psi=0.0, ktype=cv2.CV_32F
            )
            kern = kern.astype(np.float32)
            kern /= float(np.linalg.norm(kern)) + 1e-6
            kernels.append(kern)
    weight = np.stack(kernels, axis=0)[:, None, :, :]
    return np.ascontiguousarray(weight)


def export_shape_peak_onnx(path: str) -> str:
    """Write the conv embedding graph. Runtime only needs onnxruntime."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight1 = _gabor_bank()
    # Second layer keeps each orientation and downsamples. No learned mix,
    # so the embedding still describes where the edges sit.
    weight2 = np.zeros((8, 8, 3, 3), dtype=np.float32)
    for channel in range(8):
        weight2[channel, channel, 1, 1] = 1.0

    patch = helper.make_tensor_value_info(
        "patch", TensorProto.FLOAT, ["N", 1, _IN_HW, _IN_HW]
    )
    emb = helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["N", 8, 6, 6])
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Conv",
                ["patch", "W1"],
                ["c1"],
                kernel_shape=[7, 7],
                pads=[3, 3, 3, 3],
                strides=[4, 4],
            ),
            helper.make_node("Relu", ["c1"], ["r1"]),
            helper.make_node(
                "Conv",
                ["r1", "W2"],
                ["c2"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
            helper.make_node("Relu", ["c2"], ["emb"]),
        ],
        name="shape_peak",
        inputs=[patch],
        outputs=[emb],
        initializer=[
            numpy_helper.from_array(weight1, name="W1"),
            numpy_helper.from_array(weight2, name="W2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    onnx.save(model, path)
    return path


def _prep_patch(gray_u8: np.ndarray) -> np.ndarray:
    img = cv2.resize(gray_u8, (_IN_HW, _IN_HW), interpolation=cv2.INTER_AREA)
    feat = img.astype(np.float32)
    feat -= float(feat.mean())
    scale = float(feat.std())
    if scale < 1e-3:
        feat[:] = 0.0
    else:
        feat /= scale
    return feat.reshape(1, 1, _IN_HW, _IN_HW)


def _rotate_binary(sil: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = sil.shape[:2]
    pad = int(0.35 * max(h, w)) + 2
    canvas = cv2.copyMakeBorder(sil, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ch, cw = canvas.shape[:2]
    matrix = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(
        canvas, matrix, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0
    )
    ys, xs = np.where(rot > 0)
    if len(xs) < 20:
        return rot
    return rot[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


class ShapePeakModel:
    """ONNX edge embedding. Output is same-shape boxes, not the tracked target."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None
        self._ref_key = None
        self._ref_emb = None
        self._ref_angles = None

    def _ensure_session(self):
        if self._session is not None:
            return
        if not os.path.isfile(self.model_path):
            export_shape_peak_onnx(self.model_path)
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )

    def _embed(self, batch: np.ndarray) -> np.ndarray:
        self._ensure_session()
        out = self._session.run(["emb"], {"patch": batch})[0]
        flat = out.reshape(out.shape[0], -1).astype(np.float32)
        norm = np.linalg.norm(flat, axis=1, keepdims=True)
        return flat / np.maximum(norm, 1e-6)

    def _reference(self, sil: np.ndarray, hint_angle):
        key = (sil.shape, int(sil.sum()), None if hint_angle is None else int(round(hint_angle)) % 180)
        if self._ref_key == key and self._ref_emb is not None:
            return self._ref_emb, self._ref_angles
        if hint_angle is None:
            angles = list(range(0, 180, 20))
        else:
            base = int(round(float(hint_angle))) % 180
            angles = [(base + off) % 180 for off in (-40, -20, 0, 20, 40, 60, 90)]
        patches = []
        for ang in angles:
            rot = _rotate_binary(sil, ang)
            patches.append(_prep_patch(rot))
        emb = self._embed(np.concatenate(patches, axis=0))
        self._ref_key = key
        self._ref_emb = emb
        self._ref_angles = angles
        return emb, angles

    def find_peaks(
        self,
        img_bgr,
        silhouette,
        nat_wh,
        roi,
        dog_fn,
        max_peaks: int = 10,
        hint_angle=None,
    ):
        """
        Returns (score, cx, cy, angle_deg, tw, th) in full-frame coordinates.
        score is cosine similarity of the edge embedding, not a track decision.
        """
        if img_bgr is None or silhouette is None:
            return []
        dx0, dy0, dx1, dy1 = [int(v) for v in roi]
        if dx1 - dx0 < 48 or dy1 - dy0 < 48:
            return []
        gray = cv2.cvtColor(img_bgr[dy0:dy1, dx0:dx1], cv2.COLOR_BGR2GRAY)
        dog = dog_fn(gray)
        nat_w, nat_h = [max(28, int(v)) for v in nat_wh]
        boxes = self._propose(dog, nat_w, nat_h)
        if not boxes:
            return []
        ref, angles = self._reference(silhouette, hint_angle)
        batch = []
        kept_boxes = []
        for x, y, w, h in boxes:
            crop = gray[y : y + h, x : x + w]
            if crop.size < 16:
                continue
            batch.append(_prep_patch(crop))
            kept_boxes.append((x, y, w, h))
        if not batch:
            return []
        emb = self._embed(np.concatenate(batch, axis=0))
        sims = emb @ ref.T
        raw = []
        for i, (x, y, w, h) in enumerate(kept_boxes):
            best_j = int(np.argmax(sims[i]))
            score = float(sims[i, best_j])
            if score < 0.20:
                continue
            cx = dx0 + x + w / 2.0
            cy = dy0 + y + h / 2.0
            raw.append((score, cx, cy, float(angles[best_j]), int(w), int(h)))
        raw.sort(key=lambda item: -item[0])
        if not raw:
            return []
        best_score = raw[0][0]
        # One box per shape. Overlapping windows of the same blob are not extra instances.
        sep = max(70.0, 0.72 * max(nat_w, nat_h))
        kept = []
        for item in raw:
            if item[0] < max(0.42, best_score - 0.08):
                continue
            if any(
                ((item[1] - prev[1]) ** 2 + (item[2] - prev[2]) ** 2) ** 0.5 < sep
                for prev in kept
            ):
                continue
            kept.append(item)
            if len(kept) >= max_peaks:
                break
        return kept

    @staticmethod
    def _propose(dog: np.ndarray, nat_w: int, nat_h: int):
        boxes = []
        if dog.size == 0:
            return boxes
        level = float(max(18.0, np.percentile(dog, 72)))
        mask = (dog >= level).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.30 * nat_w * nat_h
        max_area = 3.4 * nat_w * nat_h
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 18 or h < 18:
                continue
            area = float(w * h)
            if area < min_area or area > max_area:
                continue
            aspect = w / float(h)
            if aspect < 0.40 or aspect > 2.4:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))

        win_w = min(nat_w, dog.shape[1] - 2)
        win_h = min(nat_h, dog.shape[0] - 2)
        if win_w >= 28 and win_h >= 28:
            stride = max(win_w, win_h)
            scored = []
            for y in range(0, dog.shape[0] - win_h + 1, stride):
                row = dog[y : y + win_h]
                for x in range(0, dog.shape[1] - win_w + 1, stride):
                    energy = float(row[:, x : x + win_w].mean())
                    scored.append((energy, x, y))
            scored.sort(key=lambda item: -item[0])
            for _energy, x, y in scored[:8]:
                boxes.append((int(x), int(y), int(win_w), int(win_h)))

        deduped = []
        for box in boxes:
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            if any(
                ((cx - (p[0] + p[2] / 2.0)) ** 2 + (cy - (p[1] + p[3] / 2.0)) ** 2) ** 0.5
                < 0.45 * max(nat_w, nat_h)
                for p in deduped
            ):
                continue
            deduped.append(box)
            if len(deduped) >= 36:
                break
        return deduped
