from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GroupSummary:
    id: int
    score: float
    group_type: str
    n_boxes: int
    time_span_px: float
    freq_span_px: float
    time_span_ratio: float
    freq_span_ratio: float
    mean_w: float
    mean_h: float
    std_log_w: float
    std_log_h: float
    mean_contrast_z: float
    std_contrast_z: float
    mean_bright_ratio: float
    std_bright_ratio: float


class MultiSignalPreprocessor:
    """
    Inference-only multi-signal postprocessor.

    Design goals:
      1) Keep current low-level robustness from the existing preprocess:
         - basic geometry filtering
         - local-energy filtering against surrounding background
      2) Replace single-cluster selection with multi-group temporal association
      3) Downweight frequency concentration; emphasize:
         - shape consistency
         - energy consistency
         - time continuity
      4) Return multiple signal groups plus human-readable summaries.
    """

    def __init__(
        self,
        min_area: int = 20,
        min_ratio: float = 0.0,
        min_width: int = 2,
        min_height: int = 2,
        max_width: int = 0,
        max_height: int = 0,
        exclude_bottom_ratio: float = 0.0,
        ring_margin: int = 5,
        min_contrast_z: float = 0.6,
        min_integrated_energy: float = 8.0,
        min_bright_ratio: float = 0.02,
        bright_z_thresh: float = 1.5,
        # association weights
        link_dt_max_ratio: float = 0.22,
        link_cost_thresh: float = 2.3,
        link_time_weight: float = 1.0,
        link_freq_weight: float = 0.35,
        link_width_weight: float = 1.4,
        link_height_weight: float = 1.4,
        link_area_weight: float = 0.8,
        link_contrast_weight: float = 1.0,
        link_bright_weight: float = 0.8,
        # group post-filtering
        min_group_len: int = 2,
        min_group_time_span_ratio: float = 0.01,
        max_group_shape_std: float = 0.80,
        max_group_energy_std: float = 1.20,
        nms_iou_thresh: float = 0.75,
        nms_cover_small_thresh: float = 0.98,
        # scoring
        score_count_weight: float = 1.2,
        score_time_span_weight: float = 1.0,
        score_shape_consistency_weight: float = 1.3,
        score_energy_consistency_weight: float = 1.0,
        score_mean_contrast_weight: float = 0.6,
        score_freq_span_penalty: float = 0.15,
    ):
        self.min_area = int(min_area)
        self.min_ratio = float(min_ratio)
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.exclude_bottom_ratio = float(exclude_bottom_ratio)

        self.ring_margin = int(ring_margin)
        self.min_contrast_z = float(min_contrast_z)
        self.min_integrated_energy = float(min_integrated_energy)
        self.min_bright_ratio = float(min_bright_ratio)
        self.bright_z_thresh = float(bright_z_thresh)

        self.link_dt_max_ratio = float(link_dt_max_ratio)
        self.link_cost_thresh = float(link_cost_thresh)
        self.link_time_weight = float(link_time_weight)
        self.link_freq_weight = float(link_freq_weight)
        self.link_width_weight = float(link_width_weight)
        self.link_height_weight = float(link_height_weight)
        self.link_area_weight = float(link_area_weight)
        self.link_contrast_weight = float(link_contrast_weight)
        self.link_bright_weight = float(link_bright_weight)

        self.min_group_len = int(min_group_len)
        self.min_group_time_span_ratio = float(min_group_time_span_ratio)
        self.max_group_shape_std = float(max_group_shape_std)
        self.max_group_energy_std = float(max_group_energy_std)
        self.nms_iou_thresh = float(nms_iou_thresh)
        self.nms_cover_small_thresh = float(nms_cover_small_thresh)

        self.score_count_weight = float(score_count_weight)
        self.score_time_span_weight = float(score_time_span_weight)
        self.score_shape_consistency_weight = float(score_shape_consistency_weight)
        self.score_energy_consistency_weight = float(score_energy_consistency_weight)
        self.score_mean_contrast_weight = float(score_mean_contrast_weight)
        self.score_freq_span_penalty = float(score_freq_span_penalty)

    # ---------- low-level helpers ----------
    @staticmethod
    def _iou(boxA, boxB) -> float:
        x1, y1, x2, y2 = [float(v) for v in boxA]
        xb1, yb1, xb2, yb2 = [float(v) for v in boxB]
        inter_w = max(0.0, min(x2, xb2) - max(x1, xb1))
        inter_h = max(0.0, min(y2, yb2) - max(y1, yb1))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        areaA = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        areaB = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        denom = areaA + areaB - inter
        return float(inter / denom) if denom > 0 else 0.0

    def _nms(self, boxes: np.ndarray) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
        order = np.argsort(-areas)
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            rest = []
            for j in order[1:]:
                iou = self._iou(arr[i], arr[j])
                area_i = max(1.0, float(areas[i]))
                area_j = max(1.0, float(areas[j]))
                x1 = max(arr[i, 0], arr[j, 0])
                y1 = max(arr[i, 1], arr[j, 1])
                x2 = min(arr[i, 2], arr[j, 2])
                y2 = min(arr[i, 3], arr[j, 3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                cover_small = inter / min(area_i, area_j)
                if iou > self.nms_iou_thresh or cover_small > self.nms_cover_small_thresh:
                    continue
                rest.append(j)
            order = np.asarray(rest, dtype=np.int64)
        return arr[np.asarray(keep, dtype=np.int64)]

    def _basic_filter(self, boxes: np.ndarray, H: int, W: int) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        widths = x2 - x1
        heights = y2 - y1
        areas = widths * heights

        mask = (widths > 0) & (heights > 0)
        mask &= areas >= self.min_area
        mask &= widths >= self.min_width
        mask &= heights >= self.min_height

        if self.min_ratio > 0:
            mask &= (widths / np.maximum(heights, 1)) >= self.min_ratio
        if self.max_width > 0:
            mask &= widths <= self.max_width
        if self.max_height > 0:
            mask &= heights <= self.max_height
        if self.exclude_bottom_ratio > 0:
            y_center = (y1 + y2) / 2.0
            max_valid_y = H * (1.0 - self.exclude_bottom_ratio)
            mask &= y_center <= max_valid_y

        return arr[mask].astype(np.int32, copy=False)

    def _box_energy_stats(self, box: np.ndarray, spectrogram: np.ndarray) -> Optional[Dict[str, float]]:
        H, W = spectrogram.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            return None

        region = spectrogram[y1:y2, x1:x2]
        if region.size == 0:
            return None

        mean_val = float(np.mean(region))
        max_val = float(np.max(region))
        var_val = float(np.var(region))
        area = float(region.size)

        m = self.ring_margin
        rx1 = max(0, x1 - m)
        ry1 = max(0, y1 - m)
        rx2 = min(W, x2 + m)
        ry2 = min(H, y2 + m)
        ring = spectrogram[ry1:ry2, rx1:rx2]
        ring_mask = np.ones_like(ring, dtype=bool)
        ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False
        bg = ring[ring_mask]

        if bg.size == 0:
            bg_mean, bg_std = 0.0, 1.0
        else:
            bg_mean = float(np.mean(bg))
            bg_std = float(np.std(bg)) + 1e-6

        contrast_z = (mean_val - bg_mean) / bg_std
        integrated_energy = float(np.maximum(region - bg_mean, 0.0).sum())
        bright_ratio = float(np.mean(region > (bg_mean + self.bright_z_thresh * bg_std)))

        return {
            "mean": mean_val,
            "max": max_val,
            "var": var_val,
            "area": area,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "contrast_z": contrast_z,
            "integrated_energy": integrated_energy,
            "bright_ratio": bright_ratio,
        }

    def _energy_filter(self, boxes: np.ndarray, spectrogram: np.ndarray) -> tuple[np.ndarray, List[Dict[str, float]]]:
        keep_boxes: List[np.ndarray] = []
        stats_list: List[Dict[str, float]] = []
        for box in boxes:
            st = self._box_energy_stats(box, spectrogram)
            if st is None:
                continue
            if st["contrast_z"] < self.min_contrast_z:
                continue
            if st["integrated_energy"] < self.min_integrated_energy:
                continue
            if st["bright_ratio"] < self.min_bright_ratio:
                continue
            keep_boxes.append(np.asarray(box, dtype=np.int32))
            stats_list.append(st)
        if len(keep_boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []
        return np.asarray(keep_boxes, dtype=np.int32), stats_list

    # ---------- descriptors ----------
    def _build_box_descriptors(self, boxes: np.ndarray, spectrogram: np.ndarray) -> List[Dict[str, float]]:
        H, W = spectrogram.shape[:2]
        descs: List[Dict[str, float]] = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = w * h
            st = self._box_energy_stats(box, spectrogram)
            if st is None:
                st = {
                    "mean": 0.0,
                    "max": 0.0,
                    "var": 0.0,
                    "area": area,
                    "bg_mean": 0.0,
                    "bg_std": 1.0,
                    "contrast_z": 0.0,
                    "integrated_energy": 0.0,
                    "bright_ratio": 0.0,
                }
            desc = {
                "box": np.asarray(box, dtype=np.int32),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "cx": float(cx), "cy": float(cy),
                "cx_n": float(cx / max(W, 1)), "cy_n": float(cy / max(H, 1)),
                "w": float(w), "h": float(h), "area": float(area),
                "w_n": float(w / max(W, 1)), "h_n": float(h / max(H, 1)),
                "area_n": float(area / max(H * W, 1)),
                "log_wh": float(math.log((w + 1e-6) / (h + 1e-6))),
                "mean": float(st["mean"]),
                "max": float(st["max"]),
                "var": float(st["var"]),
                "contrast_z": float(st["contrast_z"]),
                "integrated_energy": float(st["integrated_energy"]),
                "bright_ratio": float(st["bright_ratio"]),
            }
            descs.append(desc)
        descs.sort(key=lambda d: (d["cx"], d["cy"]))
        return descs

    def _pair_cost(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if b["cx_n"] <= a["cx_n"]:
            return float("inf")

        dt = b["cx_n"] - a["cx_n"]
        if dt > self.link_dt_max_ratio:
            return float("inf")

        df = abs(b["cy_n"] - a["cy_n"])
        dw = abs(math.log((b["w"] + 1e-6) / (a["w"] + 1e-6)))
        dh = abs(math.log((b["h"] + 1e-6) / (a["h"] + 1e-6)))
        da = abs(math.log((b["area"] + 1e-6) / (a["area"] + 1e-6)))
        dc = abs(b["contrast_z"] - a["contrast_z"])
        db = abs(b["bright_ratio"] - a["bright_ratio"])

        cost = (
            self.link_time_weight * (dt / max(self.link_dt_max_ratio, 1e-6))
            + self.link_freq_weight * df
            + self.link_width_weight * dw
            + self.link_height_weight * dh
            + self.link_area_weight * da
            + self.link_contrast_weight * dc
            + self.link_bright_weight * db
        )
        return float(cost)

    def _make_empty_group(self, first_desc: Dict[str, float]) -> Dict[str, Any]:
        return {
            "descs": [first_desc],
            "boxes": [first_desc["box"]],
        }

    def _assign_boxes_to_groups(self, descs: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for d in descs:
            best_idx = None
            best_cost = float("inf")
            for gi, g in enumerate(groups):
                tail = g["descs"][-1]
                cost = self._pair_cost(tail, d)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = gi
            if best_idx is not None and best_cost < self.link_cost_thresh:
                groups[best_idx]["descs"].append(d)
                groups[best_idx]["boxes"].append(d["box"])
            else:
                groups.append(self._make_empty_group(d))
        return groups

    # ---------- group analytics ----------
    def _group_stats(self, group: Dict[str, Any], H: int, W: int) -> Dict[str, float]:
        descs = group["descs"]
        boxes = np.asarray(group["boxes"], dtype=np.int32).reshape(-1, 4)
        cx = np.asarray([d["cx"] for d in descs], dtype=np.float32)
        cy = np.asarray([d["cy"] for d in descs], dtype=np.float32)
        w = np.asarray([d["w"] for d in descs], dtype=np.float32)
        h = np.asarray([d["h"] for d in descs], dtype=np.float32)
        contrast = np.asarray([d["contrast_z"] for d in descs], dtype=np.float32)
        bright = np.asarray([d["bright_ratio"] for d in descs], dtype=np.float32)

        log_w = np.log(w + 1e-6)
        log_h = np.log(h + 1e-6)

        return {
            "n_boxes": int(len(descs)),
            "time_span_px": float(cx.max() - cx.min()) if len(cx) > 0 else 0.0,
            "freq_span_px": float(cy.max() - cy.min()) if len(cy) > 0 else 0.0,
            "time_span_ratio": float((cx.max() - cx.min()) / max(W, 1)) if len(cx) > 0 else 0.0,
            "freq_span_ratio": float((cy.max() - cy.min()) / max(H, 1)) if len(cy) > 0 else 0.0,
            "mean_w": float(np.mean(w)) if len(w) else 0.0,
            "mean_h": float(np.mean(h)) if len(h) else 0.0,
            "std_log_w": float(np.std(log_w)) if len(log_w) else 0.0,
            "std_log_h": float(np.std(log_h)) if len(log_h) else 0.0,
            "mean_contrast_z": float(np.mean(contrast)) if len(contrast) else 0.0,
            "std_contrast_z": float(np.std(contrast)) if len(contrast) else 0.0,
            "mean_bright_ratio": float(np.mean(bright)) if len(bright) else 0.0,
            "std_bright_ratio": float(np.std(bright)) if len(bright) else 0.0,
            "boxes": boxes,
        }

    def _group_passes(self, stats: Dict[str, float]) -> bool:
        if stats["n_boxes"] < self.min_group_len:
            return False
        if stats["time_span_ratio"] < self.min_group_time_span_ratio:
            return False
        shape_std = 0.5 * (stats["std_log_w"] + stats["std_log_h"])
        if shape_std > self.max_group_shape_std:
            return False
        energy_std = 0.5 * (stats["std_contrast_z"] + stats["std_bright_ratio"])
        if energy_std > self.max_group_energy_std:
            return False
        return True

    def _score_group(self, stats: Dict[str, float]) -> float:
        shape_consistency = math.exp(-0.5 * (stats["std_log_w"] + stats["std_log_h"]))
        energy_consistency = math.exp(-0.5 * (stats["std_contrast_z"] + stats["std_bright_ratio"]))
        score = (
            self.score_count_weight * float(stats["n_boxes"])
            + self.score_time_span_weight * float(stats["time_span_ratio"])
            + self.score_shape_consistency_weight * shape_consistency
            + self.score_energy_consistency_weight * energy_consistency
            + self.score_mean_contrast_weight * float(stats["mean_contrast_z"])
            - self.score_freq_span_penalty * float(stats["freq_span_ratio"])
        )
        return float(score)

    def _infer_group_type(self, stats: Dict[str, float]) -> str:
        shape_std = 0.5 * (stats["std_log_w"] + stats["std_log_h"])
        energy_std = 0.5 * (stats["std_contrast_z"] + stats["std_bright_ratio"])
        if stats["freq_span_ratio"] < 0.08 and shape_std < 0.35:
            return "static_like"
        if stats["freq_span_ratio"] >= 0.08 and shape_std < 0.45 and energy_std < 0.8:
            return "hopping_like"
        return "unknown"

    # ---------- main API ----------
    def select_signal_groups(self, det_boxes, spectrogram: np.ndarray) -> List[Dict[str, Any]]:
        if spectrogram is None or np.asarray(spectrogram).ndim != 2:
            raise ValueError("spectrogram must be a 2D numpy array")

        spec = np.asarray(spectrogram)
        H, W = spec.shape[:2]

        boxes = self._basic_filter(det_boxes, H, W)
        if boxes.size == 0:
            return []

        energy_boxes, _ = self._energy_filter(boxes, spec)
        if energy_boxes.size == 0:
            # fallback to geometry-filtered boxes, same philosophy as current preprocess
            candidate_boxes = boxes
        else:
            candidate_boxes = energy_boxes

        descs = self._build_box_descriptors(candidate_boxes, spec)
        if len(descs) == 0:
            return []

        groups = self._assign_boxes_to_groups(descs)
        outputs: List[Dict[str, Any]] = []
        gid = 1
        for g in groups:
            stats = self._group_stats(g, H, W)
            if not self._group_passes(stats):
                continue
            boxes_nms = self._nms(stats["boxes"])
            if boxes_nms.size == 0:
                continue

            # keep descriptors whose boxes survive NMS
            keep_descs: List[Dict[str, float]] = []
            keep_boxes_set = {tuple(map(int, b.tolist())) for b in boxes_nms}
            for d in g["descs"]:
                if tuple(map(int, d["box"].tolist())) in keep_boxes_set:
                    keep_descs.append(d)
            if len(keep_descs) == 0:
                continue

            # sort by time center after NMS
            keep_descs.sort(key=lambda d: (d["cx"], d["cy"]))
            boxes_sorted = np.asarray([d["box"] for d in keep_descs], dtype=np.int32).reshape(-1, 4)
            stats = self._group_stats({"descs": keep_descs, "boxes": boxes_sorted}, H, W)
            score = self._score_group(stats)
            group_type = self._infer_group_type(stats)

            outputs.append({
                "id": gid,
                "boxes": boxes_sorted,
                "descs": keep_descs,
                "score": score,
                "group_type": group_type,
                "n_boxes": stats["n_boxes"],
                "time_span_px": stats["time_span_px"],
                "freq_span_px": stats["freq_span_px"],
                "time_span_ratio": stats["time_span_ratio"],
                "freq_span_ratio": stats["freq_span_ratio"],
                "mean_w": stats["mean_w"],
                "mean_h": stats["mean_h"],
                "std_log_w": stats["std_log_w"],
                "std_log_h": stats["std_log_h"],
                "mean_contrast_z": stats["mean_contrast_z"],
                "std_contrast_z": stats["std_contrast_z"],
                "mean_bright_ratio": stats["mean_bright_ratio"],
                "std_bright_ratio": stats["std_bright_ratio"],
            })
            gid += 1

        outputs.sort(key=lambda x: x["score"], reverse=True)
        # renumber after sorting by score
        for i, g in enumerate(outputs, start=1):
            g["id"] = i
        return outputs
