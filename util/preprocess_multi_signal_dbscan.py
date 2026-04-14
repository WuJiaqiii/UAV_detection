import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class BoxDesc:
    box: np.ndarray  # [4]
    cx: float
    cy: float
    w: float
    h: float
    area: float
    log_wh: float
    mean: float
    maxv: float
    var: float
    bg_mean: float
    bg_std: float
    contrast_z: float
    integrated_energy: float
    bright_ratio: float


class MultiSignalDBSCANPreprocessor:
    """
    Inference-only multi-signal preprocessor for .mat spectrograms.

    Strategy:
      1) basic geometry filtering
      2) local energy filtering (with fallback to geometry-only result)
      3) DBSCAN clustering on frequency centers only
      4) merge adjacent/similar clusters (frequency + shape + energy consistency)
      5) score groups and keep top-N groups
      6) optional light NMS inside each group

    This version intentionally does NOT try to model large hopping trajectories.
    It is designed for multi-signal scenes where signals are static or near-static
    in frequency, and where one signal may appear in multiple time-separated bursts.
    """

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        # basic geometry
        min_area: int = 20,
        min_ratio: float = 0.0,
        min_width: int = 2,
        min_height: int = 2,
        max_width: int = 0,
        max_height: int = 0,
        exclude_bottom_ratio: float = 0.0,
        # local energy filtering
        ring_margin: int = 5,
        min_contrast_z: float = 0.6,
        min_integrated_energy: float = 8.0,
        min_bright_ratio: float = 0.02,
        bright_z_thresh: float = 1.5,
        # DBSCAN on frequency centers
        freq_eps: float = 12.0,
        freq_min_samples: int = 1,
        # cluster merge
        merge_freq_thresh: float = 10.0,
        merge_w_log_thresh: float = 0.35,
        merge_h_log_thresh: float = 0.35,
        merge_energy_thresh: float = 1.0,
        merge_bright_thresh: float = 0.12,
        # group filtering
        min_group_len: int = 2,
        min_group_time_span_ratio: float = 0.01,
        # intra-group NMS
        nms_iou_thresh: float = 0.7,
        use_cover_small_rule: bool = False,
        # top groups
        max_groups: int = 3,
        # scoring
        score_boxes_weight: float = 1.0,
        score_time_span_weight: float = 1.2,
        score_area_weight: float = 20.0,
        score_mean_contrast_weight: float = 0.8,
        score_shape_std_weight: float = 1.0,
        score_energy_std_weight: float = 0.8,
    ) -> None:
        self.sr = float(sampling_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

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

        self.freq_eps = float(freq_eps)
        self.freq_min_samples = int(freq_min_samples)

        self.merge_freq_thresh = float(merge_freq_thresh)
        self.merge_w_log_thresh = float(merge_w_log_thresh)
        self.merge_h_log_thresh = float(merge_h_log_thresh)
        self.merge_energy_thresh = float(merge_energy_thresh)
        self.merge_bright_thresh = float(merge_bright_thresh)

        self.min_group_len = int(min_group_len)
        self.min_group_time_span_ratio = float(min_group_time_span_ratio)

        self.nms_iou_thresh = float(nms_iou_thresh)
        self.use_cover_small_rule = bool(use_cover_small_rule)

        self.max_groups = int(max_groups)

        self.score_boxes_weight = float(score_boxes_weight)
        self.score_time_span_weight = float(score_time_span_weight)
        self.score_area_weight = float(score_area_weight)
        self.score_mean_contrast_weight = float(score_mean_contrast_weight)
        self.score_shape_std_weight = float(score_shape_std_weight)
        self.score_energy_std_weight = float(score_energy_std_weight)

    # ---------- basic utils ----------
    @staticmethod
    def _iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        if inter <= 0:
            return 0.0
        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        denom = areaA + areaB - inter + 1e-6
        return float(inter / denom)

    def _nms(self, boxes: np.ndarray) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(areas)[::-1]
        kept = []
        while len(order) > 0:
            i = order[0]
            kept.append(i)
            rest = []
            for j in order[1:]:
                iou_val = self._iou(boxes[i], boxes[j])
                if self.use_cover_small_rule:
                    xA = max(boxes[i][0], boxes[j][0])
                    yA = max(boxes[i][1], boxes[j][1])
                    xB = min(boxes[i][2], boxes[j][2])
                    yB = min(boxes[i][3], boxes[j][3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    area_j = max(0, boxes[j][2] - boxes[j][0]) * max(0, boxes[j][3] - boxes[j][1])
                    cover_small = inter / (area_j + 1e-6)
                    suppress = (iou_val > self.nms_iou_thresh) or (cover_small > 0.9)
                else:
                    suppress = iou_val > self.nms_iou_thresh
                if not suppress:
                    rest.append(j)
            order = np.array(rest, dtype=np.int64)
        return boxes[kept]

    # ---------- filtering ----------
    def _basic_filter(self, boxes: np.ndarray, img_h=None, img_w=None) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = x2 - x1
        heights = y2 - y1
        areas = widths * heights

        mask = (widths > 0) & (heights > 0)
        mask &= (areas >= self.min_area)
        mask &= (widths >= self.min_width)
        mask &= (heights >= self.min_height)

        if self.min_ratio > 0:
            mask &= ((widths / (heights + 1e-6)) >= self.min_ratio)
        if self.max_width > 0:
            mask &= (widths <= self.max_width)
        if self.max_height > 0:
            mask &= (heights <= self.max_height)
        if img_h is not None and self.exclude_bottom_ratio > 0:
            y_center = (y1 + y2) / 2.0
            max_valid_y = img_h * (1.0 - self.exclude_bottom_ratio)
            mask &= (y_center <= max_valid_y)
        return boxes[mask].astype(np.int32, copy=False)

    def _box_energy_stats(self, box, spectrogram) -> Optional[Dict[str, float]]:
        H, W = spectrogram.shape
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
            bg_mean = 0.0
            bg_std = 1.0
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

    def _filter_boxes_by_energy(self, boxes: np.ndarray, spectrogram: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        keep_boxes = []
        keep_stats = []
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
            keep_boxes.append(box)
            keep_stats.append(st)

        if len(keep_boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []
        return np.asarray(keep_boxes, dtype=np.int32), keep_stats

    # ---------- descriptors ----------
    def _build_descriptors(self, boxes: np.ndarray, spectrogram: np.ndarray) -> List[BoxDesc]:
        descs: List[BoxDesc] = []
        for box in boxes:
            st = self._box_energy_stats(box, spectrogram)
            if st is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            descs.append(
                BoxDesc(
                    box=np.asarray(box, dtype=np.int32),
                    cx=(x1 + x2) / 2.0,
                    cy=(y1 + y2) / 2.0,
                    w=w,
                    h=h,
                    area=w * h,
                    log_wh=float(np.log((w + 1e-6) / (h + 1e-6))),
                    mean=st["mean"],
                    maxv=st["max"],
                    var=st["var"],
                    bg_mean=st["bg_mean"],
                    bg_std=st["bg_std"],
                    contrast_z=st["contrast_z"],
                    integrated_energy=st["integrated_energy"],
                    bright_ratio=st["bright_ratio"],
                )
            )
        return descs

    # ---------- clustering / merging ----------
    def _cluster_by_frequency(self, descs: List[BoxDesc]) -> List[List[BoxDesc]]:
        if len(descs) == 0:
            return []
        freq_centers = np.array([[d.cy] for d in descs], dtype=np.float32)
        labels = DBSCAN(eps=self.freq_eps, min_samples=self.freq_min_samples).fit_predict(freq_centers)

        groups: Dict[int, List[BoxDesc]] = {}
        # with min_samples=1, labels should all be >=0. still guard noise labels.
        next_noise_id = max(int(labels.max()) + 1, 0) if labels.size > 0 else 0
        for d, lbl in zip(descs, labels):
            if lbl < 0:
                lbl = next_noise_id
                next_noise_id += 1
            groups.setdefault(int(lbl), []).append(d)

        out = []
        for _, g in groups.items():
            g_sorted = sorted(g, key=lambda x: x.cx)
            out.append(g_sorted)
        out.sort(key=lambda g: np.mean([d.cy for d in g]))
        return out

    @staticmethod
    def _group_summary(group: List[BoxDesc]) -> Dict[str, float]:
        cys = np.array([d.cy for d in group], dtype=np.float32)
        cxs = np.array([d.cx for d in group], dtype=np.float32)
        ws = np.array([d.w for d in group], dtype=np.float32)
        hs = np.array([d.h for d in group], dtype=np.float32)
        contrasts = np.array([d.contrast_z for d in group], dtype=np.float32)
        brights = np.array([d.bright_ratio for d in group], dtype=np.float32)
        return {
            "mean_freq": float(cys.mean()),
            "freq_span": float(cys.max() - cys.min()) if len(cys) > 1 else 0.0,
            "time_span": float(cxs.max() - cxs.min()) if len(cxs) > 1 else 0.0,
            "mean_w": float(ws.mean()),
            "mean_h": float(hs.mean()),
            "std_log_w": float(np.std(np.log(ws + 1e-6))),
            "std_log_h": float(np.std(np.log(hs + 1e-6))),
            "mean_contrast": float(contrasts.mean()),
            "std_contrast": float(contrasts.std()),
            "mean_bright": float(brights.mean()),
            "std_bright": float(brights.std()),
            "n_boxes": int(len(group)),
        }

    def _can_merge_groups(self, g1: List[BoxDesc], g2: List[BoxDesc]) -> bool:
        s1 = self._group_summary(g1)
        s2 = self._group_summary(g2)
        freq_ok = abs(s1["mean_freq"] - s2["mean_freq"]) <= self.merge_freq_thresh
        w_ok = abs(math.log((s1["mean_w"] + 1e-6) / (s2["mean_w"] + 1e-6))) <= self.merge_w_log_thresh
        h_ok = abs(math.log((s1["mean_h"] + 1e-6) / (s2["mean_h"] + 1e-6))) <= self.merge_h_log_thresh
        e_ok = abs(s1["mean_contrast"] - s2["mean_contrast"]) <= self.merge_energy_thresh
        b_ok = abs(s1["mean_bright"] - s2["mean_bright"]) <= self.merge_bright_thresh
        return bool(freq_ok and w_ok and h_ok and e_ok and b_ok)

    def _merge_similar_clusters(self, clusters: List[List[BoxDesc]]) -> List[List[BoxDesc]]:
        if len(clusters) <= 1:
            return clusters

        clusters = [sorted(c, key=lambda d: d.cx) for c in clusters]
        clusters.sort(key=lambda g: self._group_summary(g)["mean_freq"])

        merged = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = list(clusters[i])
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(clusters)):
                    if used[j]:
                        continue
                    if self._can_merge_groups(cur, clusters[j]):
                        cur.extend(clusters[j])
                        cur = sorted(cur, key=lambda d: d.cx)
                        used[j] = True
                        changed = True
            merged.append(cur)
        merged.sort(key=lambda g: self._group_summary(g)["mean_freq"])
        return merged

    # ---------- scoring / typing ----------
    def _score_group(self, group: List[BoxDesc], image_shape: Tuple[int, int]) -> float:
        if len(group) == 0:
            return -1e9
        H, W = image_shape
        s = self._group_summary(group)
        total_area_ratio = float(sum(d.area for d in group) / max(float(H * W), 1.0))
        time_span_ratio = float(s["time_span"] / max(float(W), 1.0))
        score = (
            self.score_boxes_weight * s["n_boxes"]
            + self.score_time_span_weight * time_span_ratio
            + self.score_area_weight * total_area_ratio
            + self.score_mean_contrast_weight * s["mean_contrast"]
            - self.score_shape_std_weight * (s["std_log_w"] + s["std_log_h"])
            - self.score_energy_std_weight * (s["std_contrast"] + s["std_bright"])
        )
        return float(score)

    @staticmethod
    def _infer_group_type(group: List[BoxDesc], image_shape: Tuple[int, int]) -> str:
        if len(group) == 0:
            return "unknown"
        H, W = image_shape
        s = MultiSignalDBSCANPreprocessor._group_summary(group)
        freq_span_ratio = s["freq_span"] / max(float(H), 1.0)
        shape_std = s["std_log_w"] + s["std_log_h"]
        energy_std = s["std_contrast"] + s["std_bright"]

        if freq_span_ratio < 0.08 and shape_std < 0.45:
            return "static_like"
        if freq_span_ratio < 0.18 and shape_std < 0.60 and energy_std < 1.8:
            return "static_like"
        if freq_span_ratio >= 0.18 and shape_std < 0.65 and energy_std < 1.8:
            return "hopping_like"
        return "unknown"

    # ---------- main API ----------
    def select_signal_groups(self, det_boxes, spectrogram: Optional[np.ndarray] = None) -> List[Dict]:
        if spectrogram is None:
            raise ValueError("select_signal_groups requires spectrogram for multi-signal inference.")
        H, W = spectrogram.shape[:2]

        boxes = self._basic_filter(det_boxes, img_h=H, img_w=W)
        if boxes.size == 0:
            return []

        energy_boxes, _ = self._filter_boxes_by_energy(boxes, spectrogram)
        if energy_boxes.size > 0:
            boxes = energy_boxes
        # else: fallback to geometry-only boxes

        descs = self._build_descriptors(boxes, spectrogram)
        if len(descs) == 0:
            return []

        clusters = self._cluster_by_frequency(descs)
        clusters = self._merge_similar_clusters(clusters)

        outputs: List[Dict] = []
        for group in clusters:
            if len(group) < self.min_group_len:
                continue

            s = self._group_summary(group)
            time_span_ratio = float(s["time_span"] / max(float(W), 1.0))
            if time_span_ratio < self.min_group_time_span_ratio:
                continue

            boxes_np = np.stack([d.box for d in group], axis=0).astype(np.int32)
            boxes_np = self._nms(boxes_np)
            if boxes_np.size == 0:
                continue

            # keep only descs whose boxes survived NMS
            kept = []
            kept_set = {tuple(map(int, b.tolist())) for b in boxes_np}
            for d in group:
                if tuple(map(int, d.box.tolist())) in kept_set:
                    kept.append(d)
            kept = sorted(kept, key=lambda d: d.cx)
            if len(kept) < self.min_group_len:
                continue

            s = self._group_summary(kept)
            score = self._score_group(kept, image_shape=(H, W))
            gtype = self._infer_group_type(kept, image_shape=(H, W))

            outputs.append({
                "boxes": np.stack([d.box for d in kept], axis=0).astype(np.int32),
                "score": score,
                "group_type": gtype,
                "n_boxes": int(s["n_boxes"]),
                "time_span_ratio": float(s["time_span"] / max(float(W), 1.0)),
                "freq_span_ratio": float(s["freq_span"] / max(float(H), 1.0)),
                "mean_w": float(s["mean_w"]),
                "mean_h": float(s["mean_h"]),
                "std_log_w": float(s["std_log_w"]),
                "std_log_h": float(s["std_log_h"]),
                "mean_contrast_z": float(s["mean_contrast"]),
                "std_contrast_z": float(s["std_contrast"]),
                "mean_bright_ratio": float(s["mean_bright"]),
                "std_bright_ratio": float(s["std_bright"]),
                "mean_freq_center_px": float(s["mean_freq"]),
            })

        outputs.sort(key=lambda g: g["score"], reverse=True)
        return outputs[: self.max_groups]
