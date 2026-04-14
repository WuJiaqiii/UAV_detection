import numpy as np
from sklearn.cluster import DBSCAN


class SignalPreprocessor:
    """
    Multi-signal training-oriented preprocessor.

    Goal:
      - do NOT choose a single best cluster
      - generate multiple candidate signal groups from YOLO boxes
      - keep all groups that pass quality thresholds
      - provide per-group frequency/statistics for trainer-side matching

    Main pipeline:
      1) basic geometry filtering
      2) local energy filtering (with fallback to geometry-only boxes)
      3) DBSCAN on frequency centers (pixel y center)
      4) merge nearby clusters with similar shape/energy style
      5) lightweight NMS inside each group
      6) compute group statistics and quality scores
      7) keep all groups passing absolute + relative quality thresholds
    """

    def __init__(
        self,
        sampling_rate,
        n_fft,
        hop_length,
        # basic box filtering
        min_area=20,
        min_ratio=0.0,
        min_width=2,
        min_height=2,
        max_width=0,
        max_height=0,
        exclude_bottom_ratio=0.0,
        # local energy filtering
        ring_margin=5,
        min_contrast_z=0.6,
        min_integrated_energy=8.0,
        min_bright_ratio=0.02,
        bright_z_thresh=1.5,
        # cluster / group construction
        freq_eps=12.0,
        freq_min_samples=1,
        nms_iou_thresh=0.6,
        # merge nearby clusters
        merge_freq_thresh=10.0,          # MHz
        merge_w_log_thresh=0.35,
        merge_h_log_thresh=0.35,
        merge_energy_thresh=1.0,
        merge_bright_thresh=0.12,
        # group quality thresholds
        min_group_len=2,
        min_group_time_span_ratio=0.01,
        min_group_contrast=0.0,
        min_group_bright=0.0,
        max_group_w_std=0.60,
        max_group_h_std=0.60,
        max_group_contrast_std=2.50,
        score_abs_thresh=0.0,
        score_rel_thresh=0.25,
        # score weights
        score_n_boxes_weight=0.80,
        score_time_span_weight=2.00,
        score_contrast_weight=0.60,
        score_bright_weight=1.20,
        score_w_std_weight=0.50,
        score_h_std_weight=0.50,
        score_contrast_std_weight=0.25,
    ):
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
        self.nms_thresh = float(nms_iou_thresh)

        self.merge_freq_thresh = float(merge_freq_thresh)
        self.merge_w_log_thresh = float(merge_w_log_thresh)
        self.merge_h_log_thresh = float(merge_h_log_thresh)
        self.merge_energy_thresh = float(merge_energy_thresh)
        self.merge_bright_thresh = float(merge_bright_thresh)

        self.min_group_len = int(min_group_len)
        self.min_group_time_span_ratio = float(min_group_time_span_ratio)
        self.min_group_contrast = float(min_group_contrast)
        self.min_group_bright = float(min_group_bright)
        self.max_group_w_std = float(max_group_w_std)
        self.max_group_h_std = float(max_group_h_std)
        self.max_group_contrast_std = float(max_group_contrast_std)
        self.score_abs_thresh = float(score_abs_thresh)
        self.score_rel_thresh = float(score_rel_thresh)

        self.score_n_boxes_weight = float(score_n_boxes_weight)
        self.score_time_span_weight = float(score_time_span_weight)
        self.score_contrast_weight = float(score_contrast_weight)
        self.score_bright_weight = float(score_bright_weight)
        self.score_w_std_weight = float(score_w_std_weight)
        self.score_h_std_weight = float(score_h_std_weight)
        self.score_contrast_std_weight = float(score_contrast_std_weight)

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def _iou(self, boxA, boxB):
        x1, y1, x2, y2 = boxA
        x1b, y1b, x2b, y2b = boxB
        inter_w = max(0, min(x2, x2b) - max(x1, x1b))
        inter_h = max(0, min(y2, y2b) - max(y1, y1b))
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        areaA = max(0, x2 - x1) * max(0, y2 - y1)
        areaB = max(0, x2b - x1b) * max(0, y2b - y1b)
        denom = float(areaA + areaB - inter_area)
        return float(inter_area / denom) if denom > 0 else 0.0

    def _nms(self, boxes):
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        keep = []
        while boxes_sorted:
            box = boxes_sorted.pop(0)
            keep.append(box)
            new_list = []
            x1, y1, x2, y2 = box
            area_box = max(0, x2 - x1) * max(0, y2 - y1)

            for other in boxes_sorted:
                x1b, y1b, x2b, y2b = other
                inter_w = max(0, min(x2, x2b) - max(x1, x1b))
                inter_h = max(0, min(y2, y2b) - max(y1, y1b))
                inter_area = inter_w * inter_h
                if inter_area <= 0:
                    new_list.append(other)
                    continue

                area_b = max(0, x2b - x1b) * max(0, y2b - y1b)
                denom = float(area_box + area_b - inter_area)
                iou_val = float(inter_area / denom) if denom > 0 else 0.0
                cover_small = float(inter_area / float(min(area_box, area_b))) if min(area_box, area_b) > 0 else 0.0

                if iou_val > self.nms_thresh or cover_small > 0.9:
                    continue
                new_list.append(other)
            boxes_sorted = new_list
        return keep

    def _basic_filter(self, boxes, img_h=None, img_w=None):
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

    def _box_energy_stats(self, box, spectrogram):
        if spectrogram is None:
            return None

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
        region = np.asarray(region, dtype=np.float32)

        mean_val = float(np.mean(region))
        max_val = float(np.max(region))
        area = float(region.size)

        m = self.ring_margin
        rx1 = max(0, x1 - m)
        ry1 = max(0, y1 - m)
        rx2 = min(W, x2 + m)
        ry2 = min(H, y2 + m)
        ring = np.asarray(spectrogram[ry1:ry2, rx1:rx2], dtype=np.float32)
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
        bright_thr = bg_mean + self.bright_z_thresh * bg_std
        bright_ratio = float(np.mean(region > bright_thr))

        return {
            "mean": mean_val,
            "max": max_val,
            "area": area,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "contrast_z": contrast_z,
            "integrated_energy": integrated_energy,
            "bright_ratio": bright_ratio,
        }

    def _filter_boxes_by_energy(self, boxes, spectrogram):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []
        if spectrogram is None:
            boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
            return boxes, [None] * len(boxes)

        kept_boxes = []
        kept_stats = []
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
            kept_boxes.append(list(map(int, box)))
            kept_stats.append(st)

        if len(kept_boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []
        return np.asarray(kept_boxes, dtype=np.int32), kept_stats

    # ------------------------------------------------------------------
    # frequency mapping
    # ------------------------------------------------------------------
    def _pixel_y_to_freq_mhz(self, y, img_h):
        """
        Map row index to frequency in MHz.
        Assumes vertically shifted spectrum with image center ≈ 0 MHz.
        Top corresponds to +sr/2, bottom to -sr/2.
        """
        if img_h <= 1:
            return 0.0
        y = float(y)
        frac = 1.0 - (y / float(img_h - 1))  # top -> 1, bottom -> 0
        freq_hz = (frac - 0.5) * self.sr
        return float(freq_hz / 1e6)

    def _pixel_bandwidth_to_mhz(self, height_px, img_h):
        if img_h <= 1:
            return 0.0
        return float(abs(height_px) * (self.sr / float(img_h - 1)) / 1e6)

    # ------------------------------------------------------------------
    # cluster/group construction
    # ------------------------------------------------------------------
    def _cluster_boxes_by_frequency(self, boxes):
        if boxes is None or len(boxes) == 0:
            return []
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        if boxes.shape[0] == 1:
            return [np.array([0], dtype=np.int32)]

        freq_center = ((boxes[:, 1] + boxes[:, 3]) / 2.0).reshape(-1, 1)
        labels = DBSCAN(eps=self.freq_eps, min_samples=self.freq_min_samples).fit_predict(freq_center)

        if np.all(labels < 0):
            return [np.arange(boxes.shape[0], dtype=np.int32)]

        clusters = []
        for lbl in sorted(set(labels.tolist())):
            if lbl < 0:
                continue
            idx = np.where(labels == lbl)[0]
            if idx.size > 0:
                clusters.append(idx.astype(np.int32))

        # keep DBSCAN noise as singleton clusters rather than dropping them
        noise_idx = np.where(labels < 0)[0]
        for i in noise_idx.tolist():
            clusters.append(np.array([i], dtype=np.int32))

        return clusters

    def _build_group_stats(self, boxes, stats, spectrogram_shape):
        H, W = spectrogram_shape[:2]
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        n = boxes.shape[0]
        if n == 0:
            return None

        x1 = boxes[:, 0].astype(np.float32)
        y1 = boxes[:, 1].astype(np.float32)
        x2 = boxes[:, 2].astype(np.float32)
        y2 = boxes[:, 3].astype(np.float32)

        w = np.maximum(x2 - x1, 1.0)
        h = np.maximum(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = w * h

        log_w = np.log(w + 1e-6)
        log_h = np.log(h + 1e-6)

        time_span_ratio = float((cx.max() - cx.min()) / max(float(W), 1.0))
        freq_span_ratio = float((cy.max() - cy.min()) / max(float(H), 1.0))

        # frequency stats in MHz for trainer-side matching
        freq_centers_mhz = np.array([self._pixel_y_to_freq_mhz(v, H) for v in cy], dtype=np.float32)
        group_center_freq = float(np.mean(freq_centers_mhz))
        group_freq_min = float(np.min(freq_centers_mhz))
        group_freq_max = float(np.max(freq_centers_mhz))
        group_bandwidth = 2.0 * max(abs(group_freq_max - group_center_freq), abs(group_center_freq - group_freq_min))
        if group_bandwidth <= 0:
            # fallback to average box height mapped to MHz
            group_bandwidth = float(np.mean([self._pixel_bandwidth_to_mhz(v, H) for v in h]))

        mean_w = float(np.mean(w))
        mean_h = float(np.mean(h))
        std_log_w = float(np.std(log_w))
        std_log_h = float(np.std(log_h))
        total_area = float(np.sum(area))

        if stats is not None and len(stats) == n and stats[0] is not None:
            contrast = np.array([float(s["contrast_z"]) for s in stats], dtype=np.float32)
            bright = np.array([float(s["bright_ratio"]) for s in stats], dtype=np.float32)
            integrated = np.array([float(s["integrated_energy"]) for s in stats], dtype=np.float32)
            mean_contrast = float(np.mean(contrast))
            std_contrast = float(np.std(contrast))
            mean_bright = float(np.mean(bright))
            std_bright = float(np.std(bright))
            mean_integrated = float(np.mean(integrated))
        else:
            mean_contrast = 0.0
            std_contrast = 0.0
            mean_bright = 0.0
            std_bright = 0.0
            mean_integrated = 0.0

        score = (
            self.score_n_boxes_weight * float(n)
            + self.score_time_span_weight * time_span_ratio
            + self.score_contrast_weight * mean_contrast
            + self.score_bright_weight * mean_bright
            - self.score_w_std_weight * std_log_w
            - self.score_h_std_weight * std_log_h
            - self.score_contrast_std_weight * std_contrast
        )

        return {
            "boxes": boxes.astype(np.int32, copy=False),
            "stats": stats,
            "n_boxes": int(n),
            "time_span_ratio": time_span_ratio,
            "freq_span_ratio": freq_span_ratio,
            "center_freq": group_center_freq,
            "freq_min": group_freq_min,
            "freq_max": group_freq_max,
            "bandwidth": float(group_bandwidth),
            "mean_w": mean_w,
            "mean_h": mean_h,
            "std_log_w": std_log_w,
            "std_log_h": std_log_h,
            "total_area": total_area,
            "mean_contrast_z": mean_contrast,
            "std_contrast_z": std_contrast,
            "mean_bright_ratio": mean_bright,
            "std_bright_ratio": std_bright,
            "mean_integrated_energy": mean_integrated,
            "score": float(score),
        }

    def _should_merge_groups(self, g1, g2):
        if g1 is None or g2 is None:
            return False
        if abs(g1["center_freq"] - g2["center_freq"]) > self.merge_freq_thresh:
            return False
        if abs(np.log((g1["mean_w"] + 1e-6) / (g2["mean_w"] + 1e-6))) > self.merge_w_log_thresh:
            return False
        if abs(np.log((g1["mean_h"] + 1e-6) / (g2["mean_h"] + 1e-6))) > self.merge_h_log_thresh:
            return False
        if abs(g1["mean_contrast_z"] - g2["mean_contrast_z"]) > self.merge_energy_thresh:
            return False
        if abs(g1["mean_bright_ratio"] - g2["mean_bright_ratio"]) > self.merge_bright_thresh:
            return False
        return True

    def _merge_groups(self, groups, spectrogram_shape):
        if not groups:
            return []
        groups = sorted(groups, key=lambda g: g["center_freq"])

        merged_any = True
        while merged_any and len(groups) > 1:
            merged_any = False
            new_groups = []
            i = 0
            while i < len(groups):
                if i < len(groups) - 1 and self._should_merge_groups(groups[i], groups[i + 1]):
                    boxes = np.concatenate([groups[i]["boxes"], groups[i + 1]["boxes"]], axis=0)
                    stats = []
                    if groups[i]["stats"] is not None:
                        stats.extend(groups[i]["stats"])
                    if groups[i + 1]["stats"] is not None:
                        stats.extend(groups[i + 1]["stats"])
                    merged = self._build_group_stats(boxes, stats, spectrogram_shape)
                    new_groups.append(merged)
                    i += 2
                    merged_any = True
                else:
                    new_groups.append(groups[i])
                    i += 1
            groups = sorted(new_groups, key=lambda g: g["center_freq"])
        return groups

    def _group_passes_thresholds(self, g, best_score):
        if g["n_boxes"] < self.min_group_len:
            return False
        if g["time_span_ratio"] < self.min_group_time_span_ratio:
            return False
        if g["mean_contrast_z"] < self.min_group_contrast:
            return False
        if g["mean_bright_ratio"] < self.min_group_bright:
            return False
        if g["std_log_w"] > self.max_group_w_std:
            return False
        if g["std_log_h"] > self.max_group_h_std:
            return False
        if g["std_contrast_z"] > self.max_group_contrast_std:
            return False
        if g["score"] < self.score_abs_thresh:
            return False
        if best_score > 0 and g["score"] < self.score_rel_thresh * best_score:
            return False
        return True

    def _infer_group_type(self, g):
        # very lightweight heuristic for debugging / logging only
        if g["freq_span_ratio"] <= 0.08 and g["std_log_w"] <= 0.35 and g["std_log_h"] <= 0.35:
            return "static_like"
        if g["freq_span_ratio"] > 0.08 and g["std_log_w"] <= 0.45 and g["std_log_h"] <= 0.45:
            return "hopping_like"
        return "unknown"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def select_signal_groups(self, det_boxes, spectrogram=None):
        """
        Return all groups that pass quality thresholds.

        Output format:
            [
              {
                "boxes": np.ndarray [N,4],
                "score": float,
                "center_freq": float,    # MHz
                "bandwidth": float,      # MHz
                "n_boxes": int,
                "time_span_ratio": float,
                "freq_span_ratio": float,
                ...
              },
              ...
            ]
        """
        if det_boxes is None or len(det_boxes) == 0:
            return []

        H = spectrogram.shape[0] if spectrogram is not None else None
        W = spectrogram.shape[1] if spectrogram is not None else None

        boxes = self._basic_filter(det_boxes, img_h=H, img_w=W)
        if boxes.size == 0:
            return []

        if spectrogram is not None:
            e_boxes, e_stats = self._filter_boxes_by_energy(boxes, spectrogram)
            if e_boxes.size > 0:
                boxes, stats = e_boxes, e_stats
            else:
                # fallback to geometry-filtered boxes to avoid killing weak but real signals
                boxes = boxes.astype(np.int32, copy=False)
                stats = [self._box_energy_stats(b, spectrogram) for b in boxes]
        else:
            stats = [None] * len(boxes)

        if boxes.size == 0:
            return []

        if boxes.shape[0] == 1:
            g = self._build_group_stats(boxes, stats, spectrogram.shape if spectrogram is not None else (1, 1))
            if g is None:
                return []
            if g["n_boxes"] == 1 and self.min_group_len > 1:
                return []
            g["group_type"] = self._infer_group_type(g)
            return [g] if self._group_passes_thresholds(g, g["score"]) else []

        clusters = self._cluster_boxes_by_frequency(boxes)
        raw_groups = []
        for idx in clusters:
            c_boxes = boxes[idx]
            c_stats = [stats[int(i)] for i in idx.tolist()] if stats is not None else None
            # sort by time first
            time_centers = (c_boxes[:, 0] + c_boxes[:, 2]) / 2.0
            order = np.argsort(time_centers)
            c_boxes = c_boxes[order]
            if c_stats is not None:
                c_stats = [c_stats[int(i)] for i in order.tolist()]
            g = self._build_group_stats(c_boxes, c_stats, spectrogram.shape if spectrogram is not None else (1, 1))
            if g is not None:
                raw_groups.append(g)

        if not raw_groups:
            return []

        groups = self._merge_groups(raw_groups, spectrogram.shape if spectrogram is not None else (1, 1))

        # lightweight NMS inside each group, then recompute stats
        cleaned = []
        for g in groups:
            boxes_list = [list(map(int, b)) for b in g["boxes"].tolist()]
            boxes_list = self._nms(boxes_list)
            if len(boxes_list) == 0:
                continue
            boxes_arr = np.asarray(boxes_list, dtype=np.int32).reshape(-1, 4)
            time_centers = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
            order = np.argsort(time_centers)
            boxes_arr = boxes_arr[order]

            # re-link stats to kept boxes using exact box tuples
            stat_map = {}
            if g["stats"] is not None:
                for b, st in zip(g["boxes"].tolist(), g["stats"]):
                    stat_map[tuple(map(int, b))] = st
                stats_kept = [stat_map.get(tuple(map(int, b)), None) for b in boxes_arr.tolist()]
            else:
                stats_kept = None

            gg = self._build_group_stats(boxes_arr, stats_kept, spectrogram.shape if spectrogram is not None else (1, 1))
            if gg is not None:
                cleaned.append(gg)

        if not cleaned:
            return []

        best_score = max(g["score"] for g in cleaned)
        final_groups = []
        for g in cleaned:
            if self._group_passes_thresholds(g, best_score):
                g["group_type"] = self._infer_group_type(g)
                final_groups.append(g)

        final_groups = sorted(final_groups, key=lambda x: x["score"], reverse=True)
        return final_groups
