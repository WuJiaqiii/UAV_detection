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
      2) pooled-ring energy filtering (with fallback to geometry-only boxes)
      3) DBSCAN on frequency center + bandwidth
      4) merge nearby clusters with similar shape/energy style
      5) lightweight NMS inside each group
      6) compute group statistics and quality scores
      7) keep all groups passing absolute + relative quality thresholds
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

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

                if iou_val > self.config.nms_thresh or cover_small > 0.9:
                    continue
                new_list.append(other)
            boxes_sorted = new_list
        return keep

    def _basic_filter(self, boxes):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)

        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = x2 - x1
        heights = y2 - y1
        areas = widths * heights

        mask = (widths > 0) & (heights > 0)
        mask &= (areas >= int(self.config.min_area))
        mask &= (widths >= int(self.config.min_width))
        mask &= (heights >= int(self.config.min_height))

        if float(self.config.min_ratio) > 0:
            mask &= ((widths / (heights + 1e-6)) >= float(self.config.min_ratio))
        if int(self.config.max_width) > 0:
            mask &= (widths <= int(self.config.max_width))
        if int(self.config.max_height) > 0:
            mask &= (heights <= int(self.config.max_height))

        return boxes[mask].astype(np.int32, copy=False)

    # ------------------------------------------------------------------
    # pooled-ring background / energy filtering
    # ------------------------------------------------------------------
    def _build_all_boxes_mask(self, boxes, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=bool)
        if boxes is None or len(boxes) == 0:
            return mask

        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = True
        return mask

    def _collect_pooled_ring_background(self, boxes, spectrogram):
        """
        Build one pooled background from ALL box ring regions.

        Important:
          - use ring region around each box
          - exclude the current box interior
          - exclude ALL detected box interiors from the pooled ring
        """
        if spectrogram is None or boxes is None or len(boxes) == 0:
            return None

        H, W = spectrogram.shape
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        all_boxes_mask = self._build_all_boxes_mask(boxes, (H, W))

        m = int(self.config.ring_margin)
        pooled_vals = []

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            rx1 = max(0, x1 - m)
            ry1 = max(0, y1 - m)
            rx2 = min(W, x2 + m)
            ry2 = min(H, y2 + m)

            ring = np.asarray(spectrogram[ry1:ry2, rx1:rx2], dtype=np.float32)
            if ring.size == 0:
                continue

            ring_mask = np.ones_like(ring, dtype=bool)

            # exclude current box interior
            ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False

            # exclude ALL box interiors
            ring_mask &= (~all_boxes_mask[ry1:ry2, rx1:rx2])

            vals = ring[ring_mask]
            if vals.size > 0:
                pooled_vals.append(vals.reshape(-1))

        if len(pooled_vals) == 0:
            return None

        pooled_bg = np.concatenate(pooled_vals, axis=0).astype(np.float32, copy=False)
        if pooled_bg.size == 0:
            return None

        bg_mean = float(np.mean(pooled_bg))
        bg_std = float(np.std(pooled_bg)) + 1e-6

        return {
            "bg_values": pooled_bg,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
        }

    def _box_energy_stats(self, box, spectrogram, pooled_bg=None):
        """
        Updated energy stats:
          - region mean/max/area still from the box itself
          - background is NO LONGER taken from local ring
          - background is taken from pooled ring statistics over ALL boxes
        """
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

        if pooled_bg is None:
            # conservative fallback: use robust full-image stats
            whole = np.asarray(spectrogram, dtype=np.float32)
            bg_mean = float(np.mean(whole))
            bg_std = float(np.std(whole)) + 1e-6
        else:
            bg_mean = float(pooled_bg["bg_mean"])
            bg_std = float(pooled_bg["bg_std"])

        contrast_z = (mean_val - bg_mean) / bg_std

        return {
            "mean": mean_val,
            "max": max_val,
            "area": area,
            "bg_mean": bg_mean,
            "bg_std": bg_std,
            "contrast_z": contrast_z,
        }

    def _filter_boxes_by_energy(self, boxes, spectrogram):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []
        if spectrogram is None:
            boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
            return boxes, [None] * len(boxes)

        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

        pooled_bg = self._collect_pooled_ring_background(boxes, spectrogram)
        if pooled_bg is None:
            self.logger.warning("[Preprocess] pooled ring background is empty, fallback to full-image background stats.")

        kept_boxes = []
        kept_stats = []
        for box in boxes:
            st = self._box_energy_stats(box, spectrogram, pooled_bg=pooled_bg)
            if st is None:
                continue
            if st["contrast_z"] < float(self.config.min_contrast_z):
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
        if img_h <= 1:
            return 0.0
        rel = float(y) / float(img_h - 1) - 0.5
        return rel * float(self.config.sampling_rate) / 1e6

    def _pixel_bandwidth_to_mhz(self, height_px, img_h):
        if img_h <= 1:
            return 0.0
        return float(abs(height_px) * (self.config.sampling_rate / float(img_h - 1)) / 1e6)

    # ------------------------------------------------------------------
    # cluster/group construction
    # ------------------------------------------------------------------
    def _cluster_boxes_by_frequency(self, boxes, img_h):
        if boxes is None or len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        if boxes.shape[0] == 1:
            return [np.array([0], dtype=np.int32)]

        freq_centers = np.array(
            [self._pixel_y_to_freq_mhz((b[1] + b[3]) / 2.0, img_h) for b in boxes],
            dtype=np.float32
        )
        bandwidths = np.array(
            [self._pixel_bandwidth_to_mhz((b[3] - b[1]), img_h) for b in boxes],
            dtype=np.float32
        )

        bw_weight = float(getattr(self.config, "freq_bw_weight", 1.0))

        features = np.stack(
            [freq_centers, bandwidths * bw_weight],
            axis=1
        )

        labels = DBSCAN(
            eps=float(self.config.freq_eps),
            min_samples=int(self.config.freq_min_samples)
        ).fit_predict(features)

        if np.all(labels < 0):
            return [np.array([i], dtype=np.int32) for i in range(boxes.shape[0])]

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
            mean_contrast = float(np.mean(contrast))
            std_contrast = float(np.std(contrast))
        else:
            mean_contrast = 0.0
            std_contrast = 0.0

        score = (
            float(self.config.score_n_boxes_weight) * float(n)
            + float(self.config.score_time_span_weight) * time_span_ratio
            + float(self.config.score_contrast_weight) * mean_contrast
            - float(self.config.score_contrast_std_weight) * std_contrast
            - float(self.config.score_w_std_weight) * std_log_w
            - float(self.config.score_h_std_weight) * std_log_h
        )

        return {
            "boxes": boxes.astype(np.int32, copy=False),
            "stats": stats,
            "n_boxes": int(n),
            "time_span_ratio": time_span_ratio,
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
            "score": float(score),
        }

    def _should_merge_groups(self, g1, g2):
        if g1 is None or g2 is None:
            return False
        if abs(g1["center_freq"] - g2["center_freq"]) > float(self.config.merge_freq_thresh):
            return False
        # width consistency deliberately relaxed
        if abs(np.log((g1["mean_h"] + 1e-6) / (g2["mean_h"] + 1e-6))) > float(self.config.merge_h_log_thresh):
            return False
        if abs(g1["mean_contrast_z"] - g2["mean_contrast_z"]) > float(self.config.merge_energy_thresh):
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

    def _group_passes_thresholds(self, g):
        if g["n_boxes"] < int(self.config.min_group_len):
            return False
        if g["time_span_ratio"] < float(self.config.min_group_time_span_ratio):
            return False
        return True

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
                ...
              },
              ...
            ]
        """
        H, W = spectrogram.shape

        if det_boxes is None or len(det_boxes) == 0:
            return []

        # basic shape filter
        boxes = self._basic_filter(det_boxes)
        if boxes.size == 0:
            return []

        # pooled-ring energy filter
        if spectrogram is not None:
            e_boxes, e_stats = self._filter_boxes_by_energy(boxes, spectrogram)
            if e_boxes.size > 0:
                boxes, stats = e_boxes, e_stats
            else:
                # fallback to geometry-filtered boxes to avoid killing weak but real signals
                boxes = boxes.astype(np.int32, copy=False)
                pooled_bg = self._collect_pooled_ring_background(boxes, spectrogram)
                stats = [self._box_energy_stats(b, spectrogram, pooled_bg=pooled_bg) for b in boxes]
        else:
            stats = [None] * len(boxes)

        if boxes.size == 0:
            return []

        if boxes.shape[0] == 1:
            g = self._build_group_stats(boxes, stats, spectrogram.shape if spectrogram is not None else (1, 1))
            if g is None:
                return []
            return [g] if self._group_passes_thresholds(g) else []

        # cluster by frequency + bandwidth
        clusters = self._cluster_boxes_by_frequency(boxes, H)
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

        final_groups = []
        for g in cleaned:
            if self._group_passes_thresholds(g):
                final_groups.append(g)

        final_groups = sorted(final_groups, key=lambda x: x["score"], reverse=True)
        return final_groups
