import numpy as np
from sklearn.cluster import DBSCAN
from util.spectrogram_frontend import pixel_y_to_freq_mhz, pixel_height_to_bandwidth_mhz


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
      4) merge nearby clusters, or associate hopping bursts by timing and shape
      5) lightweight NMS inside each group
      6) compute group statistics and quality scores
      7) apply final size and temporal quality thresholds
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.grouping_mode = str(getattr(config, "grouping_mode", "frequency"))
        if self.grouping_mode not in {"frequency", "hopping"}:
            raise ValueError("grouping_mode must be frequency or hopping")
        self.hop_association = str(getattr(config, "hop_association", "shape"))
        if self.hop_association not in {"shape", "periodic"}:
            raise ValueError("hop_association must be shape or periodic")
        for name, default in {
            "hop_shape_log_tol": 0.35, "hop_contrast_delta": 3.0,
            "hop_period_rel_tol": 0.15, "hop_max_overlap_ratio": 0.2,
            "hop_min_timing_consistency": 0.85,
            "hop_max_conflict_fraction": 0.2,
            "hop_min_time_extent_ratio": 0.15, "hop_min_time_coverage_ratio": 0.02,
        }.items():
            value = float(getattr(config, name, default))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name not in {"hop_shape_log_tol", "hop_contrast_delta"} and value > 1:
                raise ValueError(f"{name} must be <= 1")
        if not 0 < float(getattr(config, "hop_period_rel_tol", 0.15)) < 0.5:
            raise ValueError("hop_period_rel_tol must be in (0, 0.5)")
        if int(getattr(config, "hop_max_missing_pulses", 2)) < 0:
            raise ValueError("hop_max_missing_pulses must be nonnegative")
        if int(getattr(config, "hop_min_pulses", 3)) < 3:
            raise ValueError("hop_min_pulses must be >= 3")
        self.last_diagnostics = {}

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
        return pixel_y_to_freq_mhz(y, img_h, self.config.sampling_rate,
                                   getattr(self.config, "frequency_axis", "legacy"))

    def _pixel_bandwidth_to_mhz(self, height_px, img_h):
        return pixel_height_to_bandwidth_mhz(height_px, img_h, self.config.sampling_rate,
                                             getattr(self.config, "frequency_axis", "legacy"))

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
    
    def _covered_x_ratio(self, boxes, img_w):
        if boxes is None or len(boxes) == 0:
            return 0.0

        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        intervals = []
        for b in boxes:
            x1, _, x2, _ = [int(v) for v in b]
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w))
            if x2 <= x1:
                continue
            intervals.append((x1, x2))

        if len(intervals) == 0:
            return 0.0

        intervals.sort(key=lambda t: t[0])

        merged = []
        cur_l, cur_r = intervals[0]
        for l, r in intervals[1:]:
            if l <= cur_r:
                cur_r = max(cur_r, r)
            else:
                merged.append((cur_l, cur_r))
                cur_l, cur_r = l, r
        merged.append((cur_l, cur_r))

        covered = float(sum(r - l for l, r in merged))
        return covered / max(float(img_w), 1.0)

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

        # time_span_ratio = float((cx.max() - cx.min()) / max(float(W), 1.0))
        time_span_ratio = self._covered_x_ratio(boxes, W)
        time_extent_ratio = float((x2.max() - x1.min()) / max(float(W), 1.0))

        # frequency stats in MHz for trainer-side matching
        freq_centers_mhz = np.array([self._pixel_y_to_freq_mhz(v, H) for v in cy], dtype=np.float32)
        group_center_freq = float(np.mean(freq_centers_mhz))
        group_freq_min = float(np.min(freq_centers_mhz))
        group_freq_max = float(np.max(freq_centers_mhz))
        group_bandwidth = 2.0 * max(abs(group_freq_max - group_center_freq), abs(group_center_freq - group_freq_min))
        if group_bandwidth <= 0:
            # fallback to average box height mapped to MHz
            group_bandwidth = float(np.mean([self._pixel_bandwidth_to_mhz(v, H) for v in h]))
        freq_edge_min = self._pixel_y_to_freq_mhz(float(y1.min()), H)
        freq_edge_max = self._pixel_y_to_freq_mhz(float(y2.max()), H)

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
            # Keep the legacy alias above unchanged for frequency-mode experiments.
            "time_coverage_ratio": time_span_ratio,
            "time_extent_ratio": time_extent_ratio,
            "center_freq": group_center_freq,
            "freq_min": group_freq_min,
            "freq_max": group_freq_max,
            "bandwidth": float(group_bandwidth),
            "freq_edge_min": float(freq_edge_min),
            "freq_edge_max": float(freq_edge_max),
            "mean_w": mean_w,
            "mean_h": mean_h,
            "std_log_w": std_log_w,
            "std_log_h": std_log_h,
            "total_area": total_area,
            "mean_contrast_z": mean_contrast,
            "std_contrast_z": std_contrast,
            "score": float(score),
            "group_type": "frequency",
            "frequency_cluster_count": 1,
        }

    def _pulse_period(self, boxes):
        """Estimate burst cadence from repeated short within-group intervals.

        At least two direct intervals must support the estimate. This does not
        infer unobserved subharmonics when each carrier has only return intervals.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        centers = np.sort((boxes[:, 0] + boxes[:, 2]) / 2.0)
        gaps = np.diff(centers)
        widths = boxes[:, 2] - boxes[:, 0]
        gaps = gaps[gaps >= max(1.0, 0.5 * float(np.median(widths)))]
        if len(gaps) < 2:
            return None
        short = gaps[gaps <= np.median(gaps) * 1.25]
        period = float(np.median(short))
        tolerance = float(getattr(self.config, "hop_period_rel_tol", 0.15))
        direct = np.abs(gaps - period) <= tolerance * period
        return period if np.count_nonzero(direct) >= 2 else None

    def _hopping_components(self, group):
        return group.get("hop_components") or [{
            "width": float(np.median(group["boxes"][:, 2] - group["boxes"][:, 0])),
            "height": float(np.median(group["boxes"][:, 3] - group["boxes"][:, 1])),
            "contrast": group["mean_contrast_z"],
            "period": self._pulse_period(group["boxes"]),
        }]

    def _reject_hop(self, reason):
        self._hop_rejection_reason = reason
        return None

    def _time_conflict_fraction(self, boxes):
        """Fraction of pulses involved in a substantial time overlap.

        Check all pairs in the union, so adding more carriers cannot hide a
        conflict inside an already merged group. Only temporal overlap matters;
        spatial duplicates have already been handled by per-carrier NMS.
        """
        boxes = np.asarray(boxes)
        conflicted = np.zeros(len(boxes), dtype=bool)
        order = np.argsort(boxes[:, 0], kind="stable")
        widths = boxes[:, 2] - boxes[:, 0]
        threshold = float(getattr(self.config, "hop_max_overlap_ratio", 0.2))
        for pos, i in enumerate(order):
            for j in order[pos + 1:]:
                if boxes[j, 0] >= boxes[i, 2]:
                    break
                overlap = min(boxes[i, 2], boxes[j, 2]) - boxes[j, 0]
                if overlap > threshold * min(widths[i], widths[j]):
                    conflicted[i] = conflicted[j] = True
        return float(np.mean(conflicted)) if len(boxes) else 0.0

    def _shape_hopping_evidence(self, first, second):
        """Build shape-compatible unions; enforce pulse count after association.

        A carrier may have only one observed pulse. Requiring the final minimum
        here would prevent two singleton seeds from ever growing into a train.
        """
        boxes = np.concatenate([first["boxes"], second["boxes"]])
        components = self._hopping_components(first) + self._hopping_components(second)
        distances = []
        for key in ("width", "height"):
            sizes = [component[key] for component in components]
            distances.append(float(np.log((max(sizes) + 1e-6) / (min(sizes) + 1e-6))))
        aspects = [component["width"] / component["height"] for component in components]
        distances.append(float(np.log(max(aspects) / min(aspects))))
        shape_distance = max(distances)
        if shape_distance > float(getattr(self.config, "hop_shape_log_tol", 0.35)):
            return self._reject_hop("incompatible_shape")
        contrasts = [component["contrast"] for component in components]
        contrast_delta = float(max(contrasts) - min(contrasts))
        if (bool(getattr(self.config, "hop_use_contrast_gate", False)) and
                contrast_delta > float(getattr(self.config, "hop_contrast_delta", 3.0))):
            return self._reject_hop("contrast_difference")
        conflict_fraction = self._time_conflict_fraction(boxes)
        if conflict_fraction > float(getattr(self.config, "hop_max_conflict_fraction", 0.2)):
            return self._reject_hop("too_many_time_conflicts")

        # Timing is diagnostic/tie-breaking evidence, never a shape-mode veto.
        periods = [component["period"] for component in components if component["period"] is not None]
        period, consistency = None, None
        tolerance = float(getattr(self.config, "hop_period_rel_tol", 0.15))
        if periods and max(periods) - min(periods) <= tolerance * min(periods):
            period = float(np.median(periods))
            centers = np.sort((boxes[:, 0] + boxes[:, 2]) / 2.0)
            gaps = np.diff(centers)
            multiples = np.rint(gaps / period)
            consistency = float(np.mean(
                (multiples >= 1) & (np.abs(gaps - multiples * period) <= tolerance * period)
            ))
        return {
            "hop_association": "shape", "hop_components": components,
            "hop_shape_distance": shape_distance, "hop_contrast_difference": contrast_delta,
            "hop_time_conflict_fraction": conflict_fraction,
            "hop_period_px": period, "hop_timing_consistency": consistency,
        }

    def _hopping_evidence(self, first, second):
        """Evaluate the entire proposed union under the selected association policy."""
        self._hop_rejection_reason = None
        if self.hop_association == "shape":
            return self._shape_hopping_evidence(first, second)
        return self._periodic_hopping_evidence(first, second)

    def _periodic_hopping_evidence(self, first, second):
        """Legacy strict cadence policy, retained for comparison/multi-source data."""
        a, b = first["boxes"], second["boxes"]
        if len(a) + len(b) < int(getattr(self.config, "hop_min_pulses", 3)):
            return self._reject_hop("insufficient_pulses")
        components = self._hopping_components(first) + self._hopping_components(second)
        shape_tol = float(getattr(self.config, "hop_shape_log_tol", 0.35))
        for key in ("width", "height"):
            sizes = [component[key] for component in components]
            if np.log((max(sizes) + 1e-6) / (min(sizes) + 1e-6)) > shape_tol:
                return self._reject_hop("incompatible_shape")
        contrasts = [component["contrast"] for component in components]
        if max(contrasts) - min(contrasts) > float(getattr(self.config, "hop_contrast_delta", 3.0)):
            return self._reject_hop("contrast_difference")
        periods = [component["period"] for component in components]
        if any(period is None for period in periods):
            return self._reject_hop("missing_period_evidence")
        tolerance = float(getattr(self.config, "hop_period_rel_tol", 0.15))
        if max(periods) - min(periods) > tolerance * min(periods):
            return self._reject_hop("different_periods")
        period = float(np.median(periods))
        boxes = np.concatenate([a, b])
        sources = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
        centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
        order = np.argsort(centers, kind="stable")
        boxes, sources, centers = boxes[order], sources[order], centers[order]
        # Simultaneous pulses at different frequencies contradict a single track.
        overlaps = boxes[:-1, 2] - boxes[1:, 0]
        shorter = np.minimum(boxes[:-1, 2] - boxes[:-1, 0], boxes[1:, 2] - boxes[1:, 0])
        if np.any(overlaps > float(getattr(self.config, "hop_max_overlap_ratio", 0.2)) * shorter):
            return self._reject_hop("time_overlap")
        gaps = np.diff(centers)
        multiples = np.rint(gaps / period)
        valid = (
            (multiples >= 1)
            & (multiples <= int(getattr(self.config, "hop_max_missing_pulses", 2)) + 1)
            & (np.abs(gaps - multiples * period) <= tolerance * period)
        )
        consistency = float(np.mean(valid))
        if consistency < float(getattr(self.config, "hop_min_timing_consistency", 0.85)):
            return self._reject_hop("low_timing_consistency")
        cross_transitions = int(np.count_nonzero(valid & (sources[:-1] != sources[1:])))
        # One boundary between two unrelated recordings is not enough evidence.
        if cross_transitions < 2:
            return self._reject_hop("insufficient_cross_transitions")
        return {"hop_association": "periodic", "hop_period_px": period, "hop_timing_consistency": consistency,
                "hop_cross_transitions": cross_transitions, "hop_components": components}

    def _merge_hopping_groups(self, groups, shape):
        groups = list(groups)
        while len(groups) > 1:
            best = None
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    evidence = self._hopping_evidence(groups[i], groups[j])
                    attempts = self.last_diagnostics.setdefault("association_attempts", [])
                    if len(attempts) < 256:
                        attempts.append({
                            "first_center_freq": groups[i]["center_freq"],
                            "second_center_freq": groups[j]["center_freq"],
                            "first_n_boxes": groups[i]["n_boxes"],
                            "second_n_boxes": groups[j]["n_boxes"],
                            "accepted": evidence is not None,
                            "rejection_reason": self._hop_rejection_reason,
                        })
                    else:
                        self.last_diagnostics["association_attempts_truncated"] = True
                    if evidence is None:
                        continue
                    if self.hop_association == "shape":
                        rank = (-evidence["hop_shape_distance"], -evidence["hop_time_conflict_fraction"],
                                evidence["hop_timing_consistency"] or 0.0)
                    else:
                        rank = (evidence["hop_timing_consistency"], evidence["hop_cross_transitions"])
                    if best is None or rank > best[0]:
                        best = (rank, i, j, evidence)
            if best is None:
                break
            _, i, j, evidence = best
            first, second = groups[i], groups[j]
            boxes = np.concatenate([first["boxes"], second["boxes"]])
            stats = list(first["stats"] or []) + list(second["stats"] or [])
            order = np.argsort((boxes[:, 0] + boxes[:, 2]) / 2.0, kind="stable")
            stats = [stats[int(k)] for k in order] if len(stats) == len(boxes) else None
            merged = self._build_group_stats(boxes[order], stats, shape)
            merged.update(evidence)
            merged["group_type"] = "hopping"
            merged["frequency_cluster_count"] = (
                first["frequency_cluster_count"] + second["frequency_cluster_count"]
            )
            # Physical envelope, including edge-frame bandwidth; unaffected by dwell counts.
            merged["center_freq"] = 0.5 * (merged["freq_edge_min"] + merged["freq_edge_max"])
            merged["bandwidth"] = merged["freq_edge_max"] - merged["freq_edge_min"]
            groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
        return groups

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

    def _group_rejection_reasons(self, g):
        reasons = []
        if g["n_boxes"] < int(self.config.min_group_len):
            reasons.append(f"n_boxes={g['n_boxes']} < min_group_len={self.config.min_group_len}")
        if self.grouping_mode == "hopping":
            if g.get("group_type") == "hopping":
                minimum = int(getattr(self.config, "hop_min_pulses", 3))
                if g["n_boxes"] < minimum:
                    reasons.append(f"n_boxes={g['n_boxes']} < hop_min_pulses={minimum}")
            for metric, option, default in (
                ("time_extent_ratio", "hop_min_time_extent_ratio", 0.15),
                ("time_coverage_ratio", "hop_min_time_coverage_ratio", 0.02),
            ):
                threshold = float(getattr(self.config, option, default))
                if g[metric] < threshold:
                    reasons.append(f"{metric}={g[metric]:.4f} < {option}={threshold}")
        elif g["time_span_ratio"] < float(self.config.min_group_time_span_ratio):
            reasons.append(
                f"time_coverage_ratio={g['time_span_ratio']:.4f} < "
                f"min_group_time_span_ratio={self.config.min_group_time_span_ratio}"
            )
        return reasons

    def _group_passes_thresholds(self, g):
        return not self._group_rejection_reasons(g)

    @staticmethod
    def _diagnostic_group(group):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in group.items() if key != "stats"}

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
        if spectrogram is None:
            raise ValueError("select_signal_groups requires the native spectrogram")
        H, W = spectrogram.shape
        self.last_diagnostics = {
            "grouping_mode": self.grouping_mode,
            "hop_association": self.hop_association,
            "detected_boxes": 0 if det_boxes is None else len(det_boxes),
            "input_boxes": [] if det_boxes is None else np.asarray(det_boxes).reshape(-1, 4).tolist(),
            "spectrogram_shape": [int(H), int(W)], "kept_groups": 0,
            "geometry_boxes": 0, "energy_boxes": 0,
            "raw_groups": [], "final_candidates": [],
        }

        if det_boxes is None or len(det_boxes) == 0:
            return []

        # basic shape filter
        boxes = self._basic_filter(det_boxes)
        self.last_diagnostics["geometry_boxes"] = len(boxes)
        if boxes.size == 0:
            return []

        # pooled-ring energy filter
        if spectrogram is not None:
            e_boxes, e_stats = self._filter_boxes_by_energy(boxes, spectrogram)
            self.last_diagnostics["energy_boxes"] = len(e_boxes)
            self.last_diagnostics["energy_fallback"] = not bool(e_boxes.size)
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

        self.last_diagnostics["raw_groups"] = [self._diagnostic_group(g) for g in raw_groups]
        groups = (raw_groups if self.grouping_mode == "hopping" else
                  self._merge_groups(raw_groups, spectrogram.shape))

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

        if self.grouping_mode == "hopping":
            cleaned = self._merge_hopping_groups(cleaned, spectrogram.shape)

        final_groups = []
        for g in cleaned:
            reasons = self._group_rejection_reasons(g)
            diagnostic = self._diagnostic_group(g)
            diagnostic["rejection_reasons"] = reasons
            self.last_diagnostics["final_candidates"].append(diagnostic)
            if not reasons:
                final_groups.append(g)

        self.last_diagnostics["kept_groups"] = len(final_groups)
        final_groups = sorted(final_groups, key=lambda x: x["score"], reverse=True)
        return final_groups
