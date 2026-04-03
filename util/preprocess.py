import numpy as np
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
#     from torchvision.models import resnet18, ResNet18_Weights
#     _HAS_TORCHVISION = True
# except Exception:
#     _HAS_TORCHVISION = False

# class TinyResNetPatchEncoder(nn.Module):
#     """
#     Pretrained ResNet18 truncated for tiny spectrogram patches.
#     Input:  (N,3,H,W), H=W~12~16
#     Output: (N,out_dim)
#     """
#     def __init__(self, out_dim=32, pretrained=True):
#         super().__init__()
#         if not _HAS_TORCHVISION:
#             raise ImportError("torchvision is required for ResNet patch features.")
#
#         weights = ResNet18_Weights.DEFAULT if pretrained else None
#         base = resnet18(weights=weights)
#
#         self.conv1 = base.conv1
#         self.bn1 = base.bn1
#         self.relu = base.relu
#         self.maxpool = nn.Identity()
#         self.layer1 = base.layer1
#         self.layer2 = base.layer2
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.proj = nn.Linear(128, out_dim)
#
#         for p in self.parameters():
#             p.requires_grad = False
#         self.eval()
#
#     @torch.no_grad()
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.pool(x).flatten(1)
#         x = self.proj(x)
#         return x


class SignalPreprocessor:
    def __init__(
        self,
        sampling_rate,
        n_fft,
        hop_length,
        min_area=50,
        min_ratio=0.0,
        min_width=4,
        min_height=10,
        max_width=0,
        max_height=0,
        freq_eps=25,
        freq_min_samples=1,
        nms_iou_thresh=0.5,
        ring_margin=5,
        min_contrast_z=0.6,
        min_integrated_energy=8.0,
        min_bright_ratio=0.05,
        bright_z_thresh=1.5,
        exclude_bottom_ratio=0.0,
        cluster_area_weight=0.03,
        cluster_contrast_weight=1.0,
        cluster_bright_weight=20.0,
        cluster_time_span_weight=0.1,
        cluster_freq_range_weight=0.3,
        use_patch_cnn=False,
        patch_size=16,
        patch_feat_dim=32,
        patch_batch_size=64,
        patch_device=None,
        patch_pretrained=True,
    ):
        """
        Preprocessor for detection outputs to extract main signal blocks.

        Pipeline:
          1) basic size / geometry filtering
          2) optional local energy filtering against surrounding background
          3) DBSCAN on frequency centers
          4) cluster scoring that favors larger / stronger signal clusters
          5) NMS + sort by time

        Notes:
          - min_ratio defaults to 0.0 so vertical boxes are not suppressed.
          - energy filtering uses local contrast instead of absolute intensity,
            which is more stable across different SNRs and normalization styles.
          - if energy filtering removes all boxes, the code falls back to the
            geometry-filtered boxes to avoid collapsing on weak-signal samples.
        """
        self.sr = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.min_area = int(min_area)
        self.min_ratio = float(min_ratio)
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.max_width = int(max_width)
        self.max_height = int(max_height)

        self.freq_eps = float(freq_eps)
        self.freq_min_samples = int(freq_min_samples)
        self.nms_thresh = float(nms_iou_thresh)

        self.ring_margin = int(ring_margin)
        self.min_contrast_z = float(min_contrast_z)
        self.min_integrated_energy = float(min_integrated_energy)
        self.min_bright_ratio = float(min_bright_ratio)
        self.bright_z_thresh = float(bright_z_thresh)
        self.exclude_bottom_ratio = float(exclude_bottom_ratio)

        self.cluster_area_weight = float(cluster_area_weight)
        self.cluster_contrast_weight = float(cluster_contrast_weight)
        self.cluster_bright_weight = float(cluster_bright_weight)
        self.cluster_time_span_weight = float(cluster_time_span_weight)
        self.cluster_freq_range_weight = float(cluster_freq_range_weight)

        self.use_patch_cnn = bool(use_patch_cnn)
        self.patch_size = int(patch_size)
        self.patch_feat_dim = int(patch_feat_dim)
        self.patch_batch_size = int(patch_batch_size)
        self.patch_device = patch_device
        self.patch_pretrained = bool(patch_pretrained)

        self._patch_encoder = None
        self._patch_encoder_device = None

    def _get_patch_device(self):
        if self.patch_device is not None:
            return torch.device(self.patch_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def _ensure_patch_encoder(self):
    #     if not self.use_patch_cnn:
    #         return None
    #     if self._patch_encoder is None:
    #         dev = self._get_patch_device()
    #         enc = TinyResNetPatchEncoder(
    #             out_dim=self.patch_feat_dim,
    #             pretrained=self.patch_pretrained
    #         ).to(dev)
    #         enc.eval()
    #         self._patch_encoder = enc
    #         self._patch_encoder_device = dev
    #     return self._patch_encoder

    # def _region_to_resnet_tensor(self, region: np.ndarray) -> torch.Tensor:
    #     ps = self.patch_size
    #     if region is None or region.size == 0:
    #         return torch.zeros((3, ps, ps), dtype=torch.float32)
    #
    #     x = np.asarray(region, dtype=np.float32)
    #     p1, p99 = np.percentile(x, [1, 99]) if x.size >= 4 else (float(x.min()), float(x.max()))
    #     if p99 <= p1:
    #         x = np.zeros_like(x, dtype=np.float32)
    #     else:
    #         x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    #
    #     t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    #     t = F.interpolate(t, size=(ps, ps), mode="bilinear", align_corners=False)
    #     t = t.squeeze(0)
    #     t = t.repeat(3, 1, 1)
    #
    #     mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    #     std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    #     t = (t - mean) / std
    #     return t

    # @torch.no_grad()
    # def _extract_patch_features(self, spectrogram: np.ndarray, main_boxes: np.ndarray) -> np.ndarray:
    #     if (not self.use_patch_cnn) or main_boxes.size == 0:
    #         return np.zeros((len(main_boxes), 0), dtype=np.float32)
    #
    #     enc = self._ensure_patch_encoder()
    #     dev = self._patch_encoder_device
    #     k = int(main_boxes.shape[0])
    #
    #     patch_tensors = []
    #     for (x1, y1, x2, y2) in main_boxes:
    #         region = spectrogram[y1:y2, x1:x2]
    #         patch_tensors.append(self._region_to_resnet_tensor(region))
    #
    #     out_feats = []
    #     bs = max(1, self.patch_batch_size)
    #     for i in range(0, k, bs):
    #         batch = torch.stack(patch_tensors[i:i + bs], dim=0).to(dev, non_blocking=True)
    #         feat = enc(batch)
    #         out_feats.append(feat.cpu())
    #
    #     feats = torch.cat(out_feats, dim=0).numpy().astype(np.float32, copy=False)
    #     return feats

    def _iou(self, boxA, boxB):
        """Compute IoU of two boxes [x1,y1,x2,y2]."""
        x1, y1, x2, y2 = boxA
        x1b, y1b, x2b, y2b = boxB
        inter_w = max(0, min(x2, x2b) - max(x1, x1b))
        inter_h = max(0, min(y2, y2b) - max(y1, y1b))
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        areaA = max(0, (x2 - x1)) * max(0, (y2 - y1))
        areaB = max(0, (x2b - x1b)) * max(0, (y2b - y1b))
        denom = float(areaA + areaB - inter_area)
        return float(inter_area / denom) if denom > 0 else 0.0

    def _nms(self, boxes):
        """Simple NMS. boxes: list[[x1,y1,x2,y2]] -> kept list."""
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        keep = []
        while boxes_sorted:
            box = boxes_sorted.pop(0)
            keep.append(box)
            new_list = []
            x1, y1, x2, y2 = box
            area_box = max(0, (x2 - x1)) * max(0, (y2 - y1))
            for other in boxes_sorted:
                x1b, y1b, x2b, y2b = other
                inter_w = max(0, min(x2, x2b) - max(x1, x1b))
                inter_h = max(0, min(y2, y2b) - max(y1, y1b))
                inter_area = inter_w * inter_h
                if inter_area <= 0:
                    new_list.append(other)
                    continue

                area_b = max(0, (x2b - x1b)) * max(0, (y2b - y1b))
                denom = float(area_box + area_b - inter_area)
                iou_val = float(inter_area / denom) if denom > 0 else 0.0
                cover_small = (
                    float(inter_area / float(min(area_box, area_b)))
                    if min(area_box, area_b) > 0 else 0.0
                )
                if iou_val > self.nms_thresh or cover_small > 0.9:
                    continue
                new_list.append(other)
            boxes_sorted = new_list
        return keep

    def _basic_filter(self, boxes: np.ndarray, img_h=None, img_w=None) -> np.ndarray:
        """Filter by size, shape, and optional bottom exclusion zone."""
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
        """
        Compute local energy / contrast statistics for one box.

        Returns a dict with:
          mean, max, area, bg_mean, bg_std,
          contrast_z, integrated_energy, bright_ratio
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

    def _filter_boxes_by_energy(self, boxes: np.ndarray, spectrogram):
        """
        Energy-based filtering using local background statistics.

        Returns:
          kept_boxes: np.ndarray [K,4]
          kept_stats: list[dict]
        """
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []

        if spectrogram is None:
            return np.asarray(boxes, dtype=np.int32).reshape(-1, 4), [None] * len(boxes)

        kept_boxes = []
        kept_stats = []
        for box in np.asarray(boxes, dtype=np.int32).reshape(-1, 4):
            st = self._box_energy_stats(box, spectrogram)
            if st is None:
                continue
            if st["contrast_z"] < self.min_contrast_z:
                continue
            if st["integrated_energy"] < self.min_integrated_energy:
                continue
            if st["bright_ratio"] < self.min_bright_ratio:
                continue
            kept_boxes.append(box)
            kept_stats.append(st)

        if len(kept_boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32), []

        return np.asarray(kept_boxes, dtype=np.int32), kept_stats

    def _score_cluster(self, cluster_boxes: np.ndarray, cluster_stats):
        """Score one cluster. Larger, brighter, more contrasted clusters are favored."""
        if cluster_boxes.size == 0:
            return -float("inf")

        freq_vals = (cluster_boxes[:, 1] + cluster_boxes[:, 3]) / 2.0
        freq_range = float(freq_vals.max() - freq_vals.min()) if len(freq_vals) > 0 else 0.0

        time_vals = (cluster_boxes[:, 0] + cluster_boxes[:, 2]) / 2.0
        time_span = float(time_vals.max() - time_vals.min()) if len(time_vals) > 0 else 0.0

        widths = cluster_boxes[:, 2] - cluster_boxes[:, 0]
        heights = cluster_boxes[:, 3] - cluster_boxes[:, 1]
        total_area = float(np.sum(widths * heights))

        total_contrast = 0.0
        total_bright = 0.0
        if cluster_stats is not None and len(cluster_stats) == len(cluster_boxes):
            for st in cluster_stats:
                if st is None:
                    continue
                total_contrast += float(st.get("contrast_z", 0.0))
                total_bright += float(st.get("bright_ratio", 0.0))

        score = (
            self.cluster_area_weight * total_area
            + self.cluster_contrast_weight * total_contrast
            + self.cluster_bright_weight * total_bright
            + self.cluster_time_span_weight * time_span
            - self.cluster_freq_range_weight * freq_range
        )
        return float(score)

    def select_main_boxes(self, det_boxes, spectrogram=None):
        """
        Select main boxes after:
          basic filter -> energy filter -> DBSCAN -> cluster scoring -> NMS.

        Args:
          det_boxes: array-like (N,4) in [x1,y1,x2,y2]
          spectrogram: optional 2D array for local energy filtering.

        Returns:
          main_boxes: np.ndarray (K,4) int32, sorted by time center (x).
        """
        img_h = spectrogram.shape[0] if spectrogram is not None else None
        img_w = spectrogram.shape[1] if spectrogram is not None else None

        boxes = self._basic_filter(det_boxes, img_h=img_h, img_w=img_w)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        # Try energy filtering first. If it becomes too strict for very weak samples,
        # fall back to the geometry-filtered boxes instead of returning empty.
        energy_boxes, energy_stats = self._filter_boxes_by_energy(boxes, spectrogram)
        if energy_boxes.size > 0:
            boxes = energy_boxes
            stats_list = energy_stats
        else:
            stats_list = []
            if spectrogram is not None:
                for box in boxes:
                    stats_list.append(self._box_energy_stats(box, spectrogram))
            else:
                stats_list = [None] * len(boxes)

        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        if boxes.shape[0] == 1:
            return boxes.astype(np.int32, copy=False)

        # DBSCAN on frequency centers (y)
        freq_centers = ((boxes[:, 1] + boxes[:, 3]) / 2.0).reshape(-1, 1)
        clustering = DBSCAN(eps=self.freq_eps, min_samples=self.freq_min_samples)
        labels = clustering.fit_predict(freq_centers)

        # If DBSCAN marks all as noise, treat all remaining boxes as one cluster.
        if np.all(labels < 0):
            labels = np.zeros((boxes.shape[0],), dtype=np.int32)
        else:
            valid = labels >= 0
            boxes = boxes[valid]
            labels = labels[valid]
            stats_list = [stats_list[i] for i in range(len(stats_list)) if valid[i]]

        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        unique_labels = np.unique(labels)
        best_score = -float("inf")
        best_label = unique_labels[0]

        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            cluster_boxes = boxes[idx]
            cluster_stats = [stats_list[i] for i in idx] if len(stats_list) == len(boxes) else None
            score = self._score_cluster(cluster_boxes, cluster_stats)
            if score > best_score:
                best_score = score
                best_label = lbl

        main_boxes = boxes[labels == best_label]
        if main_boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        # Sort by time before NMS to keep temporal order stable.
        time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
        main_boxes = main_boxes[np.argsort(time_centers)]

        # NMS within main cluster
        main_list = [list(b) for b in main_boxes]
        main_list = self._nms(main_list)
        main_boxes = np.asarray(main_list, dtype=np.int32).reshape(-1, 4)

        # Sort again
        if main_boxes.size > 0:
            time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
            main_boxes = main_boxes[np.argsort(time_centers)]

        return main_boxes

    def process(self, det_boxes, spectrogram, return_boxes: bool = False):
        """
        Convert detection boxes to a token sequence.

        Args:
          det_boxes: array-like (N,4) in [x1,y1,x2,y2]
          spectrogram: 2D array (freq_bins x time_frames)
          return_boxes: if True, also return the final main_boxes after filtering.

        Returns:
          features_array: (K,F) float32; if no boxes -> (0,0)
          main_boxes (optional): (K,4) int32
        """
        main_boxes = self.select_main_boxes(det_boxes, spectrogram=spectrogram)
        if main_boxes.size == 0:
            empty = np.zeros((0, 0), dtype=np.float32)
            return (empty, main_boxes) if return_boxes else empty

        features = []
        num_freq_bins = int(spectrogram.shape[0])
        freq_res = (self.sr / 2.0) / (num_freq_bins - 1) if num_freq_bins > 1 else 0.0
        time_res = self.hop_length / float(self.sr) if self.sr else 0.0

        for (x1, y1, x2, y2) in main_boxes:
            width = int(x2 - x1)
            height = int(y2 - y1)
            area = float(width * height)

            region = spectrogram[y1:y2, x1:x2]
            mean_val = float(np.mean(region)) if region.size else 0.0
            max_val = float(np.max(region)) if region.size else 0.0
            var_val = float(np.var(region)) if region.size else 0.0

            time_center_idx = (x1 + x2) / 2.0
            freq_center_idx = (y1 + y2) / 2.0
            time_center_sec = float(time_center_idx * time_res)
            freq_center_hz = float(freq_center_idx * freq_res)

            time_duration = float(width * time_res)
            freq_bandwidth = float(height * freq_res)

            # Keep the current 4D token format for compatibility.
            # If you want to restore the 8D version, switch to the line below.
            # feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth,
            #         area, mean_val, max_val, var_val]
            feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth]
            features.append(feat)

        features_array = np.asarray(features, dtype=np.float32)

        # patch_feats = self._extract_patch_features(spectrogram, main_boxes)
        # if patch_feats.shape[1] > 0:
        #     features_array = np.concatenate([features_array, patch_feats], axis=1).astype(np.float32, copy=False)
        # else:
        #     features_array = features_array.astype(np.float32, copy=False)

        return (features_array, main_boxes) if return_boxes else features_array
