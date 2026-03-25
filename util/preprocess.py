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

#         weights = ResNet18_Weights.DEFAULT if pretrained else None
#         base = resnet18(weights=weights)

#         # Keep early layers only (fast + suitable for tiny patches)
#         self.conv1 = base.conv1
#         self.bn1 = base.bn1
#         self.relu = base.relu

#         # Remove maxpool for tiny inputs (important)
#         self.maxpool = nn.Identity()

#         self.layer1 = base.layer1
#         self.layer2 = base.layer2

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.proj = nn.Linear(128, out_dim)  # layer2 output channels = 128

#         # Freeze all params (inference only)
#         for p in self.parameters():
#             p.requires_grad = False

#         self.eval()

#     @torch.no_grad()
#     def forward(self, x):
#         # x: (N,3,H,W)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = self.pool(x).flatten(1)   # (N,128)
#         x = self.proj(x)              # (N,out_dim)
#         return x

class SignalPreprocessor:
    def __init__(self, sampling_rate, n_fft, hop_length,
             min_area=20, min_ratio=2.0, freq_eps=5, freq_min_samples=1,
             nms_iou_thresh=0.5,
             use_patch_cnn=False,
             patch_size=16,              
             patch_feat_dim=32,          
             patch_batch_size=64,        
             patch_device=None,          
             patch_pretrained=True):
        
        """Preprocessor for detection outputs to extract main signal blocks.

        This class does **two** jobs:
        1) Select "main" boxes from raw detections (basic filtering -> DBSCAN on freq center -> pick best cluster -> NMS).
        2) Convert the selected boxes into a feature sequence for the Transformer.

        Parameters:
          sampling_rate (float): Sampling rate of the signal (Hz).
          n_fft (int): Number of FFT points used in STFT (defines frequency resolution).
          hop_length (int): Hop length (step in samples) for STFT (defines time resolution).
          min_area (int): Minimum area (in pixels) for a box to be considered.
          min_ratio (float): Minimum width/height ratio for a box to be considered (to filter horizontal shapes).
          freq_eps (float): DBSCAN epsilon for frequency clustering (in frequency-bin units).
          freq_min_samples (int): Minimum samples for DBSCAN clustering.
          nms_iou_thresh (float): IoU threshold for NMS to remove overlapping boxes.
        """
        self.sr = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_area = int(min_area)
        self.min_ratio = float(min_ratio)
        self.freq_eps = float(freq_eps)
        self.freq_min_samples = int(freq_min_samples)
        self.nms_thresh = float(nms_iou_thresh)
        
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
    #     """
    #     region: (h,w) float
    #     return: (3, patch_size, patch_size) float32 tensor
    #     """
    #     ps = self.patch_size

    #     if region is None or region.size == 0:
    #         return torch.zeros((3, ps, ps), dtype=torch.float32)

    #     x = np.asarray(region, dtype=np.float32)

    #     # robust normalization to [0,1]
    #     p1, p99 = np.percentile(x, [1, 99]) if x.size >= 4 else (float(x.min()), float(x.max()))
    #     if p99 <= p1:
    #         x = np.zeros_like(x, dtype=np.float32)
    #     else:
    #         x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    #     t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    #     t = F.interpolate(t, size=(ps, ps), mode="bilinear", align_corners=False)  # (1,1,ps,ps)
    #     t = t.squeeze(0)  # (1,ps,ps)

    #     # 1ch -> 3ch
    #     t = t.repeat(3, 1, 1)  # (3,ps,ps)

    #     # ImageNet normalization (optional but recommended for pretrained backbone)
    #     mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
    #     std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)
    #     t = (t - mean) / std
    #     return t
    
    # @torch.no_grad()
    # def _extract_patch_features(self, spectrogram: np.ndarray, main_boxes: np.ndarray) -> np.ndarray:
    #     """
    #     Return: (K, patch_feat_dim) float32
    #     """
    #     if (not self.use_patch_cnn) or main_boxes.size == 0:
    #         return np.zeros((len(main_boxes), 0), dtype=np.float32)

    #     enc = self._ensure_patch_encoder()
    #     dev = self._patch_encoder_device
    #     K = int(main_boxes.shape[0])

    #     patch_tensors = []
    #     for (x1, y1, x2, y2) in main_boxes:
    #         region = spectrogram[y1:y2, x1:x2]
    #         patch_tensors.append(self._region_to_resnet_tensor(region))

    #     out_feats = []
    #     bs = max(1, self.patch_batch_size)
    #     for i in range(0, K, bs):
    #         batch = torch.stack(patch_tensors[i:i+bs], dim=0).to(dev, non_blocking=True)  # (n,3,ps,ps)
    #         feat = enc(batch)  # (n, patch_feat_dim)
    #         out_feats.append(feat.cpu())

    #     feats = torch.cat(out_feats, dim=0).numpy().astype(np.float32, copy=False)
    #     return feats

    def _iou(self, boxA, boxB):
        """Compute Intersection over Union (IoU) of two boxes [x1,y1,x2,y2]."""
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
        boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
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
                cover_small = float(inter_area / float(min(area_box, area_b))) if min(area_box, area_b) > 0 else 0.0
                if iou_val > self.nms_thresh or cover_small > 0.9:
                    continue
                new_list.append(other)
            boxes_sorted = new_list
        return keep

    def _basic_filter(self, boxes: np.ndarray) -> np.ndarray:
        """Filter by area and width/height ratio."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = x2 - x1
        heights = y2 - y1
        areas = widths * heights
        mask = (areas >= self.min_area) & ((widths / (heights + 1e-6)) >= self.min_ratio)
        boxes = boxes[mask]
        return boxes.astype(np.int32, copy=False)

    def select_main_boxes(self, det_boxes):
        """Select main boxes after basic filtering + DBSCAN + NMS.

        Args:
          det_boxes: array-like (N,4) in [x1,y1,x2,y2]
        Returns:
          main_boxes: np.ndarray (K,4) int32, sorted by time center (x).
        """
        boxes = self._basic_filter(det_boxes)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        # DBSCAN on frequency centers (y)
        freq_centers = ((boxes[:, 1] + boxes[:, 3]) / 2.0).reshape(-1, 1)
        clustering = DBSCAN(eps=self.freq_eps, min_samples=self.freq_min_samples)
        labels = clustering.fit_predict(freq_centers)

        # drop noise (-1)
        valid = labels >= 0
        boxes = boxes[valid]
        labels = labels[valid]
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        # pick best cluster
        unique_labels = np.unique(labels)
        best_score = -float("inf")
        best_label = unique_labels[0]
        for lbl in unique_labels:
            idx = (labels == lbl)
            cluster_boxes = boxes[idx]
            if cluster_boxes.size == 0:
                continue
            n = cluster_boxes.shape[0]
            freq_vals = (cluster_boxes[:, 1] + cluster_boxes[:, 3]) / 2.0
            freq_range = float(freq_vals.max() - freq_vals.min())
            time_vals = (cluster_boxes[:, 0] + cluster_boxes[:, 2]) / 2.0
            time_span = float(time_vals.max() - time_vals.min())
            score = float(n + 0.5 * time_span - 0.5 * freq_range)
            if score > best_score:
                best_score = score
                best_label = lbl

        main_boxes = boxes[labels == best_label]
        if main_boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int32)

        # sort by time
        time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
        main_boxes = main_boxes[np.argsort(time_centers)]

        # NMS within main cluster
        main_list = [list(b) for b in main_boxes]
        main_list = self._nms(main_list)
        main_boxes = np.asarray(main_list, dtype=np.int32).reshape(-1, 4)

        # sort again
        if main_boxes.size > 0:
            time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
            main_boxes = main_boxes[np.argsort(time_centers)]

        return main_boxes

    def process(self, det_boxes, spectrogram, return_boxes: bool = False):
        """Convert detection boxes to a token sequence.

        Args:
          det_boxes: array-like (N,4) in [x1,y1,x2,y2]
          spectrogram: 2D array (freq_bins x time_frames)
          return_boxes: if True, also return the final main_boxes after DBSCAN+NMS.

        Returns:
          features_array: (K,F) float32; if no boxes -> (0,0)
          main_boxes (optional): (K,4) int32
        """
        main_boxes = self.select_main_boxes(det_boxes)
        if main_boxes.size == 0:
            empty = np.zeros((0, 0), dtype=np.float32)
            return (empty, main_boxes) if return_boxes else empty

        features = []
        num_freq_bins = int(spectrogram.shape[0])
        # Frequency resolution (Hz per bin) – assuming one-sided spectrogram from 0 to Nyquist
        freq_res = (self.sr / 2.0) / (num_freq_bins - 1) if num_freq_bins > 1 else 0.0
        # Time resolution (seconds per pixel column)
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

            # feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth,
            #         area, mean_val, max_val, var_val]
            feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth]
            features.append(feat)

        features_array = np.asarray(features, dtype=np.float32)
        
        # # 2) 新增 patch CNN 特征
        # patch_feats = self._extract_patch_features(spectrogram, main_boxes)  # (K, patch_feat_dim) or (K,0)

        # # 3) 拼接
        # if patch_feats.shape[1] > 0:
        #     features_array = np.concatenate([features_array, patch_feats], axis=1).astype(np.float32, copy=False)
        # else:
        #     features_array = features_array.astype(np.float32, copy=False)

        return (features_array, main_boxes) if return_boxes else features_array
