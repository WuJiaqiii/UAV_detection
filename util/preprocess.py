import numpy as np
from sklearn.cluster import DBSCAN

class SignalPreprocessor:
    def __init__(self, sampling_rate, n_fft, hop_length,
                 min_area=20, min_ratio=2.0, freq_eps=5, freq_min_samples=1,
                 nms_iou_thresh=0.5):
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

            feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth,
                    area, mean_val, max_val, var_val]
            features.append(feat)

        features_array = np.asarray(features, dtype=np.float32)
        return (features_array, main_boxes) if return_boxes else features_array
