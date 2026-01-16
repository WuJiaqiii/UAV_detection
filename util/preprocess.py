import numpy as np
from sklearn.cluster import DBSCAN

class SignalPreprocessor:
    def __init__(self, sampling_rate, n_fft, hop_length,
                 min_area=20, min_ratio=2.0, freq_eps=5, freq_min_samples=1,
                 nms_iou_thresh=0.5):
        """
        Preprocessor for detection outputs to extract main signal blocks.
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
        self.min_area = min_area
        self.min_ratio = min_ratio
        self.freq_eps = freq_eps
        self.freq_min_samples = freq_min_samples
        self.nms_thresh = nms_iou_thresh

    def _iou(self, boxA, boxB):
        """Compute Intersection over Union (IoU) of two boxes [x1,y1,x2,y2]."""
        x1, y1, x2, y2 = boxA
        x1b, y1b, x2b, y2b = boxB
        # Overlap in coordinates
        inter_w = max(0, min(x2, x2b) - max(x1, x1b))
        inter_h = max(0, min(y2, y2b) - max(y1, y1b))
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        areaA = (x2 - x1) * (y2 - y1)
        areaB = (x2b - x1b) * (y2b - y1b)
        iou = inter_area / float(areaA + areaB - inter_area)
        return iou

    def _nms(self, boxes):
        """Perform Non-Maximum Suppression (NMS) on a list of boxes. Returns filtered list of boxes."""
        if not boxes:
            return []
        # Sort boxes by area (descending) – large boxes first
        boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep = []
        while boxes_sorted:
            box = boxes_sorted.pop(0)
            keep.append(box)
            new_list = []
            # Current box parameters for overlap comparison
            x1, y1, x2, y2 = box
            area_box = (x2 - x1) * (y2 - y1)
            for other in boxes_sorted:
                x1b, y1b, x2b, y2b = other
                # Calculate overlap area with the kept box
                inter_w = max(0, min(x2, x2b) - max(x1, x1b))
                inter_h = max(0, min(y2, y2b) - max(y1, y1b))
                inter_area = inter_w * inter_h
                area_b = (x2b - x1b) * (y2b - y1b)
                # Compute IoU and coverage of smaller box
                iou_val = inter_area / float(area_box + area_b - inter_area) if inter_area > 0 else 0.0
                cover_small = 0.0
                if inter_area > 0:
                    cover_small = inter_area / float(min(area_box, area_b))
                # If boxes significantly overlap (IoU above threshold) or one mostly contains the other, drop the other
                if iou_val > self.nms_thresh or cover_small > 0.9:
                    continue  # discard "other"
                new_list.append(other)
            boxes_sorted = new_list
        return keep

    def process(self, det_boxes, spectrogram):
        """
        Process detection boxes to extract main signal block features.
        Parameters:
          det_boxes (ndarray or list): Array of detected boxes for the image (each box format [x1, y1, x2, y2]).
          spectrogram (ndarray): 2D array (freq_bins x time_frames) of STFT magnitudes for the image.
        Returns:
          features_array (ndarray): Array of shape (N_blocks, F) with features for each main signal block.
        """
        # If det_boxes is empty or None, return empty result
        if det_boxes is None or len(det_boxes) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        boxes = np.array(det_boxes, dtype=np.int32)
        # Ensure boxes are in [x1, y1, x2, y2] format
        if boxes.shape[1] == 4:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        else:
            raise ValueError("det_boxes should be of shape (N,4) for [x1,y1,x2,y2].")
        # Compute width, height for each box
        widths = x2 - x1
        heights = y2 - y1
        areas = widths * heights

        # Filter by area and shape (width/height ratio)
        mask = (areas >= self.min_area) & ((widths / (heights + 1e-6)) >= self.min_ratio)
        boxes = boxes[mask]
        if boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Cluster boxes by frequency (y-coordinate) using DBSCAN
        # Use frequency center of each box for clustering
        freq_centers = (boxes[:, 1] + boxes[:, 3]) / 2.0  # y1 and y2 mean as frequency center (pixel)
        freq_centers = freq_centers.reshape(-1, 1)
        clustering = DBSCAN(eps=self.freq_eps, min_samples=self.freq_min_samples)
        labels = clustering.fit_predict(freq_centers)
        # Ignore noise label -1
        valid_idx = labels >= 0
        boxes = boxes[valid_idx]
        labels = labels[valid_idx]
        if boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Identify the main cluster (scoring by number of boxes, frequency consistency, time span)
        main_label = None
        if labels.size > 0:
            unique_labels = np.unique(labels)
            best_score = -float('inf')
            best_label = unique_labels[0]
            for lbl in unique_labels:
                cluster_idx = (labels == lbl)
                cluster_boxes = boxes[cluster_idx]
                if cluster_boxes.size == 0:
                    continue
                n = cluster_boxes.shape[0]
                # Frequency consistency: use frequency range of cluster
                freq_vals = (cluster_boxes[:, 1] + cluster_boxes[:, 3]) / 2.0
                freq_range = freq_vals.max() - freq_vals.min()
                # Time span: span of x (time) coverage
                time_vals = (cluster_boxes[:, 0] + cluster_boxes[:, 2]) / 2.0
                time_span = time_vals.max() - time_vals.min()
                # Score: prioritize more boxes and longer time span, penalize wide freq range
                score = n + 0.5 * time_span - 0.5 * freq_range
                if score > best_score:
                    best_score = score
                    best_label = lbl
            main_label = best_label
        else:
            main_label = 0  # if somehow all were noise, treat as one cluster

        # Select main cluster boxes
        main_cluster_idx = (labels == main_label)
        main_boxes = boxes[main_cluster_idx]
        if main_boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Sort main boxes by time (x-center) to form a time sequence
        time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
        sorted_idx = np.argsort(time_centers)
        main_boxes = main_boxes[sorted_idx]

        # Apply Non-Max Suppression to remove overlapping boxes within main cluster
        main_list = [list(b) for b in main_boxes]  # convert to list of [x1,y1,x2,y2]
        main_list = self._nms(main_list)
        main_boxes = np.array(main_list, dtype=np.int32)
        if main_boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # Sort again by time after NMS (NMS might change order)
        time_centers = (main_boxes[:, 0] + main_boxes[:, 2]) / 2.0
        sorted_idx = np.argsort(time_centers)
        main_boxes = main_boxes[sorted_idx]

        if main_boxes.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Compute features for each remaining main signal box
        features = []
        # Pre-calculate conversion factors for time and frequency
        num_freq_bins = spectrogram.shape[0]
        # Frequency resolution (Hz per bin) – assuming one-sided spectrogram from 0 to Nyquist
        freq_res = (self.sr / 2.0) / (num_freq_bins - 1) if num_freq_bins > 1 else 0
        # Time resolution (seconds per pixel column)
        time_res = self.hop_length / float(self.sr)
        for (x1, y1, x2, y2) in main_boxes:
            # Pixel-based measurements
            width = x2 - x1
            height = y2 - y1
            area = width * height
            # Compute region's intensity statistics
            region = spectrogram[y1:y2, x1:x2]
            # If spectrogram is complex, take magnitude (assuming already magnitude)
            # Compute mean, max, var of intensity in this region
            mean_val = float(np.mean(region))
            max_val = float(np.max(region))
            var_val = float(np.var(region))
            # Time and frequency centers in physical units
            time_center_idx = (x1 + x2) / 2.0
            freq_center_idx = (y1 + y2) / 2.0
            time_center_sec = time_center_idx * time_res
            freq_center_hz = freq_center_idx * freq_res
            # Time duration and frequency bandwidth in physical units
            time_duration = width * time_res
            freq_bandwidth = height * freq_res
            # Feature vector for this block
            feat = [time_center_sec, freq_center_hz, time_duration, freq_bandwidth,
                    area, mean_val, max_val, var_val]
            features.append(feat)
        features_array = np.array(features, dtype=np.float32)
        return features_array
