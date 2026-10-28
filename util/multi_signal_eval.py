"""Per-group classification, native frequency geometry and conservative track voting."""
from collections import Counter
from collections.abc import Mapping
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.iq_dat_loader import BYTES_PER_COMPLEX_SAMPLE
from util.eval_common import load_spectrum, plan_iq_slices, validate_classes, vote_predictions, read_path_mapping
from util.group_predictor import GroupPredictor
from util.spectrogram_frontend import pixel_y_to_freq_mhz


def merge_bands(bands):
    merged = []
    for low, high in sorted((float(a), float(b)) for a, b in bands):
        if not math.isfinite(low + high) or high <= low:
            raise ValueError("Frequency bands must have finite increasing edges")
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def band_iou(first, second):
    first, second = merge_bands(first), merge_bands(second)
    intersection = sum(max(0., min(b, d) - max(a, c)) for a, b in first for c, d in second)
    union = sum(b-a for a, b in first + second) - intersection
    return intersection / union if union else 0.


def group_geometry(boxes, shape, sample_rate, slice_start_ms=0., duration_ms=50., rf_center_mhz=None):
    """Geometry is measured before mask centering; frequency edges follow the FFT grid."""
    height, width = shape
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Need finite nonempty group boxes")
    if not math.isfinite(slice_start_ms) or not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Invalid slice time coordinates")
    if rf_center_mhz is not None and not math.isfinite(rf_center_mhz):
        raise ValueError("RF center must be finite MHz")
    arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, width)
    arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, height)
    if ((arr[:, 2:] - arr[:, :2]) <= 0).any():
        raise ValueError("Group contains an empty or inverted box")
    pulses, bands, centers = [], [], []
    for x1, y1, x2, y2 in arr:
        low = pixel_y_to_freq_mhz(y1, height, sample_rate, "fft")
        high = pixel_y_to_freq_mhz(y2, height, sample_rate, "fft")
        centers.append((low + high) / 2.)
        bands.append([low, high])
        pulses.append({"box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                       "start_ms": slice_start_ms + x1 / width * duration_ms,
                       "end_ms": slice_start_ms + x2 / width * duration_ms,
                       "frequency_low_offset_mhz": low, "frequency_high_offset_mhz": high,
                       "frequency_low_absolute_mhz": None if rf_center_mhz is None else rf_center_mhz + low,
                       "frequency_high_absolute_mhz": None if rf_center_mhz is None else rf_center_mhz + high})
    bands = merge_bands(bands)
    low, high, center = bands[0][0], bands[-1][1], float(np.mean(centers))
    return {"start_ms": min(p["start_ms"] for p in pulses), "end_ms": max(p["end_ms"] for p in pulses),
            "frequency_center_offset_mhz": center, "frequency_low_offset_mhz": low,
            "frequency_high_offset_mhz": high, "frequency_span_mhz": high-low,
            "occupied_bandwidth_mhz": sum(b-a for a, b in bands),
            "frequency_bands_offset_mhz": bands, "rf_center_mhz": rf_center_mhz,
            "frequency_center_absolute_mhz": None if rf_center_mhz is None else rf_center_mhz + center,
            "frequency_low_absolute_mhz": None if rf_center_mhz is None else rf_center_mhz + low,
            "frequency_high_absolute_mhz": None if rf_center_mhz is None else rf_center_mhz + high,
            "frequency_bands_absolute_mhz": None if rf_center_mhz is None else [[rf_center_mhz+a, rf_center_mhz+b] for a, b in bands],
            "pulses": pulses}


class MultiSignalPredictor(GroupPredictor):
    def predict(self, spectrum, domain):
        result, groups = self.detect_groups(spectrum, domain)
        result["groups"] = self.classify_groups(spectrum, groups) if groups else []
        return result


class FrequencyTracker:
    """One-to-one frequency association, independent of instantaneous class labels.

    Tracks are local candidates, not transmitter identities. Disjoint carriers are
    deliberately not joined merely because their predicted classes are the same.
    """
    def __init__(self, frequency_tolerance_mhz=3., min_band_iou=.1, max_gap_ms=150.):
        if (not math.isfinite(frequency_tolerance_mhz) or frequency_tolerance_mhz <= 0
                or not 0 <= min_band_iou <= 1 or not math.isfinite(max_gap_ms) or max_gap_ms < 0):
            raise ValueError("Invalid frequency tracking gates")
        self.tolerance, self.min_iou, self.max_gap = frequency_tolerance_mhz, min_band_iou, max_gap_ms
        self.tracks = []

    def update(self, observations):
        if not observations:
            return
        candidates = []
        for track in self.tracks:
            previous = track["observations"][-1]
            if any(max(0., current["start_ms"] - previous["end_ms"]) <= self.max_gap
                   for current in observations):
                candidates.append(track)
        costs = np.full((len(candidates), len(observations)), 1e6)
        for i, track in enumerate(candidates):
            previous = track["observations"][-1]
            for j, current in enumerate(observations):
                if previous["slice_index"] >= current["slice_index"]:
                    continue
                distance = abs(previous["frequency_center_offset_mhz"] - current["frequency_center_offset_mhz"])
                overlap = band_iou(previous["frequency_bands_offset_mhz"], current["frequency_bands_offset_mhz"])
                gap = max(0., current["start_ms"] - previous["end_ms"])
                # Even a zero configured IoU threshold never joins disjoint bands.
                if distance <= self.tolerance and overlap > 0 and overlap >= self.min_iou and gap <= self.max_gap:
                    costs[i, j] = distance/self.tolerance + (1.-overlap)
        assigned = set()
        if costs.size:
            left, right = linear_sum_assignment(costs)
            for i, j in zip(left, right):
                if costs[i, j] >= 1e6:
                    continue
                track = candidates[i]
                observations[j]["track_id"] = track["track_id"]
                track["observations"].append(observations[j])
                assigned.add(int(j))
        for j, current in enumerate(observations):
            if j not in assigned:
                track_id = f"T{len(self.tracks)+1:04d}"
                current["track_id"] = track_id
                self.tracks.append({"track_id": track_id, "observations": [current]})

    def summarize(self, classes, vote_options=None):
        summaries = []
        for track in self.tracks:
            observations = track["observations"]
            vote = vote_predictions(observations, classes, **(vote_options or {}))
            bands = merge_bands([band for obs in observations for band in obs["frequency_bands_offset_mhz"]])
            center = float(np.mean([obs["frequency_center_offset_mhz"] for obs in observations]))
            rf = observations[0]["rf_center_mhz"]
            vote.update(track_id=track["track_id"], observation_count=len(observations),
                observation_ids=[obs["observation_id"] for obs in observations],
                start_ms=min(obs["start_ms"] for obs in observations), end_ms=max(obs["end_ms"] for obs in observations),
                frequency_center_offset_mhz=center, frequency_low_offset_mhz=bands[0][0],
                frequency_high_offset_mhz=bands[-1][1], frequency_bands_offset_mhz=bands,
                frequency_span_mhz=bands[-1][1]-bands[0][0], occupied_bandwidth_mhz=sum(b-a for a, b in bands),
                rf_center_mhz=rf, frequency_center_absolute_mhz=None if rf is None else rf+center,
                frequency_low_absolute_mhz=None if rf is None else rf+bands[0][0],
                frequency_high_absolute_mhz=None if rf is None else rf+bands[-1][1],
                frequency_bands_absolute_mhz=None if rf is None else [[rf+a, rf+b] for a, b in bands])
            summaries.append(vote)
        return summaries


def evaluate_multi_file(path, predictor, config, vote_options=None, stride_ms=50., tail_policy="drop",
                        max_slices=0, fail_fast=False, rf_center_mhz=None, track_options=None,
                        visualization_dir=None, max_visualizations=24):
    path = Path(path).resolve()
    if max_visualizations < 0:
        raise ValueError("max_visualizations must be >= 0")
    if path.suffix.lower() == ".dat":
        size = path.stat().st_size
        if size % BYTES_PER_COMPLEX_SAMPLE:
            raise ValueError("DAT size is not a whole number of int16 I/Q pairs")
        windows, sampling = plan_iq_slices(size // BYTES_PER_COMPLEX_SAMPLE, config.iq_sample_rate,
                                          stride_ms, tail_policy, max_slices)
    else:
        windows, sampling = [None], None
    visible = set(range(len(windows)) if not max_visualizations else
                  np.linspace(0, len(windows)-1, min(max_visualizations, len(windows)), dtype=int).tolist())
    tracker = FrequencyTracker(**(track_options or {}))
    rows = []
    for index, window in enumerate(windows):
        row = {"path": str(path), "slice_index": index, "slice_start_ms": (window or {}).get("start_ms", 0.),
               "slice_end_ms": (window or {}).get("end_ms", 50.), "groups": [], **(window or {})}
        try:
            spectrum, domain = load_spectrum(path, config, window)
            predicted = predictor.predict(spectrum, domain)
            if "groups" not in predicted:
                raise ValueError("Multi-signal mode requires a predictor returning groups")
            row.update(predicted)
            for group in row["groups"]:
                group.update(path=str(path), slice_index=index,
                    observation_id=f"S{index:06d}_G{group['group_id']:04d}",
                    slice_start_ms=row["slice_start_ms"], slice_end_ms=row["slice_end_ms"])
                group.update(group_geometry(group["selected_boxes"], spectrum.shape, config.sampling_rate,
                                             row["slice_start_ms"], config.iq_segment_ms, rf_center_mhz))
            tracker.update(row["groups"])
        except Exception as exc:
            if fail_fast:
                raise
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", groups=[])
        if visualization_dir is not None and index in visible and row["status"] != "error":
            try:
                from util.signal_eval_report import save_slice_preview
                destination = save_slice_preview(path, index, spectrum, domain, config, row, visualization_dir, rf_center_mhz)
                row["visualization"] = str(destination)
            except Exception as exc:
                if fail_fast:
                    raise
                row["visualization_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    tracks = tracker.summarize(config.classes, vote_options)
    decided = [track for track in tracks if track["status"] == "ok"]
    errors = sum(row["status"] == "error" for row in rows)
    status = "error" if errors == len(rows) else "ok" if decided else "unknown"
    result = {"path": str(path), "input_type": path.suffix.lower()[1:], "status": status,
              "predicted_classes": sorted({track["class_name"] for track in decided}),
              "total_slices": len(rows), "observation_count": sum(len(row["groups"]) for row in rows),
              "track_count": len(tracks), "accepted_track_count": len(decided),
              "signals": tracks, "error_slices": errors, "sampling": sampling,
              "rf_center_mhz": rf_center_mhz, "frequency_reference": "offset" if rf_center_mhz is None else "absolute_and_offset",
              "visualization_errors": sum("visualization_error" in row for row in rows)}
    return result, rows


def resolve_rf_centers(path=None, default=None):
    if default is not None and (isinstance(default, bool) or not math.isfinite(default)):
        raise ValueError("rf_center_mhz must be finite")
    centers = read_path_mapping(path) if path else {}
    for name, value in centers.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"RF center must be a finite MHz value: {name}")
    return centers


def load_multi_truth(path, inputs, classes, label_from_parent=False, allow_unknown=False, label_from_filename=False):
    truth = read_path_mapping(path) if path else {}
    for name, value in list(truth.items()):
        names = [value] if isinstance(value, str) else value
        if not isinstance(names, list) or any(not isinstance(item, str) or not item or (not allow_unknown and item not in classes) for item in names):
            raise ValueError(f"Truth must be a class name or list of known class names: {name}")
        truth[name] = sorted(set(names))
    if label_from_parent:
        for file in inputs:
            name = file.parent.name
            if (not allow_unknown and name not in classes) or (str(file) in truth and truth[str(file)] != [name]):
                raise ValueError(f"Invalid or conflicting parent-folder truth: {file}")
            truth[str(file)] = [name]
    if label_from_filename:
        from data.data_loader import UAVDataset
        for file in inputs:
            prefix = file.stem.split("-SNR-")[0]
            names = sorted({match.group("protocol") for match in UAVDataset._SIGNAL_RE.finditer(prefix)})
            if not names or (not allow_unknown and any(name not in classes for name in names)):
                raise ValueError(f"Filename must contain valid Protocol-[...] blocks: {file}")
            if str(file) in truth and truth[str(file)] != names:
                raise ValueError(f"Conflicting filename truth: {file}")
            truth[str(file)] = names
    return truth


def multi_evaluation_metrics(results, classes):
    labeled = [row for row in results if row.get("true_class_names") is not None]
    correct = sum(row["status"] != "error" and set(row["predicted_classes"]) == set(row["true_class_names"]) for row in labeled)
    counts = Counter(name for row in results for name in row["predicted_classes"])
    per_class = {}
    for name in classes:
        tp = sum(name in row["predicted_classes"] and name in row["true_class_names"] for row in labeled)
        fp = sum(name in row["predicted_classes"] and name not in row["true_class_names"] for row in labeled)
        fn = sum(name not in row["predicted_classes"] and name in row["true_class_names"] for row in labeled)
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp/(tp+fp) if tp+fp else None,
                           "recall": tp/(tp+fn) if tp+fn else None}
    return {"total_files": len(results), "classified_files": sum(row["status"] == "ok" for row in results),
            "unknown_files": sum(row["status"] == "unknown" for row in results),
            "error_files": sum(row["status"] == "error" for row in results),
            "files_with_slice_errors": sum(row.get("error_slices", 0) > 0 for row in results),
            "visualization_errors": sum(row.get("visualization_errors", 0) for row in results),
            "total_observations": sum(row["observation_count"] for row in results),
            "total_candidate_tracks": sum(row["track_count"] for row in results),
            "total_accepted_tracks": sum(row["accepted_track_count"] for row in results),
            "files_per_predicted_class": {name: counts[name] for name in classes},
            "labeled_files": len(labeled), "file_class_set_accuracy": correct/len(labeled) if labeled else None,
            "file_class_presence_metrics": per_class}
