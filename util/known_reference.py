"""Record-balanced, multimodal known-class references; no test-data fitting."""
from collections import defaultdict

import numpy as np

from util.recordings import group_recordings


def canonical_records(records):
    rows = [{**row, "index": i, "class_name": row["true_class_name"]}
            for i, row in enumerate(records)]
    return group_recordings(rows)


class RecordingClassBank:
    """Each original recording contributes at most one neighbor per class.

    A recording can retain several component exemplars (e.g. video and control).
    Its query distance is the minimum distance to those exemplars. Replicating
    a component therefore cannot increase that recording's neighbor support.
    """
    def __init__(self, reference, labels, recording_ids, classes, k, scales=None):
        self.reference = np.asarray(reference, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=str)
        self.recording_ids = np.asarray(recording_ids, dtype=str)
        self.names = [name for name, _ in sorted(classes.items(), key=lambda x: x[1])]
        self.k = int(k)
        self.scales = np.ones(len(self.names)) if scales is None else np.asarray(scales, dtype=float)
        if (self.reference.ndim != 2 or not len(self.reference) or
                len(self.labels) != len(self.reference) or len(self.recording_ids) != len(self.reference) or
                self.k < 1 or set(self.labels) != set(self.names) or
                self.scales.shape != (len(self.names),) or not np.isfinite(self.scales).all() or
                (self.scales <= 0).any() or not np.isfinite(self.reference).all() or
                not np.allclose(np.linalg.norm(self.reference, axis=1), 1., atol=1e-5)):
            raise ValueError("Invalid record-balanced reference bank")
        self.groups = []
        for name in self.names:
            groups = [(identity, np.flatnonzero((self.labels == name) & (self.recording_ids == identity)))
                      for identity in sorted(set(self.recording_ids[self.labels == name]))]
            if len(groups) < self.k:
                raise ValueError(f"Need k={self.k} independent reference recordings for {name}; got {len(groups)}")
            self.groups.append(groups)

    def distances(self, features, excluded_ids=None):
        features = np.asarray(features, dtype=np.float32)
        if (features.ndim != 2 or features.shape[1] != self.reference.shape[1] or
                not np.isfinite(features).all()):
            raise ValueError("Invalid query features")
        output = np.empty((len(features), len(self.names)), dtype=float)
        reference64 = self.reference.astype(np.float64)
        reference_norm = np.sum(reference64 ** 2, axis=1)
        for start in range(0, len(features), 128):
            query = features[start:start+128].astype(np.float64)
            squared = np.sum(query ** 2, axis=1)[:, None] + reference_norm[None, :] - 2. * (query @ reference64.T)
            distances = np.sqrt(np.maximum(0., squared))
            for c, groups in enumerate(self.groups):
                values = np.stack([distances[:, indices].min(axis=1) for _, indices in groups], axis=1)
                for i in range(len(query)):
                    if excluded_ids is None:
                        available = values[i]
                    else:
                        available = values[i, [identity != excluded_ids[start+i] for identity, _ in groups]]
                    # Leave-one-recording-out scales may have k-1 available records.
                    count = min(self.k, len(available))
                    output[start+i, c] = np.partition(available, count-1)[count-1] if count else np.nan
        return output

    def details(self, features):
        raw = self.distances(features)
        normalized = raw / self.scales[None, :]
        indices = normalized.argmin(axis=1)
        return normalized[np.arange(len(indices)), indices], [self.names[i] for i in indices], raw, normalized


def fit_record_scales(bank, prior_records=5.):
    """Reference-only scale fit, shared by deployed profiles and ablations."""
    if not np.isfinite(prior_records) or prior_records < 0:
        raise ValueError("profile_scale_prior_records must be finite and nonnegative")
    leave_out = bank.distances(bank.reference, bank.recording_ids)
    typical, counts = [], []
    for c, groups in enumerate(bank.groups):
        values = [float(np.max(leave_out[indices, c])) for _, indices in groups
                  if np.isfinite(leave_out[indices, c]).all()]
        typical.append(float(np.median(values)) if values else np.nan)
        counts.append(len(values))
    finite = [value for value in typical if np.isfinite(value)]
    pooled = max(float(np.median(finite)) if finite else 1., 1e-3)
    floor = max(1e-3, .05 * pooled)
    scales = []
    for value, count in zip(typical, counts):
        value = value if np.isfinite(value) else pooled
        denominator = count + prior_records
        scales.append(max(float((count * value + prior_records * pooled) / denominator
                                if denominator else pooled), floor))
    bank.scales = np.asarray(scales)
    return counts, floor


def build_record_bank(reference, ref_records, calibration, cal_records, classes, k,
                      minimum_calibration, maximum_reference, false_reject_rate, prior_records=5.):
    if not np.isfinite(prior_records) or prior_records < 0:
        raise ValueError("profile_scale_prior_records must be finite and nonnegative")
    ref_groups, cal_groups = canonical_records(ref_records), canonical_records(cal_records)
    selected, selected_records, labels, identities = [], [], [], []
    per_class = {}
    for name in classes:
        groups = sorted([g for g in ref_groups if g[0]["class_name"] == name], key=lambda g: g[0]["recording_id"])
        calibrated = [g for g in cal_groups if g[0]["class_name"] == name]
        if len(groups) < k:
            raise ValueError(f"Need at least k={k} independent reference recordings for known class {name}; got {len(groups)}")
        if len(calibrated) < minimum_calibration:
            raise ValueError(f"Need calibration tracks from {minimum_calibration} independent known recordings for {name}; got {len(calibrated)}")
        if len(groups) > maximum_reference:
            groups = [groups[int(i)] for i in np.linspace(0, len(groups)-1, maximum_reference, dtype=int)]
        for group in groups:
            # Keep distinct component modes; duplicate windows/exports add no support.
            indices = [row["index"] for row in group]
            _, keep = np.unique(reference[indices], axis=0, return_index=True)
            for j in sorted(keep.tolist()):
                row = group[j]
                selected.append(reference[row["index"]])
                selected_records.append({key: value for key, value in row.items() if key not in {"index", "class_name"}})
                labels.append(name)
                identities.append(row["recording_id"])
        per_class[name] = {"reference_recordings": len(groups), "calibration_recordings": len(calibrated),
                           "reference_tracks": sum(label == name for label in labels),
                           "calibration_tracks": sum(row["class_name"] == name for g in calibrated for row in g)}
    bank = RecordingClassBank(np.asarray(selected), labels, identities, classes, k)
    counts, floor = fit_record_scales(bank, prior_records)
    for c, (name, count) in enumerate(zip(bank.names, counts)):
        per_class[name].update(scale=float(bank.scales[c]), scale_fit_recordings=count,
                               leave_one_recording_out_k=min(k, max(0, len(bank.groups[c])-1)))
    track_scores, _, _, _ = bank.details(calibration)
    record_scores = np.asarray([max(track_scores[row["index"]] for row in group) for group in cal_groups])
    threshold = float(np.quantile(record_scores, 1-false_reject_rate, method="higher"))
    for name in classes:
        values = [score for score, group in zip(record_scores, cal_groups) if group[0]["class_name"] == name]
        per_class[name]["empirical_false_reject_rate"] = float(np.mean(np.asarray(values) > threshold))
    warnings = ["Calibration rates are empirical, not deployment guarantees."]
    if len(record_scores) * false_reject_rate < 5:
        warnings.append("Few independent calibration records in the requested tail; threshold uncertainty is high.")
    if any(count < 2 for count in counts):
        warnings.append("Some class scales have fewer than two leave-one-recording-out observations; pooled fallback used.")
    metadata = {"score_method": "record_class_knn", "score": "minimum class-normalized kth-distinct-recording distance",
                "calibration_unit": "maximum component score per independent single-device recording",
                "reference_unit": "one neighbor per original recording, multiple component prototypes retained",
                "class_scale_fit": "reference-only leave-one-recording-out; median record maxima; pooled shrinkage",
                "scale_prior_records": prior_records, "scale_floor": floor, "per_class": per_class,
                "reference_recording_count": len(set(identities)), "calibration_recording_count": len(cal_groups),
                "empirical_calibration_false_reject_rate": float(np.mean(record_scores > threshold)),
                "threshold": threshold, "warnings": warnings, "reference_components": selected_records,
                "class_order": bank.names}
    arrays = {"reference": bank.reference, "reference_labels": np.asarray(labels, dtype=str),
              "reference_recording_ids": np.asarray(identities, dtype=str), "class_scales": bank.scales,
              "calibration_scores": record_scores, "calibration_track_scores": track_scores,
              "calibration_recording_ids": np.asarray([g[0]["recording_id"] for g in cal_groups], dtype=str)}
    return metadata, arrays
