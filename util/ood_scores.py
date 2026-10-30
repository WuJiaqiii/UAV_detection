"""Known-only fitted OOD scores. All scores increase toward unknown.

Record-balanced variants are explicitly adaptations, not benchmark reproductions.
References and exact formulas are recorded in docs/open_set_ablation.md.
"""
from collections import Counter

import numpy as np
from scipy.special import logsumexp

from util.known_reference import RecordingClassBank, fit_record_scales
from util.open_set import normalize_features

METHODS = ("record_class_knn", "record_class_knn_unscaled", "record_knn", "global_knn",
           "maha_pp_shrink", "maxlogit", "energy", "nnguide_record", "vim")
RAW_METHODS = {"maxlogit", "energy", "nnguide_record", "vim"}


def record_weights(records):
    counts = Counter(r["recording_id"] for r in records)
    weights = np.asarray([1. / counts[r["recording_id"]] for r in records])
    return weights / weights.sum()


def record_groups(records):
    result = {}
    for i, r in enumerate(records):
        result.setdefault(r["recording_id"], []).append(i)
    return [np.asarray(ids) for _, ids in sorted(result.items())]


def squared_distances(x, y):
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None] - 2 * x @ y.T, 0.)


class FittedScore:
    def __init__(self, method, reference, k=5, shrinkage=.1, scale_prior=5., vim_dim=64, head=None):
        if method not in METHODS or isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("Invalid score method or k")
        if not 0 < shrinkage <= 1 or not np.isfinite(shrinkage):
            raise ValueError("covariance_shrinkage must be in (0,1]")
        self.method, self.k = method, k
        self.reference = reference.features.astype(np.float64)
        self.groups = record_groups(reference.records)
        self.weights = record_weights(reference.records)
        self.info = {"method": method, "fit_split": "reference_only", "k": k,
                     "reference_recordings": len(self.groups), "reference_tracks": len(reference.records)}
        if method in RAW_METHODS and reference.logits is None:
            raise ValueError(f"{method} requires original logits; enable checkpoint mask replay")
        if method in {"record_class_knn", "record_class_knn_unscaled"}:
            names = [r["true_class_name"] for r in reference.records]
            self.bank = RecordingClassBank(reference.features, names, reference.recording_ids,
                                          reference.classes, k)
            if min(len(g) for g in self.bank.groups) < k:
                raise ValueError("Every class needs at least k independent reference recordings")
            if method == "record_class_knn":
                fit_record_scales(self.bank, scale_prior)
            self.info["class_scales"] = dict(zip(self.bank.names, self.bank.scales.tolist()))
        if method in {"record_knn", "nnguide_record"} and len(self.groups) < k:
            raise ValueError("k exceeds independent reference recordings")
        if method == "global_knn" and len(self.reference) < k:
            raise ValueError("k exceeds reference tracks")
        if method == "maha_pp_shrink":
            names = np.asarray([r["true_class_name"] for r in reference.records])
            self.means = []
            residual = self.reference.copy()
            for name in reference.classes:
                keep = names == name
                if not keep.any():
                    raise ValueError(f"No reference data for {name}")
                mean = np.average(self.reference[keep], axis=0, weights=self.weights[keep])
                self.means.append(mean)
                residual[keep] -= mean
            self.means = np.asarray(self.means)
            covariance = residual.T @ (self.weights[:, None] * residual)
            dimension = covariance.shape[0]
            average_variance = max(float(np.trace(covariance)/dimension), 1e-12)
            covariance = (1-shrinkage)*covariance + shrinkage*average_variance*np.eye(dimension)
            self.precision = np.linalg.inv(covariance)
            self.info.update(covariance_shrinkage=shrinkage,
                             covariance="record-weighted shared, shrunk toward isotropic; normalized features")
        if method == "nnguide_record":
            self.reference_energy = logsumexp(reference.logits, axis=1)
            self.info["adaptation"] = "best confidence-weighted cosine per recording; mean of top-k distinct recordings"
        if method == "vim":
            if head is None or reference.raw_features is None:
                raise ValueError("ViM requires original raw features and linear head parameters")
            w, b = np.asarray(head["weight"], dtype=float), np.asarray(head["bias"], dtype=float)
            raw = reference.raw_features.astype(float)
            if w.shape != (len(reference.classes), raw.shape[1]):
                raise ValueError("ViM feature/head shapes differ")
            self.origin = -np.linalg.pinv(w) @ b
            shifted = raw - self.origin
            covariance = shifted.T @ (self.weights[:, None] * shifted)
            _, vectors = np.linalg.eigh(covariance)
            rank = int(np.linalg.matrix_rank(shifted))
            if not 1 <= vim_dim < min(rank, raw.shape[1]):
                raise ValueError(f"vim_dim must be below reference rank/dimension {min(rank, raw.shape[1])}")
            self.null_space = vectors[:, :raw.shape[1]-vim_dim]
            residual = np.linalg.norm(shifted @ self.null_space, axis=1)
            mean_residual = float(np.average(residual, weights=self.weights))
            if mean_residual <= 1e-12:
                raise ValueError("ViM residual is degenerate")
            self.alpha = float(np.average(reference.logits.max(1), weights=self.weights)/mean_residual)
            if self.alpha <= 0:
                raise ValueError("ViM requires positive reference maximum-logit scale")
            self.info.update(vim_dim=vim_dim, alpha=self.alpha, score="virtual logit minus known logsumexp",
                             adaptation="record-weighted reference moments; raw accepted-observation track means")

    def score(self, split):
        x = split.features.astype(np.float64)
        if self.method in {"record_class_knn", "record_class_knn_unscaled"}:
            return self.bank.details(x)[0]
        if self.method in {"record_knn", "global_knn"}:
            values = squared_distances(x, self.reference)
            if self.method == "record_knn":
                values = np.stack([values[:, ids].min(1) for ids in self.groups], axis=1)
            return np.sqrt(np.partition(values, self.k-1, axis=1)[:, self.k-1])
        if self.method == "maha_pp_shrink":
            distance = np.stack([np.sum(((x-mean) @ self.precision) * (x-mean), axis=1)
                                 for mean in self.means], axis=1)
            return np.maximum(distance.min(1), 0.)
        if split.logits is None:
            raise ValueError(f"Missing logits for {self.method}")
        if self.method == "maxlogit":
            return -split.logits.max(1)
        if self.method == "energy":
            return -logsumexp(split.logits, axis=1)
        if self.method == "nnguide_record":
            similarities = (x @ self.reference.T) * self.reference_energy[None]
            values = np.stack([similarities[:, ids].max(1) for ids in self.groups], axis=1)
            top = np.partition(values, values.shape[1]-self.k, axis=1)[:, -self.k:]
            return -logsumexp(split.logits, axis=1) * top.mean(1)
        residual = np.linalg.norm((split.raw_features-self.origin) @ self.null_space, axis=1)
        return self.alpha * residual - logsumexp(split.logits, axis=1)


def calibrate(scores, split, unit, false_reject_rate):
    """No test scores, unknown labels or unknown ratio enter threshold fitting."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(split.records),) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    if not 0 < false_reject_rate < 1:
        raise ValueError("known_false_reject_rate must be in (0,1)")
    if unit == "record_max":
        values = np.asarray([scores[ids].max() for ids in record_groups(split.records)])
        threshold = float(np.quantile(values, 1-false_reject_rate, method="higher"))
        empirical = float(np.mean(values > threshold))
    elif unit == "primary":
        values = scores[split.primary_indices()]
        if not len(values):
            raise ValueError("No valid primary calibration tracks")
        threshold = float(np.quantile(values, 1-false_reject_rate, method="higher"))
        empirical = float(np.mean(values > threshold))
    elif unit == "record_balanced":
        values = scores
        weights = record_weights(split.records)
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        index = min(int(np.searchsorted(cumulative, 1-false_reject_rate, side="left")), len(values)-1)
        threshold = float(values[order[index]])
        empirical = float(weights[values > threshold].sum())
    else:
        raise ValueError("calibration_unit must be primary, record_max or record_balanced")
    n = len(set(split.recording_ids))
    identity_map = {r["path"]: r["recording_id"] for r in split.records}
    identity_map.update(split.source_recording_ids or {})
    source_count = len({identity_map.get(f["path"], f["path"]) for f in split.report["files"]})
    return {"threshold": threshold, "calibration_unit": unit, "target_known_false_reject_rate": false_reject_rate,
            "calibration_recordings": n, "calibration_values": len(values),
            "calibration_source_recordings": source_count,
            "calibration_primary_without_feature": source_count-len(values) if unit == "primary" else None,
            "empirical_calibration_false_reject_rate": empirical,
            "threshold_fit": "independent_known_calibration_only",
            "warnings": ["Few independent tail recordings; empirical target is not a deployment guarantee"]
                        if n*false_reject_rate < 5 else []}
