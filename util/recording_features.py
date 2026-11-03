"""Shared recording weights, primary-track selection and known-only calibration."""
from collections import Counter

import numpy as np


PRIMARY_RULE = "most valid votes, then observations, then path/track ID; independent of truth"


def record_weights(records):
    counts = Counter(r["recording_id"] for r in records)
    weights = np.asarray([1. / counts[r["recording_id"]] for r in records])
    return weights / weights.sum()


def record_groups(records):
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(row["recording_id"], []).append(i)
    return [np.asarray(indices) for _, indices in sorted(groups.items())]


def primary_track_indices(records, files, recording_ids=None):
    """A missing/invalid winning track never falls back to a secondary track."""
    lookup = {(r["path"], r["track_id"]): i for i, r in enumerate(records)}
    identities = {r["path"]: r["recording_id"] for r in records}
    identities.update(recording_ids or {})
    grouped = {}
    for source in files:
        if source["path"] in identities:
            for track in source.get("signals", []):
                grouped.setdefault(identities[source["path"]], []).append((source["path"], track))
    selected = []
    for _, items in sorted(grouped.items()):
        path, track = min(items, key=lambda pair: (-pair[1].get("valid_votes", 0),
            -pair[1].get("observation_count", 0), pair[0], str(pair[1].get("track_id", ""))))
        key = (path, track["track_id"])
        if key in lookup:
            selected.append(lookup[key])
    return np.asarray(selected, dtype=int)


def annotated_primary_indices(records):
    if any(not isinstance(r.get("is_primary"), bool) for r in records):
        raise ValueError("Primary calibration requires primary-track annotations; regenerate reference/calibration feature archives")
    selected = np.asarray([i for i, r in enumerate(records) if r["is_primary"]], dtype=int)
    identities = [records[i]["recording_id"] for i in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("More than one primary track for an independent calibration recording")
    return selected


def calibrate_scores(scores, records, unit, false_reject_rate, primary_indices=None, source_count=None):
    """All scores and identities here must come from independent known calibration."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(records),) or not len(records) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    if not 0 < false_reject_rate < 1:
        raise ValueError("known_false_reject_rate must be in (0,1)")
    if unit == "record_max":
        values = np.asarray([scores[ids].max() for ids in record_groups(records)])
        threshold = float(np.quantile(values, 1-false_reject_rate, method="higher"))
        empirical = float(np.mean(values > threshold))
    elif unit == "primary":
        indices = annotated_primary_indices(records) if primary_indices is None else primary_indices
        values = scores[indices]
        if not len(values):
            raise ValueError("No valid primary calibration tracks")
        threshold = float(np.quantile(values, 1-false_reject_rate, method="higher"))
        empirical = float(np.mean(values > threshold))
    elif unit == "record_balanced":
        values = scores
        weights = record_weights(records)
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        index = min(int(np.searchsorted(cumulative, 1-false_reject_rate, side="left")), len(values)-1)
        threshold = float(values[order[index]])
        empirical = float(weights[values > threshold].sum())
    else:
        raise ValueError("calibration_unit must be primary, record_max or record_balanced")
    n = len(set(r["recording_id"] for r in records))
    source_count = n if source_count is None else source_count
    return {"threshold": threshold, "calibration_unit": unit, "target_known_false_reject_rate": false_reject_rate,
            "calibration_recordings": n, "calibration_values": len(values),
            "calibration_source_recordings": source_count,
            "calibration_primary_without_feature": source_count-len(values) if unit == "primary" else None,
            "empirical_calibration_false_reject_rate": empirical,
            "threshold_fit": "independent_known_calibration_only",
            "warnings": ["Few independent tail recordings; empirical target is not a deployment guarantee"]
                        if n*false_reject_rate < 5 else []}, values
