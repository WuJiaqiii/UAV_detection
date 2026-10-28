"""Frozen known-only kNN rejection, track feature archives and profile calibration.

All distance thresholds come from a disjoint known calibration split. Unknown
test features are used only for inference/discovery, never to fit this profile.
"""
import hashlib
import json
import math
from importlib.metadata import version
from collections import Counter
from pathlib import Path

import numpy as np
from util.recordings import file_sha256, recording_identity

FEATURE_VERSION = "mask-penultimate-l2-accepted-track-mean-l2-v1"


def normalize_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Features must be finite [N, D] vectors")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= 1e-12).any():
        raise ValueError("Zero embeddings cannot be used for open-set inference")
    return (x / norm).astype(np.float32)


def inference_signature(args, config):
    """Bind weights, actual feature-producing source and effective settings.

    Paths/devices/output names and truth labels do not affect the representation.
    Recording identity metadata is checked separately when building a profile.
    """
    root = Path(__file__).resolve().parents[1]
    sources = ["model/resnet.py", "util/mask_input.py", "util/preprocess.py", "util/detector.py",
               "util/spectrogram_frontend.py", "data/iq_dat_loader.py",
               "util/single_signal_eval.py", "util/multi_signal_eval.py", "util/open_set.py",
               "util/eval_common.py", "util/group_predictor.py", "util/recordings.py"]
    settings = {key: value for key, value in vars(config).items()
                if key not in {"device", "yolo_weights", "frontend_config"}}
    options = {key: getattr(args, key) for key in (
        "iq_stride_ms", "tail_policy", "max_slices", "vote_method", "min_slice_confidence",
        "min_votes", "min_vote_share", "exclude_vote_classes", "track_freq_tolerance_mhz",
        "track_min_band_iou", "track_max_gap_ms")}
    payload = {"feature_version": FEATURE_VERSION, "config": settings, "options": options,
               "checkpoint_sha256": file_sha256(args.checkpoint_path),
               "detector_sha256": file_sha256(args.yolo_weights),
               "source_sha256": {name: file_sha256(root/name) for name in sources},
               "yolov5_source_sha256": {file.relative_to(root).as_posix(): file_sha256(file)
                                        for file in sorted((root/"yolov5").rglob("*.py"))},
               "libraries": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn")}}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "details": payload}


def save_npz(path, metadata, **arrays):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("Archive/profile path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, ensure_ascii=False,
                             sort_keys=True, allow_nan=False)), **arrays)


def load_npz(path, kind):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    if metadata.get("kind") != kind or metadata.get("schema_version") != 1:
        raise ValueError(f"Expected a version-1 {kind} archive")
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature representation version mismatch; regenerate archives/profile")
    return metadata, arrays


def track_feature(track, groups):
    """One representative per associated track, independent of slice count."""
    accepted = [group for group in groups if group.get("vote_accepted")]
    if track.get("status") != "ok" or not accepted:
        return None, []
    if "_feature" in track:
        return normalize_features([track["_feature"]])[0], accepted
    if any("embedding" not in group for group in accepted):
        raise ValueError("Feature mode requires an embedding for every accepted observation")
    features = normalize_features([group["embedding"] for group in accepted])
    center = normalize_features(features.mean(axis=0, keepdims=True))[0]
    return center, accepted


def iter_track_features(results, rows_by_file):
    for result in results:
        groups_by_track = {}
        for row in rows_by_file[result["path"]]:
            for group in row.get("groups", []):
                groups_by_track.setdefault(group["track_id"], []).append(group)
        for track in result["signals"]:
            groups = groups_by_track.get(track["track_id"], [])
            center, accepted = track_feature(track, groups)
            yield result, track, center, accepted, groups


def cache_track_features(results, rows_by_file):
    """Release per-observation vectors between files during reference extraction."""
    for result, track, center, accepted, groups in iter_track_features(results, rows_by_file):
        if center is not None:
            track["_feature"] = center.tolist()
        for group in groups:
            group.pop("embedding", None)


def export_feature_archive(path, signature, classes, results, rows_by_file, recording_ids=None):
    features, records, skipped = [], [], 0
    identities = {}
    for result, track, center, accepted, groups in iter_track_features(results, rows_by_file):
        if center is None:
            skipped += 1
            continue
        source = result["path"]
        if source not in identities:
            digest = file_sha256(source)
            identities[source] = (digest, recording_identity(source, digest, recording_ids))
        digest, identity = identities[source]
        labels = result.get("true_class_names")
        # Multi-label files cannot supply unambiguous per-track known truth.
        truth = labels[0] if labels and len(labels) == 1 else None
        records.append({"path": source, "source_sha256": digest, "recording_id": identity,
                        "track_id": track["track_id"], "true_class_name": truth,
                        "known_candidate_class": track["class_name"],
                        "observation_count": len(accepted)})
        features.append(center)
    if not features:
        raise ValueError("No valid track features to export; inspect detections, groups and votes")
    metadata = {"schema_version": 1, "kind": "track_features", "feature_version": FEATURE_VERSION,
                "signature": signature, "classes": classes, "records": records,
                "skipped_tracks": skipped, "unit": "one L2-normalized mean per accepted track"}
    save_npz(path, metadata, features=normalize_features(features))
    return {"path": str(path), "track_count": len(records), "skipped_tracks": skipped}


def knn_scores(features, reference, k):
    from sklearn.neighbors import NearestNeighbors
    query, bank = normalize_features(features), normalize_features(reference)
    if query.shape[1] != bank.shape[1] or isinstance(k, bool) or not 1 <= k <= len(bank):
        raise ValueError("Invalid feature dimension or kNN neighbor count")
    # Euclidean distances on the unit sphere. No test-dependent standardization/PCA.
    neighbors = NearestNeighbors(n_neighbors=int(k), metric="euclidean", algorithm="brute").fit(bank)
    return neighbors.kneighbors(query, return_distance=True)[0][:, -1]


def _validate_feature_archive(metadata, arrays):
    records = metadata["records"]
    features = normalize_features(arrays["features"])
    if len(records) != len(features) or not records:
        raise ValueError("Archive records/features must be nonempty and aligned")
    classes = metadata["classes"]
    for record in records:
        if record.get("true_class_name") not in classes:
            raise ValueError("Profile inputs require single-known-class truth for every track; "
                             "use clean single-signal files and --label_from_filename/--label_from_parent/--labels_json")
        if not record.get("recording_id") or not record.get("source_sha256"):
            raise ValueError("Missing source/recording identity")
    return features, records


def build_profile(reference_path, calibration_path, output_path, k=5, known_false_reject_rate=.01,
                  min_calibration_per_class=20, max_reference_per_class=2000, ignore_classes=()):
    if not 0 < known_false_reject_rate < 1 or min_calibration_per_class < 1 or max_reference_per_class < k or k < 1:
        raise ValueError("Need 0 < known_false_reject_rate < 1 and positive counts")
    ref_meta, ref_arrays = load_npz(reference_path, "track_features")
    cal_meta, cal_arrays = load_npz(calibration_path, "track_features")
    if ref_meta["signature"] != cal_meta["signature"] or ref_meta["classes"] != cal_meta["classes"]:
        raise ValueError("Reference/calibration representation or class mapping mismatch")
    reference, ref_records = _validate_feature_archive(ref_meta, ref_arrays)
    calibration, cal_records = _validate_feature_archive(cal_meta, cal_arrays)
    for key in ("recording_id", "source_sha256"):
        shared = {row[key] for row in ref_records} & {row[key] for row in cal_records}
        if shared:
            raise ValueError(f"Reference and calibration overlap by {key}; split original recordings first")
    classes = ref_meta["classes"]
    ignored = set(ignore_classes)
    if ignored - classes.keys() or len(ignored) >= len(classes):
        raise ValueError("ignore_classes must be known model classes and leave at least one active class")
    ref_keep = [i for i, record in enumerate(ref_records) if record["true_class_name"] not in ignored]
    cal_keep = [i for i, record in enumerate(cal_records) if record["true_class_name"] not in ignored]
    reference, calibration = reference[ref_keep], calibration[cal_keep]
    ref_records, cal_records = [ref_records[i] for i in ref_keep], [cal_records[i] for i in cal_keep]
    ref_labels = np.array([row["true_class_name"] for row in ref_records])
    cal_labels = np.array([row["true_class_name"] for row in cal_records])
    selected, per_class = [], {}
    for name in classes:
        if name in ignored:
            continue
        indices = np.flatnonzero(ref_labels == name)
        if len(indices) < k:
            raise ValueError(f"Need at least k={k} reference tracks for known class {name}")
        n_cal = int(np.sum(cal_labels == name))
        n_recordings = len({record["recording_id"] for record in cal_records if record["true_class_name"] == name})
        if n_recordings < min_calibration_per_class:
            raise ValueError(f"Need calibration tracks from {min_calibration_per_class} independent known recordings for {name}; got {n_recordings}")
        if len(indices) > max_reference_per_class:
            indices = indices[np.linspace(0, len(indices)-1, max_reference_per_class, dtype=int)]
        selected.extend(indices.tolist())
        per_class[name] = {"reference_tracks": len(indices), "calibration_tracks": n_cal,
                           "calibration_recordings": n_recordings}
    reference = reference[selected]
    if len(reference) < k:
        raise ValueError("Reference cap is smaller than k")
    scores = knn_scores(calibration, reference, k)
    # Conservative empirical upper quantile; ties at the threshold remain known.
    threshold = float(np.quantile(scores, 1-known_false_reject_rate, method="higher"))
    for name in per_class:
        per_class[name]["empirical_false_reject_rate"] = float(np.mean(scores[cal_labels == name] > threshold))
    warnings = []
    if len(scores)*known_false_reject_rate < 5:
        warnings.append("Few calibration samples in the requested tail; threshold uncertainty is high.")
    warnings.append("Calibration rate is empirical, not a guarantee of deployment FPR or unknown recall.")
    metadata = {"schema_version": 1, "kind": "open_set_profile", "feature_version": FEATURE_VERSION,
                "signature": ref_meta["signature"], "classes": classes, "ignore_classes": sorted(ignored), "k": int(k),
                "score": "kth Euclidean neighbor distance on L2-normalized track embeddings",
                "threshold": threshold, "target_known_false_reject_rate": known_false_reject_rate,
                "empirical_calibration_false_reject_rate": float(np.mean(scores > threshold)),
                "per_class": per_class, "warnings": warnings,
                "reference_recording_count": len({r["recording_id"] for r in ref_records}),
                "calibration_recording_count": len({r["recording_id"] for r in cal_records}),
                "reference_archive_sha256": file_sha256(reference_path),
                "calibration_archive_sha256": file_sha256(calibration_path)}
    save_npz(output_path, metadata, reference=reference, calibration_scores=scores,
             reference_labels=ref_labels[selected])
    return metadata


class OpenSetProfile:
    def __init__(self, path, signature, classes):
        self.metadata, arrays = load_npz(path, "open_set_profile")
        if self.metadata["signature"] != signature or self.metadata["classes"] != classes:
            raise ValueError("Open-set profile does not match weights, source, classes or inference settings; rebuild it")
        self.reference = normalize_features(arrays["reference"])
        self.ignore_classes = set(self.metadata.get("ignore_classes", []))
        self.k, self.threshold = self.metadata["k"], self.metadata["threshold"]
        if not isinstance(self.k, int) or not 1 <= self.k <= len(self.reference) or not math.isfinite(self.threshold):
            raise ValueError("Invalid profile k/threshold")
        from sklearn.neighbors import NearestNeighbors
        self.neighbors = NearestNeighbors(n_neighbors=self.k, metric="euclidean", algorithm="brute").fit(self.reference)

    def score(self, features):
        query = normalize_features(features)
        if query.shape[1] != self.reference.shape[1]:
            raise ValueError("Profile/query feature dimension mismatch")
        return self.neighbors.kneighbors(query, return_distance=True)[0][:, -1]


def apply_open_set(profile, results, rows_by_file, min_novel_observation_share=.6):
    if not 0 <= min_novel_observation_share <= 1:
        raise ValueError("min_novel_observation_share must be in [0, 1]")
    candidates = []
    items = list(iter_track_features(results, rows_by_file))
    ignored = getattr(profile, "ignore_classes", set())
    valid = [item for item in items if item[2] is not None and item[1].get("class_name") not in ignored]
    scores = iter(profile.score([item[2] for item in valid])) if valid else iter([])
    for result, track, center, accepted, groups in items:
        track.update(closed_set_status=track["status"], known_candidate_class=track.get("class_name") or track.get("candidate_class_name"),
                     recognition_status="uncertain", unknown_cluster_id=None, ood_score=None,
                     ood_threshold=profile.threshold, novel_observation_share=None)
        if track["known_candidate_class"] in ignored:
            track.update(status="unknown", class_name=None, class_id=None, unknown_reason="ignored_profile_class")
        elif center is not None:
            score = float(next(scores))
            share = float(np.mean(profile.score([group["embedding"] for group in accepted]) > profile.threshold))
            track.update(ood_score=score, novel_observation_share=share)
            if score <= profile.threshold:
                track.update(recognition_status="known", display_label=track["class_name"])
            elif share >= min_novel_observation_share:
                track.update(status="unknown", recognition_status="novel_candidate", display_label="Unknown_unassigned",
                             class_name=None, class_id=None, unknown_reason="outside_known_feature_distribution")
                candidates.append({"path": result["path"], "track": track, "feature": center})
            else:
                track.update(status="unknown", class_name=None, class_id=None, unknown_reason="inconsistent_novelty_across_observations")
        if track["recognition_status"] == "uncertain":
            track["display_label"] = "uncertain"
        for group in groups:
            group.update(recognition_status=track["recognition_status"], display_label=track["display_label"],
                         known_candidate_class=group["class_name"], track_ood_score=track["ood_score"],
                         unknown_cluster_id=None)
    return candidates


def refresh_open_set_results(results, rows_by_file):
    for result in results:
        by_track = {track["track_id"]: track for track in result["signals"]}
        for row in rows_by_file[result["path"]]:
            for group in row.get("groups", []):
                track = by_track[group["track_id"]]
                for key in ("recognition_status", "display_label", "unknown_cluster_id", "cluster_status", "cluster_similarity", "cluster_membership"):
                    group[key] = track.get(key)
        counts = Counter(track["recognition_status"] for track in result["signals"])
        result.update(predicted_classes=sorted({track["class_name"] for track in result["signals"]
                                               if track["recognition_status"] == "known"}),
                      predicted_unknown_classes=sorted({track["unknown_cluster_id"] for track in result["signals"]
                                                        if track.get("unknown_cluster_id")}),
                      known_track_count=counts["known"], novel_track_count=counts["novel_candidate"],
                      uncertain_track_count=counts["uncertain"],
                      accepted_track_count=counts["known"] + counts["novel_candidate"])
        if result["status"] != "error":
            result["status"] = "ok" if result["accepted_track_count"] else "uncertain" if result["signals"] else "no_signal"
        if result["status"] == "no_signal":
            result["no_signal_reason"] = ("no_detection" if not any(row.get("detected_boxes", 0) for row in rows_by_file[result["path"]])
                                          else "no_valid_group")
        # Existing file-class-set accuracy cannot score anonymous novel IDs.
        if result.get("true_class_names") is not None:
            result["class_set_correct"] = None
