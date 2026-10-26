"""Offline discovery of anonymous classes; shared support rules for both backends."""
import argparse
from collections import Counter

import numpy as np

from util.open_set import FEATURE_VERSION, normalize_features, save_npz


def add_discovery_arguments(parser):
    parser.add_argument("--unknown_cluster_backend", choices=["hdbscan", "dcg"], default="hdbscan")
    parser.add_argument("--unknown_min_cluster_size", type=int, default=5)
    parser.add_argument("--unknown_min_samples", type=int, default=3, help="HDBSCAN only")
    parser.add_argument("--unknown_min_recordings", type=int, default=2)
    parser.add_argument("--unknown_allow_single_cluster", action=argparse.BooleanOptionalAction, default=True, help="HDBSCAN only; DCG naturally permits one community")
    parser.add_argument("--dcg_topk", type=int, default=15)
    parser.add_argument("--dcg_cosine_threshold", type=float, default=.85, help="Actual cosine similarity; no (cos+1)/2 remapping")
    parser.add_argument("--dcg_resolution", type=float, default=1.)
    parser.add_argument("--dcg_mutual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dcg_seed", type=int, default=42)


def discovery_kwargs(args):
    return {"backend": args.unknown_cluster_backend, "min_cluster_size": args.unknown_min_cluster_size,
            "min_samples": args.unknown_min_samples, "min_recordings": args.unknown_min_recordings,
            "allow_single_cluster": args.unknown_allow_single_cluster,
            "dcg_options": {"topk": args.dcg_topk, "cosine_threshold": args.dcg_cosine_threshold,
                            "resolution": args.dcg_resolution, "mutual": args.dcg_mutual, "seed": args.dcg_seed}}


def validate_discovery_options(backend="hdbscan", min_cluster_size=5, min_samples=3, min_recordings=2, dcg_options=None):
    if backend not in {"hdbscan", "dcg"}:
        raise ValueError("unknown_cluster_backend must be hdbscan or dcg")
    if min_cluster_size < 2 or min_samples < 1 or min_recordings < 1:
        raise ValueError("Need min_cluster_size >= 2, min_samples >= 1 and min_recordings >= 1")
    if backend == "dcg":
        from util.dcg_discovery import validate_dcg_options
        validate_dcg_options(**(dcg_options or {}))
    else:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise ImportError("Unknown discovery requires scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN)") from exc


def discover_unknown(candidates, min_cluster_size=5, min_samples=3, min_recordings=2,
                     allow_single_cluster=True, backend="hdbscan", dcg_options=None):
    validate_discovery_options(backend, min_cluster_size, min_samples, min_recordings, dcg_options)
    labels = np.full(len(candidates), -1, dtype=int)
    membership = np.zeros(len(candidates))
    diagnostics = {}
    features = normalize_features([item["feature"] for item in candidates]) if candidates else None
    for item in candidates:
        item["track"].update(unknown_cluster_id=None, cluster_status=None, cluster_membership=None,
                             cluster_similarity=None, display_label="Unknown_unassigned")
    if candidates and backend == "dcg":
        from util.dcg_discovery import cluster_dcg
        labels, diagnostics = cluster_dcg(features, **(dcg_options or {}))
    elif len(candidates) >= max(min_cluster_size, min_samples):
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                        allow_single_cluster=allow_single_cluster, copy=True)
        labels = model.fit_predict(features)
        membership = model.probabilities_
    groups = []
    for label in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == label).tolist()
        recordings = {candidates[i].get("recording_id", candidates[i]["path"]) for i in indices}
        reason = ("insufficient_cluster_tracks" if len(indices) < min_cluster_size else
                  "insufficient_independent_recordings" if len(recordings) < min_recordings else None)
        center = features[indices].mean(axis=0)
        if np.linalg.norm(center) <= 1e-12:
            reason = "degenerate_cluster_center"
        if reason:
            for i in indices:
                candidates[i]["track"]["cluster_status"] = reason
            continue
        groups.append((indices, normalize_features(center[None])[0]))
    groups.sort(key=lambda group: min((candidates[i]["path"], candidates[i]["track"]["track_id"]) for i in group[0]))
    clusters = []
    for number, (indices, center) in enumerate(groups, 1):
        name = f"Unknown_{number:03d}"
        similarities = features[indices] @ center
        for i, similarity in zip(indices, similarities):
            candidates[i]["track"].update(unknown_cluster_id=name, display_label=name, cluster_status="clustered",
                cluster_similarity=float(np.clip(similarity, -1., 1.)))
        support = Counter(candidates[i].get("recording_id", candidates[i]["path"]) for i in indices)
        representative = indices[int(np.argmax(similarities))]
        clusters.append({"unknown_cluster_id": name, "track_count": len(indices), "recording_count": len(support),
                         "files": sorted({candidates[i]["path"] for i in indices}),
                         "prototype": center.tolist(),
                         "representative": {"path": candidates[representative]["path"],
                                            "track_id": candidates[representative]["track"]["track_id"]}})
    for index, item in enumerate(candidates):
        item["track"]["cluster_membership"] = float(membership[index]) if backend == "hdbscan" else None
        if item["track"]["unknown_cluster_id"] is None:
            item["track"]["cluster_status"] = item["track"].get("cluster_status") or "noise_or_insufficient_support"
    clustered = sum(cluster["track_count"] for cluster in clusters)
    parameters = {"min_cluster_size": min_cluster_size, "min_recordings": min_recordings}
    if backend == "hdbscan":
        parameters.update(min_samples=min_samples, allow_single_cluster=allow_single_cluster)
    else:
        parameters.update(topk=15, cosine_threshold=.85, resolution=1., mutual=True, seed=42)
        parameters.update(dcg_options or {})
    return {"method": "sklearn.HDBSCAN" if backend == "hdbscan" else "DCG-Louvain", "backend": backend,
            "unit": "one feature per accepted track", "prototype_space": FEATURE_VERSION,
            "unknown_candidate_tracks": len(candidates), "estimated_unknown_class_count": len(clusters),
            "clustered_unknown_tracks": clustered, "unassigned_unknown_tracks": len(candidates)-clustered,
            "parameters": parameters, "diagnostics": diagnostics, "clusters": clusters,
            "interpretation": "Batch-local anonymous feature clusters, not verified physical class identities."}


def save_unknown_prototypes(path, discovery, signature, feature_dimension):
    clusters = discovery["clusters"]
    features = np.asarray([cluster["prototype"] for cluster in clusters], dtype=np.float32).reshape(-1, feature_dimension)
    metadata = {"kind": "unknown_prototypes", "schema_version": 1, "feature_version": FEATURE_VERSION,
                "signature": signature, "method": discovery["method"],
                "clusters": [{key: value for key, value in cluster.items() if key != "prototype"} for cluster in clusters]}
    save_npz(path, metadata, features=features)
