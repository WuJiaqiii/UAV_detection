"""Offline discovery of anonymous classes; shared support rules for both backends."""
import argparse
from collections import Counter

import numpy as np

from util.open_set import FEATURE_VERSION, normalize_features, save_npz


def add_discovery_arguments(parser):
    parser.add_argument('--competition_profile', choices=['unconstrained','task1_n1','task1_n2','task2'], default='unconstrained')
    parser.add_argument('--test_item', choices=['1-1','1-2','1-3','1-4','2-1','2-2'])
    parser.add_argument('--unknown_fixed_k', type=int, help='Optional label-free joint-selection result, within protocol bounds')
    parser.add_argument('--unknown_min_silhouette', type=float, default=.15)
    parser.add_argument("--unknown_input_mode", choices=["single", "multi"], default="multi",
                        help="single: one primary discovery seed per recording; multi: keep every signal track")
    parser.add_argument("--unknown_cluster_policy", choices=["auto", "legacy"], default="auto",
                        help="auto: independent-record density and evidence-based root check; legacy: original HDBSCAN")
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
            "input_mode": getattr(args, "unknown_input_mode", "multi"),
            "cluster_policy": getattr(args, "unknown_cluster_policy", "auto"),
            'competition_profile':getattr(args,'competition_profile','unconstrained'),
            'test_item':getattr(args,'test_item',None),
            'fixed_k':getattr(args,'unknown_fixed_k',None),
            'min_silhouette':getattr(args,'unknown_min_silhouette',.15),
            "min_samples": args.unknown_min_samples, "min_recordings": args.unknown_min_recordings,
            "allow_single_cluster": args.unknown_allow_single_cluster,
            "dcg_options": {"topk": args.dcg_topk, "cosine_threshold": args.dcg_cosine_threshold,
                            "resolution": args.dcg_resolution, "mutual": args.dcg_mutual, "seed": args.dcg_seed}}


def validate_discovery_options(backend="hdbscan", min_cluster_size=5, min_samples=3, min_recordings=2,
                               dcg_options=None, input_mode="multi", cluster_policy="auto"):
    if input_mode not in {"single", "multi"} or cluster_policy not in {"auto", "legacy"}:
        raise ValueError("Invalid unknown_input_mode or unknown_cluster_policy")
    if backend not in {"hdbscan", "dcg"}:
        raise ValueError("unknown_cluster_backend must be hdbscan or dcg")
    if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in
            (min_cluster_size, min_samples, min_recordings)) or
            min_cluster_size < 2 or min_samples < 1 or min_recordings < 1):
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
                     allow_single_cluster=True, backend="hdbscan", dcg_options=None,
                     input_mode="multi", cluster_policy="auto", competition_profile='unconstrained',
                     test_item=None, fixed_k=None, min_silhouette=.15):
    from util.competition_protocol import resolve_protocol
    protocol = resolve_protocol(competition_profile,input_mode,test_item)
    if fixed_k is not None and protocol is None:
        raise ValueError('unknown_fixed_k requires a bounded competition profile')
    validate_discovery_options(backend, min_cluster_size, min_samples, min_recordings,
                               dcg_options, input_mode, cluster_policy)
    if not isinstance(allow_single_cluster, bool):
        raise ValueError("unknown_allow_single_cluster must be boolean")
    identities, seen = [], set()
    for item in candidates:
        key = (item["path"], item["track"]["track_id"])
        identity = item.get("recording_id", item["path"])
        if key in seen or not isinstance(identity, str) or not identity:
            raise ValueError("Duplicate track or invalid discovery recording identity")
        seen.add(key)
        identities.append(identity)
    if input_mode == "single":
        if any(not isinstance(c["track"].get("discovery_is_primary"), bool) for c in candidates):
            raise ValueError("Single-signal discovery needs primary context; provide --results_json for old archives")
        selected = [i for i, c in enumerate(candidates) if c["track"]["discovery_is_primary"]]
        if len({identities[i] for i in selected}) != len(selected):
            raise ValueError("More than one primary discovery seed per recording")
    else:
        selected = list(range(len(candidates)))
    selected = np.asarray(selected, dtype=int)
    labels = np.full(len(selected), -1, dtype=int)
    membership = np.zeros(len(selected))
    diagnostics = {"input_mode": input_mode, "fit_tracks": len(selected),
                   "secondary_candidate_tracks": len(candidates)-len(selected)}
    features = normalize_features([item["feature"] for item in candidates]) if candidates else None
    for index, item in enumerate(candidates):
        item["track"].update(unknown_cluster_id=None, cluster_status=None, cluster_membership=None,
            cluster_similarity=None, display_label="Unknown_unassigned", cluster_assignment_source=None,
            discovery_role="track" if input_mode == "multi" else
                           "recording_primary" if item["track"]["discovery_is_primary"] else "recording_secondary")
    fit_features = features[selected] if len(selected) else None
    fit_ids = [identities[i] for i in selected]
    if protocol is not None:
        from util.bounded_discovery import bounded_partition
        fit = fit_features if fit_features is not None else np.empty((0,features.shape[1] if features is not None else 1))
        labels, bounded_info = bounded_partition(fit,fit_ids,protocol.minimum,protocol.maximum,
            min_cluster_size,min_recordings,min_silhouette,fixed_k=fixed_k)
        diagnostics.update(bounded_info)
    elif len(selected) and backend == "dcg":
        from util.dcg_discovery import cluster_dcg
        labels, dcg_info = cluster_dcg(fit_features, **(dcg_options or {}))
        diagnostics.update(dcg_info)
    elif len(selected) >= max(min_cluster_size, min_samples):
        from util.unknown_density import guarded_hdbscan
        labels, membership, density_info = guarded_hdbscan(fit_features, fit_ids, min_cluster_size,
            min_samples, min_recordings, allow_single_cluster, cluster_policy)
        diagnostics.update(density_info)
    groups = []
    for label in sorted(set(labels) - {-1}):
        indices = selected[np.flatnonzero(labels == label)].tolist()
        recordings = {candidates[i].get("recording_id", candidates[i]["path"]) for i in indices}
        reason = ("insufficient_cluster_tracks" if len(indices) < min_cluster_size else
                  "insufficient_independent_recordings" if len(recordings) < min_recordings else None)
        if cluster_policy == "auto":
            from util.recording_features import record_weights
            weights = record_weights([{"recording_id": identities[i]} for i in indices])
            center = np.average(features[indices], axis=0, weights=weights)
        else:
            center = features[indices].mean(axis=0)
        if np.linalg.norm(center) <= 1e-12:
            reason = "degenerate_cluster_center"
        if reason:
            for i in indices:
                candidates[i]["track"]["cluster_status"] = reason
            continue
        groups.append((indices, normalize_features(center[None])[0]))
    groups.sort(key=lambda group: min((candidates[i]["path"], candidates[i]["track"]["track_id"]) for i in group[0]))
    clusters, fit_clustered = [], 0
    for number, (seeds, center) in enumerate(groups, 1):
        indices = seeds
        if input_mode == "single":
            sources = {identities[i] for i in seeds}
            indices = [i for i, identity in enumerate(identities) if identity in sources]
        name = f"Unknown_{number:03d}"
        similarities = features[indices] @ center
        for i, similarity in zip(indices, similarities):
            candidates[i]["track"].update(unknown_cluster_id=name, display_label=name, cluster_status="clustered",
                cluster_similarity=float(np.clip(similarity, -1., 1.)),
                cluster_assignment_source="direct" if i in seeds else "recording_primary")
        fit_clustered += len(seeds)
        support = Counter(candidates[i].get("recording_id", candidates[i]["path"]) for i in indices)
        representative = seeds[int(np.argmax(features[seeds] @ center))]
        clusters.append({"unknown_cluster_id": name, "track_count": len(indices), "recording_count": len(support),
                         "fit_track_count": len(seeds),
                         "files": sorted({candidates[i]["path"] for i in indices}),
                         "prototype": center.tolist(),
                         "representative": {"path": candidates[representative]["path"],
                                            "track_id": candidates[representative]["track"]["track_id"]}})
    fit_lookup = {int(index): rank for rank, index in enumerate(selected)}
    for index, item in enumerate(candidates):
        item["track"]["cluster_membership"] = (float(membership[fit_lookup[index]])
            if backend == "hdbscan" and protocol is None and index in fit_lookup else None)
        if item["track"]["unknown_cluster_id"] is None:
            item["track"]["cluster_status"] = item["track"].get("cluster_status") or (
                "secondary_without_clustered_primary" if index not in fit_lookup else "noise_or_insufficient_support")
    clustered = sum(cluster["track_count"] for cluster in clusters)
    parameters = {"min_cluster_size": min_cluster_size, "min_recordings": min_recordings,
                  "input_mode": input_mode, "cluster_policy": cluster_policy,
                  'competition_profile':competition_profile,'test_item':test_item,
                  'fixed_k':fixed_k,'min_silhouette':min_silhouette}
    if backend == "hdbscan":
        parameters.update(min_samples=min_samples, allow_single_cluster=allow_single_cluster)
    else:
        parameters.update(topk=15, cosine_threshold=.85, resolution=1., mutual=True, seed=42)
        parameters.update(dcg_options or {})
    return {"method": 'bounded-average-cosine' if protocol else "sklearn.HDBSCAN" if backend == "hdbscan" else "DCG-Louvain", "backend": backend,
            "unit": "one primary seed per recording; secondary labels inherited" if input_mode == "single" else "one feature per accepted track",
            "prototype_space": FEATURE_VERSION, "input_mode": input_mode, "cluster_policy": cluster_policy,
            "fit_tracks": len(selected), "clustered_fit_tracks": fit_clustered,
            "inherited_secondary_tracks": sum(c["track"]["cluster_assignment_source"] == "recording_primary" for c in candidates),
            "unknown_candidate_tracks": len(candidates), "estimated_unknown_class_count": len(clusters),
            "clustered_unknown_tracks": clustered, "unassigned_unknown_tracks": len(candidates)-clustered,
            'constraint_satisfied':diagnostics.get('constraint_satisfied'),
            "parameters": parameters, "diagnostics": diagnostics, "clusters": clusters,
            "interpretation": "Batch-local anonymous feature clusters, not verified physical class identities."}


def save_unknown_prototypes(path, discovery, signature, feature_dimension):
    clusters = discovery["clusters"]
    features = np.asarray([cluster["prototype"] for cluster in clusters], dtype=np.float32).reshape(-1, feature_dimension)
    metadata = {"kind": "unknown_prototypes", "schema_version": 1, "feature_version": FEATURE_VERSION,
                "signature": signature, "method": discovery["method"],
                "input_mode": discovery.get("input_mode", "multi"),
                "parameters": discovery.get("parameters", {}), "diagnostics": discovery.get("diagnostics", {}),
                "clusters": [{key: value for key, value in cluster.items() if key != "prototype"} for cluster in clusters]}
    save_npz(path, metadata, features=features)
