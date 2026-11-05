"""Independent-record density and a label-free single-root HDBSCAN check."""
import numpy as np

ALGORITHM_VERSION = "record-density-root-check-v1"
MIN_SPLIT_COVERAGE = .5
MIN_SPLIT_SILHOUETTE = .25


def independent_reachability(features, recording_ids, min_samples):
    """kNN density counts each original recording once; no signal is discarded.

    HDBSCAN counts the query itself in min_samples. The closest component of a
    recording supplies that recording's one neighbor distance. Feeding mutual
    reachability to HDBSCAN with min_samples=1 preserves this chosen core radius.
    """
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(np.asarray(features, dtype=np.float64), metric="euclidean")
    np.fill_diagonal(distances, 0.)
    names, codes = np.unique(recording_ids, return_inverse=True)
    effective_k = min(min_samples, len(names))
    core = np.empty(len(features))
    for i, row in enumerate(distances):
        closest = np.full(len(names), np.inf)
        np.minimum.at(closest, codes, row)
        core[i] = np.partition(closest, effective_k-1)[effective_k-1]
    np.maximum(distances, core[:, None], out=distances)
    np.maximum(distances, core[None, :], out=distances)
    np.fill_diagonal(distances, 0.)
    return distances, {"density_unit": "independent_recording", "density_recordings": len(names),
                         "effective_density_neighbors": effective_k}


def supported_labels(labels, recording_ids, min_cluster_size, min_recordings):
    return [label for label in sorted(set(labels) - {-1})
            if np.sum(labels == label) >= min_cluster_size and
            len({recording_ids[i] for i in np.flatnonzero(labels == label)}) >= min_recordings]


def guarded_hdbscan(features, recording_ids, min_cluster_size, min_samples,
                    min_recordings, allow_single_cluster=True, policy="auto"):
    from sklearn.cluster import HDBSCAN
    if policy == "legacy":
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        metric="euclidean", allow_single_cluster=allow_single_cluster, copy=True)
        labels = model.fit_predict(features)
        return labels, model.probabilities_, {"algorithm_version": "legacy-track-density",
                                             "density_unit": "track"}
    matrix, diagnostics = independent_reachability(features, recording_ids, min_samples)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="precomputed",
                    allow_single_cluster=allow_single_cluster, copy=True)
    labels = model.fit_predict(matrix)
    membership = model.probabilities_
    initial = supported_labels(labels, recording_ids, min_cluster_size, min_recordings)
    guard = {"checked": False, "accepted_split": False, "initial_supported_clusters": len(initial),
             "min_record_balanced_coverage": MIN_SPLIT_COVERAGE,
             "min_record_balanced_silhouette": MIN_SPLIT_SILHOUETTE}
    diagnostics.update(algorithm_version=ALGORITHM_VERSION, single_cluster_check=guard)
    # A real single unknown class is valid. Only replace a root when supported
    # child clusters cover enough records and have separated feature geometry.
    if allow_single_cluster and len(set(labels) - {-1}) == 1 and len(initial) == 1:
        guard["checked"] = True
        alternate = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="precomputed",
                            allow_single_cluster=False, copy=True)
        children = alternate.fit_predict(matrix)
        valid = supported_labels(children, recording_ids, min_cluster_size, min_recordings)
        guard["supported_child_clusters"] = len(valid)
        if len(valid) >= 2:
            from util.recording_features import record_weights
            from sklearn.metrics import silhouette_samples
            keep = np.isin(children, valid)
            records = [{"recording_id": r} for r in recording_ids]
            coverage = float(record_weights(records)[keep].sum())
            selected = np.flatnonzero(keep)
            # Bound only the diagnostic cost, not density fitting or outputs.
            if len(selected) > 2000:
                selected = np.random.default_rng(42).choice(selected, 2000, replace=False)
            selected_labels = children[selected]
            if 1 < len(set(selected_labels)) < len(selected):
                samples = silhouette_samples(features[selected], selected_labels, metric="euclidean")
                weights = record_weights([records[i] for i in selected])
                silhouette = float(np.sum(weights * samples))
            else:
                silhouette = -1.
            guard.update(record_balanced_coverage=coverage, record_balanced_silhouette=silhouette)
            if coverage >= MIN_SPLIT_COVERAGE and silhouette >= MIN_SPLIT_SILHOUETTE:
                labels, membership = children, alternate.probabilities_
                guard.update(accepted_split=True, reason="supported_separated_children")
            else:
                guard["reason"] = "insufficient_coverage_or_separation"
        else:
            guard["reason"] = "no_supported_multicluster_evidence"
    return labels, membership, diagnostics
