"""Public-count-constrained merging of supported density cores, without truth.

Noise remains unresolved. Cores are never split or promoted from unsupported
points. Greedy pair merges maximize record-balanced whole-partition silhouette;
legal K candidates additionally receive independent-record subsampling checks.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples

VERSION = 'bounded-density-global-silhouette-merge-v1'
MAX_CORES = 32
MAX_SCORE_POINTS = 2000


def _canonical(labels):
    groups = sorted([np.flatnonzero(labels == c) for c in set(labels)-{-1}], key=lambda g: int(g[0]))
    stable = np.full(len(labels), -1, dtype=int)
    for c, indices in enumerate(groups):
        stable[indices] = c
    return stable


def bounded_density_partition(features, identities, minimum, maximum, min_size=5,
        min_recordings=2, min_silhouette=.15, min_samples=3, allow_single_cluster=True,
        fixed_k=None, bootstrap_runs=5, seed=42):
    from util.unknown_density import guarded_hdbscan, supported_labels
    from util.recording_features import record_weights
    values = np.asarray(features, dtype=float)
    if (values.ndim != 2 or len(values) != len(identities) or not np.isfinite(values).all()
            or values.shape[1] == 0 or any(not isinstance(r, str) or not r for r in identities)):
        raise ValueError('Invalid bounded density features/identities')
    if (not 0 <= minimum <= maximum or maximum < 1 or not -1 <= min_silhouette <= 1 or
            min_size < 2 or min_recordings < 1 or min_samples < 1 or bootstrap_runs < 0):
        raise ValueError('Invalid bounded density limits/support')
    if fixed_k is not None and (isinstance(fixed_k, bool) or not isinstance(fixed_k, int) or
                               not max(1, minimum) <= fixed_k <= maximum):
        raise ValueError('unknown_fixed_k must be within public bounds')
    if len(values) and not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5):
        raise ValueError('Bounded density expects L2-normalized features')
    keys = [(r, tuple(row)) for r, row in zip(identities, values)]
    unique = sorted(set(keys)); lookup = {key: i for i, key in enumerate(unique)}
    inverse = np.array([lookup[key] for key in keys], dtype=int)
    x = np.array([k[1] for k in unique], dtype=float).reshape(-1, values.shape[1])
    ids = np.array([k[0] for k in unique])
    output = np.full(len(values), -1, dtype=int)
    diagnostics = dict(policy=VERSION, allowed_K=[minimum, maximum], unique_fit_components=len(x),
        candidate_K_scores=[], constraint_satisfied=False, selected_K=None, merge_path=[],
        interpretation='Greedy supported-core merges constrained by public count bounds; no class truth is used.')
    if len(x) < max(min_size, min_samples):
        diagnostics.update(constraint_satisfied=minimum == 0 and not len(x),
            reason='no_unknown_candidates' if not len(x) else 'insufficient_components')
        return output, diagnostics
    labels, _, density = guarded_hdbscan(x, ids.tolist(), min_size, min_samples,
        min_recordings, allow_single_cluster, policy='auto')
    supported = supported_labels(labels, ids.tolist(), min_size, min_recordings)
    labels = _canonical(np.where(np.isin(labels, supported), labels, -1))
    seed_labels = labels.copy()
    count = len(supported)
    weights = record_weights([dict(recording_id=r) for r in ids])
    diagnostics.update(density=density, initial_supported_cores=count,
        noise_components=int(np.sum(labels < 0)), record_balanced_coverage=float(np.clip(weights[labels >= 0].sum(), 0, 1)),
        max_seed_cores=MAX_CORES)
    stop = fixed_k if fixed_k is not None else max(1, minimum)
    if count < stop or count > MAX_CORES:
        diagnostics['reason'] = 'insufficient_density_cores' if count < stop else 'too_many_density_cores'
        return output, diagnostics
    # Deterministic diagnostic sample only; all retained points receive labels.
    score_indices = np.flatnonzero(labels >= 0)
    if len(score_indices) > MAX_SCORE_POINTS:
        score_indices = np.sort(np.random.default_rng(seed).choice(score_indices, MAX_SCORE_POINTS, replace=False))
    matrix = np.clip(pairwise_distances(x[score_indices], metric='cosine'), 0, 2)
    np.fill_diagonal(matrix, 0.)
    score_weights = record_weights([dict(recording_id=ids[i]) for i in score_indices])
    diagnostics['silhouette_sample_components'] = len(score_indices)

    def separation(partition):
        k = len(set(partition)-{-1})
        if k == 1:
            return None
        sample = partition[score_indices]
        if len(set(sample)) != k or len(sample) <= k:
            return -1.
        return float(score_weights @ silhouette_samples(matrix, sample, metric='precomputed'))

    partitions = {}
    while count >= stop:
        if minimum <= count <= maximum and (fixed_k is None or count == fixed_k):
            partitions[count] = labels.copy()
        if count == stop:
            break
        options = []
        for a in range(count):
            for b in range(a+1, count):
                trial = labels.copy(); trial[trial == b] = a; trial = _canonical(trial)
                s = separation(trial)
                options.append((.35 if s is None else s, -a, -b, trial))
        previous = labels
        quality, a, b, labels = max(options, key=lambda item: item[:3])
        diagnostics['merge_path'].append(dict(from_K=count, to_K=count-1,
            left_core_ids=sorted(int(c) for c in set(seed_labels[previous == -a])),
            right_core_ids=sorted(int(c) for c in set(seed_labels[previous == -b])),
            merged_cluster_indices=[-a, -b], silhouette_after=quality))
        count -= 1
    candidates = []
    for k, partition in sorted(partitions.items()):
        info = dict(K=k, valid=False,
            component_support=[int(np.sum(partition == c)) for c in range(k)],
            recording_support=[len(set(ids[partition == c])) for c in range(k)])
        diagnostics['candidate_K_scores'].append(info)
        s = separation(partition)
        if k > 1 and s < min_silhouette:
            info.update(reason='weak_separation', silhouette=s); continue
        centers = np.array([x[partition == c].mean(0) for c in range(k)])
        lengths = np.linalg.norm(centers, axis=1)
        if (lengths < 1e-12).any():
            info['reason'] = 'degenerate_center'; continue
        centers /= lengths[:, None]
        if k > 1 and np.max(centers @ centers.T - 2*np.eye(k)) > 1-1e-6:
            info['reason'] = 'indistinguishable_centers'; continue
        stability, successful = [], 0
        records = sorted(set(ids)); rng = np.random.default_rng(seed)
        for _ in range(bootstrap_runs):
            chosen = set(rng.choice(records, max(1, int(len(records)*.85)), replace=False))
            keep = np.array([r in chosen for r in ids])
            alternative, details = bounded_density_partition(x[keep], ids[keep].tolist(), minimum, maximum,
                min_size, min_recordings, min_silhouette, min_samples, allow_single_cluster,
                fixed_k=k, bootstrap_runs=0, seed=seed)
            shared = (partition[keep] >= 0) & (alternative >= 0)
            if (details['constraint_satisfied'] and shared.sum() > k and
                    len(set(partition[keep][shared])) == k and len(set(alternative[shared])) == k):
                # A fit covering very few common records cannot look perfectly stable.
                overlap = float(np.sum(weights[keep][shared]) / max(np.sum(weights[keep][partition[keep] >= 0]), 1e-12))
                stability.append(float(adjusted_rand_score(partition[keep][shared], alternative[shared])) * overlap)
                successful += 1
            else:
                stability.append(0.)
        agreement = float(np.mean(stability)) if stability else 0.
        score = .35 if k == 1 else s + .1*agreement - .02*(k-1)
        info.update(valid=True, silhouette=s, stability=agreement, score=float(score),
            bootstrap_evaluated=bootstrap_runs, bootstrap_successful=successful)
        candidates.append((score, -k, partition))
    if not candidates:
        diagnostics['reason'] = 'no_supported_partition'
        return output, diagnostics
    _, negative_k, selected = max(candidates, key=lambda item: item[:2])
    diagnostics.update(constraint_satisfied=True, selected_K=-negative_k)
    return selected[inverse], diagnostics
