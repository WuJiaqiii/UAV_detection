"""Label-free small-K discovery on frozen features; no arbitrary ID compression."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_samples


def bounded_partition(features, identities, minimum, maximum, min_size=5,
                      min_recordings=2, min_silhouette=.15, bootstrap_runs=5, seed=42,
                      fixed_k=None):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) != len(identities) or not np.isfinite(values).all():
        raise ValueError('Invalid bounded discovery features/identities')
    if not 0 <= minimum <= maximum or maximum < 1 or not -1 <= min_silhouette <= 1:
        raise ValueError('Invalid bounded discovery limits')
    if fixed_k is not None and (isinstance(fixed_k, bool) or not max(1,minimum) <= fixed_k <= maximum):
        raise ValueError('unknown_fixed_k must be within the public protocol bounds')
    if bootstrap_runs < 1: raise ValueError('bootstrap_runs must be positive')
    # Exact duplicate components from one recording cannot inflate the fit/support.
    keys = [(str(r), tuple(row)) for r,row in zip(identities,values)]
    unique = sorted(set(keys))
    index = {key:i for i,key in enumerate(unique)}
    inverse = np.array([index[k] for k in keys], dtype=int)
    x = np.array([k[1] for k in unique], dtype=float).reshape(-1, values.shape[1])
    ids = np.array([k[0] for k in unique])
    output = np.full(len(values), -1, dtype=int)
    diagnostics = {'policy':'bounded-average-cosine-v1', 'allowed_K':[minimum,maximum],
                   'unique_fit_components':len(x), 'candidate_K_scores':[],
                   'constraint_satisfied':False, 'selected_K':None,
                   'interpretation':'Anonymous feature partitions subject to a public class budget; not verified model identities.'}
    candidates = []
    for k in ([fixed_k] if fixed_k is not None else range(max(1,minimum),maximum+1)):
        info = {'K':k, 'valid':False}
        diagnostics['candidate_K_scores'].append(info)
        if len(x) < max(k+1, k*min_size):
            info['reason']='insufficient_components'; continue
        labels = np.zeros(len(x),dtype=int) if k == 1 else AgglomerativeClustering(
            n_clusters=k, metric='cosine', linkage='average').fit_predict(x)
        supports = [len(set(ids[labels==i])) for i in range(k)]
        sizes = [int(np.sum(labels==i)) for i in range(k)]
        info.update(recording_support=supports, component_support=sizes)
        if min(supports) < min_recordings or min(sizes) < min_size:
            info['reason']='insufficient_independent_support'; continue
        centers = np.array([x[labels==i].mean(0) for i in range(k)])
        lengths = np.linalg.norm(centers,axis=1)
        if (lengths <= 1e-12).any(): info['reason']='degenerate_center'; continue
        centers /= lengths[:,None]
        if k > 1 and np.max((centers @ centers.T)-np.eye(k)*2) > 1-1e-6:
            info['reason']='indistinguishable_centers'; continue
        counts = {r:int(np.sum(ids==r)) for r in set(ids)}
        weights = np.array([1/counts[r] for r in ids]); weights/=weights.sum()
        # K=1 has no silhouette. Its fixed reference score competes with supported splits.
        silhouette = None if k == 1 else float(weights @ silhouette_samples(x,labels,metric='cosine'))
        if k > 1 and silhouette < min_silhouette:
            info.update(reason='weak_separation', silhouette=silhouette); continue
        stability = []
        rng = np.random.default_rng(seed)
        records = sorted(set(ids))
        for _ in range(bootstrap_runs):
            chosen = set(rng.choice(records,max(1,int(len(records)*.85)),replace=False))
            keep = np.array([r in chosen for r in ids])
            if keep.sum() <= k or len(set(labels[keep])) != k: continue
            fit = np.zeros(keep.sum(),dtype=int) if k == 1 else AgglomerativeClustering(
                n_clusters=k,metric='cosine',linkage='average').fit_predict(x[keep])
            stability.append(float(adjusted_rand_score(labels[keep],fit)))
        agreement = float(np.mean(stability)) if stability else 0.
        score = .35 if k == 1 else silhouette + .1*agreement - .02*(k-1)
        info.update(valid=True, silhouette=silhouette, stability=agreement, score=float(score),
                    bootstrap_evaluated=len(stability))
        candidates.append((score,-k,labels))
    if not candidates:
        diagnostics['constraint_satisfied'] = minimum == 0 and len(values) == 0
        diagnostics['reason']='no_unknown_candidates' if len(values)==0 else 'no_supported_partition'
        return output, diagnostics
    _,negative_k,labels=max(candidates,key=lambda a:a[:2])
    # Stable labels depend on features/record IDs, never ingestion order or truth.
    groups=sorted([np.flatnonzero(labels==i) for i in range(-negative_k)],key=lambda g:unique[int(g[0])])
    stable=np.full(len(x),-1,dtype=int)
    for i,g in enumerate(groups): stable[g]=i
    diagnostics.update(constraint_satisfied=True,selected_K=-negative_k)
    return stable[inverse], diagnostics
