"""Frozen-feature DCG: sparse cosine graph and seeded Louvain communities.

Adapted from AMR_NCD's graph/concept separation. No oracle identity mask, fixed K,
test-time feature whitening or network optimization is used here.
"""
import math

import numpy as np

from util.open_set import normalize_features


def validate_dcg_options(topk=15, cosine_threshold=.85, resolution=1., mutual=True, seed=42):
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError("dcg_topk must be an integer >= 1")
    if not math.isfinite(cosine_threshold) or not 0 <= cosine_threshold <= 1:
        raise ValueError("dcg_cosine_threshold must be in [0,1] (actual cosine similarity)")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("dcg_resolution must be finite and > 0")
    if not isinstance(mutual, bool) or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Need boolean dcg_mutual and integer dcg_seed >= 0")
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as exc:
        raise ImportError("DCG requires NetworkX >= 3.0; install requirements.txt") from exc
    return nx, louvain_communities


def cluster_dcg(features, topk=15, cosine_threshold=.85, resolution=1., mutual=True, seed=42):
    nx, louvain = validate_dcg_options(topk, cosine_threshold, resolution, mutual, seed)
    values = normalize_features(features)
    input_count = len(values)
    # Identical masks can give identical embeddings. Collapse exact duplicates so
    # mutual-kNN tie breaking cannot arbitrarily strand most of those samples.
    values, inverse, multiplicity = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    count = len(values)
    labels = np.full(count, -1, dtype=int)
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    if count > 1:
        from sklearn.neighbors import NearestNeighbors
        # On the unit sphere: cosine = 1 - distance**2 / 2.
        neighbors = NearestNeighbors(n_neighbors=min(topk+1, count), metric="euclidean", algorithm="brute")
        neighbors.fit(values)
        distances, indices = neighbors.kneighbors(values)
        neighborhoods = []
        for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            row = {}
            for distance, j in zip(row_distances, row_indices):
                if int(j) == i:
                    continue
                similarity = float(np.clip(1.-float(distance)**2/2., -1., 1.))
                if similarity >= cosine_threshold and similarity > 0:
                    row[int(j)] = similarity
                if len(row) >= topk:
                    break
            neighborhoods.append(row)
        for i, row in enumerate(neighborhoods):
            for j, similarity in row.items():
                if mutual and i not in neighborhoods[j]:
                    continue
                if not graph.has_edge(i, j):
                    graph.add_edge(i, j, weight=similarity)
    active = [node for node, degree in graph.degree() if degree > 0]
    communities = []
    if active:
        # Backend exceptions propagate; never fabricate K=1 on failure.
        communities = list(louvain(graph.subgraph(active).copy(), weight="weight", resolution=resolution, seed=seed))
        communities.sort(key=lambda nodes: min(nodes))
        for label, nodes in enumerate(communities):
            labels[list(nodes)] = label
    for node in range(count):
        if labels[node] < 0 and multiplicity[node] > 1:
            labels[node] = len(communities)
            communities.append({node})
    expanded = labels[inverse]
    return expanded, {"input_tracks": input_count, "graph_nodes": count,
                    "duplicate_tracks_collapsed": input_count-count, "graph_edges": graph.number_of_edges(),
                    "isolated_tracks": int(np.sum(expanded < 0)), "raw_community_count": len(communities),
                    "community_algorithm": "networkx.louvain_communities", "networkx_version": nx.__version__}
