# nprobe=32 is a good candidate for ivf
# nprobe=16 is the best of your tested configurations if 100% recall is required
# nprobe=4 is also interesting

# Maybe what we can experiment next is:
# - Increase to 1,000 queries and compare nprobe=4, 8, 12, 16.
# - Test IVF with 100K vectors and adjust the number of clusters.
# - Run brute-force k-NN, LSH, and HNSW on the same SIFT subset.
# - Compare their Recall@10, latency, build time, and memory.
# - Repeat the experiments using GloVe-100.

import numpy as np
from algorithms.knn import BruteForceKNN
from algorithms.lsh import LSHIndex
from algorithms.ivf import IVFIndex
from algorithms.hnsw import HNSWIndex
from evaluate import recall_at_k

rng = np.random.default_rng(42)
base = rng.standard_normal((1000, 32)).astype(np.float32)
queries = rng.standard_normal((20, 32)).astype(np.float32)
k = 10

for metric in ("euclidean", "cosine"):
    expected, _ = BruteForceKNN(metric=metric).fit(base).query(queries, k=k)
    print(f"\n{metric}  |  base={len(base)}  queries={len(queries)}  dimensions={base.shape[1]}  k={k}")
    indices = {
        "k-NN": BruteForceKNN(metric=metric),
        "LSH": LSHIndex(metric=metric, seed=42),
        "IVF": IVFIndex(metric=metric, seed=42),
        "HNSW": HNSWIndex(metric=metric, seed=42),}
    for name, model in indices.items():
        model.fit(base)
        got, distances = model.query(queries, k=k)
        assert got.shape == (20, k), (name, got.shape)
        assert distances.shape == (20, k), (name, distances.shape)
        assert ((got >= 0) & (got < len(base))).all(), name
        assert (np.diff(distances, axis=1) >= -1e-6).all(), name
        assert all(len(np.unique(row)) == k for row in got), name
        print(f"  {name:<5}  Recall@10={recall_at_k(expected, got, k=k):.3f}")
        if name == "HNSW":
            directed = sum(len(neighbors) for layer in model.layers for neighbors in layer.values())
            one_way = [(layer_id, u, v) for layer_id, layer in enumerate(model.layers)
                       for u, nbrs in layer.items() for v in nbrs
                       if u not in layer[v]]
            print(f"    HNSW: {directed} total directed links; {len(one_way)} one-way links")
            if one_way:
                print(f"    Example: layer {one_way[0][0]}, {one_way[0][1]} -> {one_way[0][2]} "
                      f"(but reverse missing)")
