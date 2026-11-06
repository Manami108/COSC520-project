from __future__ import annotations
import numpy as np
from .common import prepare_vectors, distances_to_query, nearest, output_distances

class IVFIndex:
    def __init__(self, metric: str = "euclidean", n_clusters: int = 64,
                 nprobe: int = 4, train_size: int = 10000,
                 max_iter: int = 15, seed: int = 42):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("Invalid metric")
        if n_clusters < 1 or nprobe < 1 or nprobe > n_clusters or train_size < n_clusters:
            raise ValueError("Invalid cluster, nprobe or train_size")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.metric, self.n_clusters, self.nprobe = metric, n_clusters, nprobe
        self.train_size, self.max_iter, self.seed = train_size, max_iter, seed
        self.vectors = None
        self.centroids = None
        self.lists = None

    def _assign(self, points: np.ndarray) -> np.ndarray:
        centroids = self.centroids
        if self.metric == "cosine":
            dis = 1.0 - points @ centroids.T
        else:
            sq = np.sum(points * points, axis=1, keepdims=True)
            c_sq = np.sum(centroids * centroids, axis=1)[None, :]
            dis = sq + c_sq - 2.0 * points @ centroids.T
        return np.argmin(dis, axis=1).astype(np.int32)

    def fit(self, vectors: np.ndarray) -> "IVFIndex":
        self.vectors = prepare_vectors(vectors, self.metric)
        n, d = self.vectors.shape
        if self.n_clusters > n:
            raise ValueError("n_clusters exceeds number of vectors")
        rng = np.random.default_rng(self.seed)
        train_ids = rng.choice(n, size=min(n, self.train_size), replace=False)
        train = self.vectors[train_ids]
        self.centroids = train[rng.choice(len(train), self.n_clusters, replace=False)].copy()
        for _ in range(self.max_iter):
            counts = np.zeros(self.n_clusters, dtype=np.int64)
            sums = np.zeros((self.n_clusters, d), dtype=np.float64)
            for start in range(0, len(train), 4096):
                pts = train[start:start + 4096]
                labels = self._assign(pts)
                np.add.at(counts, labels, 1)
                np.add.at(sums, labels, pts)
            updated = self.centroids.copy()
            nonempty = counts > 0
            updated[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
            empty = np.flatnonzero(~nonempty)
            if len(empty):
                updated[empty] = train[rng.choice(len(train), len(empty), replace=True)]
            if self.metric == "cosine":
                updated = prepare_vectors(updated, "cosine")
            if np.allclose(updated, self.centroids, rtol=0, atol=1e-5):
                self.centroids = updated
                break
            self.centroids = updated
        inv_lists = [[] for _ in range(self.n_clusters)]
        for start in range(0, n, 8192):
            labels = self._assign(self.vectors[start:start + 8192])
            for offset, label in enumerate(labels):
                inv_lists[int(label)].append(start + offset)
        self.lists = [np.asarray(bucket, dtype=np.int64) for bucket in inv_lists]
        return self

    @property
    def index_bytes(self) -> int:
        if self.vectors is None:
            return 0
        return int(self.vectors.nbytes + self.centroids.nbytes + sum(x.nbytes for x in self.lists))

    def query(self, queries: np.ndarray, k: int = 10):
        if self.vectors is None:
            raise RuntimeError("Call fit() first")
        if not 1 <= k <= len(self.vectors):
            raise ValueError("k out of range")
        queries = prepare_vectors(queries, self.metric)
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query dimensions differ")
        found, distances = [], []
        for q in queries:
            cdist = distances_to_query(self.centroids, q, self.metric)
            centroid_order = np.argsort(cdist, kind="stable")
            selected = []
            count = 0
            for j, center_id in enumerate(centroid_order):
                group = self.lists[int(center_id)]
                selected.append(group)
                count += len(group)
                if j + 1 >= self.nprobe and count >= k:
                    break
            ids = np.concatenate(selected)
            ds = distances_to_query(self.vectors[ids], q, self.metric)
            nearest_ids, nearest_ds = nearest(ds, ids, k)
            found.append(nearest_ids)
            distances.append(output_distances(nearest_ds, self.metric))
        return np.stack(found), np.stack(distances)
