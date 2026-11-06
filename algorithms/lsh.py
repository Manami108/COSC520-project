from __future__ import annotations
from collections import defaultdict
import sys
import numpy as np
from .common import prepare_vectors, distances_to_query, nearest, output_distances

class LSHIndex:
    def __init__(self, metric: str = "cosine", n_tables: int = 8, n_hashes: int = 6,
                 bucket_width: float = 2.0, fallback_candidates: int = 64,
                 seed: int = 42):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("Invalid metric")
        if n_tables < 1 or n_hashes < 1 or n_hashes > 63:
            raise ValueError("n_tables must be positive and n_hashes between 1 and 63")
        if bucket_width <= 0 or fallback_candidates < 0:
            raise ValueError("bucket_width must be positive and fallback_candidates nonnegative")
        self.metric, self.n_tables, self.n_hashes = metric, n_tables, n_hashes
        self.bucket_width, self.fallback_candidates = bucket_width, fallback_candidates
        self.seed = seed
        self.vectors = None
        self._projections = None
        self._offsets = None
        self._tables = []
        self._fallback_ids = None

    def _hash_rows(self, rows: np.ndarray, table_id: int):
        scores = rows @ self._projections[table_id].T
        if self.metric == "cosine":
            # Bit packing into signed-safe 64-bit integers.
            weights = np.left_shift(np.int64(1), np.arange(self.n_hashes, dtype=np.int64))
            return ((scores >= 0).astype(np.int64) * weights).sum(axis=1).tolist()
        codes = np.floor((scores + self._offsets[table_id]) / self.bucket_width)
        return [tuple(int(v) for v in row) for row in codes]

    def fit(self, vectors: np.ndarray) -> "LSHIndex":
        self.vectors = prepare_vectors(vectors, self.metric)
        rng = np.random.default_rng(self.seed)
        n, d = self.vectors.shape
        self._projections = rng.normal(size=(self.n_tables, self.n_hashes, d)).astype(np.float32)
        self._offsets = (rng.uniform(0, self.bucket_width, size=(self.n_tables, self.n_hashes))
                         .astype(np.float32) if self.metric == "euclidean" else None)
        self._tables = []
        # Keep table-building bounded in memory for million-vector inputs.
        for table_id in range(self.n_tables):
            buckets = defaultdict(list)
            for start in range(0, n, 4096):
                hashes = self._hash_rows(self.vectors[start:start + 4096], table_id)
                for j, h in enumerate(hashes, start):
                    buckets[h].append(j)
            self._tables.append(dict(buckets))
        count = min(n, max(self.fallback_candidates, 1))
        self._fallback_ids = rng.choice(n, size=count, replace=False).astype(np.int64)
        return self

    @property
    def index_bytes(self) -> int:
        if self.vectors is None:
            return 0
        # Estimated Python hash tables; actual heap footprint depends on Python runtime.
        ids_bytes = self.n_tables * len(self.vectors) * (8 + sys.getsizeof(0))
        key_bytes = sum(len(t) * 72 for t in self._tables)
        return int(self.vectors.nbytes + self._projections.nbytes + ids_bytes + key_bytes)

    def query(self, queries: np.ndarray, k: int = 10):
        if self.vectors is None:
            raise RuntimeError("Call fit() first")
        if not 1 <= k <= len(self.vectors):
            raise ValueError("k out of range")
        queries = prepare_vectors(queries, self.metric)
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query dimensions differ")
        results, distances = [], []
        for q in queries:
            candidates = set()
            for table_id, buckets in enumerate(self._tables):
                key = self._hash_rows(q[None], table_id)[0]
                candidates.update(buckets.get(key, ()))
            # Deterministic fallback sample, not full dataset search.
            if len(candidates) < max(k, self.fallback_candidates):
                candidates.update(self._fallback_ids.tolist())
            if len(candidates) < k:
                # k may exceed the configured fallback size.
                candidates.update(range(min(k, len(self.vectors))))
            ids = np.fromiter(sorted(candidates), dtype=np.int64)
            ds = distances_to_query(self.vectors[ids], q, self.metric)
            best_ids, best_ds = nearest(ds, ids, k)
            results.append(best_ids)
            distances.append(output_distances(best_ds, self.metric))
        return np.stack(results), np.stack(distances)
