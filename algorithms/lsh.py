from __future__ import annotations
from collections import defaultdict
import sys
import numpy as np
from .common import prepare_vectors, distances_to_query, nearest, output_distances

class LSHIndex:
    def __init__(self, metric="cosine", n_tables=8, n_hashes=4,
                 bucket_width=1.0, fallback_candidates=64, seed=42):
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if n_tables < 1 or not 1 <= n_hashes <= 63 or bucket_width <= 0 or fallback_candidates < 0:
            raise ValueError("Invalid LSH parameter")
        self.metric = metric
        self.n_tables = n_tables
        self.n_hashes = n_hashes
        self.bucket_width = bucket_width
        self.fallback_candidates = fallback_candidates
        self.seed = seed
        self.vectors = None
        self._projections = None
        self._scales = None
        self._offsets = None
        self._tables = []
        self._fallback_ids = None
        self.last_candidate_counts = []
        self.last_fallback_flags = []

    def _hash_rows(self, rows, table_id):
        projection = rows @ self._projections[table_id].T
        normalized = projection / self._scales[table_id]
        codes = np.floor((normalized + self._offsets[table_id]) / self.bucket_width)
        return [tuple(int(v) for v in row) for row in codes]

    def fit(self, vectors):
        self.vectors = prepare_vectors(vectors, self.metric)
        n, d = self.vectors.shape

        # Distinct, deterministic RNG streams for shared scale sample,
        # per-table projections/offsets, and shared fallback IDs. The table
        # RNG depends on its index, *not* the total number of tables.
        sample_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0]))
        sample_ids = sample_rng.choice(n, size=min(n, 2048), replace=False)
        sample = self.vectors[sample_ids]

        self._projections = np.empty(
            (self.n_tables, self.n_hashes, d), dtype=np.float32)
        self._scales = np.empty(
            (self.n_tables, self.n_hashes), dtype=np.float32)
        self._offsets = np.empty(
            (self.n_tables, self.n_hashes), dtype=np.float32)
        self._tables = []

        for t in range(self.n_tables):
            rng_table = np.random.default_rng(
                np.random.SeedSequence([self.seed, 1, t]))
            projection = rng_table.normal(size=(self.n_hashes, d)).astype(np.float32)
            self._projections[t] = projection
            self._scales[t] = np.maximum(
                np.std(sample @ projection.T, axis=0), 1e-6)
            self._offsets[t] = rng_table.uniform(
                0, self.bucket_width, size=self.n_hashes).astype(np.float32)

            buckets = defaultdict(list)
            for start in range(0, n, 4096):
                for j, key in enumerate(
                    self._hash_rows(self.vectors[start:start + 4096], t), start
                ):
                    buckets[key].append(j)
            self._tables.append(dict(buckets))

        fallback_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 2]))
        count = min(n, max(self.fallback_candidates, 1))
        self._fallback_ids = fallback_rng.choice(
            n, size=count, replace=False).astype(np.int64)
        self.last_candidate_counts = []
        self.last_fallback_flags = []
        return self

    @property
    def index_bytes(self):
        if self.vectors is None:
            return 0
        estimated_table_entries = self.n_tables * len(self.vectors) * (8 + sys.getsizeof(0))
        estimated_keys = sum(len(t) * 72 for t in self._tables)
        return int(self.vectors.nbytes + self._projections.nbytes + self._scales.nbytes
                   + self._offsets.nbytes + estimated_table_entries + estimated_keys)

    def query(self, queries, k=10):
        if self.vectors is None:
            raise RuntimeError("Call fit() first")
        if not 1 <= k <= len(self.vectors):
            raise ValueError("k out of range")
        queries = prepare_vectors(queries, self.metric)
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query dimensions differ")
        found, distances = [], []
        candidate_counts, fallback_flags = [], []
        for q in queries:
            candidates = set()
            for t, table in enumerate(self._tables):
                key = self._hash_rows(q[None], t)[0]
                candidates.update(table.get(key, ()))
            used_fallback = len(candidates) < max(k, self.fallback_candidates)
            if used_fallback:
                candidates.update(self._fallback_ids.tolist())
            if len(candidates) < k:
                candidates.update(range(min(k, len(self.vectors))))
            ids = np.fromiter(sorted(candidates), dtype=np.int64)
            ds = distances_to_query(self.vectors[ids], q, self.metric)
            result_ids, result_dists = nearest(ds, ids, k)
            found.append(result_ids)
            distances.append(output_distances(result_dists, self.metric))
            candidate_counts.append(len(ids))
            fallback_flags.append(used_fallback)
        self.last_candidate_counts = candidate_counts
        self.last_fallback_flags = fallback_flags
        return np.stack(found), np.stack(distances)
