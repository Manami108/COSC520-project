# A clear, NumPy-only exact k-nearest-neighbor reference implementation.
from __future__ import annotations
import numpy as np


class BruteForceKNN:
    def __init__(self, metric: str = "euclidean", chunk_size: int = 8192):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.metric = metric
        self.chunk_size = chunk_size
        self.vectors: np.ndarray | None = None

    def _prepare(self, values: np.ndarray) -> np.ndarray:
        a = np.asarray(values, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2 or not a.size or not np.isfinite(a).all():
            raise ValueError("Input must be a nonempty, finite, 1D/2D array")
        if self.metric == "cosine":
            norms = np.linalg.norm(a, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Cosine similarity is undefined for zero vectors")
            a = a / norms
        return np.ascontiguousarray(a)

    def fit(self, vectors: np.ndarray) -> "BruteForceKNN":
        self.vectors = self._prepare(vectors)
        return self

    def query(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        # Return (indices, distances), both shape (n_queries, k)
        if self.vectors is None:
            raise RuntimeError("Call fit() before query()")
        if not 1 <= k <= len(self.vectors):
            raise ValueError("k must be between 1 and the database size")
        queries = self._prepare(queries)
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query dimensions do not match the database")

        indices = np.empty((len(queries), k), dtype=np.int64)
        distances = np.empty((len(queries), k), dtype=np.float32)

        for qi, q in enumerate(queries):
            best_indices = np.empty(0, dtype=np.int64)
            best_sq_dists = np.empty(0, dtype=np.float32)

            for start in range(0, len(self.vectors), self.chunk_size):
                block = self.vectors[start : start + self.chunk_size]
                difference = block - q  # bounded temporary allocation
                sq_dists = np.einsum("ij,ij->i", difference, difference)

                # Only local top-k can ever enter the global top-k.
                take = min(k, len(block))
                local = np.argpartition(sq_dists, take - 1)[:take]
                candidate_indices = np.concatenate((best_indices, local + start))
                candidate_sq_dists = np.concatenate((best_sq_dists, sq_dists[local]))

                keep = min(k, len(candidate_indices))
                selected = np.argpartition(candidate_sq_dists, keep - 1)[:keep]
                # Sort by distance, then index, for a reproducible order.
                ordered = selected[np.lexsort((candidate_indices[selected], candidate_sq_dists[selected]))]
                best_indices = candidate_indices[ordered]
                best_sq_dists = candidate_sq_dists[ordered]

            indices[qi] = best_indices
            if self.metric == "euclidean":
                distances[qi] = np.sqrt(np.maximum(best_sq_dists, 0))
            else:
                # Unit vectors: ||x-q||^2 = 2 * (1 - cosine_similarity).
                distances[qi] = np.maximum(best_sq_dists / 2, 0)

        return indices, distances
