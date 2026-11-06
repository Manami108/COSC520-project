# NumPy-only exact k-nearest-neighbor reference implementation.
from __future__ import annotations
import numpy as np
from .common import prepare_vectors

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
        return prepare_vectors(values, self.metric)

    def fit(self, vectors: np.ndarray) -> "BruteForceKNN":
        self.vectors = self._prepare(vectors)
        return self

    @property
    def index_bytes(self) -> int:
        return 0 if self.vectors is None else self.vectors.nbytes

    def query(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
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
                difference = block - q
                sq_dists = np.einsum("ij,ij->i", difference, difference)
                take = min(k, len(block))
                local = np.argpartition(sq_dists, take - 1)[:take]
                candidate_indices = np.concatenate((best_indices, local + start))
                candidate_sq_dists = np.concatenate((best_sq_dists, sq_dists[local]))
                keep = min(k, len(candidate_indices))
                selected = np.argpartition(candidate_sq_dists, keep - 1)[:keep]
                ordered = selected[np.lexsort((candidate_indices[selected], candidate_sq_dists[selected]))]
                best_indices = candidate_indices[ordered]
                best_sq_dists = candidate_sq_dists[ordered]
            indices[qi] = best_indices
            if self.metric == "euclidean":
                distances[qi] = np.sqrt(np.maximum(best_sq_dists, 0))
            else:
                distances[qi] = np.maximum(best_sq_dists / 2, 0)
        return indices, distances
