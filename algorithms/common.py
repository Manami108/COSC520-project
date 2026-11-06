
from __future__ import annotations
import numpy as np

def prepare_vectors(values: np.ndarray, metric: str, *, normalize: bool = True) -> np.ndarray:
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    a = np.asarray(values, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("Vectors must be a nonempty finite 1D/2D array")
    if metric == "cosine" and normalize:
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cosine distance is undefined for zero vectors")
        a = a / norms
    return np.ascontiguousarray(a, dtype=np.float32)

def distances_to_query(vectors: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        delta = vectors - query
        return np.einsum("ij,ij->i", delta, delta).astype(np.float32)
    return np.maximum(0, 1.0 - vectors @ query).astype(np.float32)

def output_distances(raw: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.sqrt(np.maximum(raw, 0)).astype(np.float32)
    return raw.astype(np.float32)

def nearest(raw_distances: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((ids, raw_distances))[:k]
    return ids[order], raw_distances[order]
