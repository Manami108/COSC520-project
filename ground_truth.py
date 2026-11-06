"""Cache exact top-k neighbors separately for each dataset subset/query configuration."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
from algorithms.knn import BruteForceKNN


def _fingerprint(base: np.ndarray, queries: np.ndarray) -> str:
    """Quick cache guard: shape and boundary vectors; not a full-file checksum."""
    h = sha256()
    for a in (base, queries):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a[:min(3, len(a))]).tobytes())
        h.update(np.ascontiguousarray(a[-min(3, len(a)):]).tobytes())
    return h.hexdigest()[:16]


def ground_truth_file(output_dir: Path, name: str, n_base: int, n_queries: int,
                      k: int, metric: str, fingerprint: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return Path(output_dir) / f"{safe}_n{n_base}_q{n_queries}_k{k}_{metric}_{fingerprint}.npz"


def get_ground_truth(base: np.ndarray, queries: np.ndarray, *,
                     metric: str, k: int, dataset_name: str,
                     output_dir: Path = Path("results/ground_truth"),
                     chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray, Path, bool]:
    fingerprint = _fingerprint(base, queries)
    path = ground_truth_file(output_dir, dataset_name, len(base), len(queries), k,
                             metric, fingerprint)
    if path.exists():
        with np.load(path) as saved:
            ids = saved["neighbors"]
            ds = saved["distances"]
        if ids.shape == (len(queries), k) and ds.shape == ids.shape:
            return ids, ds, path, True
    if not 1 <= k <= len(base):
        raise ValueError("k must fit the database")
    output_dir.mkdir(parents=True, exist_ok=True)
    exact = BruteForceKNN(metric=metric, chunk_size=chunk_size).fit(base)
    neighbors = np.empty((len(queries), k), dtype=np.int64)
    distances = np.empty((len(queries), k), dtype=np.float32)
    for i, query in enumerate(queries):
        ids, ds = exact.query(query, k=k)
        neighbors[i], distances[i] = ids[0], ds[0]
    np.savez_compressed(path, neighbors=neighbors, distances=distances,
                        metric=np.asarray(metric), fingerprint=np.asarray(fingerprint))
    return neighbors, distances, path, False
