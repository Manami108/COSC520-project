from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from algorithms.knn import BruteForceKNN

def recall_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int = 10) -> float:
    truth = np.asarray(ground_truth)
    pred = np.asarray(predicted)
    if truth.ndim != 2 or pred.ndim != 2 or truth.shape[0] != pred.shape[0] or not len(truth):
        raise ValueError("Both arrays must be nonempty 2D with the same number of queries")
    if not 1 <= k <= truth.shape[1] or pred.shape[1] < k:
        raise ValueError("k must fit both sets of neighbors")
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(truth, pred)]))

def benchmark_queries(index, queries: np.ndarray, k: int, *, warmup: int = 2):
    if len(queries) == 0:
        raise ValueError("At least one query required")
    for query in queries[:min(warmup, len(queries))]:
        index.query(query, k=k)
    timings, neighbors, distances = [], [], []
    for query in queries:
        start = perf_counter()
        idx, dst = index.query(query, k=k)
        timings.append(perf_counter() - start)
        neighbors.append(idx[0])
        distances.append(dst[0])
    timings = np.asarray(timings)
    summary = {
        "mean_latency_ms": float(timings.mean() * 1000),
        "p95_latency_ms": float(np.percentile(timings, 95) * 1000),
        "queries_per_second": float(len(queries) / timings.sum()),}
    return np.asarray(neighbors), np.asarray(distances), summary

def _fingerprint(base: np.ndarray, queries: np.ndarray) -> str:
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
    output_dir = Path(output_dir)
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
