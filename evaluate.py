
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


def benchmark_queries(index, queries: np.ndarray, k: int, *, warmup: int = 5, repeats: int = 3):
    qarray = np.asarray(queries)
    if qarray.ndim != 2 or not len(qarray) or repeats < 1 or warmup < 0:
        raise ValueError("Need nonempty (Q,D) queries, nonnegative warmup and positive repeats")
    for q in qarray[:min(warmup, len(qarray))]:
        index.query(q, k=k)
    per_query_ms = []
    neighbors, distances = [], []
    candidate_counts, fallback_flags = [], []
    ivf_probes, ivf_candidates = [], []
    for q in qarray:
        durations = []
        for repetition in range(repeats):
            start = perf_counter()
            ids, ds = index.query(q, k=k)
            durations.append((perf_counter() - start) * 1000)
            if repetition == 0:
                neighbors.append(ids[0])
                distances.append(ds[0])
                # Diagnostics are recorded only from an untimed read after the
                # timed call, never re-running extra work in the measurement.
                if (hasattr(index, "last_candidate_counts") and hasattr(index, "last_fallback_flags")
                        and index.last_candidate_counts):
                    candidate_counts.append(index.last_candidate_counts[0])
                    fallback_flags.append(bool(index.last_fallback_flags[0]))
                if hasattr(index, "last_probe_counts") and index.last_probe_counts:
                    ivf_probes.append(index.last_probe_counts[0])
                    ivf_candidates.append(index.last_candidate_counts[0])
        per_query_ms.append(float(np.median(durations)))
    timings = np.asarray(per_query_ms, dtype=np.float64)
    mean = float(timings.mean())
    result = {
        "mean_latency_ms": mean,
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "latency_std_ms": float(timings.std(ddof=1)) if len(timings) > 1 else 0.0,
        "queries_per_second": float(1000.0 / mean) if mean > 0 else float("inf"),
    }
    if candidate_counts:
        result["mean_candidates"] = float(np.mean(candidate_counts))
        result["candidate_fraction"] = float(np.mean(candidate_counts) / len(index.vectors))
        result["fallback_rate"] = float(np.mean(fallback_flags))
    if ivf_probes:
        result["mean_clusters_probed"] = float(np.mean(ivf_probes))
        result["mean_ivf_candidates"] = float(np.mean(ivf_candidates))
    return np.asarray(neighbors), np.asarray(distances), result


def _fingerprint(base: np.ndarray, queries: np.ndarray) -> str:
    h = sha256()
    for array in (base, queries):
        array = np.ascontiguousarray(array)
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()[:16]


def ground_truth_file(output_dir: Path, name: str, n_base: int, n_queries: int,
                      k: int, metric: str, fingerprint: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return Path(output_dir) / f"{safe}_n{n_base}_q{n_queries}_k{k}_{metric}_{fingerprint}.npz"


def get_ground_truth(base: np.ndarray, queries: np.ndarray, *, metric: str, k: int,
                     dataset_name: str, output_dir: Path = Path("results/ground_truth"),
                     chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray, Path, bool]:
    if not 1 <= k <= len(base):
        raise ValueError("k must fit the database")
    output_dir = Path(output_dir)
    fingerprint = _fingerprint(base, queries)
    path = ground_truth_file(output_dir, dataset_name, len(base), len(queries), k,
                             metric, fingerprint)
    if path.exists():
        with np.load(path) as saved:
            ids, ds = saved["neighbors"], saved["distances"]
        if ids.shape == (len(queries), k) and ds.shape == ids.shape:
            return ids, ds, path, True
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
