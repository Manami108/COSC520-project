from time import perf_counter
import numpy as np

def recall_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int = 10) -> float:
    truth = np.asarray(ground_truth)
    pred = np.asarray(predicted)
    if truth.ndim != 2 or pred.ndim != 2 or truth.shape[0] != pred.shape[0]:
        raise ValueError("Both arrays must be 2D with the same number of queries")
    if not 1 <= k <= truth.shape[1] or pred.shape[1] < k:
        raise ValueError("k must fit both sets of neighbors")
    return float(np.mean([
        len(set(a[:k]) & set(b[:k])) / k
        for a, b in zip(truth, pred)
    ]))

def benchmark_queries(index, queries: np.ndarray, k: int):
    timings = []
    neighbors, distances = [], []
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
        "queries_per_second": float(len(queries) / timings.sum()),
    }
    return np.asarray(neighbors), np.asarray(distances), summary
