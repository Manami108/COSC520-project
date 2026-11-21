from __future__ import annotations
import heapq
import sys
import numpy as np
from .common import prepare_vectors, distances_to_query, output_distances


class VamanaIndex:
    def __init__(self, metric: str = "euclidean", R: int = 16,
                 L: int = 40, alpha: float = 1.2, search_L: int = 40,
                 passes: int = 2, seed: int = 42):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("Unsupported metric")
        if R < 1 or L < R or search_L < 1 or alpha < 1 or passes < 1:
            raise ValueError("Require R>=1, L>=R, search_L>=1, alpha>=1, passes>=1")
        self.metric, self.R, self.L, self.alpha = metric, R, L, float(alpha)
        self.search_L, self.passes, self.seed = search_L, passes, seed
        self.vectors = None
        self.graph: list[list[int]] = []
        self.entry_point: int | None = None

    def _distance(self, q: np.ndarray, ids: list[int] | np.ndarray) -> np.ndarray:
        return distances_to_query(self.vectors[ids], q, self.metric)

    def _greedy_search(self, q: np.ndarray, entry: int, width: int,
                       *, exclude: int | None = None) -> list[tuple[float, int]]:
        """Best-first graph traversal; return up to width best visited nodes."""
        seen = {entry}
        initial = float(self._distance(q, [entry])[0])
        frontier = [(initial, entry)]
        best = [(-initial, -entry)]
        while frontier:
            dist, vertex = heapq.heappop(frontier)
            if len(best) >= width and dist > -best[0][0]:
                break
            unseen = [v for v in self.graph[vertex] if v not in seen]
            if not unseen:
                continue
            seen.update(unseen)
            scores = self._distance(q, unseen)
            for v, d in zip(unseen, scores):
                d = float(d)
                if len(best) < width or (d, v) < (-best[0][0], -best[0][1]):
                    heapq.heappush(frontier, (d, v))
                    heapq.heappush(best, (-d, -v))
                    if len(best) > width:
                        heapq.heappop(best)
        return sorted((-d, -v) for d, v in best if -v != exclude)

    def _robust_prune(self, vertex: int, candidates: list[int], alpha: float) -> list[int]:
        ids = np.asarray(sorted(set(candidates) - {vertex}), dtype=np.int64)
        if not len(ids):
            return []
        dv = self._distance(self.vectors[vertex], ids).astype(np.float64)
        remaining = np.ones(len(ids), dtype=bool)
        chosen = []
        while remaining.any() and len(chosen) < self.R:
            selectable = np.flatnonzero(remaining)
            selected = min(selectable, key=lambda j: (dv[j], int(ids[j])))
            p = int(ids[selected])
            chosen.append(p)
            remaining[selected] = False
            other = np.flatnonzero(remaining)
            if len(other):
                pd = self._distance(self.vectors[p], ids[other])
                remaining[other[pd * alpha <= dv[other]]] = False
        return chosen

    def fit(self, vectors: np.ndarray) -> "VamanaIndex":
        self.vectors = prepare_vectors(vectors, self.metric)
        n = len(self.vectors)
        if n < 2:
            self.graph = [[] for _ in range(n)]
            self.entry_point = 0
            return self
        rng = np.random.default_rng(self.seed)
        initial_degree = min(self.R, n - 1)
        self.graph = []
        for i in range(n):
            # Sample without replacement from [0, n) excluding i.
            picks = rng.choice(n - 1, size=initial_degree, replace=False)
            picks = picks + (picks >= i)
            self.graph.append([int(v) for v in picks])
        # Approximate medoid sampled deterministically, avoids O(n^2) full medoid.
        sample_ids = rng.choice(n, size=min(n, 256), replace=False)
        centroid = self.vectors[sample_ids].mean(axis=0)
        if self.metric == "cosine":
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 0 else self.vectors[0]
        self.entry_point = int(sample_ids[np.argmin(self._distance(centroid, sample_ids))])

        for iteration in range(self.passes):
            # Original Vamana commonly uses alpha=1 in the first pass.
            pruning_alpha = 1.0 if iteration == 0 and self.passes > 1 else self.alpha
            for v in rng.permutation(n):
                v = int(v)
                discovered = [j for _, j in self._greedy_search(
                    self.vectors[v], self.entry_point, min(self.L, n), exclude=v)]
                merged = self.graph[v] + discovered
                self.graph[v] = self._robust_prune(v, merged, pruning_alpha)
                for other in self.graph[v]:
                    if v not in self.graph[other]:
                        self.graph[other].append(v)
                    if len(self.graph[other]) > self.R:
                        self.graph[other] = self._robust_prune(other, self.graph[other], pruning_alpha)
        return self

    @property
    def index_bytes(self) -> int:
        if self.vectors is None:
            return 0
        # Estimate; excludes allocations and some Python object overhead.
        return int(self.vectors.nbytes + sum(sys.getsizeof(a) +
                   sum(sys.getsizeof(v) for v in a) for a in self.graph))

    def query(self, queries: np.ndarray, k: int = 10):
        if self.vectors is None:
            raise RuntimeError("Call fit() first")
        if not 1 <= k <= len(self.vectors):
            raise ValueError("k out of range")
        queries = prepare_vectors(queries, self.metric)
        if queries.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query dimensions differ")
        neighbors, distances = [], []
        for q in queries:
            found = self._greedy_search(q, self.entry_point, max(k, self.search_L))
            # Sparse or disconnected graphs can produce fewer than k candidates.
            if len(found) < k:
                known = {v for _, v in found}
                for v in range(len(self.vectors)):
                    if v not in known:
                        found.append((float(self._distance(q, [v])[0]), v))
                    if len(found) >= k:
                        break
            selected = sorted(found)[:k]
            neighbors.append([v for _, v in selected])
            distances.append(output_distances(np.asarray([d for d, _ in selected], dtype=np.float32), self.metric))
        return np.asarray(neighbors, dtype=np.int64), np.asarray(distances, dtype=np.float32)
