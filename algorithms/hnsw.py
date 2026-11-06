"""Educational, pure-Python/NumPy HNSW implementation.

Builds nested proximity-graph layers using incremental insertion, randomized
levels, best-first search, and diversity-based neighbor pruning. This is an
in-memory reference implementation, not a performance substitute for hnswlib.

Based on Malkov & Yashunin, IEEE TPAMI (2018/2020).
"""
from __future__ import annotations
import heapq
import math
import sys
from collections.abc import Sequence
import numpy as np
from .common import prepare_vectors, distances_to_query, nearest, output_distances


class HNSWIndex:
    def __init__(self, metric: str = "euclidean", M: int = 8,
                 ef_construction: int = 40, ef_search: int = 40,
                 seed: int = 42, max_layers: int = 32):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("Invalid metric")
        if M < 2 or ef_construction < M or ef_search < 1 or max_layers < 1:
            raise ValueError("M>=2, ef_construction>=M, ef_search>=1, max_layers>=1 required")
        self.metric = metric
        self.M, self.ef_construction, self.ef_search = M, ef_construction, ef_search
        self.seed, self.max_layers = seed, max_layers
        self.vectors = None
        self.layers: list[dict[int, list[int]]] = []
        self.levels: list[int] = []
        self.entry_point: int | None = None
        self.max_level = -1
        self._rng = np.random.default_rng(seed)

    def _random_level(self) -> int:
        u = max(float(self._rng.random()), np.finfo(float).tiny)
        return min(self.max_layers - 1, int(-math.log(u) / math.log(self.M)))

    def _distance(self, q: np.ndarray, node: int) -> float:
        if self.metric == "euclidean":
            delta = self.vectors[node] - q
            return float(delta @ delta)
        return float(max(0.0, 1.0 - self.vectors[node] @ q))

    def _search_layer(self, q: np.ndarray, entry_points: Sequence[int],
                      ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first graph search; result is (raw_distance, node) sorted ascending."""
        adjacency = self.layers[layer]
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []  # max-heap of (-distance, -node)
        for node in entry_points:
            if node not in adjacency or node in visited:
                continue
            visited.add(node)
            d = self._distance(q, node)
            heapq.heappush(candidates, (d, node))
            heapq.heappush(best, (-d, -node))
        while candidates:
            d, node = heapq.heappop(candidates)
            if len(best) >= ef and d > -best[0][0]:
                break
            for other in adjacency[node]:
                if other in visited:
                    continue
                visited.add(other)
                nd = self._distance(q, other)
                worst_d = -best[0][0] if best else math.inf
                if len(best) < ef or nd < worst_d:
                    heapq.heappush(candidates, (nd, other))
                    heapq.heappush(best, (-nd, -other))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, -node) for d, node in best)

    def _select_neighbors(self, q: np.ndarray, candidates: list[tuple[float, int]],
                          limit: int, *, exclude: int | None = None) -> list[int]:
        """Diverse-neighbor heuristic, then fill unused slots from rejected points."""
        taken: list[int] = []
        rejected: list[int] = []
        seen = set()
        for dist, node in sorted(candidates):
            if node == exclude or node in seen:
                continue
            seen.add(node)
            if all(self._distance(self.vectors[node], picked) > dist for picked in taken):
                taken.append(node)
            else:
                rejected.append(node)
            if len(taken) == limit:
                break
        if len(taken) < limit:
            taken.extend(rejected[:limit - len(taken)])
        return taken

    def _connect(self, node: int, neighbors: list[int], layer: int):
        degree_limit = self.M * 2 if layer == 0 else self.M
        adj = self.layers[layer]
        adj[node] = list(neighbors)
        for other in neighbors:
            if node not in adj[other]:
                adj[other].append(node)
            if len(adj[other]) > degree_limit:
                q = self.vectors[other]
                options = [(self._distance(q, candidate), candidate)
                           for candidate in adj[other] if candidate != other]
                adj[other] = self._select_neighbors(q, options, degree_limit, exclude=other)

    def fit(self, vectors: np.ndarray) -> "HNSWIndex":
        self.vectors = prepare_vectors(vectors, self.metric)
        self.layers = []
        self.levels = []
        self.entry_point = None
        self.max_level = -1
        self._rng = np.random.default_rng(self.seed)
        for node, q in enumerate(self.vectors):
            level = self._random_level()
            self.levels.append(level)
            while len(self.layers) <= level:
                self.layers.append({})
            for layer in range(level + 1):
                self.layers[layer][node] = []
            if self.entry_point is None:
                self.entry_point = node
                self.max_level = level
                continue

            previous_top = self.max_level
            entry = [self.entry_point]
            # Start at the sparse layers, greedily finding a better entry point.
            for layer in range(previous_top, level, -1):
                best = self._search_layer(q, entry, 1, layer)
                entry = [best[0][1]]

            for layer in range(min(level, previous_top), -1, -1):
                candidates = self._search_layer(q, entry, self.ef_construction, layer)
                cap = 2 * self.M if layer == 0 else self.M
                picked = self._select_neighbors(q, candidates, cap, exclude=node)
                self._connect(node, picked, layer)
                # Carry candidate entry points to the next lower (denser) layer.
                entry = [v for _, v in candidates]

            if level > previous_top:
                self.entry_point = node
                self.max_level = level
        return self

    @property
    def index_bytes(self) -> int:
        if self.vectors is None:
            return 0
        edge_count = sum(len(neighbors) for layer in self.layers for neighbors in layer.values())
        node_count = sum(len(layer) for layer in self.layers)
        # Rough Python structure estimate; inspect process RSS for actual heap use.
        return int(self.vectors.nbytes + edge_count * (8 + sys.getsizeof(0)) + node_count * 96)

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
            entry = [self.entry_point]
            for layer in range(self.max_level, 0, -1):
                closest = self._search_layer(q, entry, 1, layer)
                entry = [closest[0][1]]
            candidates = self._search_layer(q, entry, max(self.ef_search, k), 0)
            if len(candidates) < k:
                # Small disconnected components can arise for pathological inputs;
                # seed additional searches without evaluating every database vector.
                known = {v for _, v in candidates}
                for extra in range(len(self.vectors)):
                    if extra in known:
                        continue
                    more = self._search_layer(q, [extra], max(self.ef_search, k), 0)
                    for item in more:
                        if item[1] not in known:
                            candidates.append(item)
                            known.add(item[1])
                    if len(candidates) >= k:
                        break
            selected = sorted(candidates)[:k]
            ids = np.asarray([node for _, node in selected], dtype=np.int64)
            raw_dist = np.asarray([dist for dist, _ in selected], dtype=np.float32)
            results.append(ids)
            distances.append(output_distances(raw_dist, self.metric))
        return np.stack(results), np.stack(distances)
