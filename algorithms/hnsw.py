from __future__ import annotations
import heapq
import math
import sys
from collections.abc import Sequence
import numpy as np
from .common import prepare_vectors, output_distances


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
        self.last_fallback_flags: list[bool] = []

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
            if len(best) > ef:
                heapq.heappop(best)
        while candidates:
            d, node = heapq.heappop(candidates)
            if len(best) >= ef and d > -best[0][0]:
                break
            for other in adjacency[node]:
                if other in visited:
                    continue
                visited.add(other)
                nd = self._distance(q, other)
                # Use vector ID to break equal-distance ties consistently.
                if len(best) < ef or (nd, other) < (-best[0][0], -best[0][1]):
                    heapq.heappush(candidates, (nd, other))
                    heapq.heappush(best, (-nd, -other))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, -node) for d, node in best)

    def _select_neighbors(self, q: np.ndarray, candidates: list[tuple[float, int]],
                          limit: int, *, exclude: int | None = None) -> list[int]:
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
                # Pruning can produce directed edges; do not remove reverse
                # links unnecessarily, as that can reduce reachability.
                adj[other] = self._select_neighbors(q, options, degree_limit, exclude=other)

    def fit(self, vectors: np.ndarray) -> "HNSWIndex":
        self.vectors = prepare_vectors(vectors, self.metric)
        self.layers = []
        self.levels = []
        self.entry_point = None
        self.max_level = -1
        self._rng = np.random.default_rng(self.seed)
        self.last_fallback_flags = []
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
                # Select M initial neighbours at every layer. Existing nodes
                # may grow to 2M at layer zero through reverse-edge insertion.
                picked = self._select_neighbors(q, candidates, self.M, exclude=node)
                self._connect(node, picked, layer)
                # Carry candidate entry points to the next lower (denser) layer.
                entry = [v for _, v in candidates]

            if level > previous_top:
                self.entry_point = node
                self.max_level = level
        return self

    def graph_reachability(self) -> float:
        """Fraction of base-layer vertices reachable from the entry point.

        Diagnostic only: call outside timed build and query measurements.
        Traverses directed edges without imposing a search-width limit.
        """
        if self.vectors is None or self.entry_point is None:
            raise RuntimeError("Call fit() first")
        adjacency = self.layers[0]
        visited = {self.entry_point}
        pending = [self.entry_point]
        while pending:
            node = pending.pop()
            for other in adjacency[node]:
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
        return len(visited) / len(adjacency)

    @property
    def last_fallback_rate(self) -> float:
        """Fraction of queries in the most recent query() call using fallback."""
        return (sum(self.last_fallback_flags) / len(self.last_fallback_flags)
                if self.last_fallback_flags else 0.0)

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
        fallback_flags = []
        for q in queries:
            entry = [self.entry_point]
            for layer in range(self.max_level, 0, -1):
                closest = self._search_layer(q, entry, 1, layer)
                entry = [closest[0][1]]
            candidates = self._search_layer(q, entry, max(self.ef_search, k), 0)
            used_fallback = len(candidates) < k
            if used_fallback:
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
            fallback_flags.append(used_fallback)
        self.last_fallback_flags = fallback_flags
        return np.stack(results), np.stack(distances)
