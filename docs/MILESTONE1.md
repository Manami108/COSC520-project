# COSC 520 — Milestone 1: Algorithm Analysis
## 1. Problem and Datasets

Given a database of \(n\) vectors in \(d\) dimensions, nearest-neighbour search retrieves the \(k\) vectors closest to a query. Approximate Nearest Neighbour (ANN) algorithms improve search efficiency by trading some accuracy for speed.

**Datasets:**

| Dataset | Vectors | Dimensions | Distance |
|---|---|---|---|
| SIFT1M | 1,000,000 | 128 | Euclidean |
| GloVe-100 | 1,183,514 | 100 | Cosine |

Experiments will use subsets of **10K, 100K, and 1M vectors**.

## 2. Algorithms and Complexity

All five algorithms will be implemented from scratch using Python/NumPy, without pre-built ANN libraries.

| Algorithm | Data Structure | Build Time | Query Time | Space |
|---|---|---|---|---|
| Brute-force k-NN | Vector matrix | \(O(nd)\) | \(O(nd+n+k\log k)\) | \(O(nd)\) |
| LSH | Hash tables | \(O(nLhd)\) | \(O(Lhd+bd)\) | \(O(nd+nL+Lhd)\) |
| IVF | Centroids, inverted lists | \(O(IsCd+nCd)\) | \(O(Cd+pnd/C)\) | \(O(nd+Cd+n)\) |
| HNSW | Multi-layer graph | Data-dependent | \(O(vMd)\) | \(O(nd+nM)\) expected |
| Vamana (2019) | Proximity graph | Data-dependent | \(O(vRd)\) | \(O(nd+nR)\) |

**Notation:** \(n\) = vectors, \(d\) = dimensions, \(k\) = neighbors, \(L\) = hash tables, \(h\) = hash bits, \(b\) = candidates, \(C\) = clusters, \(p\) = clusters probed, \(I\) = k-means iterations, \(s\) = training samples, \(M/R\) = graph degree bounds, \(v\) = visited nodes.

These are simplified estimates; selection overhead, graph traversal, and implementation details affect actual performance. Graph search complexity is data-dependent, with no universal logarithmic worst-case guarantee.

## 3. State-of-the-Art Algorithm: Vamana

**Vamana (Microsoft DiskANN, NeurIPS 2019)** is our selected post-2017 SOTA algorithm.

We will implement its core components from scratch:

1. Random graph initialization
2. GreedySearch
3. RobustPrune
4. Reverse-edge insertion
5. Two-pass graph refinement

The initial implementation will use an in-memory graph rather than full SSD-based DiskANN.

## 4. Evaluation Plan

**Metrics:**

- Recall@10 — Retrieval accuracy compared with exact k-NN
- Query latency (mean/p95) and QPS — Search efficiency
- Build time — Index construction cost
- Memory usage — Index storage requirements
- Scalability — Performance at 10K, 100K, and 1M vectors

**Experiments:**

| Experiment | Comparison |
|---|---|
| Scalability | Dataset size vs query time, build time, and memory |
| Accuracy vs speed | Recall@10 vs latency/QPS |
| Parameter sensitivity | Algorithm-specific parameter effects |
| Dataset comparison | SIFT1M vs GloVe-100 |

**Parameters to investigate:**

- LSH: Number of hash tables and hash bits
- IVF: Number of clusters and probed clusters
- HNSW: M, efConstruction, efSearch
- Vamana: Graph degree R, search list size L, pruning parameter α

All algorithms will use the same queries, distance metric, \(k\), random seeds, and hardware for fair comparison. Exact ground truth will be recomputed for each dataset subset.

## 5. Implementation Considerations

- All algorithms are implemented from scratch; NumPy is used for numerical operations.
- Brute-force k-NN provides exact ground truth.
- HNSW and Vamana may be computationally expensive at 1M vectors in Python.
- SIFT and GloVe are ANN benchmarks, not end-to-end RAG datasets.
- GenAI-assisted code and external references must be credited according to course requirements.

## 6. References

- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [FAISS Index Overview](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [DiskANN: NeurIPS 2019](https://papers.nips.cc/paper/9527-diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)