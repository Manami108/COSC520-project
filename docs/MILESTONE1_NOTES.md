# COSC 520 Milestone 1 — algorithm analysis notes

**Source:** *COSC 520 Projects* (September 2026), pages 1, 5, 6. Milestone 1 is optional, is due Oct 16, and concerns algorithm analysis. Final due Nov 30. Confirm any professor-specific instructions separately.

## Problem and data

Given a database X of n vectors in R^d and a query q, return the k vectors closest to q according to a fixed metric. Exact kNN finds the true closest neighbors; approximate nearest-neighbor algorithms speed up queries by using an index and can miss some exact neighbors.

Candidate datasets: SIFT (1,000,000 vectors, dimension 128, Euclidean/L2); GloVe-100 (1,183,514 vectors, dimension 100, angular/cosine). ANN-Benchmarks has independent test queries and precomputed neighbors for each full dataset. Use recomputed exact neighbors after subsetting the database.

## Algorithm table (analysis draft)

Symbols: n = dataset size, d = dimensions, k = returned neighbors, C = IVF centroids, p = IVF cells probed, s = k-means training sample size, I = k-means iterations, L = LSH hash tables, h = random hyperplanes/table, b = candidate vectors examined, M = HNSW degree bound, v = nodes visited.

| Method | Key data structures | Index / build work | Query work | Storage (including vectors) |
|---|---|---|---|---|
| Exact kNN | contiguous float32 matrix + top-k selection | O(nd) if copying into the index | O(nd + n + k log k) with full-distance linear selection; a bounded-memory chunked implementation adds merge overhead | O(nd) |
| Random-hyperplane LSH | L hash tables holding vector IDs, Lh hyperplane vectors | O(nLhd) | O(Lhd + bd) plus bucket access/selection; b may approach n | O(nd + nL + Lhd) |
| IVF-Flat | C centroid vectors + inverted lists | O(IsCd) to train k-means on s points, plus O(nCd) for naive full assignment | O(Cd + p(n/C)d) for roughly balanced lists, plus candidate selection | O(nd + Cd + n) |
| HNSW | multi-layer proximity graph and vector array | Data-dependent; roughly n repeated graph searches; no universal O(n log n) guarantee | O(vMd) distance/evaluation work with visited-node count v dependent on parameters and data | O(nd + nM) under bounded expected degree |

Time-complexity details depend on choices: exact kNN top-k heap vs partition, NumPy vectorization, hash-table collision management, centroid training schedule, and graph traversal. In particular, HNSW is often described as having near-logarithmic empirical search scaling on favorable distributions, **not a general worst-case O(log n) guarantee**.

## State-of-the-art comparison (literature only)

DiskANN (Subramanya et al., NeurIPS 2019) uses Vamana graph indexing and techniques for SSD-backed large-scale search. It is a relevant advanced graph-based reference, not necessarily the single best current method on every benchmark. Microsoft maintains the DiskANN library. Avoid promising to implement DiskANN from scratch unless instructor approves a larger scope.

## Proposed evaluation

- Exact ground truth: your NumPy brute-force implementation; cross-check small cases against an independent NumPy sort.
- Metrics: recall@10 (fraction of exact top-10 IDs returned), mean/p95 latency (ms), QPS, build time (s), RAM/index bytes, recall-vs-latency and scaling plots.
- Database sizes: 10K, 100K, 1M vectors (GloVe can use 1M subset to align scale).
- Query sets: use the SAME independent query vectors per dataset size and model.
- Controlled variables: machine, threads, metric, vector normalization, k, random seed, datasets, query batch size.
- Parameter sweeps: LSH L and h; IVF C and p; HNSW M, efConstruction, efSearch.

## Cautions

1. SIFT image features and GloVe word vectors are ANN benchmarks, but neither is a modern end-to-end RAG embedding benchmark. Describe them accurately.
2. Existing ANN comparisons are extensive. If positioning your work as a novel empirical benchmarking study, discuss a distinct research question with the professor. Your selected category is 'Algorithm of your choice', so emphasize your own implementation, data structures, and analysis instead.
3. From-scratch Python HNSW at 1M vectors may be prohibitively slow; prototype at 10K then scale only if feasible.
4. Credit code and discuss GenAI assistance: the course PDF explicitly requires disclosures and instructor discussion for reused code.

## Reference starting points

- ANN-Benchmarks dataset info: https://github.com/erikbern/ann-benchmarks
- FAISS index summaries: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- DiskANN paper: https://www.microsoft.com/en-us/research/?p=634449
- Stanford GloVe: https://nlp.stanford.edu/projects/glove/
