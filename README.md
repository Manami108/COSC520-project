# COSC 520 — Approximate Nearest Neighbour Search for RAG

**Team:** Manami Yano, Mahdieh Sadat

## 1. Project Overview

This project investigates Approximate Nearest Neighbour (ANN) search algorithms for Retrieval-Augmented Generation (RAG) systems.

RAG systems retrieve relevant information from large vector databases. However, exact nearest-neighbour search becomes computationally expensive as the number of vectors increases. Thus, we implement and compare several nearest-neighbour search algorithms in Python/NumPy, focusing on search accuracy, computational efficiency, scalability, and memory usage.

**Milestone 1 Documentation:** [View Algorithm Analysis Notes](docs/MILESTONE1.md)

**Milestone 2 Documentation:** [View Implementation and Initial Experiment Notes](docs/MILESTONE2.md)

**Final Documentation:** TBA

### Algorithms

| Algorithm | Description | Role |
|---|---|---|
| Brute-force k-NN | Exhaustive exact nearest-neighbour search | Baseline |
| LSH | Locality-Sensitive Hashing using random hyperplanes ||
| IVF | Inverted File Index using k-means clustering | |
| HNSW | Hierarchical Navigable Small World graph | |
| Vamana (2019)| Graph construction with greedy search and robust pruning | SOTA |

## 2. Project Structure

```text
project-code/
├── algorithms/
│   ├── __init__.py
│   ├── common.py
│   ├── knn.py
│   ├── lsh.py
│   ├── ivf.py
│   ├── hnsw.py
│   └── vamana.py             
│
├── tests/
│   ├── test_knn.py
│   ├── test_lsh.py
│   ├── test_ivf.py
│   ├── test_hnsw.py
│   └── test_pipeline.py
│
├── experiments/
│   ├── __init__.py
│   ├── benchmark.py
│   └── plots.py         
│
├── data/
├── results/
├── docs/
│
├── datasets.py
├── evaluate.py
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 3. Installation

### Setup (macOS / Linux)

```bash
git clone https://github.com/Manami108/COSC520-project.git  # Clone repository
cd COSC520-project                                         # Open project directory
python3 -m venv .venv                                      # Create virtual environment
source .venv/bin/activate                                  # Activate environment
python -m pip install -r requirements.txt                  # Install dependencies
```

For subsequent sessions, activate the environment using `source .venv/bin/activate`.

## 4. Running the Code
### 4.1 Download Datasets

```bash
# Download both SIFT1M (Euclidean) and GloVe-100 (Cosine)
python datasets.py --dataset both

# Or download only one dataset
python datasets.py --dataset sift
python datasets.py --dataset glove
```

### 4.2 Run Tests and Experiments

```bash
# Unit tests
python -m unittest discover -s tests -v

# Brute-force k-NN with synthetic data
python run.py --n-base 10000 --n-queries 25 --k 10

# Brute-force k-NN with SIFT1M
python run.py --algorithm knn --dataset data/sift-128-euclidean.hdf5 --metric euclidean --n-base 10000 --n-queries 25 --k 10

# Brute-force k-NN with GloVe-100
python run.py --algorithm knn --dataset data/glove-100-angular.hdf5 --metric cosine --n-base 10000 --n-queries 25 --k 10

# LSH with SIFT1M
python run.py --algorithm lsh --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10

# IVF with SIFT1M
python run.py --algorithm ivf --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10

# HNSW with SIFT1M (smaller parameters to begin with)
python run.py --algorithm hnsw --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10 --M 4 --ef-construction 16 --ef-search 30
```

### 4.3 Parameters and Results

| Parameter | Description |
|---|---|
| `--n-base` | Number of database vectors (10K, 100K, 1M) |
| `--n-queries` | Number of queries |
| `--k` | Number of nearest neighbors |
| `--metric` | Distance metric (Euclidean or cosine) |

Results are saved in `results/` as JSON performance summaries and NPZ exact-neighbor ground truth. The ground truth is generated using brute-force k-NN for each subset and reused when available.

## 5. Evaluation Metrics

All algorithms will be evaluated using the same database vectors, query vectors, distance metric, and value of k.

| Metric | Description |
|---|---|
| Recall@k | Fraction of exact top-k neighbors retrieved |
| Query latency (ms) | Average or percentile search time per query |
| Queries per second (QPS) | Search throughput |
| Build time (s) | Time required to construct an index |
| Memory usage (MB) | Index memory consumption |
| Scalability | Performance as dataset size increases |

### Recall@k

Recall@k is defined as:
\[
\text{Recall@k}=\frac{1}{Q}\sum_{i=1}^{Q}\frac{|A_i\cap G_i|}{k}
\]

Where:

- \(Q\): Number of queries
- \(A_i\): Retrieved nearest neighbors
- \(G_i\): Exact nearest neighbors from brute-force k-NN

Brute-force k-NN provides the ground truth for evaluating approximate algorithms.

For dataset subsets, exact ground truth must be recomputed using the selected subset.

## 6. Experiments

### Experiment 1 — Scalability

Measure performance at three database sizes:

- 10,000 vectors
- 100,000 vectors
- 1,000,000 vectors

Compare query latency, build time, and index memory.

### Experiment 2 — Recall vs Query Speed

Compare Recall@10 against latency or queries per second.
Analyze the trade-off between retrieval accuracy and search efficiency.

### Experiment 3 — Parameter Sensitivity

Investigate how algorithm parameters affect performance.

| Algorithm | Parameters |
|---|---|
| LSH | Number of hash tables, number of hash bits, bucket width |
| IVF | Number of clusters, number of searched clusters |
| HNSW | M, efConstruction, efSearch |
| Vamana (planned) | Graph degree, search list size, pruning parameter |

### Experiment 4 — Dataset Comparison

Compare algorithm behavior across:

- SIFT1M: Euclidean distance, 128 dimensions
- GloVe-100: Cosine distance, 100 dimensions

## 7. Expected Visualizations

Planned comparison plots include:
1. Recall@10 vs query latency
2. Recall@10 vs queries per second
3. Dataset size vs query time
4. Dataset size vs index memory
5. Index build time by algorithm
6. HNSW search-path visualization in 2D

A Streamlit application is planned for interactive comparison of algorithm performance.

## 8. Development Milestones

| Milestone | Deadline | Goal |
|---|---|---|
| Milestone 1 | October 16, 2026 | Algorithm analysis and theoretical complexity |
| Milestone 2 | November 6, 2026 | Most of the implementations completed |
| Milestone 3 | November 30, 2026 | Complete implementation, experiments,and report|
| Presentation | December 3, 2026 | Complete presentation |

**Milestone 1 Documentation:** [View Algorithm Analysis Notes](docs/MILESTONE1.md)

**Milestone 2 Documentation:** [View Implementation and Initial Experiment Notes](docs/MILESTONE2.md)

**Milestone 3 Documentation:** TBA

## 9. Development Status

- [x] Initial project structure
- [x] Brute-force k-NN starter implementation
- [x] Basic evaluation utilities
- [x] Validate baseline with SIFT1M and GloVe
- [x] Implement LSH
- [x] Implement IVF
- [x] Implement HNSW
- [x] Implement Vamana
- [x] Complete parameter experiments
- [x] Generate comparison plots
- [ ] Write final report


## 10. References

### Research Papers

- **LSH (Cosine Similarity):** Charikar, M. S. (2002). [Similarity Estimation Techniques from Rounding Algorithms](https://doi.org/10.1145/509907.509965). *STOC 2002*.
- **LSH (Euclidean Distance):** Datar, M., Immorlica, N., Indyk, P., & Mirrokni, V. S. (2004). [Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://doi.org/10.1145/997817.997857). *SCG 2004*.
- **IVF and Quantization:** Jégou, H., Douze, M., & Schmid, C. (2011). [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57). *IEEE TPAMI, 33*(1), 117–128.
- **HNSW:** Malkov, Y. A., & Yashunin, D. A. (2020). [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473). *IEEE TPAMI, 42*(4), 824–836.
- **Vamana (DiskANN):** Subramanya, S. J., Devvrit, Kadekodi, R., Krishnaswamy, R., & Simhadri, H. V. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://arxiv.org/abs/1908.01410). *NeurIPS 2019*.
- **ANN Benchmarking:** Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). [ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://doi.org/10.1016/j.is.2019.02.006). *Information Systems, 87*, 101374.
- **Retrieval-Augmented Generation:** Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). *NeurIPS 2020*.

### Datasets

- **SIFT1M:** Jégou, H., Douze, M., & Schmid, C. (2011). [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57). *IEEE TPAMI*.
- **GloVe-100:** Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/). *EMNLP 2014*.
- **ANN-Benchmarks Datasets:** [SIFT-128 Euclidean and GloVe-100 Angular](https://github.com/erikbern/ann-benchmarks).

### Software and Implementations

- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [DiskANN — Microsoft Research](https://github.com/microsoft/DiskANN)

