# COSC 520 — Approximate Nearest Neighbour Search for RAG

**Team:** Manami Yano, Mahdieh Sadat

## 1. Project Overview

This project investigates Approximate Nearest Neighbour (ANN) search algorithms for Retrieval-Augmented Generation (RAG) systems.

RAG systems retrieve relevant information from large vector databases. However, exact nearest-neighbour search becomes computationally expensive as the number of vectors increases. Thus, we implement and compare several nearest-neighbour search algorithms in Python/NumPy, focusing on search accuracy, computational efficiency, scalability, and memory usage.

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
│   ├── knn.py
│   ├── lsh.py
│   ├── ivf.py
│   └── hnsw.py
│
├── tests/
│   ├── test_knn.py
│   ├── test_lsh.py
│   ├── test_ivf.py
│   └── test_hnsw.py
│
├── experiments/
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
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** This is the planned final structure. Many algorithms, experiment scripts, and visualization components are not yet implemented.

## 3. Installation
### Step 1 — Clone the repository

```bash
git clone https://github.com/Manami108/COSC520-project.git
cd COSC520-project
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3 — Activate the environment (macOS)

```bash
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5 — Run tests

```bash
python -m unittest discover -s tests -v
```

## 4. Running the Code

Run all commands from the project root directory.

### 4.1 Brute-force k-NN using synthetic data

```bash
python run.py --n-base 10000 --n-queries 25 --k 10
```

Parameters:

- `--n-base`: Number of database vectors
- `--n-queries`: Number of query vectors
- `--k`: Number of nearest neighbors returned

This experiment uses 10,000 synthetic vectors and 25 queries, retrieving the 10 nearest neighbors for each query.

The output includes search performance measurements, index size, and build time.

Results are saved to the `results/` directory.

**Important:** Because `knn.py` is located in `algorithms/`, ensure `run.py` imports it as follows:

```python
from algorithms.knn import BruteForceKNN
```

### 4.2 Download benchmark datasets

**SIFT1M — Euclidean distance**

```bash
mkdir -p data

curl -L https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
-o data/sift-128-euclidean.hdf5
```

**GloVe-100 — Angular/Cosine distance**

```bash
curl -L https://ann-benchmarks.com/glove-100-angular.hdf5 \
-o data/glove-100-angular.hdf5
```

Dataset source: [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)

### 4.3 Run with SIFT1M

```bash
python run.py \
  --dataset data/sift-128-euclidean.hdf5 \
  --metric euclidean \
  --n-base 10000 \
  --n-queries 25 \
  --k 10
```

### 4.4 Run with GloVe-100

```bash
python run.py \
  --dataset data/glove-100-angular.hdf5 \
  --metric cosine \
  --n-base 10000 \
  --n-queries 25 \
  --k 10
```

### 4.5 Test larger dataset sizes

Change `--n-base` to:

- `10000` — 10K vectors
- `100000` — 100K vectors
- `1000000` — 1M vectors

Larger brute-force experiments can be slow and memory-intensive.

Currently, `run.py` supports the brute-force baseline. Other algorithms will be integrated in future development.

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

## 6. Planned Experiments

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
| LSH | Number of hash tables, number of hash bits |
| IVF | Number of clusters, number of searched clusters |
| HNSW | M, efConstruction, efSearch |

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
| Milestone 2 | November 6, 2026 | Half of the implementations completed |
| Final | November 30, 2026 | Complete implementation, experiments, report, and demonstration |

The first two milestones are optional checkpoints under the course project instructions.

## 9. Development Status

- [x] Initial project structure
- [x] Brute-force k-NN starter implementation
- [x] Basic evaluation utilities
- [ ] Validate baseline with SIFT1M and GloVe
- [ ] Implement LSH
- [ ] Implement IVF
- [ ] Implement HNSW
- [ ] Complete parameter experiments
- [ ] Generate comparison plots
- [ ] Implement Streamlit UI
- [ ] Complete final report and demonstration

## 10. References

- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [DiskANN (Vamana) — Microsoft Research](https://github.com/microsoft/DiskANN)

Additional academic references, dataset citations, and experimental results will be added as the project develops.
