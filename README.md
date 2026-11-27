# COSC 520 — Approximate Nearest Neighbour Search for RAG

**Team:** [Mahdieh Sadat](https://github.com/smahdieh) and [Manami Yano](https://github.com/Manami108)
**Course:** COSC 520

## 1. Project Overview

This project investigates approximate nearest-neighbour (ANN) search methods for vector retrieval, a core component of retrieval-augmented generation (RAG). Exact nearest-neighbour search evaluates distances to all database vectors, which can become costly as the database grows. ANN methods aim to reduce retrieval time while preserving high recall.

We implement five search methods using Python and NumPy and compare their retrieval accuracy, query latency, index construction time, estimated memory requirements, and scalability. We evaluate vector search directly; this project does **not** measure end-to-end RAG answer quality.

| Algorithm | Description | Role |
|---|---|---|
| Brute-force k-NN | Exhaustive exact nearest-neighbour search | Exact baseline |
| LSH | Gaussian p-stable bucket hashing | Hash-based ANN |
| IVF | Inverted file index using k-means clustering | Partition-based ANN |
| HNSW | Hierarchical Navigable Small World graph | Graph-based ANN |
| Vamana | In-memory proximity graph using greedy search and robust pruning | Advanced graph-based ANN |

Our Vamana implementation reproduces core graph construction and search ideas. It is **not** a full SSD-based DiskANN system.

### Implementation Details

- **Exact k-NN:** Chunk-based exhaustive search with deterministic tie handling based on vector IDs.
- **LSH:** Gaussian bucket hashing with data-dependent projection scaling and deterministic per-table random streams.
- **IVF:** K-means clustering with adaptive cluster probing to ensure sufficient retrieval candidates.
- **HNSW:** Hierarchical graph search with diversity-based neighbour selection, degree constraints, and graph reachability diagnostics.
- **Vamana:** Greedy graph search and robust pruning using expanded vertices during construction, with metric-specific pruning.

**Project documentation:** [Milestone 1](docs/MILESTONE1.md) · [Milestone 2](docs/MILESTONE2.md)

## 2. Project Structure

```text
COSC520-project/
├── algorithms/
│   ├── common.py
│   ├── knn.py
│   ├── lsh.py
│   ├── ivf.py
│   ├── hnsw.py
│   └── vamana.py
├── experiments/
│   ├── benchmark.py
│   ├── sensitivity.py
│   └── plots.py
├── tests/
│   ├── test_knn.py
│   ├── test_lsh.py
│   ├── test_ivf.py
│   ├── test_hnsw.py
│   ├── test_vamana.py
│   ├── test_pipeline.py
│   └── test_experiment_tools.py
├── docs/
│   ├── MILESTONE1.md
│   └── MILESTONE2.md
├── data/                  # Created when datasets are downloaded; ignored by Git
├── results/               # Created by experiments; ignored by Git
├── check_algorithms.py
├── datasets.py
├── evaluate.py
├── run.py
├── run_recall_targets.sh
├── requirements.txt
└── README.md
```

## 3. Installation

### macOS / Linux

```bash
git clone https://github.com/Manami108/COSC520-project.git
cd COSC520-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone https://github.com/Manami108/COSC520-project.git
cd COSC520-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then activate the environment again. The `run_recall_targets.sh` script requires a Bash-compatible shell (for example, Git Bash on Windows).

## 4. Running the Code

### 4.1 Download datasets

```bash
python datasets.py --dataset both
```

To download only one dataset, use `--dataset sift` or `--dataset glove`.

The datasets are obtained from [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks):

| Dataset | Vector dimension | Evaluation metric |
|---|---:|---|
| SIFT-128 Euclidean (SIFT1M) | 128 | Euclidean distance |
| GloVe-100 Angular | 100 | Cosine distance |

The experiments use subsets of these datasets, not necessarily the complete million-vector collections.

### 4.2 Run the tests

```bash
python -m unittest discover -s tests -v
```

### 4.3 Run an individual algorithm

The following examples use 10,000 SIFT database vectors, 25 queries, and `k=10`:

```bash
# Exact baseline
python run.py --algorithm knn --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10

# LSH
python run.py --algorithm lsh --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10

# IVF
python run.py --algorithm ivf --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10

# HNSW
python run.py --algorithm hnsw --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10 --M 4 --ef-construction 16 --ef-search 30

# Vamana
python run.py --algorithm vamana --dataset data/sift-128-euclidean.hdf5 --n-base 10000 --n-queries 25 --k 10
```

For synthetic-data smoke tests, omit the `--dataset` argument. Individual runs save JSON summaries in `results/`.

### 4.4 Run the main benchmark

From the repository root, run:

```bash
bash run_recall_targets.sh
```

The script runs both datasets, tunes each ANN method once per dataset, and evaluates three target recall levels: **0.80, 0.90, and 0.95**. It also generates the available benchmark plots.

Main experimental settings:

| Setting | Value |
|---|---|
| Database sizes | 1,000; 5,000; 10,000; 25,000 |
| Tuning database size | 10,000 vectors |
| Validation queries | 100 |
| Held-out test queries | 500 |
| Neighbours requested | `k=10` |
| Hyperparameter search | 12 randomly sampled configurations per ANN method |
| Tuning seed | 42 |
| Final index-build seeds | 123, 456, 789 |
| Repeated timings | 3 runs per query |
| Recall targets | 0.80, 0.90, 0.95 |

The script sets common numerical-library thread environment variables to one thread by default (`ANN_THREADS=1`). The complete benchmark may take substantial time because the graph indexes are constructed in Python.

### 4.5 Run the parameter sensitivity analysis

After running the main benchmark, use the selected 0.90-recall configuration as the starting point:

```bash
# SIFT
python -m experiments.sensitivity --metadata results/recall_targets/r900/sift-128-euclidean/metadata.json

# GloVe
python -m experiments.sensitivity --metadata results/recall_targets/r900/glove-100-angular/metadata.json
```

The sensitivity script varies one parameter per algorithm while keeping its other parameters fixed:

| Algorithm | Parameter varied |
|---|---|
| LSH | Number of hash tables (`n_tables`) |
| IVF | Number of probed clusters (`nprobe`) |
| HNSW | Query search width (`ef_search`) |
| Vamana | Query search-list size (`search_L`) |

IVF, HNSW, and Vamana reuse the same constructed index for their query-time sweeps. LSH rebuilds the index at each table count, but deterministic per-table random streams preserve the existing hash tables when additional tables are introduced. The sensitivity analysis uses the validation queries and tuning seed, so it is a **parameter analysis**, not a new held-out test evaluation.

Regenerate plots after producing `sensitivity.csv`:

```bash
python -m experiments.plots --input-dir results/recall_targets/r900
```

### 4.6 Output files

The main benchmark writes results under:

```text
results/recall_targets/
├── r800/
│   ├── sift-128-euclidean/
│   └── glove-100-angular/
├── r900/
│   ├── sift-128-euclidean/
│   └── glove-100-angular/
└── r950/
    ├── sift-128-euclidean/
    └── glove-100-angular/
```

Each dataset folder contains the following files:

| File | Contents |
|---|---|
| `tuning.csv` | Validation results for candidate configurations |
| `scaling.csv` | Held-out test measurements by algorithm, size, and seed |
| `summary.csv` | Aggregated measurements across seeds and speedup relative to exact k-NN |
| `metadata.json` | Selected parameters, seeds, recall target, and environment information |
| `plots/` | Generated figures |
| `sensitivity.csv` | Created only after running the sensitivity analysis |

Large datasets and generated results are excluded from Git by `.gitignore`.

## 5. Evaluation Methodology

### 5.1 Data and ground truth

All methods use the same database subset, queries, distance metric, and `k=10` within each experiment. SIFT is evaluated with Euclidean distance; GloVe is evaluated with cosine distance. Exact nearest neighbours are recomputed using brute-force k-NN **for each database subset**. Official full-dataset neighbours are not used as ground truth for smaller subsets.

### 5.2 Validation and hyperparameter selection

The query set is shuffled deterministically and split into **100 validation queries** and **500 held-out test queries**. For each algorithm, 12 candidate configurations are sampled without replacement from a predefined parameter grid. Hyperparameters are chosen on the validation set only.

For each target recall, the selected configuration is the one with the **lowest validation median per-query latency among configurations meeting that target**. If no tested configuration meets the target, the configuration with the **highest validation recall** is selected, with latency used to break ties. An unmet target is explicitly recorded rather than assumed to have been achieved.

The same tuning results are reused for the 0.80, 0.90, and 0.95 recall targets. This avoids repeated parameter searches for each threshold.

### 5.3 Held-out evaluation and scalability

Selected configurations are evaluated on 500 held-out queries and database sizes of 1K, 5K, 10K, and 25K vectors. Hyperparameters are kept fixed across sizes for each algorithm and recall target. Final ANN indexes are independently rebuilt with seeds 123, 456, and 789. The exact k-NN timing baseline is measured once per database size, rather than separately for each ANN seed.

Because parameters are selected at 10K vectors, this study measures **fixed-configuration scalability**, not the best separately retuned configuration at every size. Validation recall is also not guaranteed to equal test recall; target achievement is checked on the held-out results.

### 5.4 Metrics

| Metric | Definition / interpretation |
|---|---|
| Recall@10 | Average fraction of exact top-10 neighbours recovered |
| Mean query latency (ms) | Mean of per-query timing estimates |
| Median query latency (ms) | Median of per-query timing estimates |
| P95 query latency (ms) | 95th percentile across query timing estimates |
| Estimated sequential QPS | `1000 / mean_latency_ms`; derived from latency, not a concurrent load test |
| Index build time (s) | Time spent constructing an index, separate from search |
| Estimated index memory (MiB) | Implementation-specific size estimate; not measured process RSS |
| Speedup vs exact k-NN | Ratio of exact-search median latency to ANN median latency |

Recall@k is defined as:

\[
\operatorname{Recall@k}=\frac{1}{Q}\sum_{i=1}^{Q}\frac{|A_i\cap G_i|}{k},
\]

where `Q` is the number of queries, `A_i` is the retrieved top-k set, and `G_i` is the exact top-k set for query `i`.

Each query is executed three times. Its median elapsed time is used as the per-query estimate, and the reported mean, median, and p95 are then calculated across queries. Index construction and exact-ground-truth generation are excluded from query timing.

### 5.5 Reproducibility

The benchmark records random seeds, selected parameters, query-split information, Python and NumPy versions, platform information, and available numerical-library thread settings. The same experimental protocol is applied to both datasets, while results are interpreted separately because the vector distributions and distance metrics differ.

## 6. Experiments

### Experiment 1 — Scalability

Compare query latency, p95 latency, recall, build time, and estimated memory at 1K, 5K, 10K, and 25K database vectors. For each algorithm and recall target, the selected configuration remains fixed across sizes.

### Experiment 2 — Recall–latency trade-off

Evaluate the balance between recall and query performance using the configurations sampled during validation. Compare final selected configurations on held-out test queries, including exact k-NN as the performance baseline. Distinguish algorithms that reach a recall target from those that do not.

### Experiment 3 — Parameter sensitivity

Study the effect of changing one retrieval parameter at a time: LSH table count, IVF `nprobe`, HNSW `ef_search`, and Vamana `search_L`. Report how recall and query latency change. The current sensitivity script uses the validation query set.

### Experiment 4 — Dataset comparison

Repeat the benchmarking protocol on SIFT-128 (Euclidean) and GloVe-100 (cosine). Compare results within each dataset first, then discuss differences in retrieval behaviour across the two datasets. Because dataset distributions and metrics differ, differences should not be attributed to the metric alone.

## 7. Results and Visualizations

The experimental results are presented using tables and visualizations to compare retrieval accuracy, query performance, scalability, and computational requirements.

All figures use consistent algorithm-specific colours, marker shapes, and line styles. Logarithmic axes are used where appropriate to improve the comparison of measurements across different orders of magnitude. Figures are exported at 300 DPI.

### 7.1 Generated Figures

| Output | Purpose |
|---|---|
| `tuning_recall_latency.png` | Validation recall–latency comparison across sampled configurations |
| `tuning_zoom.png` | Detailed comparison in the high-recall region |
| `tuning_recall_qps.png` | Validation recall–QPS comparison |
| `test_recall_latency.png` | Held-out recall–latency comparison at the largest database size |
| `test_algorithm_comparison.png` | Aligned comparison of recall and median query latency |
| `scaling_latency.png` | Query latency across database sizes |
| `scaling_p95.png` | P95 query latency across database sizes |
| `scaling_recall.png` | Recall across database sizes |
| `scaling_build.png` | Index construction time across database sizes |
| `scaling_memory.png` | Estimated index memory across database sizes |
| `scaling_combined.png` | Combined four-panel scalability comparison |
| `sensitivity_<algorithm>.png` | Parameter sensitivity in terms of recall and query latency |

### 7.2 Recall and Query Performance

The validation plots display all sampled configurations and connect non-dominated configurations within each algorithm to form empirical Pareto frontiers.

For recall–latency comparisons, higher recall and lower latency are preferred. For recall–QPS comparisons, higher recall and higher estimated sequential QPS are preferred.

The connecting lines serve as visualization aids and do not represent measurements of untested configurations.

### 7.3 Scalability Analysis

Scalability is evaluated using fixed selected hyperparameters across database sizes of 1K, 5K, 10K, and 25K vectors.

The combined scalability figure presents four principal metrics:

- Median query latency
- Recall
- Index construction time
- Estimated index memory

Individual figures are also generated for detailed analysis, including P95 query latency.

### 7.4 Parameter Sensitivity

Parameter sensitivity figures present recall and median query latency in separate aligned panels. This allows the computational cost of improving retrieval accuracy to be examined for each algorithm.

### 7.5 Interpretation of Results

The primary result tables report held-out test recall, median and P95 query latency, speedup relative to exact k-NN, index construction time, and estimated memory.

Where applicable, results are reported using the mean and standard deviation across independent index-build seeds. The exact k-NN baseline is measured once per database size.

Configurations that fail to satisfy the specified recall target are explicitly identified and are not considered valid candidates for performance comparisons at that target.

## 8. Scope and Limitations

- The methods are educational Python/NumPy implementations; measured execution time depends on both algorithm design and implementation overhead.
- The study evaluates database subsets up to **25,000 vectors** in the default benchmark; it does not establish million-scale performance.
- Random search evaluates a fixed budget of **12 candidate configurations** per algorithm, not a guarantee of globally optimal parameters.
- Hyperparameter selection uses **one tuning seed**; final ANN evaluations use three different index-build seeds.
- The memory metric is an **estimate** obtained from each implementation, not actual resident-set-size (RSS) measurement. Estimates may differ in coverage of Python object overhead.
- The sensitivity sweep uses validation queries; it is intended to analyze parameter behaviour, not provide an independent test-set comparison.
- This project measures vector retrieval, **not** RAG generation quality, factuality, or end-to-end latency.

## 9. Development Milestones

| Milestone | Deadline | Goal |
|---|---|---|
| Milestone 1 | October 16, 2026 | Algorithm analysis and theoretical complexity |
| Milestone 2 | November 6, 2026 | Core implementations and initial experiments |
| Milestone 3 | November 30, 2026 | Complete experiments and final report |
| Presentation | December 3, 2026 | Present methods, results, and trade-offs |

**Documentation:** [Milestone 1](docs/MILESTONE1.md) · [Milestone 2](docs/MILESTONE2.md)

## 10. References

### Research papers

- **LSH (random hyperplanes):** Charikar, M. S. (2002). [Similarity Estimation Techniques from Rounding Algorithms](https://doi.org/10.1145/509907.509965). *STOC 2002*. (Background reference; the implemented LSH uses Gaussian bucket hashing instead.)
- **LSH (p-stable hashing):** Datar, M., Immorlica, N., Indyk, P., & Mirrokni, V. S. (2004). [Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://doi.org/10.1145/997817.997857). *SCG 2004*.
- **IVF / quantization:** Jégou, H., Douze, M., & Schmid, C. (2011). [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57). *IEEE TPAMI, 33*(1), 117–128.
- **HNSW:** Malkov, Y. A., & Yashunin, D. A. (2020). [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473). *IEEE TPAMI, 42*(4), 824–836.
- **Vamana / DiskANN:** Subramanya, S. J., Devvrit, Kadekodi, R., Krishnaswamy, R., & Simhadri, H. V. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://arxiv.org/abs/1908.01410). *NeurIPS 2019*.
- **ANN benchmarking:** Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). [ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://doi.org/10.1016/j.is.2019.02.006). *Information Systems, 87*, 101374.
- **RAG:** Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). *NeurIPS 2020*.

### Datasets and software

- **SIFT1M:** Jégou, Douze, & Schmid (2011), *Product Quantization for Nearest Neighbor Search*.
- **GloVe:** Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/). *EMNLP 2014*.
- [ANN-Benchmarks datasets and evaluation framework](https://github.com/erikbern/ann-benchmarks)
- [FAISS](https://github.com/facebookresearch/faiss)
- [DiskANN](https://github.com/microsoft/DiskANN)
