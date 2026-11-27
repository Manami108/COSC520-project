# COSC 520 — Approximate Nearest Neighbour Search for RAG

**Team:** [Mahdieh Sadat](https://github.com/smahdieh) and [Manami Yano](https://github.com/Manami108)

Welcome to our COSC 520 final project! 🌎

Finding your nearest neighbor sounds easy... until you have a million of them. In this project, we explore how different nearest neighbor search algorithms tackle this challenge and evaluate their efficiency, accuracy, and scalability in the context of Retrieval-Augmented Generation (RAG).

To all the LLM lovers out there who are curious about how RAG works behind the scenes: don't worry! We've spent countless hours experimenting so you can spend five minutes reading 😉✨

| Algorithm | Approach |
|---|---|
| Exact k-NN | Exhaustive search (baseline) |
| LSH | Locality-sensitive hashing (Gaussian bucket hashing) |
| IVF | Inverted file index with k-means clustering |
| HNSW | Hierarchical navigable small-world graph |
| Vamana | Proximity graph with greedy search and robust pruning |

Vamana is an in-memory implementation of the core graph algorithm, not the full SSD-based DiskANN system.

## 1. Installation

Run from the repository root (macOS/Linux):

```bash
git clone https://github.com/Manami108/COSC520-project.git
cd COSC520-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `python -m venv .venv` and `.\.venv\Scripts\Activate.ps1` in PowerShell. The benchmark shell script requires Bash.

## 2. Unit Tests

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests verify the correctness of all five search algorithms, the evaluation pipeline, and experimental utilities. No dataset download is required.

## 3. Datasets

```bash
python datasets.py --dataset both
```

| Dataset | Dimension | Distance |
|---|---:|---|
| [SIFT-128](https://ann-benchmarks.com/sift-128-euclidean.hdf5) | 128 | Euclidean |
| [GloVe-100](https://ann-benchmarks.com/glove-100-angular.hdf5) | 100 | Cosine |

Datasets are provided by [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) and stored in `data/`. Exact ground truth is recomputed for each database subset.

## 4. Interactive Demo

```bash
# Synthetic data (no download required)
python demo.py

# SIFT-128
python demo.py --dataset data/sift-128-euclidean.hdf5 --n-base 2000 --n-queries 20 --k 10
```

Compare retrieved neighbours, recall and latency across all five algorithms. Use the query navigation and search-effort buttons to inspect the results. PCA is used only for visualization, and retrieval uses the original vector space.

Use `--no-show` to save the figures without opening an interactive window. Demo figures and metrics are saved in `results/demo/`, independently of benchmark results.

## 5. Experiments

```bash
# Run the tests
python -m unittest discover -s tests -v

# Run the main benchmark, sensitivity analysis and plots
bash run_recall_targets.sh
```

| Setting | Value |
|---|---|
| Database sizes | 1K, 5K, 10K, 25K |
| Tuning database size | 10K |
| Validation / test queries | 100 / 500 (disjoint) |
| Neighbours | k = 10 |
| Hyperparameter search | 24 configurations per ANN algorithm (4 index profiles × 6 effort levels) |
| Recall targets | 0.80, 0.90, 0.95 |
| Tuning seed | 42 |
| Final ANN index-build seeds | 123, 456, 789 |
| Timing repetitions | 3 per query |

All algorithms use identical database subsets and queries within each dataset. For each recall target, the fastest configuration satisfying validation recall is selected. If none qualifies, the highest-recall configuration is selected. Parameters remain fixed across database sizes, and missed test recall targets are recorded.

Sensitivity analysis varies one parameter at a time on the validation queries: LSH (`n_tables`), IVF (`nprobe`), HNSW (`ef_search`) and Vamana (`search_L`).

The script defaults to one numerical-library thread and may take considerable time due to Python graph construction. The original 12-trial random-search protocol remains available with `bash run_recall_targets.sh original`. It uses a different query pool, so differences cannot be attributed solely to the tuning strategy.

## 6. Results

The main experiment saves outputs in `results/structured_tuning/`, grouped by recall target (`r800`, `r900`, `r950`) and dataset.

| Output | Contents |
|---|---|
| `tuning.csv` | Validation measurements |
| `scaling.csv` | Held-out measurements by size and seed |
| `summary.csv` | Aggregated metrics and speedup versus exact k-NN |
| `tuning_cost.csv` | Configuration counts, index builds and tuning time |
| `metadata.json` | Parameters, query indices, seeds and environment |
| `sensitivity.csv` | Parameter sensitivity results (90% target) |
| `plots/` | Individual figures |

Regenerate the combined figures without rerunning the benchmark:

```bash
python -m experiments.plots --input-dir results/structured_tuning --report-only
```

The `report_figures/` directory contains `report_recall_qps.png`, `report_scalability.png` and `report_sensitivity.png`.

**Metrics:** Recall@10, mean/median/P95 query latency, estimated sequential QPS (1000 / mean latency in ms), build time, estimated index memory (MiB), and speedup versus exact k-NN. Data and generated results are excluded from Git.

## 7. Repository Structure

```text
COSC520-project/
├── algorithms/
│   ├── __init__.py
│   ├── common.py
│   ├── knn.py
│   ├── lsh.py
│   ├── ivf.py
│   ├── hnsw.py
│   └── vamana.py
├── experiments/
│   ├── __init__.py
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
├── data/                  # Downloaded datasets (generated)
├── results/               # Experimental and demo outputs (generated)
├── datasets.py
├── evaluate.py
├── demo.py
├── run.py
├── run_recall_targets.sh
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Limitations

- We use educational Python/NumPy implementations, so the measured runtimes may not reflect the performance of optimized production systems.
- We use fixed configurations and a single tuning seed, which limit hyperparameter optimization. Equal trial counts do not necessarily imply equal tuning costs.
- We limit database subsets to 25K vectors and estimate memory usage rather than directly measuring process RSS.
- We focus on vector retrieval performance rather than end-to-end RAG answer quality.

## 9. References and Credits

- LSH: Datar et al. (2004), [Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://doi.org/10.1145/997817.997857).
- IVF and SIFT: Jégou et al. (2011), [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57).
- HNSW: Malkov and Yashunin (2020), [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473).
- Vamana: Subramanya et al. (2019), [DiskANN](https://arxiv.org/abs/1908.01410).
- Benchmarking: Aumüller et al. (2020), [ANN-Benchmarks](https://doi.org/10.1016/j.is.2019.02.006).
- GloVe: Pennington et al. (2014), [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/).
- RAG: Lewis et al. (2020), [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401).

Project documentation: [Milestone 1](docs/MILESTONE1.md) · [Milestone 2](docs/MILESTONE2.md).

If you've made it this far, congratulations! You've officially survived our README. See you at the presentation!
— Mahdieh & Manami