# COSC 520: Exact k-NN baseline (starter)

Starter components for comparing approximate nearest neighbor methods: LSH, IVF, HNSW. Written as an educational draft to adapt and understand, not as a substitute for independent development.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Run without any download

```bash
python run.py --n-base 10000 --n-queries 25 --k 10
```

## Download real benchmark datasets

From https://github.com/erikbern/ann-benchmarks (dataset licenses and citations must be checked):

```bash
mkdir -p data
curl -L https://ann-benchmarks.com/sift-128-euclidean.hdf5 -o data/sift-128-euclidean.hdf5
curl -L https://ann-benchmarks.com/glove-100-angular.hdf5 -o data/glove-100-angular.hdf5
```

SIFT uses Euclidean distance:

```bash
python run.py --dataset data/sift-128-euclidean.hdf5 --metric euclidean --n-base 10000 --n-queries 25 --k 10
```

GloVe uses cosine distance (normalization performed by `BruteForceKNN`):

```bash
python run.py --dataset data/glove-100-angular.hdf5 --metric cosine --n-base 10000 --n-queries 25 --k 10
```

For full scale, change `--n-base` to `1000000` (SIFT) or `1183514` (GloVe), starting with a small query count. Full-scale exact search in NumPy is computationally expensive.

## Interface for future approximate methods

Aim for a common class interface:

- `fit(vectors)` builds the index.
- `query(queries, k=10)` returns `(indices, distances)` of shape `(n_queries, k)`.

Use `evaluate.recall_at_k(exact_indices, approx_indices, k)` for the same test queries, base vectors, and metric. Exact result files produced by `run.py` work as ground truth **only for the dataset and size used in that run**.

The initial `index_bytes` counts NumPy vectors held by the baseline index; it is not peak process memory. To fairly compare methods later, track index-specific memory, peak RAM, build time, mean/p95 latency, QPS, recall@10, hardware, thread count, and parameter choices.

## Academic integrity

The assignment instructs students to discuss code reuse with the instructor and disclose and credit any use of GenAI code. Review and adapt this starter, retain appropriate attribution, and follow your instructor's instructions before submitting.


cosc520_ann_baseline_starter/
├── knn.py
├── datasets.py
├── evaluate.py
├── run.py
├── requirements.txt
├── README.md
├── MILESTONE1_NOTES.md
└── tests/
    └── test_knn.py