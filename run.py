
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from datasets import load_hdf5
from evaluate import benchmark_queries, recall_at_k
from algorithms.knn import BruteForceKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, help="ANN-Benchmarks HDF5 file; omit for synthetic test")
    parser.add_argument("--n-base", type=int, default=10_000)
    parser.add_argument("--n-queries", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.dataset:
        base, queries, official_truth = load_hdf5(args.dataset, args.n_base, args.n_queries)
    else:
        rng = np.random.default_rng(42)
        base = rng.standard_normal((args.n_base, 32)).astype("float32")
        queries = rng.standard_normal((args.n_queries, 32)).astype("float32")
        official_truth = None

    model = BruteForceKNN(metric=args.metric, chunk_size=args.chunk_size)
    start = perf_counter()
    model.fit(base)
    build_seconds = perf_counter() - start
    neighbors, distances, search_metrics = benchmark_queries(model, queries, args.k)
    results = {
        "algorithm": "numpy_brute_force_knn",
        "dataset": str(args.dataset) if args.dataset else "synthetic_normal_seed_42",
        "metric": args.metric,
        "n_base": len(base),
        "n_queries": len(queries),
        "dimensions": base.shape[1],
        "k": args.k,
        "chunk_size": args.chunk_size,
        "build_seconds": build_seconds,
        "index_bytes": model.vectors.nbytes,
        **search_metrics,
        "official_recall_at_k": (
            recall_at_k(official_truth, neighbors, k=args.k)
            if official_truth is not None else None
        ),
    }
    args.output_dir.mkdir(exist_ok=True, parents=True)
    tag = (args.dataset.stem if args.dataset else "synthetic") + f"_n{len(base)}_k{args.k}"
    (args.output_dir / f"{tag}_summary.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        args.output_dir / f"{tag}_exact_neighbors.npz",
        neighbors=neighbors,
        distances=distances,
        # These exact neighbor IDs serve as ground truth for future ANN runs.
    )
    print(json.dumps(results, indent=2))
    print(f"Saved exact-neighbor ground truth in {args.output_dir}/")


if __name__ == "__main__":
    main()
