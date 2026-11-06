"""Benchmark exact k-NN, LSH, IVF, or HNSW on the same data and ground truth."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np

from datasets import load_hdf5, infer_metric
from evaluate import benchmark_queries, recall_at_k
from ground_truth import get_ground_truth
from algorithms.knn import BruteForceKNN
from algorithms.lsh import LSHIndex
from algorithms.ivf import IVFIndex
from algorithms.hnsw import HNSWIndex


def build_index(args, metric):
    if args.algorithm == "knn":
        return BruteForceKNN(metric=metric, chunk_size=args.chunk_size), {"chunk_size": args.chunk_size}
    if args.algorithm == "lsh":
        kw = dict(metric=metric, n_tables=args.lsh_tables, n_hashes=args.lsh_hashes,
                  bucket_width=args.bucket_width, fallback_candidates=args.lsh_fallback,
                  seed=args.seed)
        return LSHIndex(**kw), kw
    if args.algorithm == "ivf":
        kw = dict(metric=metric, n_clusters=args.n_clusters, nprobe=args.nprobe,
                  train_size=args.train_size, max_iter=args.kmeans_iter, seed=args.seed)
        return IVFIndex(**kw), kw
    kw = dict(metric=metric, M=args.M, ef_construction=args.ef_construction,
              ef_search=args.ef_search, seed=args.seed)
    return HNSWIndex(**kw), kw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["knn", "lsh", "ivf", "hnsw"], default="knn")
    parser.add_argument("--dataset", type=Path, help="ANN-Benchmarks HDF5; omit for synthetic")
    parser.add_argument("--n-base", type=int, default=10_000)
    parser.add_argument("--n-queries", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["auto", "euclidean", "cosine"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--ground-truth-only", action="store_true")
    # LSH
    parser.add_argument("--lsh-tables", type=int, default=8)
    parser.add_argument("--lsh-hashes", type=int, default=6)
    parser.add_argument("--bucket-width", type=float, default=2.0)
    parser.add_argument("--lsh-fallback", type=int, default=64)
    # IVF
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--kmeans-iter", type=int, default=15)
    # HNSW
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--ef-search", type=int, default=40)
    args = parser.parse_args(argv)
    if args.n_base < 1 or args.n_queries < 1 or args.k < 1 or args.k > args.n_base:
        parser.error("n-base, n-queries, k must be positive, with k <= n-base")

    if args.dataset:
        base, queries, official_truth = load_hdf5(args.dataset, args.n_base, args.n_queries)
        metric = infer_metric(args.dataset) if args.metric == "auto" else args.metric
        dataset_name = args.dataset.stem
    else:
        rng = np.random.default_rng(args.seed)
        base = rng.standard_normal((args.n_base, 32)).astype("float32")
        queries = rng.standard_normal((args.n_queries, 32)).astype("float32")
        official_truth = None
        metric = "euclidean" if args.metric == "auto" else args.metric
        dataset_name = f"synthetic_seed_{args.seed}"

    if args.algorithm == "knn" and not args.ground_truth_only:
        ground_truth, truth_distances, truth_path, cached = get_ground_truth(
            base, queries, metric=metric, k=args.k, dataset_name=dataset_name,
            output_dir=args.output_dir / "ground_truth", chunk_size=args.chunk_size)
    else:
        print("Preparing exact top-k ground truth (may be slow for large subsets)...", flush=True)
        ground_truth, truth_distances, truth_path, cached = get_ground_truth(
            base, queries, metric=metric, k=args.k, dataset_name=dataset_name,
            output_dir=args.output_dir / "ground_truth", chunk_size=args.chunk_size)
    if args.ground_truth_only:
        print(json.dumps({"ground_truth": str(truth_path), "cached": cached}, indent=2))
        return

    model, parameters = build_index(args, metric)
    start = perf_counter()
    model.fit(base)
    build_seconds = perf_counter() - start
    neighbors, distances, search_metrics = benchmark_queries(model, queries, args.k)
    stats = {
        "algorithm": args.algorithm, "dataset": dataset_name, "metric": metric,
        "n_base": len(base), "n_queries": len(queries), "dimensions": base.shape[1],
        "k": args.k, "parameters": parameters,
        "build_seconds": build_seconds,
        "index_bytes_estimate": model.index_bytes,
        "index_memory_mb_estimate": model.index_bytes / (1024 * 1024),
        "recall_at_k": recall_at_k(ground_truth, neighbors, args.k),
        "ground_truth_file": str(truth_path),
        "ground_truth_cached": cached,
        "official_recall_at_k": (recall_at_k(official_truth, neighbors, args.k)
                                 if official_truth is not None else None),
        **search_metrics,
    }
    args.output_dir.mkdir(exist_ok=True, parents=True)
    digest = sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:10]
    label = f"{dataset_name}_n{len(base)}_q{len(queries)}_k{args.k}_{metric}_{args.algorithm}_{digest}"
    output = args.output_dir / f"{label}.json"
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved result: {output}")


if __name__ == "__main__":
    main()
