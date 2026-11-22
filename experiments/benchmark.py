"""ANN hyperparameter tuning + fixed-configuration scalability benchmark.

Run as: python -m experiments.benchmark --dataset data/sift-128-euclidean.hdf5
No changes to the algorithm classes are required.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import infer_metric, load_hdf5
from evaluate import get_ground_truth, recall_at_k
from algorithms.knn import BruteForceKNN
from algorithms.lsh import LSHIndex
from algorithms.ivf import IVFIndex
from algorithms.hnsw import HNSWIndex

try:
    from algorithms.vamana import VamanaIndex
except ImportError:
    VamanaIndex = None

INDEXES = {"knn": BruteForceKNN, "lsh": LSHIndex, "ivf": IVFIndex, "hnsw": HNSWIndex}
if VamanaIndex is not None:
    INDEXES["vamana"] = VamanaIndex

# Small, sensible default spaces. These are candidate values, not optimal values.
SPACES = {
    "lsh": {"n_tables": [4, 8, 16, 24], "n_hashes": [2, 4, 6], "bucket_width": [0.25, 0.5, 1.0, 2.0, 4.0]},
    "ivf": {"n_clusters": [16, 32, 64, 128], "nprobe": [1, 2, 4, 8, 16, 32]},
    "hnsw": {"M": [4, 8, 12], "ef_construction": [40, 80], "ef_search": [10, 20, 40, 80]},
    "vamana": {"R": [8, 16, 24], "L": [20, 40, 80], "alpha": [1., 1.2]},
}


def configs_for(algorithm: str, method: str, trials: int, seed: int, metric: str):
    space = SPACES[algorithm].copy()
    keys = tuple(space)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]
    if algorithm == "ivf":
        combinations = [c for c in combinations if c["nprobe"] <= c["n_clusters"]]
    if algorithm == "vamana":
        combinations = [c for c in combinations if c["L"] >= c["R"]]
    if method == "random":
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(combinations), size=min(trials, len(combinations)), replace=False)
        combinations = [combinations[int(i)] for i in ids]
    return combinations


def make_model(algorithm: str, metric: str, params: dict, seed: int, k: int):
    cls = INDEXES[algorithm]
    kwargs = dict(metric=metric, seed=seed, **params) if algorithm != "knn" else dict(metric=metric)
    signature = inspect.signature(cls)
    accepted = signature.parameters
    if algorithm == "vamana":
        # Different teaching implementations use different constructor names.
        aliases = {"R": ("R", "degree", "max_degree"), "L": ("L", "search_list_size", "L_search", "search_L"),
                   "alpha": ("alpha", "pruning_alpha")}
        translated = {"metric": metric, "seed": seed}
        for key, value in params.items():
            options = aliases.get(key, (key,))
            match = next((x for x in options if x in accepted), None)
            if match is None:
                raise ValueError(f"Vamana constructor does not support {key}: {signature}. Adjust SPACES['vamana'] to match.")
            translated[match] = value
        # Couple the query beam width to L so changing L affects search, not only build.
        if "search_L" in accepted:
            translated["search_L"] = params["L"]
        kwargs = translated
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    if algorithm == "hnsw":
        kwargs["ef_search"] = max(k, kwargs.get("ef_search", k))
    return cls(**kwargs)


def measure(model, queries, truth, k, repeats):
    # One warmup, followed by repeated timed passes over the same queries.
    model.query(queries[:1], k=k)
    times = []
    predictions = None
    for _ in range(repeats):
        start = perf_counter()
        ids, _ = model.query(queries, k=k)
        elapsed = perf_counter() - start
        times.append(elapsed * 1000 / len(queries))
        if predictions is None:
            predictions = ids
    times = np.asarray(times, dtype=float)
    result = {
        "recall_at_k": recall_at_k(truth, predictions, k),
        "mean_latency_ms": float(np.mean(times)),
        "median_latency_ms": float(np.median(times)),
        "latency_std_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.,
        "queries_per_second": float(1000. / np.mean(times)),
    }
    # Diagnostic metrics for LSH (not included in timed passes).
    if hasattr(model, "last_candidate_counts") and model.last_candidate_counts:
        result["mean_candidates"] = float(np.mean(model.last_candidate_counts))
        result["candidate_fraction"] = float(np.mean(model.last_candidate_counts) / len(model.vectors))
        result["fallback_rate"] = float(np.mean(model.last_fallback_flags))
    return result


def evaluate(algorithm, base, queries, truth, metric, k, params, seed, repeats):
    model = make_model(algorithm, metric, params, seed, k)
    start = perf_counter()
    model.fit(base)
    build_s = perf_counter() - start
    metrics = measure(model, queries, truth, k, repeats)
    return {
        "algorithm": algorithm, "parameters": params, "build_seconds": build_s,
        "index_memory_mb_estimate": float(model.index_bytes / (1024 ** 2)),
        **metrics,
    }


def select_best(rows, target):
    feasible = [r for r in rows if r["recall_at_k"] >= target]
    if feasible:
        return min(feasible, key=lambda r: (r["median_latency_ms"], -r["recall_at_k"])), True
    return min(rows, key=lambda r: (-r["recall_at_k"], r["median_latency_ms"])), False


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["all", *INDEXES], default="all")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    parser.add_argument("--tune-size", type=int, default=10000)
    parser.add_argument("--tune-queries", type=int, default=100)
    parser.add_argument("--test-queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimized"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.trials < 1 or args.k < 1 or args.tune_queries < 1 or args.test_queries < 1:
        parser.error("k, trials, repeats and query counts must be positive")
    if not 0 <= args.target_recall <= 1:
        parser.error("target-recall must be between 0 and 1")
    if any(n < args.k for n in args.sizes + [args.tune_size]):
        parser.error("all database sizes must be at least k")
    metric = infer_metric(args.dataset)
    # Exact KNN remains available explicitly, but --algorithm all compares ANN methods only.
    algorithms = [a for a in INDEXES if a != "knn"] if args.algorithm == "all" else [args.algorithm]
    max_n = max(args.sizes + [args.tune_size])
    n_queries = args.tune_queries + args.test_queries
    if args.dry_run:
        for algorithm in algorithms:
            n = 1 if algorithm == "knn" else len(configs_for(algorithm, args.method, args.trials, args.seed, metric))
            print(f"{algorithm}: {n} tuning configurations, {len(args.sizes)} final sizes")
        return
    base, queries, _ = load_hdf5(args.dataset, n_base=max_n, n_queries=n_queries)
    validation = queries[:args.tune_queries]
    test = queries[args.tune_queries:]
    output = args.output_dir / args.dataset.stem
    output.mkdir(parents=True, exist_ok=True)
    config_rows, scale_rows, choices = [], [], {}
    # Exact ground truth is recomputed for every database subset.
    tune_truth, _, _, _ = get_ground_truth(base[:args.tune_size], validation, metric=metric,
        k=args.k, dataset_name=args.dataset.stem + "_validation", output_dir=output / "truth")
    test_truth_by_size = {}
    for size in sorted(set(args.sizes)):
        test_truth_by_size[size], _, _, _ = get_ground_truth(base[:size], test, metric=metric,
            k=args.k, dataset_name=args.dataset.stem + "_test", output_dir=output / "truth")
    for algorithm in algorithms:
        candidates = [{}] if algorithm == "knn" else configs_for(algorithm, args.method, args.trials, args.seed, metric)
        if algorithm == "ivf":
            candidates = [c for c in candidates if c["n_clusters"] <= args.tune_size]
        print(f"\nTuning {algorithm}: {len(candidates)} configurations on {args.tune_size} vectors", flush=True)
        valid_rows = []
        for i, params in enumerate(candidates, 1):
            try:
                result = evaluate(algorithm, base[:args.tune_size], validation, tune_truth, metric,
                                  args.k, params, args.seed, args.repeats)
            except (ValueError, TypeError) as exc:
                print(f"  skip configuration {i}: {exc}", flush=True)
                continue
            result.update(dataset=args.dataset.stem, metric=metric, stage="tuning", n_base=args.tune_size,
                          n_queries=len(validation), k=args.k, seed=args.seed)
            config_rows.append(result)
            valid_rows.append(result)
            print(f"  {i}/{len(candidates)} recall={result['recall_at_k']:.3f} median_ms={result['median_latency_ms']:.4f}", flush=True)
        if not valid_rows:
            print(f"Skipping {algorithm}: no valid configurations")
            continue
        best, met = select_best(valid_rows, args.target_recall)
        choices[algorithm] = {"parameters": best["parameters"], "validation_recall": best["recall_at_k"],
                              "validation_latency_ms": best["median_latency_ms"], "target_met": met}
        print(f"Selected {algorithm}: {choices[algorithm]}", flush=True)
        for size in sorted(set(args.sizes)):
            result = evaluate(algorithm, base[:size], test, test_truth_by_size[size], metric,
                              args.k, best["parameters"], args.seed, args.repeats)
            result.update(dataset=args.dataset.stem, metric=metric, stage="test", n_base=size,
                          n_queries=len(test), k=args.k, seed=args.seed, target_met_on_validation=met)
            scale_rows.append(result)
            print(f"  test n={size} recall={result['recall_at_k']:.3f} median_ms={result['median_latency_ms']:.4f}", flush=True)
    write_csv(output / "tuning.csv", config_rows)
    write_csv(output / "scaling.csv", scale_rows)
    metadata = {"lsh_hash_family": "Gaussian p-stable bucket hashing on Euclidean vectors (unit-normalized for cosine)",
                "knn_role": "Exact ground-truth baseline, excluded from ANN comparison by default",
                "dataset": str(args.dataset), "metric": metric, "tune_size": args.tune_size,
                "tune_queries": args.tune_queries, "test_queries": args.test_queries,
                "sizes": sorted(set(args.sizes)), "selection": "minimum validation median latency subject to recall threshold; otherwise highest recall",
                "target_recall": args.target_recall, "method": args.method, "trials": args.trials,
                "seed": args.seed, "repeats": args.repeats, "python": platform.python_version(),
                "numpy": np.__version__, "choices": choices}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved tuning, scaling, and metadata files to {output}")


if __name__ == "__main__":
    main()
