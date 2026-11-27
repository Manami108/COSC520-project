from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from algorithms.hnsw import HNSWIndex
from algorithms.ivf import IVFIndex
from algorithms.knn import BruteForceKNN
from algorithms.lsh import LSHIndex
from datasets import infer_metric, load_hdf5
from evaluate import benchmark_queries, get_ground_truth, recall_at_k

try:
    from algorithms.vamana import VamanaIndex
except ImportError:
    VamanaIndex = None

INDEXES = {"knn": BruteForceKNN, "lsh": LSHIndex, "ivf": IVFIndex, "hnsw": HNSWIndex}
if VamanaIndex is not None:
    INDEXES["vamana"] = VamanaIndex

# Original random/grid search retained for reproducing Experiment A.
SPACES = {
    "lsh": {"n_tables": [4, 8, 16, 24], "n_hashes": [2, 4, 6],
            "bucket_width": [0.25, 0.5, 1.0, 2.0, 4.0]},
    "ivf": {"n_clusters": [16, 32, 64, 128], "nprobe": [1, 2, 4, 8, 16, 32]},
    "hnsw": {"M": [4, 8, 12], "ef_construction": [40, 80],
             "ef_search": [10, 20, 40, 80]},
    "vamana": {"R": [8, 16, 24], "L": [20, 40, 80],
               "alpha": [1.0, 1.2], "search_L": [20, 40, 80]},
}

# Experiment B: four index profiles per method, six effort levels per profile.
# These are predefined representative profiles, not exhaustive optimal settings.
INDEX_PROFILES = {
    "lsh": [{"n_hashes": 2, "bucket_width": .5},
            {"n_hashes": 4, "bucket_width": 1.0},
            {"n_hashes": 4, "bucket_width": 4.0},
            {"n_hashes": 6, "bucket_width": 2.0}],
    "ivf": [{"n_clusters": n} for n in [32, 64, 96, 128]],
    "hnsw": [{"M": m, "ef_construction": ef}
             for m, ef in itertools.product([8, 16], [40, 120])],
    "vamana": [{"R": 8, "L": 40, "alpha": 1.0},
               {"R": 16, "L": 80, "alpha": 1.2},
               {"R": 24, "L": 80, "alpha": 1.0},
               {"R": 32, "L": 120, "alpha": 1.2}],
}
EFFORTS = {
    "lsh": ("n_tables", [2, 4, 8, 12, 16, 24]),
    "ivf": ("nprobe", [1, 2, 4, 8, 16, 32]),
    "hnsw": ("ef_search", [10, 20, 40, 80, 120, 200]),
    "vamana": ("search_L", [20, 40, 80, 120, 160, 200]),
}
STRUCTURED_TRIALS = 24


def configs_for(algorithm: str, method: str, trials: int, seed: int, metric: str):
    del metric  # Same search spaces for both metrics.
    if algorithm == "knn":
        return [{}]
    if method == "structured":
        if trials != STRUCTURED_TRIALS:
            raise ValueError("Structured search requires exactly 24 trials")
        param, levels = EFFORTS[algorithm]
        return [{**profile, param: value}
                for profile in INDEX_PROFILES[algorithm] for value in levels]
    space = SPACES[algorithm]
    keys = tuple(space)
    configs = [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]
    if algorithm == "ivf":
        configs = [c for c in configs if c["nprobe"] <= c["n_clusters"]]
    if algorithm == "hnsw":
        configs = [c for c in configs if c["ef_construction"] >= c["M"]]
    if algorithm == "vamana":
        configs = [c for c in configs if c["L"] >= c["R"]]
    if method == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(configs), size=min(trials, len(configs)), replace=False)
        configs = [configs[int(i)] for i in selected]
    return configs


def make_model(algorithm: str, metric: str, params: dict, seed: int, k: int):
    if algorithm not in INDEXES:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    cls = INDEXES[algorithm]
    kwargs = {"metric": metric} if algorithm == "knn" else {"metric": metric, "seed": seed, **params}
    accepted = inspect.signature(cls).parameters
    if algorithm == "vamana":
        aliases = {"R": ("R", "degree", "max_degree"),
                   "L": ("L", "search_list_size", "L_search"),
                   "alpha": ("alpha", "pruning_alpha"),
                   "search_L": ("search_L", "query_search_L")}
        translated = {"metric": metric, "seed": seed}
        for key, value in params.items():
            match = next((name for name in aliases.get(key, (key,)) if name in accepted), None)
            if match is None:
                raise ValueError(f"Vamana constructor does not accept {key}: {inspect.signature(cls)}")
            translated[match] = value
        kwargs = translated
    unknown = set(kwargs) - set(accepted)
    if unknown:
        raise ValueError(f"Unsupported parameters for {algorithm}: {sorted(unknown)}")
    if algorithm == "hnsw":
        kwargs["ef_search"] = max(k, kwargs.get("ef_search", k))
    if algorithm == "vamana" and "search_L" in kwargs:
        kwargs["search_L"] = max(k, kwargs["search_L"])
    return cls(**kwargs)


def measure(model, queries, truth, k: int, repeats: int):
    found, _, metrics = benchmark_queries(model, queries, k, warmup=5, repeats=repeats)
    metrics["recall_at_k"] = recall_at_k(truth, found, k)
    return metrics


def measure_configuration(model, algorithm, params, build_s, queries, truth, k, repeats):
    metrics = measure(model, queries, truth, k, repeats)
    return {"algorithm": algorithm, "parameters": dict(params),
            "build_seconds": float(build_s),
            "index_memory_mb_estimate": float(model.index_bytes / (1024 ** 2)),
            "memory_note": "Implementation-specific estimate, not measured process RSS",
            **metrics}


def evaluate(algorithm, base, queries, truth, metric, k, params, seed, repeats):
    model = make_model(algorithm, metric, params, seed, k)
    start = perf_counter()
    model.fit(base)
    build_s = perf_counter() - start
    return measure_configuration(model, algorithm, params, build_s, queries, truth, k, repeats)


def structured_evaluations(algorithm, base, queries, truth, metric, k, configs, seed, repeats):
    """Reuse a built index only when the varied parameter is query-only."""
    param, _ = EFFORTS[algorithm]
    model, previous_profile, build_s = None, None, None
    for params in configs:
        profile = {key: value for key, value in params.items() if key != param}
        # LSH table count changes the index itself; every setting is rebuilt.
        reused = algorithm != "lsh" and profile == previous_profile
        if not reused:
            model = make_model(algorithm, metric, params, seed, k)
            start = perf_counter()
            model.fit(base)
            build_s = perf_counter() - start
        else:
            setattr(model, param, params[param])
        previous_profile = profile
        row = measure_configuration(model, algorithm, params, build_s, queries, truth, k, repeats)
        row["index_reused_during_tuning"] = reused
        yield row


def select_best(rows, target):
    if not rows:
        raise ValueError("No candidate configurations to select")
    feasible = [r for r in rows if r["recall_at_k"] >= target]
    if feasible:
        return min(feasible, key=lambda r: (r["median_latency_ms"], -r["recall_at_k"])), True
    return min(rows, key=lambda r: (-r["recall_at_k"], r["median_latency_ms"])), False


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def summarize_scaling(rows):
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], int(row["n_base"]))].append(row)
    metrics = ("recall_at_k", "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
               "queries_per_second", "build_seconds", "index_memory_mb_estimate")
    summary = []
    for (algorithm, size), group in sorted(grouped.items()):
        result = {"algorithm": algorithm, "n_base": size,
                  "dataset": group[0]["dataset"], "metric": group[0]["metric"],
                  "target_recall": group[0]["target_recall"], "n_index_seeds": len(group),
                  "validation_target_met": all(r["target_met_on_validation"] for r in group),
                  "test_target_met_all_seeds": all(r["test_target_met"] for r in group),
                  "test_target_meeting_fraction": sum(r["test_target_met"] for r in group)/len(group),
                  "parameters": group[0]["parameters"]}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    exact_by_size = {item["n_base"]: item for item in summary if item["algorithm"] == "knn"}
    for item in summary:
        baseline = exact_by_size.get(item["n_base"])
        item["speedup_vs_exact_median"] = (
            baseline["median_latency_ms_mean"] / item["median_latency_ms_mean"]
            if baseline and item["median_latency_ms_mean"] > 0 else None)
    return summary


def label_for_target(target: float) -> str:
    return "r" + str(round(target * 1000)).zfill(3)


def query_split_ids(offset, n_validation, n_test, seed):
    """Absolute row IDs in the HDF5 test array; offset 0 preserves Experiment A."""
    ids = np.random.default_rng(seed).permutation(n_validation + n_test) + offset
    return ids[:n_validation], ids[n_validation:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["all", *INDEXES], default="all")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000])
    parser.add_argument("--tune-size", type=int, default=10000)
    parser.add_argument("--tune-queries", type=int, default=100)
    parser.add_argument("--test-queries", type=int, default=500)
    parser.add_argument("--query-offset", type=int, default=None,
                        help="First eligible query row (default: 600 for structured, 0 otherwise)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", choices=["random", "grid", "structured"], default="random")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--target-recall", type=float, default=.90)
    parser.add_argument("--target-recalls", type=float, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    structured = args.method == "structured"
    if args.trials is None:
        args.trials = STRUCTURED_TRIALS if structured else 12
    if args.query_offset is None:
        args.query_offset = 600 if structured else 0
    if args.output_dir is None:
        args.output_dir = Path("results/structured_tuning" if structured else "results/optimized")
    targets = sorted(set(args.target_recalls if args.target_recalls is not None else [args.target_recall]))
    if not targets or any(not 0 <= t <= 1 for t in targets):
        parser.error("Each recall target must be between 0 and 1")
    if len({label_for_target(t) for t in targets}) < len(targets):
        parser.error("Recall thresholds are too close to distinguish in output names")
    if (args.repeats < 1 or args.trials < 1 or args.k < 1 or args.tune_queries < 1
            or args.test_queries < 1 or not args.seeds or args.query_offset < 0):
        parser.error("Positive counts and a nonnegative query offset required")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values")
    if any(n < args.k for n in args.sizes + [args.tune_size]):
        parser.error("Database sizes must be >= k")
    if structured:
        if args.trials != STRUCTURED_TRIALS:
            parser.error("Structured search requires exactly 24 trials")
        if args.k > 10:
            parser.error("Structured effort levels are defined for k <= 10")
        if args.algorithm in ("all", "ivf") and min(args.sizes + [args.tune_size]) < 128:
            parser.error("Structured IVF needs at least 128 database vectors at every size")
        if args.algorithm == "all" and "vamana" not in INDEXES:
            parser.error("Structured all-algorithm comparison requires Vamana")
    metric = infer_metric(args.dataset)
    algorithms = ([a for a in INDEXES if a != "knn"] if args.algorithm == "all"
                  else ([] if args.algorithm == "knn" else [args.algorithm]))
    include_knn = not args.no_knn or args.algorithm == "knn"
    if args.dry_run:
        for algorithm in algorithms:
            print(f"{algorithm}: {len(configs_for(algorithm, args.method, args.trials, args.seed, metric))} "
                  f"tuning configurations; {len(args.sizes)} sizes; {len(args.seeds)} seeds")
        print(f"Recall targets: {targets}, include exact baseline: {include_knn}; query offset: {args.query_offset}")
        return

    max_n = max(args.sizes + [args.tune_size])
    requested_queries = args.query_offset + args.tune_queries + args.test_queries
    base, queries, _ = load_hdf5(args.dataset, n_base=max_n, n_queries=requested_queries)
    validation_ids, test_ids = query_split_ids(args.query_offset, args.tune_queries, args.test_queries, args.seed)
    validation, test = queries[validation_ids], queries[test_ids]

    truth_dir = args.output_dir / "truth" / args.dataset.stem
    validation_truth, _, _, _ = get_ground_truth(
        base[:args.tune_size], validation, metric=metric, k=args.k,
        dataset_name=args.dataset.stem + "_validation", output_dir=truth_dir)
    test_truth = {}
    for size in sorted(set(args.sizes)):
        test_truth[size], _, _, _ = get_ground_truth(
            base[:size], test, metric=metric, k=args.k,
            dataset_name=args.dataset.stem + "_test", output_dir=truth_dir)

    tuning_rows, candidates_by_algorithm, tuning_cost = [], {}, {}
    for algorithm in algorithms:
        configs = configs_for(algorithm, args.method, args.trials, args.seed, metric)
        if algorithm == "ivf":
            configs = [c for c in configs if c["n_clusters"] <= args.tune_size]
        print(f"\nTuning {algorithm}: {len(configs)} configurations", flush=True)
        successful = []
        tuning_start = perf_counter()
        if structured:
            # Fail loudly rather than silently giving one algorithm fewer trials.
            iterator = structured_evaluations(algorithm, base[:args.tune_size], validation,
                validation_truth, metric, args.k, configs, args.seed, args.repeats)
        else:
            def legacy_rows():
                for i, params in enumerate(configs, 1):
                    try:
                        yield evaluate(algorithm, base[:args.tune_size], validation,
                            validation_truth, metric, args.k, params, args.seed, args.repeats)
                    except (ValueError, TypeError) as exc:
                        print(f"  skip {i}: {exc}", flush=True)
            iterator = legacy_rows()
        for i, row in enumerate(iterator, 1):
            row.update(dataset=args.dataset.stem, metric=metric, stage="tuning",
                       n_base=args.tune_size, n_queries=len(validation), k=args.k, seed=args.seed)
            successful.append(row)
            tuning_rows.append(row)
            print(f"  {i}/{len(configs)} recall={row['recall_at_k']:.3f} "
                  f"median={row['median_latency_ms']:.4f} ms", flush=True)
        if structured and len(successful) != STRUCTURED_TRIALS:
            raise RuntimeError(f"{algorithm}: expected 24 completed configurations")
        candidates_by_algorithm[algorithm] = successful
        tuning_cost[algorithm] = {"attempted_configurations": len(configs),
            "completed_configurations": len(successful),
            "index_builds": sum(not r.get("index_reused_during_tuning", False) for r in successful),
            "wall_seconds": perf_counter() - tuning_start}

    cache, baseline_rows = {}, {}
    if include_knn:
        for size in sorted(set(args.sizes)):
            row = evaluate("knn", base[:size], test, test_truth[size], metric,
                           args.k, {}, args.seed, args.repeats)
            row.update(dataset=args.dataset.stem, metric=metric, stage="test", n_base=size,
                       n_queries=len(test), k=args.k, seed=args.seed, target_met_on_validation=True)
            baseline_rows[size] = row
            print(f"Exact baseline n={size} median={row['median_latency_ms']:.4f} ms", flush=True)
    for target in targets:
        selected = {}
        rows = [dict(baseline_rows[s], target_recall=target,
                     test_target_met=baseline_rows[s]["recall_at_k"] >= target)
                for s in sorted(baseline_rows)]
        for algorithm in algorithms:
            valid = candidates_by_algorithm[algorithm]
            if not valid:
                print(f"WARNING: no valid configurations for {algorithm}", file=sys.stderr)
                continue
            best, meets_validation = select_best(valid, target)
            params = best["parameters"]
            selected[algorithm] = {"parameters": params, "validation_recall": best["recall_at_k"],
                                   "validation_latency_ms": best["median_latency_ms"],
                                   "validation_target_met": meets_validation}
            for size in sorted(set(args.sizes)):
                for seed in args.seeds:
                    cache_key = (algorithm, size, seed, json.dumps(params, sort_keys=True))
                    if cache_key not in cache:
                        row = evaluate(algorithm, base[:size], test, test_truth[size],
                                       metric, args.k, params, seed, args.repeats)
                        row.update(dataset=args.dataset.stem, metric=metric, stage="test",
                                   n_base=size, n_queries=len(test), k=args.k, seed=seed)
                        cache[cache_key] = row
                    measured = cache[cache_key]
                    rows.append(dict(measured, target_recall=target,
                                     target_met_on_validation=meets_validation,
                                     test_target_met=measured["recall_at_k"] >= target))
        root = args.output_dir / label_for_target(target) if len(targets) > 1 else args.output_dir
        folder = root / args.dataset.stem
        write_csv(folder / "tuning.csv", tuning_rows)
        write_csv(folder / "scaling.csv", rows)
        write_csv(folder / "summary.csv", summarize_scaling(rows))
        write_csv(folder / "tuning_cost.csv", [{"algorithm": a, **cost} for a, cost in tuning_cost.items()])
        metadata = {
            "dataset": str(args.dataset), "metric": metric, "k": args.k,
            "tune_size": args.tune_size, "tune_queries": args.tune_queries,
            "test_queries": args.test_queries, "query_split": "shuffled, disjoint",
            "query_offset": args.query_offset, "validation_query_ids": validation_ids.tolist(),
            "test_query_ids": test_ids.tolist(), "query_split_seed": args.seed,
            "sizes": sorted(set(args.sizes)), "target_recall": target, "all_targets": targets,
            "selection": "min validation median per-query latency at target recall; otherwise highest recall",
            "selection_seed": args.seed, "final_seeds": args.seeds,
            "repeats_per_query": args.repeats, "method": args.method, "trials": args.trials,
            "structured_index_profiles": INDEX_PROFILES if structured else None,
            "structured_effort_levels": EFFORTS if structured else None,
            "tuning_cost": tuning_cost, "knn_timed_baseline": include_knn,
            "latency_definition": "median per-query latency; independent repeated single-query calls",
            "qps_definition": "reciprocal of mean single-query latency (sequential)",
            "memory_definition": "algorithm-specific estimate including vectors, not measured RSS",
            "lsh_hash_family": "Gaussian p-stable bucket hashing; cosine operates on unit-normalized vectors",
            "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "thread_env": {key: os.environ.get(key) for key in
                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "selected": selected,
        }
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved threshold {target:.3f} to {folder}", flush=True)


if __name__ == "__main__":
    main()
