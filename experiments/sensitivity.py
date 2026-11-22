
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from datasets import load_hdf5
from evaluate import get_ground_truth
from experiments.benchmark import make_model, measure, write_csv

SWEEPS = {
    "lsh": ("n_tables", [4, 8, 16, 24]),
    "ivf": ("nprobe", [1, 2, 4, 8, 16, 32]),
    "hnsw": ("ef_search", [10, 20, 40, 80, 120]),
    "vamana": ("search_L", [20, 40, 80, 120]),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.repeats < 1:
        parser.error("repeats must be positive")
    dataset = Path(metadata["dataset"])
    n = int(metadata["tune_size"])
    k = int(metadata["k"])
    n_validation = int(metadata["tune_queries"])
    n_test = int(metadata["test_queries"])
    seed = int(metadata["selection_seed"])
    base, queries, _ = load_hdf5(dataset, n_base=n, n_queries=n_validation+n_test)
    order = np.random.default_rng(metadata["query_split_seed"]).permutation(len(queries))
    validation = queries[order[:n_validation]]
    truth, _, _, _ = get_ground_truth(base, validation, metric=metadata["metric"], k=k,
        dataset_name=dataset.stem+"_validation", output_dir=args.metadata.parent/"truth")

    results = []
    for algorithm, choice in metadata["selected"].items():
        if algorithm not in SWEEPS:
            continue
        param, values = SWEEPS[algorithm]
        original = choice["parameters"]
        if algorithm == "ivf":
            values = [v for v in values if v <= original["n_clusters"]]
        values = sorted(set([*values, original[param]]))
        # All query-only parameters can be swept on ONE fixed, identical graph/index.
        # For LSH n_tables, the hash index must be rebuilt for each value.
        if algorithm != "lsh":
            model = make_model(algorithm, metadata["metric"], original, seed, k)
            start = perf_counter()
            model.fit(base)
            base_build = perf_counter() - start
        for value in values:
            params = {**original, param: value}
            if algorithm == "lsh":
                model = make_model(algorithm, metadata["metric"], params, seed, k)
                start = perf_counter()
                model.fit(base)
                build_s = perf_counter() - start
            else:
                setattr(model, param, value)
                build_s = base_build
            metrics = measure(model, validation, truth, k, args.repeats)
            results.append({"algorithm": algorithm, "dataset": dataset.stem,
                            "n_base": n, "n_queries": len(validation), "k": k,
                            "seed": seed, "parameter_name": param, "parameter_value": value,
                            "parameters": params, "build_seconds": build_s,
                            "index_memory_mb_estimate": model.index_bytes/(1024**2),
                            **metrics})
            print(f"{algorithm:6s} {param}={value}: recall={metrics['recall_at_k']:.3f}, "
                  f"latency={metrics['median_latency_ms']:.4f} ms", flush=True)
    output = args.metadata.parent / "sensitivity.csv"
    write_csv(output, results)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
