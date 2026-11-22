
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NUMERIC = {"n_base", "recall_at_k", "mean_latency_ms", "median_latency_ms",
           "p95_latency_ms", "index_memory_mb_estimate", "build_seconds",
           "queries_per_second", "target_recall", "seed"}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for name in NUMERIC:
            if row.get(name) not in (None, ""):
                row[name] = float(row[name])
        if isinstance(row.get("parameters"), str):
            row["parameters"] = json.loads(row["parameters"])
        for name in ("test_target_met", "target_met_on_validation"):
            if name in row:
                row[name] = row[name].lower() == "true"
    return rows


def frontier(rows):
    return sorted((r for r in rows if not any(
        (s["recall_at_k"] >= r["recall_at_k"] and s["median_latency_ms"] <= r["median_latency_ms"])
        and (s["recall_at_k"] > r["recall_at_k"] or s["median_latency_ms"] < r["median_latency_ms"])
        for s in rows)), key=lambda r: r["recall_at_k"])


def grouped_by_algorithm_size(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], int(row["n_base"]))].append(float(row[field]))
    return {key: (float(np.mean(values)),
                  float(np.std(values, ddof=1)) if len(values) > 1 else 0., len(values))
            for key, values in groups.items()}


def plot_dataset(folder):
    folder = Path(folder)
    tuning_path, scaling_path = folder / "tuning.csv", folder / "scaling.csv"
    if not tuning_path.exists() or not scaling_path.exists():
        return []
    tuning, scaling = read_rows(tuning_path), read_rows(scaling_path)
    output = folder / "plots"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    algorithms = sorted({r["algorithm"] for r in tuning if r["algorithm"] != "knn"})

    # Tuning tradeoff is displayed on validation data (not held-out test data).
    for zoom in (False, True):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for algorithm in algorithms:
            subset = [r for r in tuning if r["algorithm"] == algorithm]
            ax.scatter([r["recall_at_k"] for r in subset],
                       [r["median_latency_ms"] for r in subset],
                       label=algorithm.upper(), alpha=.65, s=28)
            pf = frontier(subset)
            if len(pf) > 1:
                ax.plot([r["recall_at_k"] for r in pf],
                        [r["median_latency_ms"] for r in pf], alpha=.6)
        ax.set_xlim(.8 if zoom else -.02, 1.01)
        ax.set_xlabel("Validation Recall@10")
        ax.set_ylabel("Median per-query latency (ms)")
        ax.set_title(f"{folder.name}: validation recall–latency tradeoff")
        ax.grid(alpha=.25)
        ax.legend(title="Algorithm")
        fig.tight_layout()
        path = output / ("tuning_zoom.png" if zoom else "tuning_recall_latency.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    for field, ylabel, filename in [
        ("median_latency_ms", "Median per-query latency (ms)", "scaling_latency.png"),
        ("p95_latency_ms", "95th-percentile per-query latency (ms)", "scaling_p95.png"),
        ("recall_at_k", "Test Recall@10", "scaling_recall.png"),
        ("build_seconds", "Index build time (s)", "scaling_build.png"),
        ("index_memory_mb_estimate", "Estimated index memory (MiB)", "scaling_memory.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        by_size = grouped_by_algorithm_size(scaling, field)
        for algorithm in sorted({r["algorithm"] for r in scaling}):
            keys = sorted((key for key in by_size if key[0] == algorithm), key=lambda key: key[1])
            if not keys:
                continue
            x = [k[1] for k in keys]
            y = [by_size[k][0] for k in keys]
            errors = [by_size[k][1] for k in keys]
            ax.errorbar(x, y, yerr=errors, marker="o", markersize=4,
                        capsize=2 if any(errors) else 0, label=algorithm.upper())
        ax.set_xlabel("Database vectors")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{folder.name}: {ylabel} (fixed selected parameters)")
        ax.grid(alpha=.25)
        ax.legend()
        if field == "recall_at_k":
            ax.set_ylim(0, 1.03)
            if scaling and "target_recall" in scaling[0]:
                ax.axhline(float(scaling[0]["target_recall"]), color="grey", ls="--", lw=1,
                           label="Target recall")
        if field == "build_seconds":
            ax.set_yscale("symlog", linthresh=.01)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    # Held-out test results at largest database size; annotate failed recall targets.
    max_n = max(int(r["n_base"]) for r in scaling)
    largest = [r for r in scaling if int(r["n_base"]) == max_n]
    summary = defaultdict(list)
    for row in largest:
        summary[row["algorithm"]].append(row)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for algorithm, rows in sorted(summary.items()):
        recalls = np.asarray([r["recall_at_k"] for r in rows], float)
        latencies = np.asarray([r["median_latency_ms"] for r in rows], float)
        met = all(r.get("test_target_met", True) for r in rows)
        ax.errorbar(float(recalls.mean()), float(latencies.mean()),
                    xerr=float(recalls.std(ddof=1)) if len(rows) > 1 else 0.,
                    yerr=float(latencies.std(ddof=1)) if len(rows) > 1 else 0.,
                    marker="o" if met else "x", capsize=3,
                    label=f"{algorithm.upper()}" + (" (target missed)" if not met else ""))
    ax.set_xlim(-.02, 1.03)
    ax.set_xlabel("Held-out test Recall@10")
    ax.set_ylabel("Median per-query latency (ms)")
    ax.set_title(f"{folder.name}: held-out test performance at n={max_n:,}")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    path = output / "test_recall_latency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    sensitivity_path = folder / "sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = read_rows(sensitivity_path)
        for algorithm in sorted({r["algorithm"] for r in sensitivity}):
            rows = [r for r in sensitivity if r["algorithm"] == algorithm]
            rows.sort(key=lambda r: float(r["parameter_value"]))
            param = rows[0]["parameter_name"]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot([float(r["parameter_value"]) for r in rows],
                    [r["recall_at_k"] for r in rows], marker="o")
            ax.set_xlabel(param)
            ax.set_ylabel("Validation Recall@10")
            ax.set_ylim(0, 1.03)
            ax.set_title(f"{algorithm.upper()} parameter sensitivity (controlled sweep)")
            ax.grid(alpha=.25)
            fig.tight_layout()
            path = output / f"sensitivity_{algorithm}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)
    print(f"Saved {len(written)} figures in {output}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/optimized"))
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Missing input directory: {args.input_dir}")
    for folder in sorted(args.input_dir.iterdir()):
        if folder.is_dir():
            plot_dataset(folder)


if __name__ == "__main__":
    main()
