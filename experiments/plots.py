"""Plots for optimized benchmark output (tuning.csv and scaling.csv)."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in ("n_base", "recall_at_k", "median_latency_ms", "index_memory_mb_estimate", "build_seconds"):
            if k in row and row[k] != "":
                row[k] = float(row[k])
    return rows


def frontier(rows):
    # Lower latency and higher recall are better; show nondominated configs only.
    return sorted([r for r in rows if not any(
        (s["recall_at_k"] >= r["recall_at_k"] and s["median_latency_ms"] <= r["median_latency_ms"])
        and (s["recall_at_k"] > r["recall_at_k"] or s["median_latency_ms"] < r["median_latency_ms"])
        for s in rows)], key=lambda r: r["recall_at_k"])


def plot_dataset(folder):
    tuning_path, scaling_path = folder / "tuning.csv", folder / "scaling.csv"
    if not tuning_path.exists() or not scaling_path.exists():
        return
    tuning, scaling = read_rows(tuning_path), read_rows(scaling_path)
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    # KNN stays in the exact ground truth but is not an ANN competitor.
    algorithms = sorted(set(r["algorithm"] for r in tuning if r["algorithm"] != "knn"))
    for zoom in (False, True):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for a in algorithms:
            rows = [r for r in tuning if r["algorithm"] == a]
            ax.scatter([r["recall_at_k"] for r in rows], [r["median_latency_ms"] for r in rows],
                       label=a.upper(), alpha=.65, s=28)
            line = frontier(rows)
            if len(line) > 1:
                ax.plot([r["recall_at_k"] for r in line], [r["median_latency_ms"] for r in line], alpha=.7)
        if zoom:
            ax.set_xlim(.8, 1.01)
        else:
            ax.set_xlim(-.02, 1.02)
        ax.set_xlabel("Validation Recall@10")
        ax.set_ylabel("Median query latency (ms)")
        ax.set_title(f"{folder.name}: tuning trade-off" + (" (high-recall zoom)" if zoom else ""))
        ax.grid(alpha=.25)
        ax.legend(title="Algorithm")
        fig.tight_layout()
        fig.savefig(out / ("tuning_zoom.png" if zoom else "tuning_recall_latency.png"), dpi=180)
        plt.close(fig)
    for field, ylabel, filename in [
        ("median_latency_ms", "Median query latency (ms)", "scaling_latency.png"),
        ("recall_at_k", "Test Recall@10", "scaling_recall.png"),
        ("build_seconds", "Index build time (s)", "scaling_build.png"),
        ("index_memory_mb_estimate", "Estimated index memory (MiB)", "scaling_memory.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for a in sorted(set(r["algorithm"] for r in scaling if r["algorithm"] != "knn")):
            rows = sorted((r for r in scaling if r["algorithm"] == a), key=lambda r: r["n_base"])
            ax.plot([r["n_base"] for r in rows], [r[field] for r in rows], marker="o", label=a.upper())
        ax.set_xlabel("Database vectors")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{folder.name}: {ylabel} (fixed tuning-selected parameters)")
        ax.grid(alpha=.25)
        ax.legend()
        if field == "build_seconds":
            ax.set_yscale("symlog", linthresh=.01)
        if field == "recall_at_k":
            ax.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    print("Saved figures to", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/optimized"))
    args = parser.parse_args()
    for folder in sorted(args.input_dir.iterdir()):
        if folder.is_dir():
            plot_dataset(folder)


if __name__ == "__main__":
    main()
