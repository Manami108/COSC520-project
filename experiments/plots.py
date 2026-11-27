from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

NUMERIC = {"n_base", "recall_at_k", "mean_latency_ms", "median_latency_ms",
           "p95_latency_ms", "index_memory_mb_estimate", "build_seconds",
           "queries_per_second", "target_recall", "seed", "k", "parameter_value"}

# Consistent colours, markers and line styles inspired by ANN benchmark papers.
STYLES = {
    "lsh":    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "ivf":    {"color": "#ff7f0e", "marker": "v", "linestyle": "--"},
    "hnsw":   {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "vamana": {"color": "#d62728", "marker": "<", "linestyle": ":"},
    "knn":    {"color": "#666666", "marker": "s", "linestyle": "--"},
}
LABELS = {"knn": "Exact k-NN", "lsh": "LSH", "ivf": "IVF", "hnsw": "HNSW", "vamana": "Vamana"}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for name in NUMERIC:
            if row.get(name) not in (None, ""):
                row[name] = float(row[name])
        if isinstance(row.get("parameters"), str) and row["parameters"]:
            row["parameters"] = json.loads(row["parameters"])
        for name in ("test_target_met", "target_met_on_validation"):
            if name in row:
                row[name] = row[name].lower() == "true"
    return rows


def frontier(rows, field="median_latency_ms", *, higher_is_better=False):
    """Non-dominated sampled configurations, ordered by increasing recall."""
    sign = -1 if higher_is_better else 1
    return sorted((r for r in rows if not any(
        s["recall_at_k"] >= r["recall_at_k"] and sign * s[field] <= sign * r[field]
        and (s["recall_at_k"] > r["recall_at_k"] or sign * s[field] < sign * r[field])
        for s in rows)), key=lambda r: r["recall_at_k"])


def grouped_by_algorithm_size(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], int(row["n_base"]))].append(float(row[field]))
    return {key: (float(np.mean(values)),
                  float(np.std(values, ddof=1)) if len(values) > 1 else 0., len(values))
            for key, values in groups.items()}


def ordered_algorithms(rows):
    present = {r["algorithm"] for r in rows}
    return [a for a in STYLES if a in present]


def format_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#b0b0b0", alpha=.60, linewidth=.7)
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=.45, linewidth=.5)
    ax.tick_params(direction="in", length=4)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(.8)


def positive_scale(ax, values, *, axis="y"):
    # Zero measurements cannot be displayed on a logarithmic axis.
    if values and all(np.isfinite(v) and v > 0 for v in values):
        (ax.set_yscale if axis == "y" else ax.set_xscale)("log")


def save_figure(fig, path, written):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(path)


def plot_tuning(tuning, field, ylabel, k, dataset, targets, ax, *, zoom=False,
                higher_is_better=False):
    for algorithm in ordered_algorithms(tuning):
        if algorithm == "knn":
            continue
        subset = [r for r in tuning if r["algorithm"] == algorithm and field in r]
        if not subset:
            continue
        style = STYLES[algorithm]
        # Faint points show all measured configurations; the line connects only
        # that method's non-dominated points, not arbitrary parameter settings.
        ax.scatter([r["recall_at_k"] for r in subset], [r[field] for r in subset],
                   color=style["color"], marker=style["marker"], alpha=.28, s=22)
        best = frontier(subset, field, higher_is_better=higher_is_better)
        ax.plot([r["recall_at_k"] for r in best], [r[field] for r in best],
                **style, linewidth=1.5, markersize=5, label=LABELS[algorithm])
    for target in targets:
        ax.axvline(target, color="#888888", linestyle="--", linewidth=.8, alpha=.6)
    ax.set_xlim(.8 if zoom else -.02, 1.02)
    positive_scale(ax, [float(r[field]) for r in tuning if field in r])
    ax.set_xlabel(f"Validation Recall@{k}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset}: validation configurations")
    format_axes(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", framealpha=.9)


def plot_scaling(ax, scaling, field, ylabel, target):
    by_size = grouped_by_algorithm_size(scaling, field)
    for algorithm in ordered_algorithms(scaling):
        keys = sorted((key for key in by_size if key[0] == algorithm), key=lambda key: key[1])
        x = [key[1] for key in keys]
        y = [by_size[key][0] for key in keys]
        errors = [by_size[key][1] for key in keys]
        ax.errorbar(x, y, yerr=errors, **STYLES[algorithm], linewidth=1.5,
                    markersize=5, capsize=2.5 if any(errors) else 0,
                    label=LABELS[algorithm])
    sizes = sorted({int(r["n_base"]) for r in scaling})
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}K" if x >= 1000 else f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Database vectors")
    ax.set_ylabel(ylabel)
    if field == "recall_at_k":
        ax.set_ylim(0, 1.03)
        ax.axhline(target, color="#777777", ls="--", lw=.9, label=f"Target ({target:.0%})")
    elif field in {"median_latency_ms", "p95_latency_ms", "build_seconds"}:
        positive_scale(ax, [float(r[field]) for r in scaling])
    format_axes(ax)


def plot_dataset(folder):
    folder = Path(folder)
    tuning_path, scaling_path = folder / "tuning.csv", folder / "scaling.csv"
    if not tuning_path.exists() or not scaling_path.exists():
        return []
    tuning, scaling = read_rows(tuning_path), read_rows(scaling_path)
    if not scaling:
        return []
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    k = int(metadata.get("k", scaling[0].get("k", 10)))
    target = float(metadata.get("target_recall", scaling[0].get("target_recall", .90)))
    targets = metadata.get("all_targets", [target])
    dataset = {"sift-128-euclidean": "SIFT-128", "glove-100-angular": "GloVe-100"}.get(folder.name, folder.name)
    output = folder / "plots"
    output.mkdir(parents=True, exist_ok=True)
    written = []

    # Validation: benchmark-style Pareto curves. QPS is computed from the
    # recorded mean latency, not the reciprocal of median latency.
    for zoom in (False, True):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        plot_tuning(tuning, "median_latency_ms", "Median query latency (ms)", k,
                    dataset, targets, ax, zoom=zoom)
        save_figure(fig, output / ("tuning_zoom.png" if zoom else "tuning_recall_latency.png"), written)
    if tuning and all(r.get("queries_per_second") not in (None, "") for r in tuning):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        plot_tuning(tuning, "queries_per_second", "Estimated sequential QPS (1/s)",
                    k, dataset, targets, ax, higher_is_better=True)
        save_figure(fig, output / "tuning_recall_qps.png", written)

    metrics = [
        ("median_latency_ms", "Median query latency (ms)", "scaling_latency.png"),
        ("p95_latency_ms", "P95 query latency (ms)", "scaling_p95.png"),
        ("recall_at_k", f"Test Recall@{k}", "scaling_recall.png"),
        ("build_seconds", "Index build time (s)", "scaling_build.png"),
        ("index_memory_mb_estimate", "Estimated index memory (MiB)", "scaling_memory.png"),
    ]
    # Retain the original individual figures for compatibility.
    for field, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        plot_scaling(ax, scaling, field, ylabel, target)
        ax.set_title(f"{dataset}: fixed parameters, target {target:.0%}")
        ax.legend(loc="best", framealpha=.9)
        save_figure(fig, output / filename, written)

    # A compact paper-ready view of the four principal scalability metrics.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), layout="constrained")
    for i, (ax, (field, ylabel, _)) in enumerate(zip(axes.flat, [metrics[j] for j in (0, 2, 3, 4)])):
        plot_scaling(ax, scaling, field, ylabel, target)
        ax.set_title(f"({chr(97+i)}) {['Query latency', 'Recall', 'Construction', 'Memory'][i]}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="best", fontsize=8)
    axes[0, 1].legend([axes[0, 1].lines[-1]], [f"Target ({target:.0%})"], loc="lower left", fontsize=8)
    fig.suptitle(f"{dataset}: held-out scalability (target {target:.0%})")
    save_figure(fig, output / "scaling_combined.png", written)

    max_n = max(int(r["n_base"]) for r in scaling)
    largest = [r for r in scaling if int(r["n_base"]) == max_n]
    summary = defaultdict(list)
    for row in largest:
        summary[row["algorithm"]].append(row)

    def mean_sd(values):
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.

    selected = []
    for algorithm in ordered_algorithms(largest):
        rows = summary[algorithm]
        recall, recall_sd = mean_sd([r["recall_at_k"] for r in rows])
        latency, latency_sd = mean_sd([r["median_latency_ms"] for r in rows])
        passed = all(r.get("test_target_met", r["recall_at_k"] >= target) for r in rows)
        selected.append(dict(algorithm=algorithm, recall=recall, recall_sd=recall_sd,
                             latency=latency, latency_sd=latency_sd, passed=passed))

    # Held-out measurements are independent points, not a continuous curve.
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for r in selected:
        style = STYLES[r["algorithm"]]
        ax.errorbar(r["recall"], r["latency"], xerr=r["recall_sd"], yerr=r["latency_sd"],
                    fmt=style["marker"], color=style["color"], markersize=7, capsize=3,
                    markerfacecolor=style["color"] if r["passed"] else "none",
                    label=LABELS[r["algorithm"]] + (" (target missed)" if not r["passed"] else ""))
    ax.axvline(target, color="#777777", ls="--", lw=.9, label=f"Target ({target:.0%})")
    ax.set_xlim(-.02, 1.03)
    positive_scale(ax, [r["latency"] for r in selected])
    ax.set_xlabel(f"Held-out test Recall@{k}")
    ax.set_ylabel("Median query latency (ms)")
    ax.set_title(f"{dataset}: held-out test (n={max_n:,})")
    format_axes(ax)
    ax.legend(loc="best", framealpha=.9)
    save_figure(fig, output / "test_recall_latency.png", written)

    # Aligned dot plots retain the same algorithm markers; hollow = target missed.
    selected.sort(key=lambda r: (r["algorithm"] == "knn", r["latency"]))
    positions = np.arange(len(selected))
    fig, (ax_recall, ax_latency) = plt.subplots(1, 2, figsize=(10, max(3.7, len(selected)*.65+1.7)),
                                             sharey=True, layout="constrained")
    ax_recall.axvline(target, color="#777777", ls="--", lw=1, label=f"Target ({target:.0%})")
    for y, r in zip(positions, selected):
        style = STYLES[r["algorithm"]]
        kwargs = dict(fmt=style["marker"], color=style["color"], markersize=7, capsize=3,
                      markerfacecolor=style["color"] if r["passed"] else "none", zorder=3)
        ax_recall.errorbar(r["recall"], y, xerr=r["recall_sd"], **kwargs)
        ax_latency.errorbar(r["latency"], y, xerr=r["latency_sd"], **kwargs)
    ax_recall.set_yticks(positions, labels=[LABELS[r["algorithm"]] + (" (missed)" if not r["passed"] else "") for r in selected])
    ax_recall.invert_yaxis()
    ax_recall.set_xlim(-.02, 1.04)
    ax_recall.set_xlabel(f"Test Recall@{k}")
    ax_latency.set_xlabel("Median query latency (ms)")
    ax_recall.set_title("Accuracy")
    ax_latency.set_title("Search speed")
    positive_scale(ax_latency, [r["latency"] for r in selected], axis="x")
    ax_recall.legend(loc="best", fontsize=8)
    for ax in (ax_recall, ax_latency):
        format_axes(ax)
    fig.suptitle(f"{dataset}: selected configurations, held-out test (n={max_n:,})\n"
                 "Error bars: ±1 SD across builds; hollow markers: target missed")
    save_figure(fig, output / "test_algorithm_comparison.png", written)

    sensitivity_path = folder / "sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = read_rows(sensitivity_path)
        for algorithm in ordered_algorithms(sensitivity):
            rows = sorted((r for r in sensitivity if r["algorithm"] == algorithm), key=lambda r: r["parameter_value"])
            param = rows[0]["parameter_name"]
            x = [r["parameter_value"] for r in rows]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.3), sharex=True, layout="constrained")
            ax1.plot(x, [r["recall_at_k"] for r in rows], **STYLES[algorithm], markersize=5)
            ax1.axhline(target, color="#777777", ls="--", lw=.9, label=f"Target ({target:.0%})")
            ax1.set_ylabel(f"Validation Recall@{k}")
            ax1.set_ylim(0, 1.03)
            ax1.legend(loc="best", fontsize=8)
            ax2.plot(x, [r["median_latency_ms"] for r in rows], **STYLES[algorithm], markersize=5)
            ax2.set_ylabel("Median query latency (ms)")
            ax2.set_xlabel(param)
            ax2.set_xticks(x)
            for ax in (ax1, ax2):
                format_axes(ax)
            fig.suptitle(f"{dataset}: {LABELS[algorithm]} parameter sensitivity (validation)")
            save_figure(fig, output / f"sensitivity_{algorithm}.png", written)
    print(f"Saved {len(written)} figures in {output}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/optimized"))
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Missing input directory: {args.input_dir}")
    with plt.style.context("default"):
        for folder in sorted(args.input_dir.iterdir()):
            if folder.is_dir():
                plot_dataset(folder)


if __name__ == "__main__":
    main()
