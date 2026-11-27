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


# Combined figures for the report (read the existing benchmark CSV files).
REPORT_DATASETS = {
    "sift-128-euclidean": "SIFT-128",
    "glove-100-angular": "GloVe-100",
}
REPORT_TARGETS = (0.80, 0.90, 0.95)


def report_data(root, filename, target=.90):
    """Read available results from one recall-target directory."""
    label = f"r{round(target * 1000):03d}"
    datasets = {}
    for name, title in REPORT_DATASETS.items():
        folder = root / label / name
        path = folder / filename
        if not path.is_file():
            print(f"Missing {path}; report data unavailable.")
            continue
        rows = read_rows(path)
        if not rows:
            print(f"Empty {path}; report data unavailable.")
            continue
        meta_path = folder / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if ("target_recall" in metadata and
                not np.isclose(float(metadata["target_recall"]), target)):
            raise ValueError(f"Recall target in {meta_path} does not match {label}")
        datasets[name] = (title, rows, metadata)
    return datasets


def report_k(datasets):
    values = {int(meta.get("k", rows[0].get("k", 10)))
              for _, rows, meta in datasets.values()}
    if len(values) != 1:
        raise ValueError("Combined figures require the same k for both datasets")
    return values.pop()


def report_legend(datasets, *, exact=True):
    from matplotlib.lines import Line2D
    present = {r["algorithm"] for _, rows, _ in datasets.values() for r in rows}
    return [Line2D([], [], **STYLES[a], markersize=5, linewidth=1.5,
                   label=LABELS[a]) for a in STYLES
            if a in present and (exact or a != "knn")]


def report_qps(root, output, written):
    datasets = report_data(root, "tuning.csv")
    if len(datasets) != len(REPORT_DATASETS):
        return
    if any(not all(r.get("queries_per_second") not in (None, "")
                   and np.isfinite(float(r["queries_per_second"]))
                   and float(r["queries_per_second"]) > 0 for r in rows)
           for _, rows, _ in datasets.values()):
        print("Skipping report_recall_qps.png: missing or invalid QPS values.")
        return
    k = report_k(datasets)
    sizes = {int(r["n_base"]) for _, rows, _ in datasets.values() for r in rows}
    if len(sizes) != 1:
        raise ValueError("Combined tuning figure requires a common tuning database size")
    n = sizes.pop()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    for i, (ax, (title, rows, _)) in enumerate(zip(axes, datasets.values())):
        plot_tuning(rows, "queries_per_second", "Estimated sequential QPS (1/s)",
                    k, title, REPORT_TARGETS, ax, higher_is_better=True)
        ax.set_title(f"({chr(97+i)}) {title}")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        if i:
            ax.set_ylabel("")
    fig.suptitle(f"Recall–QPS comparison on validation queries (n={n:,})", fontsize=13)
    handles = report_legend(datasets, exact=False)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_figure(fig, output / "report_recall_qps.png", written)


def report_scalability(root, output, written):
    target = .90
    datasets = report_data(root, "scaling.csv", target)
    if len(datasets) != len(REPORT_DATASETS):
        return
    k = report_k(datasets)
    metrics = [
        ("median_latency_ms", "Median query latency (ms)"),
        ("recall_at_k", f"Test Recall@{k}"),
        ("build_seconds", "Index build time (s)"),
        ("index_memory_mb_estimate", "Estimated memory (MiB)"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(10.6, 12), sharey="row")
    for j, (title, rows, _) in enumerate(datasets.values()):
        for i, (field, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            plot_scaling(ax, rows, field, ylabel, target)
            if i == 0:
                ax.set_title(title, fontweight="bold")
            if j:
                ax.set_ylabel("")
            if i < len(metrics) - 1:
                ax.set_xlabel("")
    # A metric has the same scale and limits in both dataset panels.
    for i, (field, _) in enumerate(metrics):
        if field in {"median_latency_ms", "build_seconds"}:
            values = [float(r[field]) for _, rows, _ in datasets.values() for r in rows]
            scale = "log" if all(np.isfinite(v) and v > 0 for v in values) else "linear"
            for ax in axes[i]:
                ax.set_yscale(scale)
                format_axes(ax)
    from matplotlib.lines import Line2D
    handles = report_legend(datasets)
    handles.append(Line2D([], [], color="#777777", ls="--", lw=.9, label="90% recall target"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Held-out scalability: configurations selected for the 90% validation recall target\n"
                 "Error bars: ±1 SD across index-build seeds", fontsize=13)
    fig.tight_layout(rect=(0, .065, 1, .945))
    save_figure(fig, output / "report_scalability.png", written)


def report_speedup(root, output, written):
    """Compare speedups only when every evaluated index seed meets the test target."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    results = {t: report_data(root, "scaling.csv", t) for t in REPORT_TARGETS}
    available = [rows for datasets in results.values() for _, rows, _ in datasets.values()]
    if not available:
        return
    # Use the largest common size so all populated cells refer to the same N.
    common_sizes = set.intersection(*({int(r["n_base"]) for r in rows} for rows in available))
    if not common_sizes:
        print("Skipping report_speedup_heatmap.png: no common database size.")
        return
    n = max(common_sizes)
    ks = {int(meta.get("k", rows[0].get("k", 10)))
          for datasets in results.values() for _, rows, meta in datasets.values()}
    if len(ks) != 1:
        raise ValueError("Speedup heatmap requires the same k across targets and datasets")
    k = ks.pop()
    algorithms = [a for a in STYLES if a != "knn"]
    matrices, statuses = {}, {}
    for name in REPORT_DATASETS:
        values = np.full((len(algorithms), len(REPORT_TARGETS)), np.nan)
        status = np.full(values.shape, "N/A", dtype=object)
        for j, target in enumerate(REPORT_TARGETS):
            if name not in results[target]:
                continue
            _, rows, meta = results[target][name]
            largest = [r for r in rows if int(r["n_base"]) == n]
            baseline = [r["median_latency_ms"] for r in largest if r["algorithm"] == "knn"]
            expected_seeds = set(meta.get("final_seeds", []))
            for i, algorithm in enumerate(algorithms):
                group = [r for r in largest if r["algorithm"] == algorithm]
                if not group:
                    continue
                if expected_seeds and {int(r["seed"]) for r in group} != expected_seeds:
                    continue  # incomplete seed results are not a valid comparison
                if not all(r["recall_at_k"] >= target and r.get("test_target_met", True)
                           for r in group):
                    status[i, j] = "Missed"
                    continue
                latency = float(np.mean([r["median_latency_ms"] for r in group]))
                if baseline and np.isfinite(latency) and latency > 0:
                    exact = float(np.mean(baseline))
                    if np.isfinite(exact) and exact > 0:
                        values[i, j] = exact / latency
                        status[i, j] = "OK"
        matrices[name], statuses[name] = values, status
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    norm = Normalize(vmin=0, vmax=max(1., float(finite.max())) if finite.size else 1.)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), layout="constrained")
    for ax, (name, title) in zip(axes, REPORT_DATASETS.items()):
        values, status = matrices[name], statuses[name]
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="equal")
        for i, algorithm in enumerate(algorithms):
            for j, target in enumerate(REPORT_TARGETS):
                v = values[i, j]
                if np.isfinite(v):
                    label = f"{v:.2f}×" if v < 10 else f"{v:.1f}×"
                    rgba = cmap(norm(v))
                    luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
                    color = "white" if luminance < .45 else "#222222"
                else:
                    label, color = status[i, j], "#555555"
                    if label == "Missed":
                        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, hatch="///",
                                               facecolor="none", edgecolor="#bbbbbb", lw=0))
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks(range(len(REPORT_TARGETS)), [f"{t:.0%}" for t in REPORT_TARGETS])
        ax.set_yticks(range(len(algorithms)), [LABELS[a] for a in algorithms])
        ax.set_xlabel(f"Required Recall@{k}")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(-.5, len(REPORT_TARGETS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(algorithms), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=.75, label="Speedup vs. exact k-NN (×)")
    fig.suptitle(f"Held-out speedup at n={n:,}\n"
                 "Missed = at least one seed below target; N/A = unavailable or incomplete", fontsize=12)
    save_figure(fig, output / "report_speedup_heatmap.png", written)


def report_sensitivity(root, output, written):
    datasets = report_data(root, "sensitivity.csv")
    if len(datasets) != len(REPORT_DATASETS):
        print("Run the 90% sensitivity analysis for both datasets before generating this figure.")
        return
    k = report_k(datasets)
    algorithms = [a for a in STYLES if a != "knn"]
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, len(algorithms), figsize=(13.8, 6.5), sharex="col", sharey="row")
    for j, a in enumerate(algorithms):
        style = STYLES[a]
        parameters, ticks = set(), set()
        for d, (_, rows, _) in enumerate(datasets.values()):
            group = sorted((r for r in rows if r["algorithm"] == a), key=lambda r: r["parameter_value"])
            if not group:
                continue
            parameters.update(r["parameter_name"] for r in group)
            x = [r["parameter_value"] for r in group]
            ticks.update(x)
            # Preserve algorithm colour, marker and line style. Dataset is
            # distinguished by filled versus hollow markers.
            kwargs = dict(**style, markersize=5, linewidth=1.5,
                          markerfacecolor=style["color"] if d == 0 else "white",
                          markeredgewidth=1.2, alpha=1. if d == 0 else .7)
            axes[0, j].plot(x, [r["recall_at_k"] for r in group], **kwargs)
            axes[1, j].plot(x, [r["median_latency_ms"] for r in group], **kwargs)
        if len(parameters) > 1:
            raise ValueError(f"Inconsistent sensitivity parameter for {a}: {parameters}")
        axes[0, j].set_title(LABELS[a], fontweight="bold", color=style["color"])
        axes[0, j].axhline(.90, color="#777777", ls="--", lw=.9)
        axes[0, j].set_ylim(0, 1.03)
        axes[1, j].set_xlabel(next(iter(parameters), "Parameter"))
        axes[1, j].set_xticks(sorted(ticks))
        for ax in axes[:, j]:
            format_axes(ax)
            if not ticks:
                ax.text(.5, .5, "No results", transform=ax.transAxes, ha="center")
    values = [r["median_latency_ms"] for _, rows, _ in datasets.values() for r in rows]
    if values and all(np.isfinite(v) and v > 0 for v in values):
        for ax in axes[1]:
            ax.set_yscale("log")
            format_axes(ax)
    axes[0, 0].set_ylabel(f"Validation Recall@{k}")
    axes[1, 0].set_ylabel("Median query latency (ms)")
    handles = [Line2D([], [], color="#444444", marker="o", label="SIFT-128 (filled)", lw=1.5),
               Line2D([], [], color="#444444", marker="o", markerfacecolor="white",
                      alpha=.7, label="GloVe-100 (hollow)", lw=1.5),
               Line2D([], [], color="#777777", ls="--", lw=.9, label="90% recall target")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Parameter sensitivity on validation queries\n"
                 "Other parameters fixed at each dataset's configuration selected for the 90% target", fontsize=13)
    fig.tight_layout(rect=(0, .08, 1, .90))
    save_figure(fig, output / "report_sensitivity.png", written)


def plot_report(root):
    root = Path(root)
    output = root / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for plot in (report_qps, report_scalability, report_speedup, report_sensitivity):
        plot(root, output, written)
    print(f"Saved {len(written)} combined report figures in {output}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/optimized"))
    parser.add_argument("--report-only", action="store_true",
                        help="Generate combined figures from the recall_targets root without rerunning experiments.")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Missing input directory: {args.input_dir}")
    with plt.style.context("default"):
        if args.report_only or any((args.input_dir / f"r{round(t*1000):03d}").is_dir()
                                   for t in REPORT_TARGETS):
            plot_report(args.input_dir)
        else:
            for folder in sorted(args.input_dir.iterdir()):
                if folder.is_dir():
                    plot_dataset(folder)


if __name__ == "__main__":
    main()
