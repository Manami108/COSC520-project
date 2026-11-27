"""COSC 520: interactive demonstration of the project's five search algorithms.

Run from the repository root: python demo.py
The demo uses the existing algorithms and evaluation code, not a separate ANN
implementation. The 2-D PCA projection is for display only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import numpy as np

from algorithms.knn import BruteForceKNN
from algorithms.lsh import LSHIndex
from algorithms.ivf import IVFIndex
from algorithms.hnsw import HNSWIndex
from algorithms.vamana import VamanaIndex
from algorithms.common import prepare_vectors
from datasets import infer_metric, load_hdf5
from evaluate import benchmark_queries, recall_at_k

ORDER = ("k-NN", "LSH", "IVF", "HNSW", "Vamana")
COLORS = dict(zip(ORDER, ("#666666", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")))


def get_data(args):
    if args.dataset:
        base, queries, _ = load_hdf5(args.dataset, n_base=args.n_base,
                                      n_queries=args.query_offset + args.n_queries)
        return (base, queries[args.query_offset:], infer_metric(args.dataset),
                args.dataset.stem)
    rng = np.random.default_rng(args.seed)
    # Independent database and query vectors, with no copied database queries.
    base = rng.normal(size=(args.n_base, 32)).astype(np.float32)
    queries = rng.normal(size=(args.n_queries, 32)).astype(np.float32)
    return base, queries, args.metric, "Synthetic (32 dimensions)"


def build(model, base, name):
    print(f"Building {name} ...", flush=True)
    start = perf_counter()
    model.fit(base)
    elapsed = perf_counter() - start
    return elapsed


def measure(model, queries, truth, k, repeats, build_s, params):
    ids, ds, metrics = benchmark_queries(model, queries, k, warmup=5, repeats=repeats)
    if ids.shape != (len(queries), k) or ds.shape != ids.shape:
        raise RuntimeError("Unexpected query output shape")
    if any(len(set(row)) != k for row in ids) or np.any(ids < 0) or np.any(ids >= len(model.vectors)):
        raise RuntimeError("Invalid or duplicate neighbour IDs")
    if not np.isfinite(ds).all() or np.any(np.diff(ds, axis=1) < -1e-6):
        raise RuntimeError("Invalid or unsorted distances")
    return {"ids": ids, "distances": ds, "recall": recall_at_k(truth, ids, k),
            "latency": metrics["median_latency_ms"], "build": build_s,
            "memory": model.index_bytes / 2**20, "params": params, "metrics": metrics}


def run_demo(base, queries, metric, k, seed, repeats):
    """Two predefined settings per ANN method; identical input and timing protocol."""
    exact = BruteForceKNN(metric=metric)
    exact_build = build(exact, base, "exact k-NN")
    truth, _ = exact.query(queries, k=k)
    baseline = measure(exact, queries, truth, k, repeats, exact_build, "Exact search")
    results = {mode: {"k-NN": baseline} for mode in ("Low", "High")}
    # LSH table count changes the index, so each setting is built separately.
    for mode, tables in (("Low", 4), ("High", 24)):
        model = LSHIndex(metric=metric, n_tables=tables, n_hashes=4,
                         bucket_width=1.0, seed=seed)
        seconds = build(model, base, f"LSH ({tables} tables)")
        results[mode]["LSH"] = measure(model, queries, truth, k, repeats, seconds,
                                        f"n_tables={tables}")
    # Query-only settings reuse exactly the same index for both effort levels.
    n_clusters = min(32, len(base))
    specs = [
        ("IVF", IVFIndex(metric=metric, n_clusters=n_clusters, nprobe=1, seed=seed),
         "nprobe", 1, min(16, n_clusters)),
        ("HNSW", HNSWIndex(metric=metric, M=8, ef_construction=40,
                           ef_search=max(10, k), seed=seed),
         "ef_search", max(10, k), max(120, k)),
        ("Vamana", VamanaIndex(metric=metric, R=16, L=80, alpha=1.2,
                               search_L=max(20, k), seed=seed),
         "search_L", max(20, k), max(120, k)),
    ]
    for name, model, param, low, high in specs:
        seconds = build(model, base, name)
        for mode, value in (("Low", low), ("High", high)):
            setattr(model, param, value)
            results[mode][name] = measure(model, queries, truth, k, repeats, seconds,
                                          f"{param}={value}")
    return truth, results


def print_results(results, k):
    for mode, rows in results.items():
        print(f"\n{mode.upper()} SEARCH EFFORT (same query set)")
        print(f"{'Algorithm':<10} {'Setting':<17} {'Recall@'+str(k):>10} "
              f"{'Median ms':>11} {'Build s':>10} {'MiB est.':>10}")
        print("-" * 75)
        for name in ORDER:
            r = rows[name]
            print(f"{name:<10} {r['params']:<17} {r['recall']:>10.3f} "
                  f"{r['latency']:>11.4f} {r['build']:>10.3f} {r['memory']:>10.2f}")
    print("\nDemo settings are illustrative, not tuned benchmark configurations.")
    print("Build times for IVF/HNSW/Vamana are shared between settings; LSH is rebuilt.")


def project_2d(base, queries, metric):
    # Normalize for cosine, matching the algorithms' representation.
    # Fit the PCA basis on database vectors only; retrieval never uses this basis.
    x = prepare_vectors(base, metric)
    q = prepare_vectors(queries, metric)
    center = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - center, full_matrices=False)
    basis = vt[:2].T
    xp, qp = (x - center) @ basis, (q - center) @ basis
    if xp.shape[1] == 1:
        xp = np.column_stack((xp, np.zeros(len(xp))))
        qp = np.column_stack((qp, np.zeros(len(qp))))
    return xp, qp


class DemoView:
    """Switch between real measured retrieval outputs without rebuilding indexes."""

    def __init__(self, base, queries, metric, dataset, k, truth, results):
        self.x, self.q = project_2d(base, queries, metric)
        self.truth, self.results = truth, results
        self.k, self.mode, self.query = k, "Low", 0
        self.dataset, self.metric = dataset, metric
        self.dimension = base.shape[1]
        self.fig, self.axes = plt.subplots(2, 3, figsize=(13.8, 8.3))
        self.fig.subplots_adjust(left=.055, right=.975, bottom=.16, top=.84,
                                 wspace=.22, hspace=.32)
        self.fig.patch.set_facecolor("white")
        self.fig.suptitle("Exact vs approximate nearest-neighbour search", fontsize=19,
                          fontweight="bold", y=.975)
        self.subtitle = self.fig.text(.5, .918, "", ha="center", fontsize=11)
        handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor="#b8bec6",
                   markeredgecolor="none", label="Database vector", markersize=6),
            Line2D([], [], marker="o", color="none", markerfacecolor="#2a9d65",
                   markeredgecolor="white", label="Correct retrieval", markersize=8),
            Line2D([], [], marker="x", color="#d62728", linestyle="none",
                   label="Missed exact neighbour", markersize=8),
            Line2D([], [], marker="o", color="none", markerfacecolor="#ff9b32",
                   markeredgecolor="white", label="Other retrieved vector", markersize=8),
            Line2D([], [], marker="*", color="none", markerfacecolor="#222222",
                   markeredgecolor="white", label="Query", markersize=12),
        ]
        self.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .084),
                        ncol=5, frameon=False, fontsize=9)
        self.fig.text(.5, .068, "PCA is for display only; distances and recall use the original vector space.",
                      ha="center", fontsize=9, color="#555555")
        self.buttons = []  # retain widget references
        for label, left, callback in (("Previous", .27, lambda _: self.move(-1)),
                                      ("Next query", .39, lambda _: self.move(1)),
                                      ("Low effort", .51, lambda _: self.set_mode("Low")),
                                      ("High effort", .63, lambda _: self.set_mode("High"))):
            button = Button(self.fig.add_axes([left, .018, .105, .037]), label,
                            color="#eef1f5", hovercolor="#dce6f0")
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        points = np.vstack((self.x, self.q))
        self.limits = [(float(points[:, i].min()), float(points[:, i].max())) for i in (0, 1)]
        self.draw()

    def on_key(self, event):
        if event.key == "right":
            self.move(1)
        elif event.key == "left":
            self.move(-1)
        elif event.key in ("l", "h"):
            self.set_mode("Low" if event.key == "l" else "High")

    def move(self, delta):
        self.query = (self.query + delta) % len(self.q)
        self.draw()

    def set_mode(self, mode):
        self.mode = mode
        self.draw()

    def scatter_ids(self, ax, ids, **kwargs):
        if ids:
            pts = self.x[sorted(ids)]
            ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

    def draw(self):
        qi, mode = self.query, self.mode
        exact = set(map(int, self.truth[qi]))
        self.subtitle.set_text(f"{self.dataset} | {len(self.x):,} vectors | d={self.dimension} | "
                               f"{self.metric} | k={self.k} | Query {qi+1}/{len(self.q)} | {mode} effort")
        for ax, name in zip(self.axes.flat, ORDER):
            ax.clear()
            r = self.results[mode][name]
            returned = set(map(int, r["ids"][qi]))
            hits = exact & returned
            ax.scatter(self.x[:, 0], self.x[:, 1], s=7, alpha=.32, c="#aeb6c0",
                       linewidths=0, rasterized=True)
            self.scatter_ids(ax, returned - exact, s=52, c="#ff9b32", edgecolors="white", linewidths=.5, zorder=3)
            self.scatter_ids(ax, hits, s=58, c="#2a9d65", edgecolors="white", linewidths=.5, zorder=4)
            self.scatter_ids(ax, exact - returned, s=61, c="#d62728", marker="x", linewidths=1.5, zorder=5)
            ax.scatter(*self.q[qi], s=170, c="#222222", marker="*", edgecolors="white", linewidths=.6, zorder=6)
            ax.set_title(f"{name} | {len(hits)}/{self.k} correct\n{r['params']}",
                         color=COLORS[name], fontsize=12, fontweight="bold")
            for j, (lo, hi) in enumerate(self.limits):
                pad = max((hi-lo)*.04, .01)
                (ax.set_xlim if j == 0 else ax.set_ylim)(lo-pad, hi+pad)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("PC 1", fontsize=8); ax.set_ylabel("PC 2", fontsize=8)
            for spine in ax.spines.values():
                spine.set_color("#c4c9d0")
        ax = self.axes[1, 2]
        ax.clear(); ax.axis("off")
        ax.set_title(f"Summary across {len(self.q)} queries", fontsize=12, fontweight="bold")
        cells = [[name, f"{self.results[mode][name]['recall']:.3f}",
                  f"{self.results[mode][name]['latency']:.3f}"] for name in ORDER]
        tab = ax.table(cellText=cells, colLabels=["Method", f"Recall@{self.k}", "Median ms"],
                       cellLoc="center", colWidths=[.33, .32, .35], bbox=[0, .26, 1, .70])
        tab.auto_set_font_size(False); tab.set_fontsize(10)
        for (i, j), cell in tab.get_celld().items():
            cell.set_edgecolor("#dce0e5")
            cell.set_facecolor("#eef1f5" if i == 0 else "white")
            if i == 0:
                cell.get_text().set_weight("bold")
            elif j == 0:
                cell.get_text().set_color(COLORS[ORDER[i-1]])
                cell.get_text().set_weight("bold")
        ax.text(.5, .13, "Recall: higher is better.  Latency: lower is better.\n"
                "Switch effort to compare measured outputs.\n"
                "No guaranteed monotonic latency or recall.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#555555")
        self.fig.canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="ANN-Benchmarks HDF5; omit for synthetic data")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="Metric for synthetic data; real data uses its dataset metric")
    parser.add_argument("--n-base", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--no-show", action="store_true", help="Save both views without opening a window")
    args = parser.parse_args(argv)
    if args.n_base < 32 or not 1 <= args.k <= args.n_base or args.n_queries < 1 or args.repeats < 1 or args.query_offset < 0:
        parser.error("Require n-base >=32, 1<=k<=n-base, positive queries/repeats, nonnegative offset")
    if not args.dataset and args.query_offset:
        parser.error("query-offset is only supported with a real dataset")
    base, queries, metric, dataset = get_data(args)
    print(f"\nCOSC 520 demo | {dataset} | {len(base):,} vectors | d={base.shape[1]} | {metric}", flush=True)
    truth, results = run_demo(base, queries, metric, args.k, args.seed, args.repeats)
    print_results(results, args.k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the demonstration separate from formal benchmark outputs.
    summary = {mode: {name: {key: val for key, val in r.items() if key not in ("ids", "distances")}
                      for name, r in rows.items()} for mode, rows in results.items()}
    (args.output_dir / "demo_metrics.json").write_text(json.dumps({"dataset": dataset, "metric": metric,
        "n_base": len(base), "n_queries": len(queries), "k": args.k, "seed": args.seed,
        "query_offset": args.query_offset, "results": summary}, indent=2), encoding="utf-8")
    with plt.style.context("default"):
        view = DemoView(base, queries, metric, dataset, args.k, truth, results)
        for mode in ("Low", "High"):
            view.set_mode(mode)
            view.fig.savefig(args.output_dir / f"demo_{mode.lower()}.png", dpi=180, facecolor="white")
        view.set_mode("Low")
        print(f"\nSaved demo figures and measured metrics to {args.output_dir}", flush=True)
        if args.no_show:
            plt.close(view.fig)
        else:
            print("Use buttons or arrow keys to change query; L/H to change effort.", flush=True)
            plt.show()


if __name__ == "__main__":
    main()
