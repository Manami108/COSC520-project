"""Create reproducible comparison figures from run.py JSON result files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(input_dir: Path, output_dir: Path):
    records = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        needed = ("algorithm", "dataset", "n_base", "recall_at_k", "mean_latency_ms")
        if all(key in row for key in needed):
            records.append(row)
    if not records:
        raise ValueError(f"No benchmark JSON files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = sorted({r['algorithm'] for r in records})
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    metrics = [
        ('mean_latency_ms', 'Mean query latency (ms)', 'recall_vs_latency'),
        ('queries_per_second', 'Queries per second', 'recall_vs_qps'),
    ]
    datasets = sorted({r['dataset'] for r in records})
    for dataset in datasets:
        rows = [r for r in records if r['dataset'] == dataset]
        # Separate database sizes to prevent misleading cross-scale curves.
        for size in sorted({r['n_base'] for r in rows}):
            selected = [r for r in rows if r['n_base'] == size]
            for ykey, ylabel, filename in metrics:
                fig, ax = plt.subplots(figsize=(8, 5.5))
                for i, alg in enumerate(algorithms):
                    subset = sorted((r for r in selected if r['algorithm'] == alg and ykey in r),
                                    key=lambda r: r['recall_at_k'])
                    if subset:
                        ax.plot([r['recall_at_k'] for r in subset],
                                [r[ykey] for r in subset], marker=markers[i % len(markers)],
                                label=alg.upper(), linewidth=1.5)
                ax.set(xlabel='Recall@k', ylabel=ylabel,
                       title=f'{dataset} — {size:,} vectors')
                ax.grid(True, alpha=.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f'{filename}_{dataset}_n{size}.png', dpi=200)
                plt.close(fig)
        # For scalability, multiple parameter configurations are summarized by
        # choosing highest recall at each (algorithm, n_base), not average performance.
        for key, ylabel, filename in [
            ('mean_latency_ms', 'Mean query latency (ms)', 'scaling_latency'),
            ('index_memory_mb_estimate', 'Estimated index memory (MiB)', 'scaling_memory'),
            ('build_seconds', 'Index build time (s)', 'scaling_build')]:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for i, alg in enumerate(algorithms):
                points = {}
                for r in rows:
                    if r['algorithm'] == alg and key in r:
                        old = points.get(r['n_base'])
                        if old is None or r['recall_at_k'] > old['recall_at_k']:
                            points[r['n_base']] = r
                if points:
                    sizes = sorted(points)
                    ax.plot(sizes, [points[s][key] for s in sizes],
                            marker=markers[i % len(markers)], label=alg.upper())
            ax.set(xlabel='Number of database vectors', ylabel=ylabel,
                   title=f'{dataset} — highest-recall configuration per size')
            ax.grid(True, alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f'{filename}_{dataset}.png', dpi=200)
            plt.close(fig)
    print(f'Created figures in {output_dir} from {len(records)} benchmark runs')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=Path('results'))
    p.add_argument('--output-dir', type=Path, default=Path('results/plots'))
    args = p.parse_args()
    plot_results(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
