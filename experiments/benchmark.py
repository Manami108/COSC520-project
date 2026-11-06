"""Parameter sensitivity sweeps using the common run.py benchmark pipeline.

Example: python -m experiments.benchmark --algorithm hnsw --dry-run
         python -m experiments.benchmark --algorithm hnsw --dataset data/...hdf5
"""
import argparse
import itertools
import subprocess
import sys


def configurations(algorithm: str):
    if algorithm == "hnsw":
        for m, ef_c, ef_s in itertools.product((4, 8), (40, 80), (20, 60)):
            yield ["--M", str(m), "--ef-construction", str(ef_c), "--ef-search", str(ef_s)]
    elif algorithm == "ivf":
        for n_clusters, nprobe in itertools.product((32, 64), (1, 4, 8)):
            yield ["--n-clusters", str(n_clusters), "--nprobe", str(nprobe)]
    else:
        for tables, hashes in itertools.product((4, 8, 16), (4, 6)):
            yield ["--lsh-tables", str(tables), "--lsh-hashes", str(hashes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["lsh", "ivf", "hnsw"], required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--n-base", type=int, default=10000)
    parser.add_argument("--n-queries", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for config in configurations(args.algorithm):
        cmd = [sys.executable, "run.py", "--algorithm", args.algorithm,
               "--n-base", str(args.n_base), "--n-queries", str(args.n_queries),
               "--k", str(args.k)]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        cmd += config
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
