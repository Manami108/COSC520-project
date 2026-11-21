"""Sweep parameters using the repository's run.py, saving each run's JSON."""
import argparse
import itertools
import subprocess
import sys


def configurations(algorithm):
    if algorithm == 'knn':
        yield []
    elif algorithm == 'hnsw':
        for m, efc, efs in itertools.product((4, 8), (40, 80), (20, 60)):
            if efc >= m:
                yield ['--M', str(m), '--ef-construction', str(efc), '--ef-search', str(efs)]
    elif algorithm == 'ivf':
        for clusters, probes in itertools.product((32, 64), (1, 4, 8)):
            yield ['--n-clusters', str(clusters), '--nprobe', str(probes)]
    elif algorithm == 'lsh':
        for tables, hashes in itertools.product((4, 8, 16), (4, 6)):
            yield ['--lsh-tables', str(tables), '--lsh-hashes', str(hashes)]
    elif algorithm == 'vamana':
        for r, l, alpha, search_l in itertools.product((8, 16), (40,), (1.1, 1.2), (20, 60)):
            yield ['--vamana-R', str(r), '--vamana-L', str(l),
                   '--vamana-alpha', str(alpha), '--vamana-search-L', str(search_l)]
    else:
        raise ValueError(algorithm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=['all', 'knn', 'lsh', 'ivf', 'hnsw', 'vamana'], required=True)
    p.add_argument('--dataset')
    p.add_argument('--n-base', type=int, default=10000)
    p.add_argument('--n-queries', type=int, default=25)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--output-dir', default='results')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    names = ['knn', 'lsh', 'ivf', 'hnsw', 'vamana'] if args.algorithm == 'all' else [args.algorithm]
    for alg in names:
        for config in configurations(alg):
            cmd = [sys.executable, 'run.py', '--algorithm', alg, '--n-base', str(args.n_base),
                   '--n-queries', str(args.n_queries), '--k', str(args.k),
                   '--output-dir', args.output_dir]
            if args.dataset:
                cmd += ['--dataset', args.dataset]
            cmd += config
            print(' '.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
