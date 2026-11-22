
import json
import tempfile
import unittest
from pathlib import Path
from experiments.benchmark import configurations
from experiments.plots import fixed_scaling_config, pareto_frontier, plot_results


def record(alg, n, recall, latency, config=None):
    return dict(algorithm=alg, dataset='unit-test', metric='euclidean', k=10,
                n_base=n, n_queries=500, parameters=config or {'nprobe': 4},
                recall_at_k=recall, mean_latency_ms=latency,
                build_seconds=1.0, index_memory_mb_estimate=2.0)


class TestExperimentTools(unittest.TestCase):
    def test_pareto_drops_dominated(self):
        data = [record('ivf', 1000, .5, .3), record('ivf', 1000, .7, .2),
                record('ivf', 1000, .8, .4)]
        out = pareto_frontier(data)
        self.assertEqual(len(out), 2)
        self.assertEqual([x['recall_at_k'] for x in out], [.7, .8])

    def test_fixed_config_cannot_switch_by_size(self):
        rows = [record('ivf', 1000, .9, .4, {'a': 1}),
                record('ivf', 2000, .6, .4, {'a': 1}),
                record('ivf', 1000, .6, .3, {'a': 2}),
                record('ivf', 2000, .95, .3, {'a': 2})]
        sig, data = fixed_scaling_config(rows, [1000, 2000])
        self.assertEqual(json.loads(sig), {'a': 2})
        self.assertEqual(set(data), {1000, 2000})
        self.assertEqual(data[1000]['parameters'], data[2000]['parameters'])

    def test_no_one_point_scalability_plots(self):
        with tempfile.TemporaryDirectory() as folder:
            inp, out = Path(folder) / 'results', Path(folder) / 'plots'
            inp.mkdir()
            (inp / 'a.json').write_text(json.dumps(record('ivf', 1000, .9, .2)))
            plots = plot_results(inp, out)
            self.assertEqual(len(plots), 1)
            self.assertTrue(plots[0].name.startswith('recall_latency_'))

    def test_two_point_scaling_plots(self):
        with tempfile.TemporaryDirectory() as folder:
            inp, out = Path(folder) / 'results', Path(folder) / 'plots'
            inp.mkdir()
            for i, n in enumerate([1000, 2000]):
                (inp / f'{i}.json').write_text(json.dumps(record('ivf', n, .8, .2 + .1*i)))
            self.assertEqual(len(plot_results(inp, out)), 5)

    def test_lsh_width_sweep(self):
        configs = list(configurations('lsh', 'euclidean'))
        self.assertTrue(any('16.0' in c for c in configs))
        self.assertTrue(any('--lsh-hashes' in c for c in configs))


if __name__ == '__main__':
    unittest.main()
