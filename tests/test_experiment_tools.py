
import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from algorithms.ivf import IVFIndex
from evaluate import benchmark_queries
from experiments.benchmark import (configs_for, label_for_target, main as benchmark_main,
                                   make_model, select_best)
from experiments.plots import frontier, grouped_by_algorithm_size, plot_dataset


class TestExperimentTools(unittest.TestCase):
    @staticmethod
    def record(recall, latency, algorithm="ivf", n=100):
        return {"algorithm": algorithm, "recall_at_k": recall,
                "median_latency_ms": latency, "n_base": n,
                "build_seconds": .01, "p95_latency_ms": latency * 1.1,
                "index_memory_mb_estimate": .25, "seed": 42,
                "parameters": {"nprobe": 2}, "target_recall": .9,
                "test_target_met": recall >= .9}

    def test_pareto_drops_dominated_configurations(self):
        rows = [self.record(.5, .3), self.record(.7, .2), self.record(.8, .4)]
        result = frontier(rows)
        self.assertEqual([r["recall_at_k"] for r in result], [.7, .8])

    def test_target_selection_and_unmet_flag(self):
        rows = [self.record(.79, .1), self.record(.91, .4), self.record(.95, .6)]
        best, met = select_best(rows, .90)
        self.assertTrue(met)
        self.assertEqual(best["recall_at_k"], .91)
        best, met = select_best(rows, .99)
        self.assertFalse(met)
        self.assertEqual(best["recall_at_k"], .95)

    def test_vamana_construction_and_query_width_decoupled(self):
        configs = configs_for("vamana", "grid", 1, 42, "euclidean")
        self.assertIn("search_L", configs[0])
        self.assertTrue(any(c["search_L"] != c["L"] for c in configs))
        different = next(c for c in configs if c["L"] != c["search_L"])
        model = make_model("vamana", "euclidean", different, 42, 10)
        self.assertNotEqual(model.L, model.search_L)

    def test_ivf_probe_diagnostics_are_exposed(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(100, 5)).astype(np.float32)
        index = IVFIndex(n_clusters=10, nprobe=1, seed=2).fit(x)
        idx, d, summary = benchmark_queries(index, x[:5], k=7, warmup=1, repeats=2)
        self.assertEqual(idx.shape, (5, 7))
        self.assertEqual(d.shape, (5, 7))
        self.assertIn("p95_latency_ms", summary)
        self.assertGreaterEqual(summary["mean_clusters_probed"], 1)
        self.assertGreaterEqual(summary["mean_ivf_candidates"], 7)
        self.assertLessEqual(summary["median_latency_ms"], summary["p95_latency_ms"])

    def test_grouping_seeds(self):
        rows = [self.record(.8, 1), self.record(.9, 2)]
        summary = grouped_by_algorithm_size(rows, "recall_at_k")
        self.assertAlmostEqual(summary[("ivf", 100)][0], .85)
        self.assertEqual(summary[("ivf", 100)][2], 2)

    def test_target_label(self):
        self.assertEqual(label_for_target(.8), "r800")
        self.assertEqual(label_for_target(.95), "r950")

    def test_end_to_end_tune_once_multiple_targets_and_seeds(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "synthetic-euclidean.hdf5"
            with h5py.File(dataset, "w") as f:
                f.attrs["distance"] = "euclidean"
                f.create_dataset("train", data=rng.normal(size=(256, 8)).astype("float32"))
                f.create_dataset("test", data=rng.normal(size=(16, 8)).astype("float32"))
            out = root / "results"
            benchmark_main([
                "--dataset", str(dataset), "--algorithm", "ivf",
                "--tune-size", "256", "--sizes", "128", "256",
                "--tune-queries", "5", "--test-queries", "8", "--k", "5",
                "--trials", "1", "--seed", "42", "--seeds", "42", "123",
                "--target-recalls", ".8", ".95", "--repeats", "2",
                "--output-dir", str(out),
            ])
            for label in ("r800", "r950"):
                folder = out / label / dataset.stem
                self.assertTrue((folder / "tuning.csv").exists())
                self.assertTrue((folder / "summary.csv").exists())
                self.assertTrue((folder / "metadata.json").exists())
                with (folder / "scaling.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 6)  # 2 exact + 2 sizes x 2 IVF seeds
                self.assertEqual({r["algorithm"] for r in rows}, {"knn", "ivf"})
                self.assertTrue(all("p95_latency_ms" in r for r in rows))
                figpaths = plot_dataset(folder)
                self.assertGreaterEqual(len(figpaths), 6)
                self.assertTrue(all(p.exists() for p in figpaths))
                metadata = json.loads((folder / "metadata.json").read_text())
                self.assertEqual(metadata["final_seeds"], [42, 123])
                self.assertTrue(metadata["knn_timed_baseline"])


if __name__ == "__main__":
    unittest.main()
