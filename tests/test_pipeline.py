import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from datasets import load_hdf5, infer_metric
from evaluate import recall_at_k, get_ground_truth

class TestPipeline(unittest.TestCase):
    def test_subset_ground_truth_cached(self):
        rng = np.random.default_rng(91)
        base = rng.normal(size=(100, 7)).astype(np.float32)
        queries = rng.normal(size=(6, 7)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            a, _, path, cached = get_ground_truth(base, queries, metric="euclidean", k=6,
                                                   dataset_name="sample", output_dir=Path(temp))
            self.assertFalse(cached)
            b, _, path2, cached2 = get_ground_truth(base, queries, metric="euclidean", k=6,
                                                    dataset_name="sample", output_dir=Path(temp))
            self.assertTrue(cached2)
            self.assertEqual(path, path2)
            self.assertEqual(recall_at_k(a, b, 6), 1.0)

    def test_hdf5_subset_has_no_invalid_official_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "glove-100-angular.hdf5"
            with h5py.File(path, "w") as f:
                f.attrs["distance"] = "angular"
                f.create_dataset("train", data=np.ones((20, 4), dtype=np.float32))
                f.create_dataset("test", data=np.ones((5, 4), dtype=np.float32))
                f.create_dataset("neighbors", data=np.tile(np.arange(10), (5, 1)))
            base, q, truth = load_hdf5(path, n_base=10, n_queries=3)
            self.assertEqual(infer_metric(path), "cosine")
            self.assertEqual(base.shape, (10, 4))
            self.assertEqual(q.shape, (3, 4))
            self.assertIsNone(truth)
            self.assertEqual(load_hdf5(path, n_base=20, n_queries=3)[2].shape, (3, 10))

    def test_cli_smoke_all_algorithms(self):
        with tempfile.TemporaryDirectory() as temp:
            for alg in ("knn", "lsh", "ivf", "hnsw"):
                cmd = [sys.executable, "run.py", "--algorithm", alg, "--n-base", "140",
                       "--n-queries", "5", "--k", "5", "--output-dir", temp,
                       "--n-clusters", "10", "--M", "4", "--ef-construction", "16"]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            summaries = [p for p in Path(temp).glob("*.json")]
            self.assertEqual(len(summaries), 4)
            for path in summaries:
                result = json.loads(path.read_text())
                self.assertTrue(0 <= result["recall_at_k"] <= 1)
                self.assertTrue(result["mean_latency_ms"] > 0)
                self.assertTrue(result["build_seconds"] >= 0)

if __name__ == "__main__":
    unittest.main()
