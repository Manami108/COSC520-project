import unittest
import numpy as np
from algorithms.ivf import IVFIndex
from algorithms.knn import BruteForceKNN
from evaluate import recall_at_k

class TestIVF(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(70)
        self.base = rng.normal(size=(250, 12)).astype(np.float32)
        self.q = rng.normal(size=(8, 12)).astype(np.float32)

    def test_all_clusters_probe_matches_exact_l2(self):
        exact, _ = BruteForceKNN().fit(self.base).query(self.q, k=10)
        ids, d = IVFIndex(n_clusters=10, nprobe=10, max_iter=6).fit(self.base).query(self.q, k=10)
        np.testing.assert_array_equal(ids, exact)
        self.assertTrue(np.isfinite(d).all())

    def test_all_clusters_probe_matches_exact_cosine(self):
        exact, _ = BruteForceKNN(metric="cosine").fit(self.base).query(self.q, k=7)
        ids, d = IVFIndex(metric="cosine", n_clusters=12, nprobe=12, max_iter=5).fit(self.base).query(self.q, k=7)
        self.assertEqual(recall_at_k(exact, ids, 7), 1.0)
        self.assertTrue(np.isfinite(d).all())

    def test_partial_probing_output(self):
        m = IVFIndex(n_clusters=16, nprobe=1, max_iter=4).fit(self.base)
        ids, _ = m.query(self.q, k=15)
        self.assertEqual(ids.shape, (8, 15))
        self.assertTrue(all(len(set(row)) == 15 for row in ids))
        self.assertGreater(m.index_bytes, self.base.nbytes)

    def test_errors(self):
        with self.assertRaises(ValueError):
            IVFIndex(n_clusters=10, nprobe=11)
        with self.assertRaises(ValueError):
            IVFIndex(n_clusters=11, train_size=10)
        with self.assertRaises(ValueError):
            IVFIndex(n_clusters=300).fit(self.base)

if __name__ == "__main__":
    unittest.main()
