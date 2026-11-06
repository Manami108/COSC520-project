import unittest
import numpy as np
from algorithms.lsh import LSHIndex
from algorithms.knn import BruteForceKNN
from evaluate import recall_at_k

class TestLSH(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.base = rng.normal(size=(320, 12)).astype(np.float32)
        self.queries = self.base[::32].copy()

    def test_cosine_returns_valid_neighbors(self):
        model = LSHIndex(metric="cosine", n_tables=12, n_hashes=4, seed=4).fit(self.base)
        ids, ds = model.query(self.queries, k=8)
        self.assertEqual(ids.shape, (10, 8))
        self.assertEqual(ds.shape, ids.shape)
        self.assertTrue(np.all(ids >= 0) and np.all(ids < len(self.base)))
        self.assertTrue(all(len(set(row)) == 8 for row in ids))
        self.assertTrue(np.all(ds >= 0))
        exact, _ = BruteForceKNN(metric="cosine").fit(self.base).query(self.queries, k=8)
        self.assertGreater(recall_at_k(exact, ids, 8), 0.3)

    def test_euclidean_reproducible(self):
        a = LSHIndex(metric="euclidean", n_tables=10, n_hashes=3, bucket_width=3, seed=9).fit(self.base)
        b = LSHIndex(metric="euclidean", n_tables=10, n_hashes=3, bucket_width=3, seed=9).fit(self.base)
        ids, d = a.query(self.queries, k=10)
        ids2, _ = b.query(self.queries, k=10)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_allclose(d[:, 0], 0, atol=1e-6)

    def test_errors(self):
        with self.assertRaises(ValueError):
            LSHIndex(n_hashes=64)
        model = LSHIndex()
        with self.assertRaises(RuntimeError):
            model.query(self.queries)
        model.fit(self.base)
        with self.assertRaises(ValueError):
            model.query(self.queries, k=400)

if __name__ == "__main__":
    unittest.main()
