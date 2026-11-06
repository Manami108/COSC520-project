import unittest
import numpy as np
from algorithms.hnsw import HNSWIndex
from algorithms.knn import BruteForceKNN
from evaluate import recall_at_k

class TestHNSW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(96)
        cls.base = rng.normal(size=(230, 10)).astype(np.float32)
        cls.query_vectors = cls.base[::23] + 0.02

    def test_hierarchy_and_degree_cap(self):
        m = HNSWIndex(M=6, ef_construction=48, ef_search=70, seed=12).fit(self.base)
        self.assertEqual(len(m.levels), len(self.base))
        self.assertIsNotNone(m.entry_point)
        self.assertEqual(m.max_level, max(m.levels))
        for layer_id, layer in enumerate(m.layers):
            degree_cap = 12 if layer_id == 0 else 6
            for v, nbrs in layer.items():
                self.assertLessEqual(len(nbrs), degree_cap)
                self.assertNotIn(v, nbrs)
                self.assertEqual(len(nbrs), len(set(nbrs)))
                self.assertTrue(all(other in layer for other in nbrs))

    def test_accuracy_and_shapes_l2(self):
        m = HNSWIndex(M=8, ef_construction=65, ef_search=90, seed=5).fit(self.base)
        ids, d = m.query(self.query_vectors, k=10)
        exact, _ = BruteForceKNN().fit(self.base).query(self.query_vectors, k=10)
        self.assertEqual(ids.shape, (10, 10))
        self.assertTrue(np.all(np.diff(d, axis=1) >= 0))
        self.assertGreater(recall_at_k(exact, ids, k=10), 0.80)
        self.assertGreater(m.index_bytes, self.base.nbytes)

    def test_cosine_and_reproducibility(self):
        a = HNSWIndex(metric="cosine", M=6, ef_construction=40, ef_search=60, seed=8).fit(self.base)
        b = HNSWIndex(metric="cosine", M=6, ef_construction=40, ef_search=60, seed=8).fit(self.base)
        ids, d = a.query(self.query_vectors[:2], k=6)
        ids2, d2 = b.query(self.query_vectors[:2], k=6)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_allclose(d, d2)
        self.assertTrue(np.all((d >= 0) & (d <= 2 + 1e-5)))

    def test_invalid_args_and_unfitted(self):
        with self.assertRaises(ValueError):
            HNSWIndex(M=1)
        with self.assertRaises(RuntimeError):
            HNSWIndex().query(self.query_vectors)
        m = HNSWIndex(M=4, ef_construction=10, ef_search=20).fit(self.base[:15])
        with self.assertRaises(ValueError):
            m.query(self.query_vectors, k=16)

if __name__ == "__main__":
    unittest.main()
