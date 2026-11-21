import unittest
import numpy as np
from algorithms.vamana import VamanaIndex


class TestVamana(unittest.TestCase):
    def test_both_metrics(self):
        rng = np.random.default_rng(7)
        base = rng.standard_normal((100, 12)).astype('float32')
        for metric in ('euclidean', 'cosine'):
            model = VamanaIndex(metric=metric, R=8, L=20, search_L=30, passes=2).fit(base)
            ids, distances = model.query(base[:5], k=5)
            self.assertEqual(ids.shape, (5, 5))
            self.assertEqual(distances.shape, (5, 5))
            self.assertTrue(np.all(np.isfinite(distances)))
            self.assertTrue(all(len(set(row)) == 5 for row in ids))
            self.assertTrue(all(len(neighbors) <= 8 for neighbors in model.graph))
            self.assertTrue(all(int(i) in ids[j] for j, i in enumerate(range(5))))

    def test_deterministic(self):
        x = np.random.default_rng(1).normal(size=(50, 5)).astype('float32')
        a = VamanaIndex(R=6, L=12, seed=12).fit(x)
        b = VamanaIndex(R=6, L=12, seed=12).fit(x)
        self.assertEqual(a.graph, b.graph)
        np.testing.assert_array_equal(a.query(x[:3], 3)[0], b.query(x[:3], 3)[0])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            VamanaIndex(R=10, L=5)


if __name__ == '__main__':
    unittest.main()
