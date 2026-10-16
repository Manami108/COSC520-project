import unittest
import numpy as np

from algorithms.knn import BruteForceKNN
from evaluate import recall_at_k


class TestExactKNN(unittest.TestCase):
    def test_euclidean_matches_independent_reference(self):
        rng = np.random.default_rng(123)
        base = rng.standard_normal((103, 8)).astype(np.float32)
        queries = rng.standard_normal((4, 8)).astype(np.float32)
        model = BruteForceKNN(chunk_size=17).fit(base)
        found, _ = model.query(queries, k=7)
        expected = np.argsort(np.sum((base[None] - queries[:, None]) ** 2, axis=2), axis=1)[:, :7]
        np.testing.assert_array_equal(found, expected)
        self.assertEqual(recall_at_k(expected, found, k=7), 1.0)

    def test_cosine_matches_reference(self):
        rng = np.random.default_rng(1234)
        base = rng.standard_normal((65, 7)).astype(np.float32)
        query = rng.standard_normal((3, 7)).astype(np.float32)
        found, _ = BruteForceKNN(metric="cosine", chunk_size=11).fit(base).query(query, k=5)
        similarities = (query @ base.T) / (
            np.linalg.norm(query, axis=1)[:, None] * np.linalg.norm(base, axis=1)[None, :])
        expected = np.argsort(-similarities, axis=1)[:, :5]
        np.testing.assert_array_equal(found, expected)

    def test_reject_invalid_k(self):
        model = BruteForceKNN().fit(np.eye(3, dtype=np.float32))
        with self.assertRaises(ValueError):
            model.query(np.ones(3), k=4)

if __name__ == "__main__":
    unittest.main()
