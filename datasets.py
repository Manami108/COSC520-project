"""Load ANN-Benchmarks HDF5 datasets with separate database and test queries."""
from pathlib import Path
import numpy as np
import h5py


def load_hdf5(path: str | Path, n_base: int = 10_000, n_queries: int = 100):
    """Return (database, test_queries, official_ground_truth_or_None).

    Official ground truth applies ONLY to the full database, not to its subsets.
    Ground truth for subsets should be regenerated using the exact baseline.
    """
    with h5py.File(path, "r") as f:
        total_base = len(f["train"])
        if not 1 <= n_base <= total_base:
            raise ValueError(f"n_base must be between 1 and {total_base}")
        if not 1 <= n_queries <= len(f["test"]):
            raise ValueError(f"n_queries must be between 1 and {len(f['test'])}")
        base = np.asarray(f["train"][:n_base], dtype=np.float32)
        test = np.asarray(f["test"][:n_queries], dtype=np.float32)
        truth = (np.asarray(f["neighbors"][:n_queries], dtype=np.int64)
                 if n_base == total_base and "neighbors" in f else None)
    return base, test, truth
