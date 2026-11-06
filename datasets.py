"""Load consistent subsets of ANN-Benchmarks' SIFT1M and GloVe-100 HDF5 files."""
from pathlib import Path
import h5py
import numpy as np

DATASET_URLS = {
    "sift": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove": "https://ann-benchmarks.com/glove-100-angular.hdf5",
}


def infer_metric(path: str | Path) -> str:
    """Infer metric from ANN-Benchmarks HDF5 distance metadata (or filename)."""
    with h5py.File(path, "r") as f:
        attr = f.attrs.get("distance", "")
    if isinstance(attr, bytes):
        attr = attr.decode()
    attr = str(attr).lower()
    if attr in {"angular", "cosine"}:
        return "cosine"
    if attr in {"euclidean", "l2"}:
        return "euclidean"
    name = str(path).lower()
    if "glove" in name or "angular" in name or "cosine" in name:
        return "cosine"
    if "sift" in name or "euclidean" in name:
        return "euclidean"
    raise ValueError("Cannot infer metric. Pass --metric euclidean or --metric cosine")


def load_hdf5(path: str | Path, n_base: int = 10_000, n_queries: int = 100):
    with h5py.File(path, "r") as f:
        total_base = len(f["train"])
        if not 1 <= n_base <= total_base:
            raise ValueError(f"n_base must be between 1 and {total_base}")
        if not 1 <= n_queries <= len(f["test"]):
            raise ValueError(f"n_queries must be between 1 and {len(f['test'])}")
        base = np.asarray(f["train"][:n_base], dtype=np.float32)
        test = np.asarray(f["test"][:n_queries], dtype=np.float32)
        # Official neighbor IDs are only valid when using the *full* base set.
        truth = (np.asarray(f["neighbors"][:n_queries], dtype=np.int64)
                 if n_base == total_base and "neighbors" in f else None)
    return base, test, truth
