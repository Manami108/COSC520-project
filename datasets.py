import argparse
from pathlib import Path
from urllib.request import urlretrieve
import h5py
import numpy as np

DATASET_URLS = {"sift": "https://ann-benchmarks.com/sift-128-euclidean.hdf5", "glove": "https://ann-benchmarks.com/glove-100-angular.hdf5",}

def download_dataset(dataset="both", output_dir="data"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset not in {"sift", "glove", "both"}:
        raise ValueError("Dataset must be 'sift', 'glove', or 'both'")
    selected = DATASET_URLS.keys() if dataset == "both" else [dataset]

    for name in selected:
        url = DATASET_URLS[name]
        dest = output_dir / url.rsplit("/", 1)[-1]
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {name}: {url}", flush=True)

        try:
            urlretrieve(url, dest)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        print(f"Downloaded: {dest}")

def infer_metric(path: str | Path) -> str:
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
    raise ValueError("Cannot infer distance metric")

def load_hdf5(path: str | Path, n_base=10_000, n_queries=100):
    with h5py.File(path, "r") as f:
        total_base = len(f["train"])
        if not 1 <= n_base <= total_base:
            raise ValueError(f"n_base must be between 1 and {total_base}")
        if not 1 <= n_queries <= len(f["test"]):
            raise ValueError(f"n_queries must be between 1 and {len(f['test'])}")
        base = np.asarray(f["train"][:n_base], dtype=np.float32)
        queries = np.asarray(f["test"][:n_queries], dtype=np.float32)
        # Official ground truth is valid only for the full dataset.
        truth = (np.asarray(f["neighbors"][:n_queries], dtype=np.int64)
            if n_base == total_base and "neighbors" in f else None)
    return base, queries, truth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sift", "glove", "both"], default="both",)
    parser.add_argument("--output-dir", type=Path, default=Path("data"),)
    args = parser.parse_args()
    download_dataset(args.dataset, args.output_dir)

if __name__ == "__main__":
    main()
