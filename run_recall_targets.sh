#!/usr/bin/env bash
set -euo pipefail

# Execute from the root of COSC520-project using: bash run_recall_targets.sh
# Requires the updated experiments.benchmark supporting the flags below.

for dataset in sift-128-euclidean glove-100-angular; do
  for target in 0.80 0.90 0.95; do
    label="r${target/./}"
    out="results/recall_targets/${label}"
    echo "=== ${dataset} | target recall ${target} ==="
    python -m experiments.benchmark \
      --algorithm all \
      --dataset "data/${dataset}.hdf5" \
      --sizes 1000 5000 10000 25000 \
      --tune-size 10000 \
      --tune-queries 100 \
      --test-queries 500 \
      --method random \
      --trials 30 \
      --target-recall "${target}" \
      --repeats 3 \
      --seed 42 \
      --output-dir "${out}"
  done
done

# Your plotting module should support --input-dir as in the previous package.
for target in 0.80 0.90 0.95; do
  label="r${target/./}"
  python -m experiments.plots --input-dir "results/recall_targets/${label}"
done

echo "Finished. Results: results/recall_targets/"
