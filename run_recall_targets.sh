#!/usr/bin/env bash
set -euo pipefail

# From the repository root: bash run_recall_targets.sh
# Tuning occurs ONCE per dataset, for all targets. Final tests use 3 seeds.
THREADS="${ANN_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

for dataset in sift-128-euclidean glove-100-angular; do
  echo "=== ${dataset}: tune once, evaluate recall 80/90/95% ==="
  python -m experiments.benchmark \
    --algorithm all \
    --dataset "data/${dataset}.hdf5" \
    --sizes 1000 5000 10000 25000 \
    --tune-size 10000 \
    --tune-queries 100 \
    --test-queries 500 \
    --method random \
    --trials 12 \
    --target-recalls 0.80 0.90 0.95 \
    --repeats 3 \
    --seed 42 \
    --seeds 123 456 789 \
    --output-dir results/recall_targets
done

for label in r800 r900 r950; do
  python -m experiments.plots --input-dir "results/recall_targets/${label}"
done

echo "Finished. Results: results/recall_targets/"
