#!/usr/bin/env bash
set -euo pipefail

# From the repository root:
#   bash run_recall_targets.sh            # Experiment B (structured)
#   bash run_recall_targets.sh original   # Experiment A (original random search)
THREADS="${ANN_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
# Also limit Apple's Accelerate backend, where applicable.
export VECLIB_MAXIMUM_THREADS="$THREADS"

case "${1:-structured}" in
  structured)
    METHOD=structured
    TRIALS=24
    QUERY_OFFSET=600
    OUTPUT="results/structured_tuning"
    ;;
  original)
    METHOD=random
    TRIALS=12
    QUERY_OFFSET=0
    # Separate directory to avoid overwriting previously generated results.
    OUTPUT="results/original_reproduction"
    ;;
  *) echo "Usage: bash run_recall_targets.sh [structured|original]" >&2; exit 2 ;;
esac

for dataset in sift-128-euclidean glove-100-angular; do
  echo "=== ${dataset}: ${METHOD} tuning, ${TRIALS} configurations, query offset ${QUERY_OFFSET} ==="
  python -m experiments.benchmark \
    --algorithm all \
    --dataset "data/${dataset}.hdf5" \
    --sizes 1000 5000 10000 25000 \
    --tune-size 10000 \
    --tune-queries 100 \
    --test-queries 500 \
    --query-offset "$QUERY_OFFSET" \
    --method "$METHOD" \
    --trials "$TRIALS" \
    --target-recalls 0.80 0.90 0.95 \
    --repeats 3 \
    --seed 42 \
    --seeds 123 456 789 \
    --output-dir "$OUTPUT"
  python -m experiments.sensitivity --metadata "$OUTPUT/r900/${dataset}/metadata.json"
done

# Regenerate the same individual and combined plots using the existing plots.py.
for label in r800 r900 r950; do
  python -m experiments.plots --input-dir "$OUTPUT/${label}"
done
python -m experiments.plots --input-dir "$OUTPUT" --report-only

echo "Finished. Results: $OUTPUT/"
