#!/bin/bash
set -e  # stop if any benchmark fails

TAGS=("wads_test" "hp_test" "h_test" "p_test" "mix_test")

for TAG in "${TAGS[@]}"; do
    echo "=== Benchmarking $TAG ==="
    python benchmark.py --tag $TAG --dataset both
    echo "=== Finished $TAG ==="
done

echo "✅ All benchmarks completed successfully!"
