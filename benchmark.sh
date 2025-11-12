#!/bin/bash
set -e  # stop if any benchmark fails

TAGS=("wads_baseline" "livox_hp" "livox_h" "livox_p" "livox_mix")

for TAG in "${TAGS[@]}"; do
    echo "=== Benchmarking $TAG ==="
    python benchmark.py --tag $TAG --dataset both
    echo "=== Finished $TAG ==="
done

echo "✅ All benchmarks completed successfully!"
