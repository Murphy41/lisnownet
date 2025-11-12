#!/bin/bash
set -e  # stop if any eval fails

# List of all experiment tags
TAGS=("wads_baseline" "livox_hp" "livox_h" "livox_p" "livox_mix")

# Batch size for evaluation
BATCH=16

# Optional log folder
mkdir -p logs_eval

for TAG in "${TAGS[@]}"; do
    echo "=== Evaluating tag: $TAG on WADS dataset ==="
    python eval.py --dataset wads --batch_size=$BATCH --tag $TAG | tee logs_eval/${TAG}_wads.log

    echo "=== Evaluating tag: $TAG on LIVOX dataset ==="
    python eval.py --dataset livox --batch_size=$BATCH --tag $TAG | tee logs_eval/${TAG}_livox.log

    echo "=== Finished evaluations for $TAG ==="
done

echo "✅ All evaluations completed successfully!"
