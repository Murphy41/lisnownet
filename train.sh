#!/bin/bash
set -e  # stop if any command fails

# Optional: create a log directory
mkdir -p logs

echo "=== Running WADS training ==="
python train.py --dataset wads --batch_size=16 --tag wads_test | tee logs/wads.log

echo "=== Running Livox HP training ==="
python train.py --dataset livox --split_mode hp --batch_size=16 --tag hp_test | tee logs/hp.log

echo "=== Running Livox H training ==="
python train.py --dataset livox --split_mode h --batch_size=16 --tag h_test | tee logs/h.log

echo "=== Running Livox P training ==="
python train.py --dataset livox --split_mode p --batch_size=16 --tag p_test | tee logs/p.log

echo "=== Running Livox MIX training ==="
python train.py --dataset livox --split_mode mix --batch_size=16 --tag mix_test | tee logs/mix.log

echo "=== All trainings completed successfully! ==="
