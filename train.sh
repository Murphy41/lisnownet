#!/bin/bash
set -e  # exit on first error

# directory to store console logs
mkdir -p sweep_logs

########################################################################
# 1) (Optional) train WADS once
########################################################################
echo "=== (optional) WADS training ==="
# comment this out if you really don't want to re-train WADS
python train.py \
  --dataset wads \
  --batch_size 16 \
  --tag wads_baseline \
  | tee sweep_logs/wads_baseline.log

########################################################################
# 2) parameter grids for LIVOX
########################################################################

# split modes you want to fine-tune
LIVOX_SPLITS=("hp" "h" "p" "mix")

# outer weight via alpha (how much sparsity vs residual)
ALPHAS=(1.1 2 5)

# inner sparsity mix (DWT vs FFT)
BETAS=(0.3 0.5 0.8)

# learning rates
LRS=(0.0005 0.001)

# decay – we will special-case 0.98 to run 30 epochs
DECAYS=(0.95 0.98)

########################################################################
# 3) sweep
########################################################################
for split in "${LIVOX_SPLITS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for beta in "${BETAS[@]}"; do
      for lr in "${LRS[@]}"; do
        for decay in "${DECAYS[@]}"; do

          # if decay is 0.98, train a bit longer, else 20
          if [[ "$decay" == "0.98" ]]; then
            EPOCHS=30
          else
            EPOCHS=20
          fi

          tag="livox_${split}_a${alpha}_b${beta}_lr${lr}_dec${decay}_e${EPOCHS}"

          echo "=== Training $tag ==="

          python train.py \
            --dataset livox \
            --split_mode "$split" \
            --batch_size 16 \
            --alpha "$alpha" \
            --beta "$beta" \
            --lr "$lr" \
            --lr_decay "$decay" \
            --num_epochs "$EPOCHS" \
            --tag "$tag" \
            | tee "sweep_logs/${tag}.log"

        done
      done
    done
  done
done

echo "=== All sweeps done. ==="
