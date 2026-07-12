"""
===============================================================================
Model Ranking Utility for LiSnowNet Experiments
===============================================================================

Author: Murphy
Last Updated: 2025-11-12
Repository: https://github.com/Murphy41/lisnownet

Purpose
-------
This script automatically scans all experiment logs under `./logs/`, extracts
the recorded loss curves from TensorBoard event files, and ranks models based
on their best (minimum) total validation loss. It groups the results by data
split (e.g., HP, H, P, MIX) inferred from the run folder name.

Each run directory is expected to contain:
    - `config.json`         : experiment configuration (hyperparameters)
    - `events.out.tfevents*`: TensorBoard log file with loss curves
    - `*.pth`               : trained model checkpoints (not directly used here)

Supported loss components (summed to form total loss):
    - loss/val/DWT
    - loss/val/FFT
    - loss/val/Residual
If validation losses are missing, training losses are used instead.

Output
------
The script prints a concise leaderboard of the best-performing run per split:
    SPLIT  | best=<min loss> | last=<final loss> | from <val/train> | run='<folder>'
             alpha=<α>, beta=<β>, lr=<learning rate>, decay=<lr_decay>, epochs=<N>

Example:
    === Best models per split (using val->train, summed) ===
    HP     | best=0.243501 | last=0.250321 | from val | run='livox_hp_a1.1_b0.3_lr0.0005_dec0.95_e20'
             alpha=1.1, beta=0.3, lr=0.0005, decay=0.95, epochs=20

How It Works
------------
1. Iterates through all subdirectories in `LOG_ROOT` (default: `./logs`).
2. For each run, loads the TensorBoard event file via the EventAccumulator API.
3. Aggregates the three loss components (DWT, FFT, Residual) to get total loss.
4. Tracks the minimum loss and last loss for that run.
5. Determines which dataset split the run belongs to (hp/h/p/mix) based on
   substring matching in the folder name.
6. Loads configuration details from `config.json`.
7. Keeps the best run (lowest loss) per split and prints the summary.

Customization
-------------
- Change `LOG_ROOT` to point to a different experiment directory.
- Update the `VAL_TAGS` / `TRAIN_TAGS` lists if your TensorBoard tag names differ.
- Modify `infer_split()` if your folder naming pattern changes (e.g., different delimiters).
- You can adapt this code to plot loss curves or export results as CSV/JSON.

Dependencies
------------
- Python ≥ 3.8
- TensorBoard (for `event_accumulator`)
    pip install tensorboard

Run Command
-----------
    python rank.py

===============================================================================
"""



import os
import json
from tensorboard.backend.event_processing import event_accumulator

LOG_ROOT = "./logs"

# we will sum these if available
VAL_TAGS = [
    "loss/val/DWT",
    "loss/val/FFT",
    "loss/val/Residual",
]
TRAIN_TAGS = [
    "loss/train/DWT",
    "loss/train/FFT",
    "loss/train/Residual",
]

def infer_split(run_name: str) -> str:
    name = run_name.lower()
    # check more specific first
    if "_hp_" in name or name.endswith("_hp"):
        return "hp"
    if "_mix_" in name or name.endswith("_mix"):
        return "mix"
    if "_p_" in name or name.endswith("_p"):
        return "p"
    if "_h_" in name or name.endswith("_h"):
        return "h"
    # fallback
    return "unknown"


best_per_split = {}

for run_name in os.listdir(LOG_ROOT):
    run_dir = os.path.join(LOG_ROOT, run_name)
    if not os.path.isdir(run_dir):
        continue

    # find tfevents
    event_files = [f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        continue
    event_path = os.path.join(run_dir, event_files[0])

    try:
        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()
    except Exception as e:
        print(f"Failed to read {run_name}: {e}")
        continue

    scalar_tags = ea.Tags().get("scalars", [])

    # prefer val
    if all(tag in scalar_tags for tag in VAL_TAGS):
        used_tags = VAL_TAGS
        source = "val"
    elif all(tag in scalar_tags for tag in TRAIN_TAGS):
        used_tags = TRAIN_TAGS
        source = "train"
    else:
        print(f"{run_name}: no complete loss set, has {scalar_tags}")
        continue

    # get summed loss over steps
    components = [ea.Scalars(tag) for tag in used_tags]
    total_losses = []
    for i in range(len(components[0])):
        total = 0.0
        for comp in components:
            total += comp[i].value
        total_losses.append(total)

    best_loss = min(total_losses)
    last_loss = total_losses[-1]

    split = infer_split(run_name)

    # load config if exists
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    current_best = best_per_split.get(split)
    if current_best is None or best_loss < current_best[0]:
        best_per_split[split] = (best_loss, last_loss, run_name, cfg, source)

print("\n=== Best models per split (using val->train, summed) ===")
for split, (best_loss, last_loss, run_name, cfg, source) in sorted(best_per_split.items()):
    print(f"{split.upper():6s} | best={best_loss:.6f} | last={last_loss:.6f} | from {source} | run='{run_name}'")
    print(f"         alpha={cfg.get('alpha')}, beta={cfg.get('beta')}, lr={cfg.get('lr')}, decay={cfg.get('lr_decay')}, epochs={cfg.get('num_epochs')}")
