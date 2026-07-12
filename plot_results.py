import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1) config
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOG_ROOT = os.path.join(ROOT_DIR, "logs")

# Requested models:
# - lisnownet on hp
# - lisnownet on wads
# - alior on hp
# - lior on hp
MODELS = [
    {
        "name": "lisnownet_hp",
        "tag": "lisnownet_livox_hp_a2_b0.5_lr0.001_dec0.98_e30",
    },
    {
        "name": "lisnownet_wads",
        "tag": "lisnownet_wads_alpha=5.5",
    },
    {
        "name": "alior_hp",
        "tag": "20260203_085525__livox_vx0.2_r0.1_v10.0005_v20.0001_b256_phi1.2_k500",
    },
    {
        "name": "lior_hp",
        "tag": "lior_best_hp_thr0.01_dirlt_full",
    },
]

# Requested test groups: HP, H, P, Mix, WADS
# Here we map:
# - H -> HD+LD
# - P -> LP+LD
TEST_GROUPS = [
    ("HP", "HP"),
    ("H", "HD+LD"),
    ("P", "LP+LD"),
    ("Mix", "mix_val"),
    ("WADS", "WADS overall"),
]

OUT_PNG = os.path.join(LOG_ROOT, "compare_hp_h_p_mix_wads_f1.png")
OUT_CSV = os.path.join(LOG_ROOT, "compare_hp_h_p_mix_wads_f1.csv")


# --------------------------------------------------
# 2) parser
# --------------------------------------------------
number_re = re.compile(r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def read_metrics_from_file(path):
    if not os.path.isfile(path):
        return {}

    sections = {}
    current = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("===") and line.endswith("==="):
                current = line.strip("=").strip()
                sections[current] = {}
                continue

            if current is None or ":" not in line:
                continue

            k, v = line.split(":", 1)
            m = number_re.match(v.strip())
            if m:
                sections[current][k.strip()] = float(m.group(1))
    return sections


def resolve_livox_path(tag):
    # lisnownet/lior: logs/<tag>/results_livox.txt
    p1 = os.path.join(LOG_ROOT, tag, "results_livox.txt")
    # alior sweep layout: logs/<tag>/livox/results_livox.txt
    p2 = os.path.join(LOG_ROOT, tag, "livox", "results_livox.txt")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return p1


def resolve_wads_path(tag):
    # most runs: logs/<tag>/results_wads.txt
    p1 = os.path.join(LOG_ROOT, tag, "results_wads.txt")
    # optional nested layout
    p2 = os.path.join(LOG_ROOT, tag, "wads", "results_wads.txt")
    if os.path.isfile(p1):
        return p1
    if os.path.isfile(p2):
        return p2
    return p1


# --------------------------------------------------
# 3) collect requested metrics
# --------------------------------------------------
model_metrics = {}
for model in MODELS:
    tag = model["tag"]
    livox_file = resolve_livox_path(tag)
    wads_file = resolve_wads_path(tag)

    merged = {}
    merged.update(read_metrics_from_file(livox_file))
    merged.update(read_metrics_from_file(wads_file))
    model_metrics[model["name"]] = merged


# --------------------------------------------------
# 4) plot F1 comparison (one figure)
# --------------------------------------------------
x = np.arange(len(TEST_GROUPS))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5))

for i, model in enumerate(MODELS):
    name = model["name"]
    values = []
    for _label, section in TEST_GROUPS:
        v = model_metrics.get(name, {}).get(section, {}).get("f1", 0.0)
        values.append(v)
    ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, values, width=width, label=name)

ax.set_xticks(x)
ax.set_xticklabels([label for label, _ in TEST_GROUPS])
ax.set_ylabel("F1")
ax.set_ylim(0, 1.0)
ax.set_title("HP/H/P/Mix/WADS F1 Comparison")
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)


# --------------------------------------------------
# 5) save CSV table
# --------------------------------------------------
with open(OUT_CSV, "w") as f:
    f.write("group," + ",".join([m["name"] for m in MODELS]) + "\n")
    for label, section in TEST_GROUPS:
        vals = []
        for model in MODELS:
            name = model["name"]
            v = model_metrics.get(name, {}).get(section, {}).get("f1", 0.0)
            vals.append(f"{v:.6f}")
        f.write(label + "," + ",".join(vals) + "\n")

print(f"saved_plot: {OUT_PNG}")
print(f"saved_csv: {OUT_CSV}")
