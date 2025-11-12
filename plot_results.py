# plot_results.py
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1) config
# --------------------------------------------------
LOG_ROOT = "./logs"

# put your 5 runs here
TAGS = [
    "wads_baseline",
    "livox_hp",
    "livox_h",
    "livox_p",
    "livox_mix",
]


# the 9 test groups we want to show (order matters for plotting)
TEST_GROUPS = [
    "wads",
    "livox",
    "HP",
    "HD",
    "LP",
    "LD",
    "LP+LD",
    "HD+LD",
    "mix_val",
]

# --------------------------------------------------
# 2) tiny parser for the results_*.txt you wrote in benchmark.py
# --------------------------------------------------
number_re = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")

def read_metrics_from_file(path):
    """
    returns {section_name: {metric: value, ...}, ...}
    e.g. {
        'Livox overall': {'precision': 0.9, ...},
        'HP': {...},
        ...
    }
    """
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
                # section header: "=== Livox overall ==="
                name = line.strip("=").strip()
                current = name
                sections[current] = {}
            else:
                # metric line: "precision: 0.1234"
                if current is None:
                    continue
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                m = number_re.match(v)
                if m:
                    sections[current][k] = float(m.group(1))
    return sections

# --------------------------------------------------
# 3) collect metrics for each tag
# --------------------------------------------------
# metrics[tag][test][metric_name] = value
metrics = {}

for tag in TAGS:
    tag_dir = os.path.join(LOG_ROOT, tag)
    wads_file = os.path.join(tag_dir, "results_wads.txt")
    livox_file = os.path.join(tag_dir, "results_livox.txt")

    tag_dict = {}

    # WADS overall
    wads_sections = read_metrics_from_file(wads_file)
    if "WADS overall" in wads_sections:
        tag_dict["wads"] = wads_sections["WADS overall"]

    # Livox + subsets
    livox_sections = read_metrics_from_file(livox_file)
    # normalize names
    if "Livox overall" in livox_sections:
        tag_dict["livox"] = livox_sections["Livox overall"]

    for name in ["HP", "HD", "LP", "LD", "LP+LD", "HD+LD", "mix_val"]:
        if name in livox_sections:
            tag_dict[name] = livox_sections[name]

    metrics[tag] = tag_dict

# --------------------------------------------------
# 4) helper to make grouped bar chart
# --------------------------------------------------
def plot_by_tag(metric_name):
    """
    x-axis: tags
    each tag: 9 bars (tests)
    """
    tags = TAGS
    tests = TEST_GROUPS
    n_tags = len(tags)
    n_tests = len(tests)

    x = np.arange(n_tags)  # tag positions
    width = 0.075  # bar width – small because 9 bars per group

    fig, ax = plt.subplots(figsize=(12, 5))

    for j, test in enumerate(tests):
        values = []
        for tag in tags:
            v = metrics.get(tag, {}).get(test, {}).get(metric_name, 0.0)
            values.append(v)
        ax.bar(x + (j - n_tests/2)*width + width/2, values, width, label=test)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} by tag (9 tests per tag)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(0, 1.05)  # because these are ratios
    fig.tight_layout()


def plot_by_test(metric_name):
    """
    x-axis: tests
    each test: 5 bars (tags)
    """
    tests = TEST_GROUPS
    tags = TAGS
    n_tests = len(tests)
    n_tags = len(tags)

    x = np.arange(n_tests)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, tag in enumerate(tags):
        values = []
        for test in tests:
            v = metrics.get(tag, {}).get(test, {}).get(metric_name, 0.0)
            values.append(v)
        ax.bar(x + (i - n_tags/2)*width + width/2, values, width, label=tag)

    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=20)
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} by test (5 tags per test)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

# --------------------------------------------------
# 5) actually make the 4× plots (acc, prec, recall, f1)
# --------------------------------------------------
for m in ["accuracy", "precision", "recall", "f1"]:
    plot_by_tag(m)
    plot_by_test(m)

plt.show()
