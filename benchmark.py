#!/usr/bin/env python3
import argparse
import os
from glob import glob
from collections import defaultdict
import numpy as np

# ---------- metrics helpers ----------
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def metrics_from_counts(tp, fp, fn, total):
    # total = tp + fp + fn + tn  (we infer tn)
    P = safe_div(tp, tp + fp)
    R = safe_div(tp, tp + fn)
    IOU = safe_div(tp, tp + fp + fn)
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
    tn = total - tp - fp - fn
    ACC = safe_div(tp + tn, total)
    return {
        "precision": P,
        "recall": R,
        "iou": IOU,
        "f1": F1,
        "accuracy": ACC,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": total,
    }

# ---------- WADS helpers ----------
def iter_wads_pred_files(run_dir):
    # logs/<tag>/<seq>/pred/*.pred.label
    for seq_dir in sorted(glob(os.path.join(run_dir, "*"))):
        pred_dir = os.path.join(seq_dir, "pred")
        if not os.path.isdir(pred_dir):
            continue
        for p in sorted(glob(os.path.join(pred_dir, "*.pred.label"))):
            yield p  # .../<tag>/<seq>/pred/000123.pred.label

def wads_paths_from_pred(pred_path, wads_root):
    # logs/<tag>/<seq>/pred/000123.pred.label -> ./data/wads/<seq>/velodyne/000123.bin
    seq = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))   # <seq>
    base = os.path.basename(pred_path).replace(".pred.label", "")        # 000123
    pts_bin = os.path.join(wads_root, f"{seq}/velodyne/{base}.bin")
    gt_lbl  = os.path.join(wads_root, f"{seq}/labels/{base}.label")
    return pts_bin, gt_lbl, seq

def load_wads_gt_aligned(pts_bin, gt_lbl):
    # replicate WADS.read_files: dedup points, subset labels with idx_unique
    pts = np.fromfile(pts_bin, dtype=np.float32).reshape(-1, 4)
    _, idx_unique = np.unique(pts, axis=0, return_index=True)
    lbl = np.fromfile(gt_lbl, dtype=np.int32)
    lbl = lbl[idx_unique]
    return lbl

# ---------- Livox helpers ----------
def iter_livox_pred_files(run_dir):
    # logs/<tag>/{HP,HD,LP,LD}/pred/*.pred.u8
    for subset in ("HP", "HD", "LP", "LD"):
        pred_dir = os.path.join(run_dir, subset, "pred")
        if not os.path.isdir(pred_dir):
            continue
        for p in sorted(glob(os.path.join(pred_dir, "*.pred.u8"))):
            yield p, subset

def livox_paths_from_pred(pred_path, livox_root):
    # logs/<tag>/<SUBSET>/pred/cloud17.pred.u8 -> ./data/livox/<SUBSET>/cloud17.pcd
    subset = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))
    base = os.path.splitext(os.path.basename(pred_path))[0].replace(".pred", "")
    pcd = os.path.join(livox_root, subset, base + ".pcd")
    rel = os.path.join(subset, base + ".pcd")  # for 'mix' set membership
    return pcd, subset, rel

def read_livox_pcd_labels(pcd_path):
    # fast ASCII reader for your format: x y z intensity dust
    with open(pcd_path, "r") as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DATA"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"DATA section not found in {pcd_path}")
    dust = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            dust.append(int(parts[4]))
        except Exception:
            continue
    return np.array(dust, dtype=np.int32)

def read_livox_mix_val_list(livox_root):
    """
    Read data/livox/splits/val_mix.txt and normalise lines to 'HD/cloud10'
    (no leading ./, no 'data/livox/', no .pcd).
    """
    val_file = os.path.join(livox_root, "splits", "val_mix.txt")
    if not os.path.isfile(val_file):
        return set()

    mix_set = set()
    with open(val_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # normalise slashes
            line = line.replace("\\", "/")

            # remove leading './'
            if line.startswith("./"):
                line = line[2:]

            # remove leading 'data/livox/'
            if line.startswith("data/livox/"):
                line = line[len("data/livox/"):]

            # drop extension (.pcd)
            line = os.path.splitext(line)[0]

            # now should look like 'HD/cloud10'
            mix_set.add(os.path.normpath(line))
    return mix_set


# ---------- per-dataset runners ----------
def run_wads_eval(args, run_dir):
    snow_id = args.snow_id if args.snow_id is not None else 110

    tp_all = fp_all = fn_all = tot_all = 0

    pred_files = list(iter_wads_pred_files(run_dir))
    if not pred_files:
        print(f"[WARN] no WADS preds under {run_dir}/<seq>/pred/*.pred.label")
        return

    for i, pred_path in enumerate(pred_files, 1):
        pts_bin, gt_lbl, _seq = wads_paths_from_pred(pred_path, args.wads_root)
        if not (os.path.isfile(pts_bin) and os.path.isfile(gt_lbl)):
            print(f"[WARN] missing raw/gt for {pred_path}; skipping")
            continue

        gt = load_wads_gt_aligned(pts_bin, gt_lbl)      # (N,)
        pr = np.fromfile(pred_path, dtype=np.uint32)    # (N,)
        if pr.size != gt.size:
            print(f"[WARN] size mismatch: pred={pr.size}, gt={gt.size} for {pred_path}; skipping")
            continue

        gt_pos = (gt == snow_id)
        pr_pos = (pr != 0)

        tp = int(np.sum(pr_pos & gt_pos))
        fp = int(np.sum(pr_pos & ~gt_pos))
        fn = int(np.sum(~pr_pos & gt_pos))
        tot = int(gt.size)

        tp_all += tp; fp_all += fp; fn_all += fn; tot_all += tot

        if (i % 200 == 0) or (i == len(pred_files)):
            m = metrics_from_counts(tp_all, fp_all, fn_all, tot_all)
            print(f"[{i:5d}/{len(pred_files):5d}] P={m['precision']:.4f} R={m['recall']:.4f} IOU={m['iou']:.4f} (pts={tot_all})", end="\r")

    m = metrics_from_counts(tp_all, fp_all, fn_all, tot_all)
    print("\n\n=== WADS overall ===")
    for k, v in m.items():
        print(f"{k}: {v}")

    out_path = os.path.join(run_dir, "results_wads.txt")
    with open(out_path, "w") as f:
        f.write("=== WADS overall ===\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")
    print(f"[INFO] WADS results written to {out_path}")


def run_livox_eval(args, run_dir):
    snow_id = args.snow_id if args.snow_id is not None else 1
    tp_all = fp_all = fn_all = tot_all = 0
    per_subset = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tot=0))
    mix_set = read_livox_mix_val_list(args.livox_root)

    pred_entries = list(iter_livox_pred_files(run_dir))
    if not pred_entries:
        print(f"[ERROR] no Livox preds under {run_dir}/(HP|HD|LP|LD)/pred/*.pred.u8")
        return

    tp_mix = fp_mix = fn_mix = tot_mix = 0

    for i, (pred_path, subset) in enumerate(pred_entries, 1):
        pcd_path, subset2, rel_pcd = livox_paths_from_pred(pred_path, args.livox_root)
        if subset != subset2:
            print(f"[WARN] subset mismatch for {pred_path}")
        if not os.path.isfile(pcd_path):
            print(f"[WARN] missing PCD {pcd_path}; skipping")
            continue

        gt = read_livox_pcd_labels(pcd_path)        # (N,)
        pr = np.fromfile(pred_path, dtype=np.uint8) # (N,)
        if pr.size != gt.size:
            print(f"[WARN] size mismatch: pred={pr.size}, gt={gt.size} for {pred_path}; skipping")
            continue

        # Livox GT: usually 1 for snow/dust
        gt_pos = (gt == snow_id)
        pr_pos = (pr != 0)

        tp = int(np.sum(pr_pos & gt_pos))
        fp = int(np.sum(pr_pos & ~gt_pos))
        fn = int(np.sum(~pr_pos & gt_pos))
        tot = int(gt.size)

        tp_all += tp; fp_all += fp; fn_all += fn; tot_all += tot

        d = per_subset[subset]
        d["tp"] += tp; d["fp"] += fp; d["fn"] += fn; d["tot"] += tot

        # count for mix val
        rel_key = os.path.splitext(os.path.normpath(rel_pcd))[0]
        if mix_set and rel_key in mix_set:
            tp_mix += tp
            fp_mix += fp
            fn_mix += fn
            tot_mix += tot

        if (i % 200 == 0) or (i == len(pred_entries)):
            m = metrics_from_counts(tp_all, fp_all, fn_all, tot_all)
            print(f"[{i:5d}/{len(pred_entries):5d}] P={m['precision']:.4f} R={m['recall']:.4f} IOU={m['iou']:.4f} (pts={tot_all})", end="\r")

    # build named groups
    groups = {}
    # 1-4: individual
    for key in ("HP", "HD", "LP", "LD"):
        if key in per_subset:
            d = per_subset[key]
            groups[key] = (d["tp"], d["fp"], d["fn"], d["tot"])

    # LP+LD
    if "LP" in per_subset or "LD" in per_subset:
        tp = per_subset.get("LP", {}).get("tp", 0) + per_subset.get("LD", {}).get("tp", 0)
        fp = per_subset.get("LP", {}).get("fp", 0) + per_subset.get("LD", {}).get("fp", 0)
        fn = per_subset.get("LP", {}).get("fn", 0) + per_subset.get("LD", {}).get("fn", 0)
        tot = per_subset.get("LP", {}).get("tot", 0) + per_subset.get("LD", {}).get("tot", 0)
        groups["LP+LD"] = (tp, fp, fn, tot)

    # HD+LD
    if "HD" in per_subset or "LD" in per_subset:
        tp = per_subset.get("HD", {}).get("tp", 0) + per_subset.get("LD", {}).get("tp", 0)
        fp = per_subset.get("HD", {}).get("fp", 0) + per_subset.get("LD", {}).get("fp", 0)
        fn = per_subset.get("HD", {}).get("fn", 0) + per_subset.get("LD", {}).get("fn", 0)
        tot = per_subset.get("HD", {}).get("tot", 0) + per_subset.get("LD", {}).get("tot", 0)
        groups["HD+LD"] = (tp, fp, fn, tot)

    # mix val
    if mix_set:
        groups["mix_val"] = (tp_mix, fp_mix, fn_mix, tot_mix)

    # write everything to results_livox.txt
    out_path = os.path.join(run_dir, "results_livox.txt")
    with open(out_path, "w") as f:
        # overall
        overall = metrics_from_counts(tp_all, fp_all, fn_all, tot_all)
        f.write("=== Livox overall ===\n")
        for k, v in overall.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        # each named group
        for name, (tp, fp, fn, tot) in groups.items():
            m = metrics_from_counts(tp, fp, fn, tot)
            f.write(f"=== {name} ===\n")
            for k, v in m.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\n[INFO] Livox results written to {out_path}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="./logs", help="Root logs dir")
    ap.add_argument("--tag", required=True, help="Run tag under log_dir")
    ap.add_argument(
        "--dataset",
        choices=["wads", "livox", "both"],
        default="both",
        help="Dataset(s) to evaluate: wads, livox, or both (default: both)",
    )
    ap.add_argument("--wads_root", default="./data/wads")
    ap.add_argument("--livox_root", default="./data/livox")
    ap.add_argument(
        "--snow_id",
        type=int,
        default=None,
        help="Positive class id for GT (default: 110 for WADS, 1 for Livox)",
    )
    args = ap.parse_args()

    run_dir = os.path.join(args.log_dir, args.tag)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[ERROR] run dir not found: {run_dir}")

    if args.dataset in ("wads", "both"):
        print("\n\n========== Evaluating WADS ==========")
        run_wads_eval(args, run_dir)

    if args.dataset in ("livox", "both"):
        print("\n\n========== Evaluating Livox ==========")
        run_livox_eval(args, run_dir)


if __name__ == "__main__":
    main()
