#!/usr/bin/env python3
import argparse
import os
from glob import glob
from collections import defaultdict
import numpy as np

# ---------- metrics helpers ----------
def safe_div(a, b): return float(a) / float(b) if b else 0.0
def reduce_counts(tp, fp, fn):
    P = safe_div(tp, tp + fp)
    R = safe_div(tp, tp + fn)
    IOU = safe_div(tp, tp + fp + fn)
    return P, R, IOU, P * R * IOU

# ---------- WADS helpers ----------
def iter_wads_pred_files(run_dir):
    # logs/<tag>/<seq>/pred/*.pred.label
    for seq_dir in sorted(glob(os.path.join(run_dir, "*"))):
        pred_dir = os.path.join(seq_dir, "pred")
        if not os.path.isdir(pred_dir): continue
        for p in sorted(glob(os.path.join(pred_dir, "*.pred.label"))):
            yield p  # .../<tag>/<seq>/pred/000123.pred.label

def wads_paths_from_pred(pred_path, wads_root):
    seq = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))   # <seq>
    base = os.path.basename(pred_path).replace(".pred.label", "")        # 000123
    pts_bin = os.path.join(wads_root, f"{seq}/velodyne/{base}.bin")
    gt_lbl  = os.path.join(wads_root, f"{seq}/labels/{base}.label")
    return pts_bin, gt_lbl, seq

def load_wads_gt_aligned(pts_bin, gt_lbl):
    # replicate WADS.read_files: dedup points, subset labels with idx_unique
    pts = np.fromfile(pts_bin, dtype=np.float32).reshape(-1, 4)
    # dedup exactly as in your code
    _, idx_unique = np.unique(pts, axis=0, return_index=True)
    lbl = np.fromfile(gt_lbl, dtype=np.int32)
    lbl = lbl[idx_unique]
    return lbl

# ---------- Livox helpers ----------
def iter_livox_pred_files(run_dir):
    # logs/<tag>/{HP,HD,LP,LD}/pred/*.pred.u8
    for subset in ("HP", "HD", "LP", "LD"):
        pred_dir = os.path.join(run_dir, subset, "pred")
        if not os.path.isdir(pred_dir): continue
        for p in sorted(glob(os.path.join(pred_dir, "*.pred.u8"))):
            yield p, subset

def livox_paths_from_pred(pred_path, livox_root):
    # .../<tag>/<SUBSET>/pred/cloud17.pred.u8 -> ./data/livox/<SUBSET>/cloud17.pcd
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
        if len(parts) < 5: continue
        try:
            dust.append(int(parts[4]))
        except Exception:
            continue
    return np.array(dust, dtype=np.int32)

def read_livox_mix_val_list(livox_root):
    val_file = os.path.join(livox_root, "splits", "val_mix.txt")
    if not os.path.isfile(val_file): return set()
    with open(val_file, "r") as f:
        return set(os.path.normpath(line.strip()) for line in f if line.strip())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="./logs", help="Root logs dir")
    ap.add_argument("--tag", required=True, help="Run tag under log_dir")
    ap.add_argument("--dataset", required=True, choices=["wads", "livox"])
    ap.add_argument("--wads_root", default="./data/wads")
    ap.add_argument("--livox_root", default="./data/livox")
    ap.add_argument("--snow_id", type=int, default=110, help="Positive class id for GT (WADS=110, Livox usually 1)")
    args = ap.parse_args()

    run_dir = os.path.join(args.log_dir, args.tag)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[ERROR] run dir not found: {run_dir}")

    tp_all = fp_all = fn_all = tot_all = 0

    if args.dataset == "wads":
        pred_files = list(iter_wads_pred_files(run_dir))
        if not pred_files:
            raise SystemExit(f"[ERROR] no WADS preds under {run_dir}/<seq>/pred/*.pred.label")

        for i, pred_path in enumerate(pred_files, 1):
            pts_bin, gt_lbl, _seq = wads_paths_from_pred(pred_path, args.wads_root)
            if not (os.path.isfile(pts_bin) and os.path.isfile(gt_lbl)):
                print(f"[WARN] missing raw/gt for {pred_path}; skipping"); continue

            gt = load_wads_gt_aligned(pts_bin, gt_lbl)              # (N,)
            pr = np.fromfile(pred_path, dtype=np.uint32)            # (N,)
            if pr.size != gt.size:
                print(f"[WARN] size mismatch: pred={pr.size}, gt={gt.size} for {pred_path}; skipping"); continue

            gt_pos = (gt == args.snow_id)
            pr_pos = (pr != 0)

            tp = int(np.sum(pr_pos & gt_pos))
            fp = int(np.sum(pr_pos & ~gt_pos))
            fn = int(np.sum(~pr_pos & gt_pos))
            tot = int(gt.size)
            tp_all += tp; fp_all += fp; fn_all += fn; tot_all += tot

            if (i % 200 == 0) or (i == len(pred_files)):
                P, R, IOU, _ = reduce_counts(tp_all, fp_all, fn_all)
                print(f"[{i:5d}/{len(pred_files):5d}] P={P:.4f} R={R:.4f} IOU={IOU:.4f} (pts={tot_all})", end="\r")

        print("\n\n=== WADS overall ===")
        P, R, IOU, S = reduce_counts(tp_all, fp_all, fn_all)
        print(f"Precision: {P:.4f}\nRecall:    {R:.4f}\nIOU:       {IOU:.4f}\nScore:     {S:.4f}\nPoints:    {tot_all}")

    else:  # Livox
        per_subset = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tot=0))
        mix_set = read_livox_mix_val_list(args.livox_root)

        pred_entries = list(iter_livox_pred_files(run_dir))
        if not pred_entries:
            raise SystemExit(f"[ERROR] no Livox preds under {run_dir}/(HP|HD|LP|LD)/pred/*.pred.u8")

        tp_mix = fp_mix = fn_mix = tot_mix = 0

        for i, (pred_path, subset) in enumerate(pred_entries, 1):
            pcd_path, subset2, rel_pcd = livox_paths_from_pred(pred_path, args.livox_root)
            if subset != subset2:  # sanity
                print(f"[WARN] subset mismatch for {pred_path}")
            if not os.path.isfile(pcd_path):
                print(f"[WARN] missing PCD {pcd_path}; skipping"); continue

            gt = read_livox_pcd_labels(pcd_path)       # (N,)
            pr = np.fromfile(pred_path, dtype=np.uint8) # (N,)
            if pr.size != gt.size:
                print(f"[WARN] size mismatch: pred={pr.size}, gt={gt.size} for {pred_path}; skipping"); continue

            gt_pos = (gt == args.snow_id) if args.snow_id is not None else (gt != 0)
            pr_pos = (pr != 0)

            tp = int(np.sum(pr_pos & gt_pos))
            fp = int(np.sum(pr_pos & ~gt_pos))
            fn = int(np.sum(~pr_pos & gt_pos))
            tot = int(gt.size)

            tp_all += tp; fp_all += fp; fn_all += fn; tot_all += tot
            d = per_subset[subset]
            d["tp"] += tp; d["fp"] += fp; d["fn"] += fn; d["tot"] += tot

            # mix (val 20%) if list exists
            if mix_set and os.path.normpath(rel_pcd) in mix_set:
                tp_mix += tp; fp_mix += fp; fn_mix += fn; tot_mix += tot

            if (i % 200 == 0) or (i == len(pred_entries)):
                P, R, IOU, _ = reduce_counts(tp_all, fp_all, fn_all)
                print(f"[{i:5d}/{len(pred_entries):5d}] P={P:.4f} R={R:.4f} IOU={IOU:.4f} (pts={tot_all})", end="\r")

        print("\n\n=== Livox overall ===")
        P, R, IOU, S = reduce_counts(tp_all, fp_all, fn_all)
        print(f"Precision: {P:.4f}\nRecall:    {R:.4f}\nIOU:       {IOU:.4f}\nScore:     {S:.4f}\nPoints:    {tot_all}")

        print("\n=== Livox subsets ===")
        for key in ("HP", "HD", "LP", "LD"):
            if key in per_subset:
                d = per_subset[key]
                p, r, iou, s = reduce_counts(d["tp"], d["fp"], d["fn"])
                print(f"[{key}]  P={p:.4f}  R={r:.4f}  IOU={iou:.4f}  Score={s:.4f}  Points={d['tot']}")

        if mix_set:
            print("\n=== Livox mix (val 20%) ===")
            Pm, Rm, IOUm, Sm = reduce_counts(tp_mix, fp_mix, fn_mix)
            print(f"Precision: {Pm:.4f}\nRecall:    {Rm:.4f}\nIOU:       {IOUm:.4f}\nScore:     {Sm:.4f}\nPoints:    {tot_mix}")
        else:
            print("\n[Note] No 'splits/val_mix.txt' found; skipping 'mix' report.")

if __name__ == "__main__":
    main()
