#!/usr/bin/env python3
import argparse
import time
from multiprocessing import Pool, cpu_count
import os
from glob import glob
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.datasets.wads import WADS
from tools.datasets.cadc import CADC
from tools.models import LiSnowNet
from tools.utils import image2points
from tools.datasets.livoxMid70 import LivoxMid70


def save_results(frame, log_dir, zmin=-2, zmax=6, axlim=30, save_bev=False, reverse=False):
    fid, points = frame
    if reverse:
        points = points[::-1, :]
    print(f'\t\t\t{fid}', end='\r')

    xyzi, res = points[:, :4], points[:, 4:6]
    idx_pr = points[:, 7].astype(bool)

    base_dir = os.path.dirname(fid)
    base_name, ext = os.path.splitext(os.path.basename(fid))  # .bin for WADS/CADC, .pcd for Livox

    # --- save filtered cloud ---
    out_xyz_dir = os.path.join(log_dir, base_dir, 'velodyne')
    os.makedirs(out_xyz_dir, exist_ok=True)

    # For KITTI-style (.bin) it's safe to keep .bin.
    # For Livox (.pcd), DO NOT write a fake .pcd via tofile; write a .bin instead.
    if ext.lower() == '.bin':
        out_cloud = os.path.join(out_xyz_dir, base_name + '.bin')
    else:  # '.pcd' or anything else
        out_cloud = os.path.join(out_xyz_dir, base_name + '.bin')

    xyzi[~idx_pr, :].tofile(out_cloud)

    if not save_bev:
        return  # BEV optional: stop here
    
    # --- save BEV image ---
    bev_dir = os.path.join(log_dir, base_dir, 'bev')
    os.makedirs(bev_dir, exist_ok=True)
    fname_png = os.path.join(bev_dir, base_name + '.png')

    # robust figure id (works for all datasets)
    figure_id = hash(fid) & 0x7fffffff
    fig = plt.figure(figure_id, figsize=(8, 4.5), tight_layout=True)
    axes = [fig.add_subplot(1, 2, k + 1) for k in range(2)]

    for idx, ax in enumerate(axes):
        if idx:
            ax.set_title('Denoised')
            ax.set_yticklabels([])
            pts_disp = xyzi[~idx_pr, :]
        else:
            ax.set_title('Raw')
            ax.set_ylabel('y [m]')
            pts_disp = xyzi

        ax.scatter(
            pts_disp[:, 0], pts_disp[:, 1], c=pts_disp[:, 2],
            s=0.1, vmin=zmin, vmax=zmax, alpha=0.9, marker=','
        )
        ax.axis('scaled')
        ax.set_xlim(-axlim, axlim)
        ax.set_ylim(-axlim, axlim)
        ax.set_xlabel('x [m]')

    fig.savefig(fname_png, dpi=240)
    plt.close(fig)

# --- unified labeling of raw points from a 2D grid (and optional per-pixel payloads) ---
def label_points_from_grid(points_xyzi, dataset, pr_img_2d, res_d_2d=None, res_i_2d=None):
    i0, i1, valid = dataset.project_points(points_xyzi)

    N = points_xyzi.shape[0]
    pr_per_point = np.zeros(N, dtype=bool)
    pr_per_point[valid] = pr_img_2d[i0[valid], i1[valid]]

    resd_per_point = resi_per_point = None
    if res_d_2d is not None:
        resd_per_point = np.zeros(N, dtype=points_xyzi.dtype)
        resd_per_point[valid] = res_d_2d[i0[valid], i1[valid]]
    if res_i_2d is not None:
        resi_per_point = np.zeros(N, dtype=points_xyzi.dtype)
        resi_per_point[valid] = res_i_2d[i0[valid], i1[valid]]

    return pr_per_point, resd_per_point, resi_per_point, valid

# build absolute path to the raw file from fid + dataset type
def fid_to_path(fid, dataset_kind):
    if dataset_kind == 'wads':
        # fid = "<seq>/<frame>.bin"
        seq, name = os.path.split(fid)
        return os.path.join('./data/wads', seq, 'velodyne', name)
    elif dataset_kind == 'cadc':
        return os.path.join('./data/cadcd', fid)  # not used in current eval path
    elif dataset_kind == 'livox':
        return os.path.join('./data/livox', fid)  # e.g. "HP/cloud17.pcd"
    else:
        raise ValueError(dataset_kind)

def write_livox_pred_sidecar(in_pcd_path, pred_bool, log_dir, fid):
    """
    Writes a tiny sidecar file with one 0/1 per point, aligned to the original PCD rows.
    Path: logs/<tag>/<subdirs-of-fid>/pred/<basename>.pred.u8
    """
    out_dir = os.path.join(log_dir, os.path.dirname(fid), 'pred')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_pcd_path))[0]
    out_path = os.path.join(out_dir, base + '.pred.u8')
    pred_bool.astype(np.uint8).tofile(out_path)
    return out_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=1.2e-2)
    parser.add_argument('--z_ground', type=float, default=-1.8)
    parser.add_argument('--snow_id', type=int, default=110)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataset', type=str, default='livox', choices=['cadc', 'wads', 'livox'])
    parser.add_argument('--save_bev', action='store_true',
                    help='Also save BEV PNGs. Off by default.')
    config = parser.parse_args()

    config.tag = config.tag.split('/')[-1]
    log_dir = os.path.join(config.log_dir, config.tag)

    if config.tag:
        checkpoints = sorted(glob(os.path.join(log_dir, '*.pth')))
    else:
        # auto-pick latest run that has any .pth
        checkpoints = []
        run_dirs = sorted(
            [d for d in glob(os.path.join(config.log_dir, '*')) if os.path.isdir(d)],
            key=os.path.getmtime, reverse=True
        )
        for d in run_dirs:
            cks = sorted(glob(os.path.join(d, '*.pth')))
            if cks:
                log_dir = d
                checkpoints = cks
                print(f"[eval] Using latest run: {log_dir}")
                break

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {log_dir} (or in {config.log_dir} if no tag).")

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'font.size': 10
    })

    device = torch.device('cuda')

    if config.dataset == 'cadc':
        dataset = CADC(data_dir='./data/cadcd', training=False, skip=1)
    elif config.dataset == 'wads':
        dataset = WADS(data_dir='./data/wads', training=False, skip=1)
    elif config.dataset == 'livox':
        dataset = LivoxMid70(data_dir='./data/livox', training=False, skip=1)

    # dataset-specific thresholds
    if config.dataset == 'cadc':
        base_thresh = config.threshold                # keep CLI
        z_ground = config.z_ground
        i_thresh = 2 / 255
        d_thresh = 2.5
    elif config.dataset == 'wads':
        base_thresh = config.threshold                # 1.2e-2 from original paper/code
        z_ground = config.z_ground
        i_thresh = 2 / 255
        d_thresh = 2.5
    elif config.dataset == 'livox':
        # Livox intensities and ranges differ → use your looser ones
        base_thresh = 1e-4
        z_ground = config.z_ground     # keep it for now
        i_thresh = 2.0                 # your livox edit
        d_thresh = 0.1                 # your livox edit

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=cpu_count() // 2,
        pin_memory=False,
        drop_last=False
    )

    # Using multiple GPUs
    model = nn.DataParallel(
        LiSnowNet(),
        device_ids=range(torch.cuda.device_count())
    ).to(device)

    ckpt = checkpoints[-1]
    print(f'\nLoading the last checkpoint {ckpt:s}')
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    model.eval()

    runtime, frames = [], []
    with torch.no_grad():
        for index, (fid, range_img, xyz_img, lbl_img) in enumerate(loader):
            range_img = range_img.to(device)
            xyz_img, lbl_img = xyz_img.to(device), lbl_img.to(device)

            # Forward
            t0 = time.time()

            idx_valid, y = model(range_img)
            residual_img = (y - range_img) * idx_valid
            delta_d, delta_i = [residual_img[:, k, :, :] for k in range(2)]

            # convert back to actual readings
            range_img = range_img.pow(3)

            # predictions
            pr_img = delta_d * delta_i.pow(3) > base_thresh
            # snowflakes are higher than the ground plane
            pr_img &= xyz_img[:, 2, :, :] > z_ground
            # snowflakes are very dark
            pr_img &= range_img[:, 1, :, :] < i_thresh
            # points within a small distance are 100% snowflakes
            pr_img |= range_img[:, 0, :, :] < d_thresh

            runtime.append((time.time() - t0) / range_img.shape[0])

            # residual_img: (B, 2, H, W) ; take channels for convenience
            res_d = residual_img[:, 0]          # (B,H,W)
            res_i = residual_img[:, 1]          # (B,H,W)

            if config.dataset in ('wads', 'livox'):
                for b, _fid in enumerate(fid):
                    pr2d  = pr_img[b].detach().cpu().numpy()
                    rd2d  = res_d[b].detach().cpu().numpy()
                    ri2d  = res_i[b].detach().cpu().numpy()

                    raw_path = fid_to_path(_fid, config.dataset)
                    pts, gt = dataset.read_files(raw_path)
                    pr_point, rd_point, ri_point, valid = label_points_from_grid(
                        pts, dataset, pr2d, rd2d, ri2d
                    )

                    arr = np.zeros((pts.shape[0], 8), dtype=np.float32)
                    arr[:, 0:4] = pts
                    if rd_point is not None:
                        arr[:, 4] = rd_point
                    if ri_point is not None:
                        arr[:, 5] = ri_point

                    if config.dataset == 'wads':
                        arr[:, 6] = (gt == config.snow_id).astype(np.float32)
                    else:
                        # livox (or anything already 0/1)
                        arr[:, 6] = (gt.astype(bool)).astype(np.float32)
                    
                    arr[:, 7] = pr_point.astype(np.float32)
                    arr = arr[np.isfinite(arr).all(axis=-1)]

                    # --- write compact per-point sidecars ---
                    if config.dataset == 'wads':
                        out_lab_dir = os.path.join(log_dir, os.path.dirname(_fid), 'pred')
                        os.makedirs(out_lab_dir, exist_ok=True)
                        out_lab = os.path.join(out_lab_dir, os.path.basename(raw_path).replace('.bin', '.pred.label'))
                        # preds are 0/1 -> uint8 is enough
                        arr[:, 7].astype(np.uint32).tofile(out_lab)

                    elif config.dataset == 'livox':
                        in_path = fid_to_path(_fid, 'livox')
                        write_livox_pred_sidecar(in_path, arr[:, 7].astype(bool), log_dir, _fid)

                    # --- write filtered point cloud (and optional BEV) right now ---
                    save_results((_fid, arr), log_dir, save_bev=config.save_bev)

                    print(', '.join([
                        f'[{index + 1:4d}/{len(loader):4d}] {_fid}',
                        f'FPS = {1 / np.median(runtime):.4f}',
                        f'num_points = {arr.shape[0]:d}'
                    ]), end='\r')

                # do NOT accumulate; we already wrote disk outputs
                continue  # skip image2points path

            # results to be saved (for CADC / no per-point GT)
            gt_img, pr_img = (lbl_img == config.snow_id), pr_img.unsqueeze(1)
            output_img = torch.cat([
                xyz_img,
                range_img[:, 1, :, :].unsqueeze(1),
                residual_img,
                gt_img,
                pr_img
            ], dim=1)
            idx_valid = idx_valid[:, 0, :, :].unsqueeze(1).expand_as(output_img)
            output_img[~idx_valid] = -1

            p_out = image2points(output_img)
            p_out = p_out.detach().cpu().numpy()
            for _fid, p1 in zip(fid, p_out):
                p1 = p1[np.isfinite(p1).all(axis=-1), :]

                print(', '.join([
                    f'[{index + 1:4d}/{len(loader):4d}] {_fid}',
                    f'FPS = {1 / np.median(runtime):.4f}',
                    f'num_points = {p1.shape[0]:d}'
                ]), end='\r')

                frames.append((_fid, p1))

    print('')
    num_proc = min(3 * cpu_count() // 4, 64)

    if config.dataset == 'wads' or config.dataset == 'livox':
        print('\nDone.')
        exit(0)

    print(f'No GT point-wise labels for {dataset.name:s}. Skipping the de-noising benchmark.')
    print('Saving results ... ', end='\r')
    num_proc = min(3 * cpu_count() // 4, 64)
    with Pool(num_proc) as pool:
        pool.map(partial(save_results, log_dir=log_dir, save_bev=config.save_bev, reverse=True), frames)
    print('\nDone.')
