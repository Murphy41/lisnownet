import os
from glob import glob
import numpy as np
from .base import Base

# Estimated resolution for 70-degree circular FOV
FOV_DEG = 70.4
RESOLUTION = 32

# Square image grid
HEIGHT = RESOLUTION
WIDTH = RESOLUTION

# Symmetric vertical sampling (placeholder, not used for inc2ring in Livox)
INC = np.deg2rad(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, HEIGHT))


class LivoxMid70(Base):
    def __init__(self, data_dir, name='LivoxMid70', inc=INC, width=WIDTH, training=True,
                 split_mode='mix', skip=1, return_points=False, filter=''):
        self.split_mode = split_mode
        super().__init__(data_dir, name=name, inc=inc, width=width,
                         training=training, skip=skip, return_points=return_points, filter=filter)

    def read_file_list(self, data_dir):
        def get_files(subfolder):
            return sorted(glob(os.path.join(data_dir, subfolder, '*.pcd')))

        HP = get_files('HP')
        HD = get_files('HD')
        LP = get_files('LP')
        LD = get_files('LD')

        if self.split_mode == 'hp':
            # train on HP, val on non-HP (HD+LP+LD) — adjust if you intended differently
            return HP if self.training else HD + LP + LD

        elif self.split_mode == 'h':
            # train on H (HP+HD), val on P (LP+LD)
            return HP + HD if self.training else LP + LD

        elif self.split_mode == 'p':
            # train on P (HP+LP), val on D (HD+LD)
            return HP + LP if self.training else HD + LD

        elif self.split_mode == 'mix':
            # reproducible 80/20 split across all files
            split_dir = os.path.join(data_dir, 'splits')
            os.makedirs(split_dir, exist_ok=True)
            train_file = os.path.join(split_dir, 'train_mix.txt')
            val_file   = os.path.join(split_dir, 'val_mix.txt')

            if os.path.exists(train_file) and os.path.exists(val_file):
                file_list = train_file if self.training else val_file
                with open(file_list, 'r') as f:
                    return [line.strip() for line in f if line.strip()]

            all_files = HP + HD + LP + LD
            rng = np.random.default_rng(seed=42)
            rng.shuffle(all_files)
            split_idx = int(0.8 * len(all_files))
            train_files = sorted(all_files[:split_idx])
            val_files   = sorted(all_files[split_idx:])

            with open(train_file, 'w') as f:
                f.writelines(fn + '\n' for fn in train_files)
            with open(val_file, 'w') as f:
                f.writelines(fn + '\n' for fn in val_files)

            return train_files if self.training else val_files

        else:
            raise ValueError(f"Unknown split_mode '{self.split_mode}'. Choose from ['hp', 'h', 'p', 'mix'].")

    # ---- key override so eval can use dataset.project_points(...) ----
    def project_points(self, points, fov_deg=70.4):
        """
        Map raw points to raster bins using Livox cone FOV:
          - azimuth clamped to ±fov/2
          - same vertical mapping via inc2ring
        Returns (i0, i1, valid) like Base.project_points.
        """
        depth = np.linalg.norm(points[:, :3], axis=-1)
        depth_safe = np.maximum(depth, 1e-9)
        inclination = np.arcsin(points[:, 2] / depth_safe)     # [-pi/2, pi/2]
        azimuth     = np.arctan2(points[:, 1], points[:, 0])   # [-pi, pi]

        half = np.deg2rad(fov_deg / 2.0)

        # optional cone check (consistent with points2image)
        angle_radius = np.sqrt(inclination**2 + azimuth**2)
        valid_cone = angle_radius <= half

        # vertical binning
        ring = self.inc2ring(inclination).round().astype(np.int32)
        i0 = (self.num_beams - 1) - ring

        # azimuth -> clamp to cone then normalize to [0,width)
        az_clamped = np.clip(azimuth, -half, half)
        i1 = (az_clamped + half) / (2 * half) * self.width
        i1 = np.floor(i1).astype(np.int32)

        valid = (i0 >= 0) & (i0 < self.num_beams) & (i1 >= 0) & (i1 < self.width) & valid_cone
        return i0, i1, valid

    def points2image(self, points, labels, interleave=True):
        """
        Same ordering as Base, but the binning comes from self.project_points()
        so eval and rasterization stay perfectly in sync.
        """
        # depth & ordering (same as Base)
        depth = np.linalg.norm(points[:, :3], axis=-1)
        if self.training:
            order = np.arange(depth.size, dtype=np.int32)
            self.rng.shuffle(order)
        else:
            order = np.argsort(depth)[::-1]
            if interleave:
                num_split = self.num_beams * 4
                order = np.hstack([order[k::num_split] for k in range(num_split)])

        points, labels = points[order, :], labels[order]
        depth = depth[order]

        # binning via the shared helper
        i0_all, i1_all, valid = self.project_points(points)
        i0, i1 = i0_all[valid], i1_all[valid]

        # allocate outputs
        points_dtype = points.dtype if points.size else np.float32
        labels_dtype = labels.dtype if labels.size else np.int32
        range_img = np.full([2, self.num_beams, self.width], -1, dtype=points_dtype)
        xyz_img   = np.full([3, self.num_beams, self.width], -np.inf, dtype=points_dtype)
        lbl_img   = np.full([self.num_beams, self.width], -1, dtype=labels_dtype)

        if i0.size > 0:
            # fill (use shrink like Base)
            rng_depth = self.shrink(depth[valid])
            rng_inten = self.shrink(points[valid, -1])
            range_img[0, i0, i1] = rng_depth
            range_img[1, i0, i1] = rng_inten
            for c in range(3):
                xyz_img[c, i0, i1] = points[valid, c]
            lbl_img[i0, i1] = labels[valid]

        lbl_img = np.expand_dims(lbl_img, 0)
        return range_img, xyz_img, lbl_img

    @staticmethod
    def get_file_id(file_name):
        # e.g., data/livox/HP/cloud17.pcd → HP/cloud17.pcd
        return os.path.join(*file_name.strip().split(os.sep)[-2:])

    @staticmethod
    def read_files(file_name):
        """
        Read a Livox .pcd file with x y z intensity dust (ascii)
        Return: points (N, 4), labels (N,)
        """
        with open(file_name, 'r') as f:
            lines = f.readlines()

        # find "DATA ..." line
        for i, line in enumerate(lines):
            if line.strip().startswith("DATA"):
                data_start = i + 1
                break
        else:
            raise ValueError(f"DATA section not found in {file_name}")

        pts, lbls = [], []
        for line in lines[data_start:]:
            try:
                x, y, z, intensity, dust = line.strip().split()
                pts.append([float(x), float(y), float(z), float(intensity)])
                lbls.append(int(dust))
            except ValueError:
                continue

        points = np.array(pts, dtype=np.float32)
        points[:, 3] /= 255.0  # keep intensity consistent with WADS scaling
        labels = np.array(lbls, dtype=np.int32)
        return points, labels
