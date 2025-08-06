import os
from glob import glob
import numpy as np
from .base import Base

# Estimated resolution for 70-degree circular FOV
FOV_DEG = 70.4
RESOLUTION = 400

# Square image grid
HEIGHT = RESOLUTION
WIDTH = RESOLUTION

# Symmetric vertical sampling (placeholder, not used for inc2ring in Livox)
INC = np.deg2rad(np.linspace(-FOV_DEG / 2, FOV_DEG / 2, HEIGHT))


class LivoxMid70(Base):
    def __init__(self, data_dir, name='LivoxMid70', training=True,
                 split_mode='mix', skip=1, return_points=False, filter=''):
        self.split_mode = split_mode.lower()  # custom split logic
        self.training = training  # must remain bool
        self.fn_points = self.read_file_list(data_dir)[::skip]
        self.fn_points = [fn for fn in self.fn_points if filter in fn]

        assert len(self.fn_points) > 0

        super().__init__(
            data_dir, name=name, inc=INC, width=WIDTH,
            training=training, skip=skip,
            return_points=return_points, filter=filter
        )

    def read_file_list(self, data_dir):
        def get_files(subfolder):
            return sorted(glob(os.path.join(data_dir, subfolder, '*.pcd')))

        HP = get_files('HP')
        HD = get_files('HD')
        LP = get_files('LP')
        LD = get_files('LD')

        if self.split_mode == 'hp':
            return HP if self.training else HD + LP + LD

        elif self.split_mode == 'h':
            return HP + HD if self.training else LP + LD

        elif self.split_mode == 'p':
            return HP + LP if self.training else HD + LD

        elif self.split_mode == 'mix':
            # Handle saved split for reproducibility
            split_dir = os.path.join(data_dir, 'splits')
            os.makedirs(split_dir, exist_ok=True)

            train_file = os.path.join(split_dir, f'train_mix.txt')
            val_file   = os.path.join(split_dir, f'val_mix.txt')

            # Check for saved files ONLY in mix mode
            if os.path.exists(train_file) and os.path.exists(val_file):
                file_list = train_file if self.training else val_file
                with open(file_list, 'r') as f:
                    return [line.strip() for line in f.readlines()]

            # Generate split if not saved
            all_files = HP + HD + LP + LD
            self.rng = np.random.default_rng(seed=42)
            self.rng.shuffle(all_files)
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


    @staticmethod
    def get_file_id(file_name):
        # e.g., data/livox/HP/cloud17.pcd → HP/cloud17.pcd
        return os.path.join(*file_name.strip().split(os.sep)[-2:])

    @staticmethod
    def read_files(file_name):
        """
        Read a Livox .pcd file with x y z intensity dust label (ascii)
        Return: points (N, 4), labels (N,)
        """
        with open(file_name, 'r') as f:
            lines = f.readlines()

        # Skip header
        for i, line in enumerate(lines):
            if line.strip().startswith("DATA"):
                data_start = i + 1
                break
        else:
            raise ValueError(f"DATA section not found in {file_name}")

        points = []
        labels = []
        for line in lines[data_start:]:
            try:
                x, y, z, intensity, dust = line.strip().split()
                points.append([float(x), float(y), float(z), float(intensity)])
                labels.append(int(dust))
            except ValueError:
                continue

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        return points, labels
