"""
KIT Motion-Language Dataset for VQ-VAE Training

Uses the preprocessed KIT-ML data (50-dim MMM features per frame)
produced by dvq/data_preprocessing/kit_preprocess.py.

Feature layout per frame (50 dims):
    RootPosition (3) + RootRotation (3) + JointPosition (44)

Directory layout expected under data_root:
    new_joint_vecs/   ← *.npy files, shape (T, 50)
    texts/            ← *.txt files (HumanML3D format)
    Mean.npy
    Std.npy
    train.txt / val.txt / test.txt
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class KITDataset(data.Dataset):
    """
    KIT Motion-Language dataset loader for VQ-VAE training.
    Mirrors the interface of HumanML3DDataset but with KIT-specific defaults.
    """

    # KIT-ML 251-dim features (same pipeline as HumanML3D but 21 joints):
    # root_vel_y(1) + root_vel_xz(2) + root_y(1) + local_pos(60) + cont6d(120) + vel(63) + foot(4)
    KIT_FEATURES_DIM = 251

    def __init__(self, data_root, split='train', window_size=64, unit_length=4,
                 features_dim=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.data_root = data_root

        self.features_dim = features_dim or self.KIT_FEATURES_DIM

        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        # Aligned with MotionGPT3: KIT 100fps downsampled 8x -> 12.5fps
        self.max_motion_length = 196
        self.min_motion_length = 24

        try:
            mean = np.load(pjoin(self.data_root, 'Mean.npy'))
            std = np.load(pjoin(self.data_root, 'Std.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Mean.npy and Std.npy not found.  "
                "Run  dvq/data_preprocessing/kit_preprocess.py  first."
            )

        split_file = pjoin(self.data_root, f'{split}.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    id_list.append(line)

        print(f"[KIT] Loading data from {split}.txt ({len(id_list)} entries)...")
        new_name_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion) < self.min_motion_length
                        or len(motion) < self.window_size
                        or len(motion) >= self.max_motion_length):
                    continue
                self.data.append(motion)
                self.lengths.append(motion.shape[0] - self.window_size)
                new_name_list.append(name)
            except Exception:
                pass

        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()

        if self.mean.shape[0] != self.features_dim or self.std.shape[0] != self.features_dim:
            raise ValueError(
                f"Dimension mismatch – actual: {self.mean.shape[0]}, expected: {self.features_dim}")

        print(f"[KIT] Loaded {split} set: {len(self.data)} clips  "
              f"(features_dim={self.features_dim})")

    def inv_transform(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]

        if len(motion) - self.window_size > 0:
            idx = random.randint(0, len(motion) - self.window_size)
            motion_window = motion[idx: idx + self.window_size]
        else:
            motion_window = motion[:self.window_size]

        motion_tensor = torch.from_numpy(motion_window).float()
        motion_normalized = (motion_tensor - self.mean) / self.std
        return motion_normalized
