from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.io_nifti import read_nifti_gz


class PetACDataset(Dataset):
    def __init__(self, split_root, augment=False, input_clip_max=1.0, target_clip_max=1.0):
        self.split_root = Path(split_root)
        self.augment = augment
        self.input_clip_max = input_clip_max
        self.target_clip_max = target_clip_max
        self.nonac_paths = sorted((self.split_root / "nonac").glob("*.nii.gz"))
        self.gt_paths = sorted((self.split_root / "ac_gt").glob("*.nii.gz"))

        if not self.nonac_paths or not self.gt_paths:
            raise RuntimeError(f"No data found in {self.split_root}")

        if len(self.nonac_paths) != len(self.gt_paths):
            raise RuntimeError("nonac and ac_gt counts do not match")

    def __len__(self):
        return len(self.nonac_paths)

    @staticmethod
    def _compute_input_scaler(inp):
        p1 = float(np.percentile(inp, 1))
        p99 = float(np.percentile(inp, 99))
        scale = max(p99 - p1, 1e-6)
        return p1, scale

    @staticmethod
    def _normalize_with_scaler(vol, p1, scale, clip_max=1.0):
        out = (vol - p1) / scale
        if clip_max is None:
            return np.clip(out, 0.0, None).astype(np.float32)
        return np.clip(out, 0.0, clip_max).astype(np.float32)

    @staticmethod
    def _augment(inp, gt):
        if np.random.rand() < 0.5:
            inp = np.flip(inp, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()
        if np.random.rand() < 0.5:
            inp = np.flip(inp, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.9, 1.1)
            inp = np.power(inp, gamma, dtype=np.float32)
        return inp, gt

    def __getitem__(self, idx):
        inp = read_nifti_gz(self.nonac_paths[idx])
        gt = read_nifti_gz(self.gt_paths[idx])

        p1, scale = self._compute_input_scaler(inp)
        inp = self._normalize_with_scaler(inp, p1, scale, clip_max=self.input_clip_max)
        gt = self._normalize_with_scaler(gt, p1, scale, clip_max=self.target_clip_max)

        if self.augment:
            inp, gt = self._augment(inp, gt)

        inp_t = torch.from_numpy(inp[None, ...])
        gt_t = torch.from_numpy(gt[None, ...])
        case_id = self.nonac_paths[idx].name.replace(".nii.gz", "")
        p1_t = torch.tensor([p1], dtype=torch.float32)
        scale_t = torch.tensor([scale], dtype=torch.float32)
        return inp_t, gt_t, case_id, p1_t, scale_t
