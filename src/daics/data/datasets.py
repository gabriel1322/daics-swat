"""
Torch datasets for DAICS-SWaT.

We keep the dataset thin: it delegates window creation to windowing.py
so that the math (paper notation W/H/S) lives in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from daics.data.windowing import WindowingConfig, build_windows


@dataclass(frozen=True)
class WindowArtifacts:
    """Useful metadata for reporting/debugging."""
    feature_cols: list[str]
    cfg: WindowingConfig
    parquet_path: str


class WindowDataset(Dataset):
    """
    Dataset returning (x, y, lab) where:
      x   : (W, D)
      y   : (D,)
      lab : scalar (0/1)
    """
    def __init__(self, X: np.ndarray, L: np.ndarray, cfg: WindowingConfig):
        self.x, self.y, self.lab = build_windows(X, L, cfg)

        # Convert once to tensors (faster than converting at each __getitem__)
        self.x_t = torch.from_numpy(self.x).float()
        self.y_t = torch.from_numpy(self.y).float()
        self.lab_t = torch.from_numpy(self.lab).long()

    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_t[idx], self.y_t[idx], self.lab_t[idx]
