from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ======================================================================================
# WindowingConfig
# ======================================================================================

@dataclass(frozen=True)
class WindowingConfig:
    """
    This repo uses DAICS paper notation:

      - Win  : input time window length
      - Wout : output time window length (future window to predict, sensors only)
      - H    : horizon (gap between input end and output start)
      - S    : stride between consecutive windows
      - label_agg : aggregation of point labels into one window label

    Backward-compatibility:
      Some older helper tests in this repo expect a simpler config:
        WindowingConfig(W=..., H=..., S=...)
      where W is equivalent to Win and the output is a single step at t_out.

    Therefore:
      - We accept W as an alias for Win via the __init__ signature (see __post_init__).
      - If Wout is not set explicitly, it defaults to 1 in "legacy helpers".
    """
    # Paper fields
    Win: int = 60
    Wout: int = 4
    H: int = 50
    S: int = 1
    label_agg: Literal["any", "all", "last"] = "any"

    # Legacy alias field (not used by paper pipeline directly)
    # NOTE: dataclasses don't accept unknown kwargs; we include W explicitly.
    W: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # If user provided legacy W=..., map to Win.
        if self.W is not None:
            object.__setattr__(self, "Win", int(self.W))
        # sanity
        if self.Win <= 0:
            raise ValueError("Win must be > 0")
        if self.Wout <= 0:
            raise ValueError("Wout must be > 0")
        if self.H < 0:
            raise ValueError("H must be >= 0")
        if self.S <= 0:
            raise ValueError("S must be > 0")


# ======================================================================================
# Legacy helpers (kept to satisfy tests/test_windowing.py)
# ======================================================================================

def num_windows(T: int, cfg: WindowingConfig) -> int:
    """
    LEGACY semantics (used by tests/test_windowing.py):

      x window: [t, t+W)  where W == cfg.Win
      y target: X[t + W + H - 1]  (single time-step)

    Valid if:
      t + W + H - 1 < T  <=>  t <= T - (W + H)

    Number of windows with stride S:
      N = 0 if T < (W + H)
          else 1 + floor((T - (W + H)) / S)
    """
    W = cfg.Win
    needed = W + cfg.H
    if T < needed:
        return 0
    last_start = T - needed
    return 1 + (last_start // cfg.S)


def build_windows(
    X: np.ndarray,
    L: np.ndarray,
    cfg: WindowingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY window builder (exactly what tests expect):

    Inputs:
      X: (T, D)
      L: (T,) point labels 0/1
      cfg: WindowingConfig(W=..., H=..., S=..., label_agg=...)

    Outputs:
      x   : (N, W, D)
      y   : (N, D)      -> single-step target at t_out = t + W + H
      lab : (N,)        -> aggregated label over the *input* window [t, t+W)
                           (test assumes anomaly inside the input window marks it)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T,D). Got {X.shape}")
    if L.ndim != 1:
        raise ValueError(f"L must be 1D (T,). Got {L.shape}")
    if len(X) != len(L):
        raise ValueError("X and L must have same length T.")

    T, D = X.shape
    W = cfg.Win

    N = num_windows(T, cfg)
    if N == 0:
        return (
            np.zeros((0, W, D), dtype=np.float32),
            np.zeros((0, D), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    xs = np.empty((N, W, D), dtype=np.float32)
    ys = np.empty((N, D), dtype=np.float32)
    labs = np.empty((N,), dtype=np.int64)

    for k in range(N):
        t = k * cfg.S
        t_out = t + W + cfg.H - 1

        xs[k] = X[t : t + W].astype(np.float32, copy=False)
        ys[k] = X[t_out].astype(np.float32, copy=False)

        lab_seg = L[t : t + W]  # IMPORTANT: legacy aggregation over INPUT window
        if cfg.label_agg == "any":
            labs[k] = 1 if np.any(lab_seg != 0) else 0
        elif cfg.label_agg == "all":
            labs[k] = 1 if np.all(lab_seg != 0) else 0
        elif cfg.label_agg == "last":
            labs[k] = int(lab_seg[-1] != 0)
        else:
            raise ValueError(f"Unknown label_agg='{cfg.label_agg}'")

    return xs, ys, labs


# ======================================================================================
# Paper-aligned helpers (used by DAICS pipeline)
# ======================================================================================

def num_windows_paper(T: int, cfg: WindowingConfig) -> int:
    """
    PAPER semantics:

      input  : [t_in, t_in+Win)
      output : [t_out, t_out+Wout) where t_out = t_in + Win + H

    Valid if:
      t_in + Win + H + Wout <= T
    """
    needed = cfg.Win + cfg.H + cfg.Wout
    if T < needed:
        return 0
    last_start = T - needed
    return 1 + (last_start // cfg.S)


def build_windows_paper(
    X: np.ndarray,
    y_point: np.ndarray,
    sensor_idx: np.ndarray,
    cfg: WindowingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PAPER window builder:

    Returns:
      x   : (N, Win, m)
      y   : (N, Wout, mse)  (sensors only)
      lab : (N,) aggregated label over the OUTPUT window (Wout)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T,m). Got {X.shape}")
    if y_point.ndim != 1:
        raise ValueError(f"y_point must be 1D (T,). Got {y_point.shape}")
    if len(X) != len(y_point):
        raise ValueError("X and y_point must have same length T.")
    if sensor_idx.ndim != 1 or sensor_idx.size == 0:
        raise ValueError("sensor_idx must be a non-empty 1D index array.")

    T, m = X.shape
    mse = int(sensor_idx.size)

    N = num_windows_paper(T, cfg)
    if N == 0:
        return (
            np.zeros((0, cfg.Win, m), dtype=np.float32),
            np.zeros((0, cfg.Wout, mse), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    xs = np.empty((N, cfg.Win, m), dtype=np.float32)
    ys = np.empty((N, cfg.Wout, mse), dtype=np.float32)
    labs = np.empty((N,), dtype=np.int64)

    for k in range(N):
        t_in = k * cfg.S
        t_out = t_in + cfg.Win + cfg.H

        xs[k] = X[t_in : t_in + cfg.Win].astype(np.float32, copy=False)
        ys[k] = X[t_out : t_out + cfg.Wout, sensor_idx].astype(np.float32, copy=False)

        lab_seg = y_point[t_out : t_out + cfg.Wout]
        if cfg.label_agg == "any":
            labs[k] = 1 if np.any(lab_seg != 0) else 0
        elif cfg.label_agg == "all":
            labs[k] = 1 if np.all(lab_seg != 0) else 0
        elif cfg.label_agg == "last":
            labs[k] = int(lab_seg[-1] != 0)
        else:
            raise ValueError(f"Unknown label_agg='{cfg.label_agg}'")

    return xs, ys, labs


# ======================================================================================
# Dataset used by DAICS training code (paper-aligned)
# ======================================================================================

class SlidingWindowDataset(Dataset):
    """
    Paper-aligned dataset used by dataloaders/training:

      __getitem__ -> (x, y, lab)
        x   : (Win, m)
        y   : (Wout, mse) sensors only
        lab : scalar (aggregated label over output window)

    Signature expected by your current dataloaders:
      SlidingWindowDataset(X, y_point, sensor_idx, cfg)
    """

    def __init__(self, X: np.ndarray, y_point: np.ndarray, sensor_idx: np.ndarray, cfg: WindowingConfig) -> None:
        self.X = X.astype(np.float32, copy=False)
        self.y_point = y_point.astype(np.int64, copy=False)
        self.sensor_idx = sensor_idx.astype(np.int64, copy=False)
        self.cfg = cfg

        self._N = num_windows_paper(len(self.X), cfg)

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, k: int):
        t_in = k * self.cfg.S
        t_out = t_in + self.cfg.Win + self.cfg.H

        x_win = self.X[t_in : t_in + self.cfg.Win]
        y_win = self.X[t_out : t_out + self.cfg.Wout, self.sensor_idx]
        lab_seg = self.y_point[t_out : t_out + self.cfg.Wout]

        if self.cfg.label_agg == "any":
            lab = 1 if np.any(lab_seg != 0) else 0
        elif self.cfg.label_agg == "all":
            lab = 1 if np.all(lab_seg != 0) else 0
        elif self.cfg.label_agg == "last":
            lab = int(lab_seg[-1] != 0)
        else:
            raise ValueError(f"Unknown label_agg='{self.cfg.label_agg}'")

        return (
            torch.from_numpy(x_win),
            torch.from_numpy(y_win),
            torch.tensor(lab, dtype=torch.int64),
        )
