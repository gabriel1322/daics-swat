from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowingConfig:
    """
    Paper-style notation (common in ICS / time-series anomaly papers):

    - W  : lookback window length (number of past timesteps)
    - H  : forecast horizon (how far in the future we predict); DAICS typically uses H=1
    - S  : stride for sampling windows (S=1 for dense sliding windows)
    """
    W: int = 16
    H: int = 1
    S: int = 1


class SlidingWindowDataset(Dataset):
    """
    Builds samples from a multivariate time series.

    Given a normalized time-series X[t] in R^N (N sensors),
    each sample is:
        X_win(t) = [ X[t-W], ..., X[t-1] ] in R^{W x N}
        y(t)     = X[t+H-1]              in R^{N}
        lab(t)   = label at time t+H-1   in {0,1} (optional)

    Implementation notes:
    - We keep X as np.float32 for speed and convert to torch in __getitem__.
    - Shapes returned:
        x:  (W, N)
        y:  (N,)
        lab:(,) int64
    """

    def __init__(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray],
        cfg: WindowingConfig,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D [T, N], got shape={X.shape}")

        self.X = X.astype(np.float32, copy=False)
        self.labels = None if labels is None else labels.astype(np.int64, copy=False)

        self.cfg = cfg
        self.T, self.N = self.X.shape

        if self.cfg.W <= 0:
            raise ValueError("W must be > 0")
        if self.cfg.H <= 0:
            raise ValueError("H must be > 0")
        if self.cfg.S <= 0:
            raise ValueError("S must be > 0")

        # Last index we predict is t + H - 1, so t must satisfy:
        # t - W >= 0  and  t + H - 1 < T
        self.start_t = self.cfg.W
        self.end_t_excl = self.T - (self.cfg.H - 1)

        if self.end_t_excl <= self.start_t:
            raise ValueError(
                f"Time series too short for W={cfg.W}, H={cfg.H}: T={self.T}"
            )

        self.ts = np.arange(self.start_t, self.end_t_excl, self.cfg.S, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.ts[idx])

        # window: [t-W, ..., t-1]
        x = self.X[t - self.cfg.W : t]  # (W, N)

        # target: X[t+H-1]
        y_t = t + self.cfg.H - 1
        y = self.X[y_t]  # (N,)

        if self.labels is None:
            lab = 0
        else:
            lab = int(self.labels[y_t])

        return (
            torch.from_numpy(x),  # (W, N)
            torch.from_numpy(y),  # (N,)
            torch.tensor(lab, dtype=torch.long),
        )
