from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from daics.models.ttnn import TTNN


@dataclass(frozen=True)
class TTNNTrainConfig:
    """
    Paper Table 4 (TTNN):
      lr = 0.01
      batch_size = 32
      epochs = 1
      median_kernel = 59   (used in Algorithm 1 input smoothing)
    """
    lr: float = 1e-2
    batch_size: int = 32
    epochs: int = 1
    out_dir: str = "runs/ttnn"
    leaky_slope: float = 0.01


class _TTNNDataset(Dataset):
    """
    Each item is (x, y) where:
      x: (Win,)   input time series of past prediction errors
      y: (1,)     target next-step prediction error

    Paper notation alignment:
      input is X = MSĒg,[t0+i, t0+Win+i)
      output is Y_hat[0] ~ predicted error proxy for threshold.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        if X.ndim != 2:
            raise ValueError("TTNN dataset X must be 2D (N, Win).")
        if y.ndim != 1:
            raise ValueError("TTNN dataset y must be 1D (N,).")
        if len(X) != len(y):
            raise ValueError("TTNN dataset X and y length mismatch.")
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])            # (Win,)
        t = torch.tensor([self.y[idx]], dtype=torch.float32)  # (1,)
        return x, t


def _median_filter_1d(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Paper Algorithm 1, line 3: median filter to reduce short-term spikes.
    We implement a simple sliding median with odd kernel size.
    Edge handling: replicate borders (pad with edge values).
    """
    if k <= 1:
        return arr.copy()
    if k % 2 == 0:
        raise ValueError("median_kernel must be odd (e.g., 59 as in paper).")
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        out[i] = np.median(padded[i : i + k])
    return out.astype(np.float32, copy=False)


def build_ttnn_training_pairs(
    mse_series: np.ndarray,
    win: int,
    median_kernel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised pairs from a univariate MSE time series (validation set, benign only).

    We use:
      X[i] = median(MSE[i : i+Win])      (length Win window)
      y[i] = MSE[i+Win]                  (next point after the window)

    This is a pragmatic "next-step" regression target consistent with Fig.4 intent:
    model the MSE time series trend to estimate upcoming error.

    Returns:
      X: (N, Win)
      y: (N,)
    """
    mse = np.asarray(mse_series, dtype=np.float32)
    if mse.ndim != 1:
        raise ValueError("mse_series must be 1D.")
    if len(mse) <= win + 1:
        raise ValueError("mse_series too short to build TTNN pairs.")

    # Apply median filter to the full series first (paper: median over Eg windows).
    mse_med = _median_filter_1d(mse, median_kernel)

    N = len(mse_med) - win
    X = np.zeros((N, win), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        X[i] = mse_med[i : i + win]
        y[i] = mse[i + win]  # predict raw next error (keeps sensitivity to real changes)
    return X, y


def train_ttnn_one_section(
    mse_series_val: np.ndarray,
    win: int,
    wout: int,
    cfg: TTNNTrainConfig,
    device: torch.device,
) -> Tuple[TTNN, Dict[str, float]]:
    """
    Train one TTNN instance for one WDNN output section g.

    Input:
      mse_series_val: MSE_g,t over validation set (normal-only)

    Output:
      trained TTNN model + training stats
    """
    X, y = build_ttnn_training_pairs(mse_series_val, win=win, median_kernel=getattr(cfg, "median_kernel", 59))
    ds = _TTNNDataset(X, y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = TTNN(win=win, wout=wout, leaky_slope=cfg.leaky_slope).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    running = 0.0
    steps = 0

    for epoch in range(cfg.epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)  # (B,1)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

    stats = {"loss": running / max(1, steps), "steps": float(steps)}
    return model, stats


def save_ttnn_checkpoint(
    model: TTNN,
    out_path: Path,
    meta: Dict[str, float] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "win": model.win,
        "wout": model.wout,
        "leaky_slope": model.leaky_slope,
        "meta": meta or {},
    }
    torch.save(payload, out_path)
