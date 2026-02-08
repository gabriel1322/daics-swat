from __future__ import annotations

"""
Threshold tuning helpers — DAICS Algorithm 1.

Paper:
- Tbase = mean(MSE_val) + std(MSE_val)   (validation is benign only)
- Tg = Tbase + max(T_est), where T_est is produced by TTNN sliding over
  median-filtered error history.

This module is used later for computing thresholds per section.
For now (G=1) it still works as-is.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from daics.models.ttnn import TTNN


@dataclass(frozen=True)
class ThresholdTuningConfig:
    """
    Paper Algorithm 1 parameters.
    """
    win: int
    wout: int
    median_kernel: int = 59


def _median_filter_1d(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Simple 1D median filter with edge padding.
    Paper uses median filtering to smooth short spikes before TTNN input.
    """
    if k <= 1:
        return arr.copy()
    if k % 2 == 0:
        raise ValueError("median_kernel must be odd.")
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        out[i] = np.median(padded[i : i + k])
    return out.astype(np.float32, copy=False)


def load_ttnn_checkpoint(path: str, device: torch.device) -> TTNN:
    """
    Load TTNN checkpoint saved by train_ttnn.py / ttnn_trainer.save_ttnn_checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    win = int(ckpt["win"])
    wout = int(ckpt["wout"])
    leaky_slope = float(ckpt.get("leaky_slope", 0.01))
    model = TTNN(win=win, wout=wout, leaky_slope=leaky_slope).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def compute_Tbase(mse_val: np.ndarray) -> float:
    """
    Paper Algorithm 1:
      Tbase = mean(MSE_val) + std(MSE_val)
    """
    mse_val = np.asarray(mse_val, dtype=np.float32)
    return float(mse_val.mean() + mse_val.std(ddof=0))


@torch.no_grad()
def tune_threshold_Tg(
    ttnn: TTNN,
    Eg: np.ndarray,
    Tbase: float,
    cfg: ThresholdTuningConfig,
    device: torch.device,
) -> float:
    """
    Paper Algorithm 1 (offline form).

    Input:
      Eg: 1D history of past MSE values for section g.
    Output:
      Tg = Tbase + max(T_est)

    Implementation:
      - Median filter Eg
      - Slide Win-sized windows over Eg_med (INCLUDING the last possible window)
      - TTNN predicts scalar per window
      - add max prediction to Tbase
    """
    Eg = np.asarray(Eg, dtype=np.float32)
    if Eg.ndim != 1:
        raise ValueError("Eg must be 1D.")
    if len(Eg) < cfg.win + 1:
        raise ValueError("Eg too short for threshold tuning (need >= Win+1).")

    Eg_med = _median_filter_1d(Eg, cfg.median_kernel)

    preds: List[float] = []
    last_start = len(Eg_med) - cfg.win  # inclusive start index of last possible window

    # IMPORTANT: +1 to include the last window
    for i in range(last_start + 1):
        x = Eg_med[i : i + cfg.win]                      # (Win,)
        xb = torch.from_numpy(x).unsqueeze(0).to(device) # (1, Win)
        yhat = ttnn(xb)                                  # (1, 1)
        preds.append(float(yhat[0, 0].item()))

    return float(Tbase + max(preds)) if preds else float(Tbase)
