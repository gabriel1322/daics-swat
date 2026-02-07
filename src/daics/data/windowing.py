"""
DAICS-SWaT — Windowing utilities

Paper notation (we will reuse these symbols consistently):
- W : window length (number of time steps in the input segment)
- H : forecasting horizon (how many steps ahead to predict; H=1 => next step)
- S : stride (step between consecutive windows)

Given a multivariate time series X[t] in R^D and labels L[t] in {0,1},
we build windows:

    x_i = X[t : t+W]          shape (W, D)
    y_i = X[t+W+H-1]          shape (D,)   (next-step target if H=1)
    lab_i = agg( L[t : t+W] ) shape ()     (window label)

For lab aggregation, typical anomaly detection setting:
- If any time step in the window is anomalous => window is anomalous (OR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


LabelAgg = Literal["any", "last"]


@dataclass(frozen=True)
class WindowingConfig:
    """Windowing hyperparameters in paper notation."""
    W: int = 16
    H: int = 1
    S: int = 1
    label_agg: str = "any"

    def __post_init__(self) -> None:
        if self.W <= 0:
            raise ValueError("W must be > 0")
        if self.H <= 0:
            raise ValueError("H must be > 0")
        if self.S <= 0:
            raise ValueError("S must be > 0")
        if self.label_agg not in ("any", "last"):
            raise ValueError("label_agg must be 'any' or 'last'")


def num_windows(T: int, cfg: WindowingConfig) -> int:
    """
    Compute the number of valid windows for a series length T.

    Need indices up to (t + W + H - 1) to exist.
    Last valid t satisfies: t + W + H - 1 <= T - 1
                          => t <= T - (W + H)
    With stride S, count windows: floor((T - (W+H)) / S) + 1 if T >= W+H else 0
    """
    max_start = T - (cfg.W + cfg.H)
    if max_start < 0:
        return 0
    return (max_start // cfg.S) + 1


def build_windows(
    X: np.ndarray,
    L: np.ndarray,
    cfg: WindowingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build windowed dataset (x, y, lab) from raw arrays.

    Parameters
    ----------
    X : array, shape (T, D)
        Multivariate time series.
    L : array, shape (T,)
        Per-timestep labels (0/1).
    cfg : WindowingConfig
        Windowing parameters (paper notation).

    Returns
    -------
    x : array, shape (N, W, D)
    y : array, shape (N, D)
    lab : array, shape (N,)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, D)")
    if L.ndim != 1:
        raise ValueError("L must be 1D (T,)")
    if X.shape[0] != L.shape[0]:
        raise ValueError("X and L must have same length T")

    T, D = X.shape
    N = num_windows(T, cfg)
    if N == 0:
        # Return empty arrays with consistent shapes
        return (
            np.empty((0, cfg.W, D), dtype=X.dtype),
            np.empty((0, D), dtype=X.dtype),
            np.empty((0,), dtype=L.dtype),
        )

    x = np.empty((N, cfg.W, D), dtype=X.dtype)
    y = np.empty((N, D), dtype=X.dtype)
    lab = np.empty((N,), dtype=L.dtype)

    # paper mapping: start index t_i = i * S
    for i in range(N):
        t = i * cfg.S
        x[i] = X[t : t + cfg.W]
        # Target at horizon H: index = t + W + H - 1
        y[i] = X[t + cfg.W + cfg.H - 1]

        if cfg.label_agg == "any":
            lab[i] = 1 if np.any(L[t : t + cfg.W] > 0) else 0
        elif cfg.label_agg == "last":
            lab[i] = L[t + cfg.W - 1]
        else:
            raise RuntimeError("Unsupported label_agg")  # should not happen

    return x, y, lab
