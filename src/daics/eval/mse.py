from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_section_mse_series(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[float]:
    """
    Compute MSE_g,t time series for a single output section g (we start with G=1).

    Paper Eq. (2):
      MSE_{g,t} = (1 / M^se_g) * sum_i (Y_t[i] - Ŷ_t[i])^2

    In our data pipeline:
      - batch gives (x, y, lab)
        x: (B, Win, m)
        y: (B, Wout, mse)   [WDNN predicts sensors only]
      - model(x) should return y_hat with same shape as y

    For TTNN training we need a *univariate* series, so we aggregate:
      mse_batch_point = mean over (Wout, mse) of squared error
    i.e. one scalar per sample window.
    """
    model.eval()
    out: List[float] = []

    for x, y, _lab in loader:
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)  # expected (B, Wout, mse)
        if yhat.shape != y.shape:
            raise ValueError(f"WDNN output shape {tuple(yhat.shape)} != target shape {tuple(y.shape)}")

        se = (yhat - y) ** 2  # (B, Wout, mse)
        mse_per_sample = se.mean(dim=(1, 2))  # (B,)
        out.extend([float(v.item()) for v in mse_per_sample])

    return out
