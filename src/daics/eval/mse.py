from __future__ import annotations

"""
Compute WDNN prediction error time series for TTNN training.

Context:
- In your WDNN implementation, forward(x) returns:
    List[Tensor]   (one Tensor per output section g)
  where each Tensor has shape (B, Wout, mse_g).

- In your current stage (G=1), model(x) returns a list with a single element:
    [ (B, Wout, mse) ]

For TTNN training we want a univariate series MSE_g,t:
- Paper Eq.(2): MSE_{g,t} = (1 / M^se_g) * sum_i (Y_t[i] - Ŷ_t[i])^2
- We additionally average over the output horizon Wout to get one scalar per sample/window:
    mse_per_sample = mean over (Wout, sensors) of squared error
"""

from typing import List, Sequence, Union

import torch
from torch.utils.data import DataLoader


def _unwrap_wdnn_output(yhat: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    """
    Make compute_section_mse_series robust to two common model outputs:
      1) Tensor (B, Wout, mse)          [single-head models]
      2) List[Tensor] of length G       [your WDNN: one tensor per section]

    For now we assume G=1 and return the first section.
    Later, if you move to G>1, we can implement a multi-section version.
    """
    if isinstance(yhat, torch.Tensor):
        return yhat

    # If it's a list/tuple of tensors (sections)
    if isinstance(yhat, (list, tuple)):
        if len(yhat) == 0:
            raise ValueError("WDNN output is an empty list/tuple.")
        if not isinstance(yhat[0], torch.Tensor):
            raise TypeError("WDNN output list does not contain tensors.")
        return yhat[0]

    raise TypeError(f"Unsupported WDNN output type: {type(yhat)}")


@torch.no_grad()
def compute_section_mse_series(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[float]:
    """
    Compute a univariate MSE time series for section g=0 (G=1).

    Inputs:
      loader yields (x, y, lab)
        x: (B, Win, m)
        y: (B, Wout, mse)     target sensors only
      model(x) returns either:
        - Tensor (B, Wout, mse)
        - OR List[Tensor] where element 0 is (B, Wout, mse)

    Output:
      list of floats of length ~ (#windows in loader),
      where each float is one sample-window MSE averaged over (Wout, mse).
    """
    model.eval()
    out: List[float] = []

    for x, y, _lab in loader:
        x = x.to(device)
        y = y.to(device)

        yhat_raw = model(x)
        yhat = _unwrap_wdnn_output(yhat_raw)  # (B, Wout, mse) for G=1

        if yhat.shape != y.shape:
            raise ValueError(
                "WDNN output shape mismatch.\n"
                f"  got     : {tuple(yhat.shape)}\n"
                f"  expected: {tuple(y.shape)}\n"
                "If you switched to multi-section outputs (G>1), you must adapt this function "
                "to select the right section and match target slicing."
            )

        # Squared error then mean over time + sensors => one scalar per sample/window
        se = (yhat - y) ** 2                  # (B, Wout, mse)
        mse_per_sample = se.mean(dim=(1, 2))  # (B,)

        out.extend([float(v.item()) for v in mse_per_sample])

    return out
