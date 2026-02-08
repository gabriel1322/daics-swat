from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# WDNN (Wide & Deep Neural Network) — DAICS paper
# ============================================================
#
# PAPER NOTATION:
# - Input time window length: Win
# - Output time window length: Wout
# - Horizon: H (handled by windowing/dataset, not inside the model)
# - Number of features per time step: m = mse + mac
# - Number of sensors: mse
# - Output sections: g in {1..G}, each outputs a group of sensors (mse^g sensors)
#
# The WDNN described in Section V-A consists of:
# - Feature extractor:
#   - Wide branch: DL1 (memorization via cross-product transformations)
#   - Deep branch: DL2 -> CL1 -> MP1 -> CL2 -> MP2 -> DL3 (generalization over Win)
# - Aggregation: concat(wide, deep) -> DL4
# - Output sections: for each g:
#   DL5 -> DL6 -> DL7 -> predict sensors of that section over Wout
#
# Activations:
# - Fully connected layers: LeakyReLU
# - Conv layers: LeakyReLU(Conv1d(...))
#


@dataclass(frozen=True)
class WDNNConfig:
    """
    Paper notation (Section V-A + Table 4):
      Win  : input window length
      Wout : output window length
      m    : total input features (mse + mac)
      mse  : number of sensors (targets)
    """
    Win: int
    Wout: int
    m: int
    mse: int

    # Table 4 defaults (SWaT/WADI in paper)
    dl1: int | None = None       # wide branch neurons (DL1) ~ 3*Win
    dl2: int | None = None       # deep pre-proj neurons (DL2) ~ 3*m
    dl4: int = 80                # aggregation (DL4)

    # Conv hyperparams (paper uses 1D conv + maxpool)
    cl1_channels: int = 64
    cl1_kernel: int = 2
    cl2_channels: int = 128
    cl2_kernel: int = 2

    # Leak for LeakyReLU (paper: 0.01)
    leaky_slope: float = 0.01


class WDNN(nn.Module):
    """
    Wide + Deep NN (paper Figure 2) for predicting future sensors.

    Input:
      x: (B, Win, m)

    Output:
      list of sections, each:
        y_hat_g: (B, Wout, mse_g)
    """

    def __init__(self, cfg: WDNNConfig, sensor_sections: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.cfg = cfg
        self.sensor_sections = [list(s) for s in sensor_sections]
        self.G = len(self.sensor_sections)

        dl1 = cfg.dl1 if cfg.dl1 is not None else 3 * cfg.Win
        dl2 = cfg.dl2 if cfg.dl2 is not None else 3 * cfg.m

        # --- Wide branch (memorization)
        # Paper: cross-product transformations. We approximate with a learnable projection
        # on the flattened window; later we can add explicit pairwise crosses if needed.
        self.wide_fc = nn.Linear(cfg.Win * cfg.m, dl1)

        # --- Deep branch (generalization)
        # Project per-time-step features to a bigger space then convolve along time.
        self.deep_fc = nn.Linear(cfg.m, dl2)

        self.cl1 = nn.Conv1d(in_channels=dl2, out_channels=cfg.cl1_channels, kernel_size=cfg.cl1_kernel, stride=1)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        self.cl2 = nn.Conv1d(in_channels=cfg.cl1_channels, out_channels=cfg.cl2_channels, kernel_size=cfg.cl2_kernel, stride=1)
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        # After conv/pool, flatten then map to Wout (paper DL3 reshapes to allow concat)
        # We implement DL3 as a projection to (Wout) * (something small), then we concat with wide.
        self.deep_to_feat = nn.Linear(cfg.cl2_channels, cfg.Wout)

        # --- Aggregation (DL4)
        self.agg_fc = nn.Linear(dl1 + cfg.Wout, cfg.dl4)

        # --- Output sections (DL5, DL6, DL7) per group of sensors
        heads: List[nn.Module] = []
        for sec in self.sensor_sections:
            mse_g = len(sec)
            dl5 = int(round(2.25 * mse_g))
            dl6 = int(round(1.5 * mse_g))
            dl7 = mse_g  # final produces mse_g per time step; we will output Wout*mse_g

            heads.append(
                nn.Sequential(
                    nn.Linear(cfg.dl4, dl5),
                    nn.LeakyReLU(cfg.leaky_slope, inplace=True),
                    nn.Linear(dl5, dl6),
                    nn.LeakyReLU(cfg.leaky_slope, inplace=True),
                    nn.Linear(dl6, cfg.Wout * dl7),
                )
            )
        self.heads = nn.ModuleList(heads)

        self.act = nn.LeakyReLU(cfg.leaky_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, Win, m)
        returns list of G tensors: (B, Wout, mse_g)
        """
        B, Win, m = x.shape
        if Win != self.cfg.Win or m != self.cfg.m:
            raise ValueError(f"Expected x shape (B,{self.cfg.Win},{self.cfg.m}), got {tuple(x.shape)}")

        # --- Wide branch
        wide_in = x.reshape(B, Win * m)
        wide = self.act(self.wide_fc(wide_in))  # (B, DL1)

        # --- Deep branch
        # Apply per-timestep FC then conv across time dimension.
        # x -> (B, Win, dl2) -> transpose to (B, dl2, Win)
        deep = self.act(self.deep_fc(x))              # (B, Win, dl2)
        deep = deep.transpose(1, 2)                   # (B, dl2, Win)

        deep = self.act(self.cl1(deep))               # (B, C1, L1)
        deep = self.mp1(deep)                         # (B, C1, L1/2)

        deep = self.act(self.cl2(deep))               # (B, C2, L2)
        deep = self.mp2(deep)                         # (B, C2, L2/2)

        # Reduce time dimension by global max (keeps it stable across Win variations)
        deep = torch.amax(deep, dim=-1)               # (B, C2)

        deep_feat = self.act(self.deep_to_feat(deep)) # (B, Wout)

        # --- Aggregation (DL4)
        agg = torch.cat([wide, deep_feat], dim=1)     # (B, DL1 + Wout)
        agg = self.act(self.agg_fc(agg))              # (B, DL4)

        # --- Output heads
        outs: List[torch.Tensor] = []
        for head, sec in zip(self.heads, self.sensor_sections):
            mse_g = len(sec)
            y_flat = head(agg)                        # (B, Wout*mse_g)
            y = y_flat.view(B, self.cfg.Wout, mse_g)  # (B, Wout, mse_g)
            outs.append(y)
        return outs


def wdnn_loss_mse_per_section(
    y_hat_sections: Sequence[torch.Tensor],
    y_true_sensors: torch.Tensor,
    sensor_sections: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Paper Eq.(1) style loss:
      c_g = (1/s) * Σ_t (1/mse_g) * Σ_i ( (ŷ_t[i] - y_t[i])^2 )
    Here we also average over Wout time steps.

    Inputs:
      y_hat_sections: list of (B, Wout, mse_g)
      y_true_sensors: (B, Wout, mse)  (all sensors)
      sensor_sections: indices of sensors for each group g
    """
    losses_g: List[torch.Tensor] = []
    total = torch.tensor(0.0, device=y_true_sensors.device)

    for y_hat, sec in zip(y_hat_sections, sensor_sections):
        idx = torch.tensor(sec, device=y_true_sensors.device, dtype=torch.long)
        y_true_g = y_true_sensors.index_select(dim=2, index=idx)  # (B, Wout, mse_g)

        # MSE averaged over batch, time, and sensors => matches Eq.(1) spirit
        lg = F.mse_loss(y_hat, y_true_g, reduction="mean")
        losses_g.append(lg)
        total = total + lg

    return total, losses_g
