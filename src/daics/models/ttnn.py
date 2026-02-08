from __future__ import annotations

import torch
import torch.nn as nn


class TTNN(nn.Module):
    """
    TTNN = Threshold Tuning Neural Network (Paper Fig. 4).

    Paper intent:
      - Input: a univariate time series of past prediction errors
               over a window of length Win.
        In Algorithm 1 the input is X = MSĒg,[t0+i, t0+Win+i)
      - Output: a scalar "estimated prediction error" / delta threshold
               We keep only Y_hat[0] (Algorithm 1, line 6).

    Architecture (paper):
      CL1 -> MP1 -> CL2 -> MP2 -> DL (1 neuron)
      - kernel size = 2, stride = 1 for convs
      - pooling size = 2 (downsample by factor 2)

    Implementation detail:
      - We use Conv1d over shape (B, C=1, L=Win)
      - Activation: LeakyReLU with slope 0.01 (paper uses LeakyReLU throughout)
    """

    def __init__(
        self,
        win: int,
        wout: int,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if win < 4:
            raise ValueError("TTNN win should be >= 4 (otherwise pooling collapses).")

        self.win = int(win)
        self.wout = int(wout)
        self.leaky_slope = float(leaky_slope)

        # Paper: CL1 kernels=2, kernel=2; MP1 pooling=2
        self.cl1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=1)
        self.mp1 = nn.MaxPool1d(kernel_size=2)

        # Paper: CL2 kernels=Wout, kernel=2; MP2 pooling=2
        self.cl2 = nn.Conv1d(in_channels=2, out_channels=self.wout, kernel_size=2, stride=1)
        self.mp2 = nn.MaxPool1d(kernel_size=2)

        self.act = nn.LeakyReLU(negative_slope=self.leaky_slope)

        # Compute flattened size after conv/pool to build final FC layer.
        # We do it deterministically to avoid silent shape bugs.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.win)  # (B=1,C=1,L=Win)
            z = self._forward_features(dummy)
            flat = int(z.numel())

        # Paper: DL neurons = 1 (scalar)
        self.dl = nn.Linear(flat, 1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, Win)
        z = self.act(self.cl1(x))
        z = self.mp1(z)
        z = self.act(self.cl2(z))
        z = self.mp2(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Win) or (B, 1, Win)
        returns: (B, 1)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,Win)
        elif x.ndim != 3:
            raise ValueError("TTNN forward expects x with shape (B,Win) or (B,1,Win).")

        z = self._forward_features(x)
        z = z.flatten(1)
        out = self.dl(z)
        return out
