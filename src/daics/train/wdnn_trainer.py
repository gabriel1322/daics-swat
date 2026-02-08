from __future__ import annotations

"""
WDNN trainer (paper-aligned)

Paper:
- Loss is MSE (Eq.(1)) summed across output sections g
- Optimizer is SGD
- Train/val on normal-only (one-class)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from daics.models.wdnn import WDNN, WDNNConfig, wdnn_loss_mse_per_section


@dataclass(frozen=True)
class WDNNTrainConfig:
    """
    Paper alignment:
      - Optimizer: SGD
      - Loss: MSE (Eq.(1)) summed over output sections
    """
    lr: float = 1e-3
    epochs: int = 30
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    seed: int = 42

    out_dir: str = "runs/wdnn"
    save_best: bool = True
    save_last: bool = True


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def evaluate_wdnn(
    model: WDNN,
    loader: DataLoader,
    sensor_sections: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate average loss on a loader.
    We also report per-section losses for debugging.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    sec_sums: List[float] = [0.0 for _ in sensor_sections]

    for x, y_sensors, _lab in loader:
        x = x.to(device, non_blocking=True)
        y_sensors = y_sensors.to(device, non_blocking=True)

        y_hat_sections = model(x)  # list of (B,Wout,mse_g)
        loss, losses_g = wdnn_loss_mse_per_section(y_hat_sections, y_sensors, sensor_sections)

        total_loss += float(loss.item())
        for i, lg in enumerate(losses_g):
            sec_sums[i] += float(lg.item())
        n_batches += 1

    out: Dict[str, float] = {}
    if n_batches == 0:
        out["loss"] = float("nan")
        return out

    out["loss"] = total_loss / n_batches
    for i, s in enumerate(sec_sums):
        out[f"loss_g{i+1}"] = s / n_batches
    return out


def save_checkpoint(
    path: Path,
    model: WDNN,
    wdnn_cfg: WDNNConfig,
    sensor_sections: Sequence[Sequence[int]],
    optim: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save everything needed to resume / evaluate:
      - model weights
      - optimizer state
      - wdnn_cfg (as dict)
      - sensor_sections
    """
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "wdnn_cfg": wdnn_cfg.__dict__,
        "sensor_sections": [list(s) for s in sensor_sections],
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_wdnn_checkpoint(path: str, device: torch.device) -> WDNN:
    """
    Load a WDNN checkpoint saved by this trainer (best.pt or last.pt).

    Returns the model in eval() mode on the specified device.
    """
    ckpt = torch.load(path, map_location=device)

    wdnn_cfg = WDNNConfig(**ckpt["wdnn_cfg"])
    sensor_sections = ckpt["sensor_sections"]

    model = WDNN(wdnn_cfg, sensor_sections=sensor_sections).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def train_wdnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    wdnn_cfg: WDNNConfig,
    train_cfg: WDNNTrainConfig,
    sensor_sections: Sequence[Sequence[int]],
    device: str = "auto",
) -> Tuple[WDNN, Dict[str, object]]:
    """
    Train WDNN using:
      - SGD optimizer (paper)
      - MSE loss summed over output sections (paper Eq.(1))
    """
    _set_seed(train_cfg.seed)
    dev = _auto_device(device)

    model = WDNN(wdnn_cfg, sensor_sections=sensor_sections).to(dev)

    # Paper: SGD
    optim = torch.optim.SGD(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for x, y_sensors, _lab in train_loader:
            x = x.to(dev, non_blocking=True)
            y_sensors = y_sensors.to(dev, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            y_hat_sections = model(x)
            loss, _losses_g = wdnn_loss_mse_per_section(y_hat_sections, y_sensors, sensor_sections)

            loss.backward()

            if train_cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

            optim.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_metrics = evaluate_wdnn(model, val_loader, sensor_sections, dev)
        val_loss = float(val_metrics["loss"])

        print(f"[WDNN] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if train_cfg.save_best and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(
                out_dir / "best.pt",
                model,
                wdnn_cfg,
                sensor_sections,
                optim,
                epoch,
                best_val,
                extra={"val_metrics": val_metrics},
            )

        if train_cfg.save_last:
            save_checkpoint(
                out_dir / "last.pt",
                model,
                wdnn_cfg,
                sensor_sections,
                optim,
                epoch,
                best_val,
                extra={"val_metrics": val_metrics},
            )

    artifacts: Dict[str, object] = {
        "out_dir": str(out_dir),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "wdnn_cfg": wdnn_cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "sensor_sections": [list(s) for s in sensor_sections],
    }
    return model, artifacts
