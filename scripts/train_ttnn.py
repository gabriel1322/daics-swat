from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from daics.config import load_config
from daics.train.ttnn_trainer import TTNNTrainConfig, save_ttnn_checkpoint, train_ttnn_one_section
from daics.train.wdnn_trainer import load_wdnn_checkpoint  # doit exister chez toi
from daics.data.dataloaders import make_dataloaders
from daics.eval.mse import compute_section_mse_series  # on ajoute juste après

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_ttnn")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="path to trained WDNN checkpoint (.pt)")
    ap.add_argument("--out_dir", default=None, help="override runs/ttnn")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu") if cfg.train.device == "cpu" else (
        torch.device("cuda") if (cfg.train.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    )
    logger.info("Device: %s", device)

    # Build loaders and get validation set (normal-only split)
    train_loader, val_loader, test_loader, artifacts = make_dataloaders(cfg)
    win = int(cfg.windowing.Win)
    wout = int(cfg.windowing.Wout)

    # Load WDNN and compute MSE series on validation (paper: TTNN trained on validation benign records)
    wdnn = load_wdnn_checkpoint(args.wdnn_ckpt, device=device)
    mse_val = compute_section_mse_series(
        model=wdnn,
        loader=val_loader,
        device=device,
    )
    mse_val_np = np.asarray(mse_val, dtype=np.float32)
    logger.info("Validation MSE series length: %d", len(mse_val_np))
    logger.info("Validation MSE mean/std: %.6f / %.6f", float(mse_val_np.mean()), float(mse_val_np.std()))

    # Train one TTNN (G=1 for now)
    tcfg = TTNNTrainConfig(
        lr=cfg.train.ttnn.lr if hasattr(cfg.train, "ttnn") else 1e-2,
        batch_size=cfg.train.ttnn.batch_size if hasattr(cfg.train, "ttnn") else 32,
        epochs=cfg.train.ttnn.epochs if hasattr(cfg.train, "ttnn") else 1,
        out_dir=args.out_dir or (cfg.train.ttnn.out_dir if hasattr(cfg.train, "ttnn") else "runs/ttnn"),
    )

    model, stats = train_ttnn_one_section(
        mse_series_val=mse_val_np,
        win=win,
        wout=wout,
        cfg=tcfg,
        device=device,
    )
    logger.info("TTNN trained: loss=%.6f steps=%d", stats["loss"], int(stats["steps"]))

    out_dir = Path(tcfg.out_dir)
    out_path = out_dir / "section_0.pt"
    save_ttnn_checkpoint(model, out_path, meta={"loss": stats["loss"]})
    logger.info("Saved TTNN checkpoint: %s", out_path.resolve())


if __name__ == "__main__":
    main()
