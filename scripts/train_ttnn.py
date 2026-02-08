# scripts/train_ttnn.py
from __future__ import annotations

"""
Train TTNN (Threshold Tuning Neural Network) — DAICS paper-aligned.

Paper recap (Section V-D, Algorithm 1):
- For each WDNN output section g, we train one TTNN[g].
- TTNN is trained on the validation set prediction error MSE_g,t (benign only).
- Online, TTNN predicts an estimated error trend; the threshold is:
    Tg = Tbase + max(T_est)
  where Tbase = mean(MSE_val) + std(MSE_val).

In this repo (current stage):
- We start with G=1 (all sensors in one section) => one TTNN checkpoint.
- Few-time-steps and full threshold tuning loop will be added after.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict
from daics.train.ttnn_trainer import TTNNTrainConfig, save_ttnn_checkpoint, train_ttnn_one_section
from daics.train.wdnn_trainer import load_wdnn_checkpoint  # must exist in your repo
from daics.eval.mse import compute_section_mse_series      # we'll implement right after if needed

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_ttnn")


def _auto_device(requested: str) -> torch.device:
    """
    Keep it consistent with WDNN trainer behavior.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "auto"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="path to trained WDNN checkpoint (e.g., runs/wdnn/best.pt)")
    ap.add_argument("--out_dir", default=None, help="override output dir (default: cfg.train.ttnn.out_dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = _auto_device(str(cfg.train.device))
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Build dataloaders (paper: TTNN trained on validation, benign-only)
    # ------------------------------------------------------------------
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, artifacts = make_dataloaders_paper_strict(
        normal_parquet_path=cfg.data.processed_normal_path,
        attack_parquet_path=cfg.data.processed_attack_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
        test_mode=str(cfg.eval.test_mode),
        normal_tail_rows=int(cfg.eval.normal_tail_rows),
    )

    win = int(cfg.windowing.Win)
    wout = int(cfg.windowing.Wout)

    logger.info("Window params: Win=%d Wout=%d H=%d S=%d", win, wout, int(cfg.windowing.H), int(cfg.windowing.S))
    logger.info("Detection params (for later): Wanom=%d Wgrace=%d", int(cfg.detection.Wanom), int(cfg.detection.Wgrace))

    # ------------------------------------------------------------------
    # Load WDNN checkpoint (best.pt is recommended)
    # ------------------------------------------------------------------
    wdnn = load_wdnn_checkpoint(args.wdnn_ckpt, device=device)
    wdnn.eval()
    logger.info("Loaded WDNN checkpoint: %s", Path(args.wdnn_ckpt).resolve())

    # ------------------------------------------------------------------
    # Compute validation MSE time series for section g=0 (G=1 for now)
    # Each element is MSE_g,t computed on one validation batch/window.
    # ------------------------------------------------------------------
    logger.info("Computing validation MSE series (section 0)...")
    mse_val = compute_section_mse_series(
        model=wdnn,
        loader=val_loader,
        device=device,
    )
    mse_val_np = np.asarray(mse_val, dtype=np.float32)

    if len(mse_val_np) < (win + 2):
        raise RuntimeError(
            f"Validation MSE series too short ({len(mse_val_np)} points) to train TTNN with Win={win}. "
            "Reduce Win or ensure validation split produces enough windows."
        )

    mu = float(mse_val_np.mean())
    sigma = float(mse_val_np.std())
    tbase = mu + sigma  # Paper: Tbase = mean + std on validation set

    logger.info("Validation MSE series length: %d", len(mse_val_np))
    logger.info("Validation MSE mean/std: %.6f / %.6f  => Tbase=%.6f", mu, sigma, tbase)

    # ------------------------------------------------------------------
    # Train one TTNN instance (G=1 for now)
    # ------------------------------------------------------------------
    tcfg = TTNNTrainConfig(
        lr=float(cfg.train.ttnn.lr),
        batch_size=int(cfg.train.ttnn.batch_size),
        epochs=int(cfg.train.ttnn.epochs),
        out_dir=str(args.out_dir or cfg.train.ttnn.out_dir),
        leaky_slope=0.01,
    )

    logger.info("Training TTNN: lr=%.4g batch=%d epochs=%d out_dir=%s", tcfg.lr, tcfg.batch_size, tcfg.epochs, tcfg.out_dir)

    model, stats = train_ttnn_one_section(
        mse_series_val=mse_val_np,
        win=win,
        wout=wout,
        cfg=tcfg,
        device=device,
    )
    logger.info("TTNN trained: loss=%.6f steps=%d", float(stats["loss"]), int(stats["steps"]))

    # ------------------------------------------------------------------
    # Save
    # We save extra metadata needed later for threshold computation.
    # ------------------------------------------------------------------
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "section_0.pt"
    save_ttnn_checkpoint(
        model,
        out_path,
        meta={
            "loss": float(stats["loss"]),
            "tbase": float(tbase),
            "mse_val_mean": float(mu),
            "mse_val_std": float(sigma),
            "win": int(win),
            "wout": int(wout),
            "wdnn_ckpt": str(Path(args.wdnn_ckpt).as_posix()),
        },
    )
    logger.info("Saved TTNN checkpoint: %s", out_path.resolve())
    logger.info("[OK] train_ttnn finished.")


if __name__ == "__main__":
    main()
