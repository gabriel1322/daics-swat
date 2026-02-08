# scripts/tune_thresholds.py
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict
from daics.eval.mse import compute_section_mse_series
from daics.eval.thresholds import ThresholdTuningConfig, compute_Tbase, load_ttnn_checkpoint, tune_threshold_Tg
from daics.train.wdnn_trainer import load_wdnn_checkpoint

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tune_thresholds")


def _pick_device(device_str: str) -> torch.device:
    d = (device_str or "auto").lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="runs/wdnn/best.pt (or last.pt)")
    ap.add_argument("--ttnn_ckpt", default="runs/ttnn/section_0.pt", help="runs/ttnn/section_0.pt")
    ap.add_argument("--out", default="runs/thresholds.json", help="Output JSON file with Tbase/Tg")

    # Experiment knobs
    ap.add_argument("--median_kernel", type=int, default=None, help="Override median kernel (paper default 59). Must be odd.")
    ap.add_argument("--few_steps", type=int, default=None, help="If set, use only last K TTNN predictions to compute Tg.")

    args = ap.parse_args()

    cfg = load_config(args.config)
    device = _pick_device(cfg.train.device)

    logger.info("Device: %s", device)
    logger.info("Building dataloaders...")
    logger.info(
        "Window params: Win=%d Wout=%d H=%d S=%d",
        int(cfg.windowing.Win), int(cfg.windowing.Wout), int(cfg.windowing.H), int(cfg.windowing.S),
    )
    logger.info(
        "Detection params (for later): Wanom=%d Wgrace=%d",
        int(cfg.detection.Wanom), int(cfg.detection.Wgrace),
    )

    # Dataloaders
    _train_loader, val_loader, _test_loader, artifacts = make_dataloaders_paper_strict(
        normal_parquet_path=cfg.data.processed_normal_path,
        attack_parquet_path=cfg.data.processed_attack_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
        test_mode=str(cfg.eval.test_mode),
        normal_tail_rows=int(cfg.eval.normal_tail_rows),
    )

    # Load WDNN
    wdnn_path = Path(args.wdnn_ckpt)
    wdnn = load_wdnn_checkpoint(str(wdnn_path), device=device)
    logger.info("Loaded WDNN checkpoint: %s", wdnn_path.resolve())

    # Compute validation MSE series (section 0 since G=1)
    logger.info("Computing validation MSE series (section 0)...")
    mse_val = compute_section_mse_series(model=wdnn, loader=val_loader, device=device)
    mse_val_np = np.asarray(mse_val, dtype=np.float32)

    Tbase = compute_Tbase(mse_val_np)
    logger.info("Validation MSE series length: %d", int(len(mse_val_np)))
    logger.info(
        "Validation MSE mean/std: %.6f / %.6f  => Tbase=%.6f",
        float(mse_val_np.mean()), float(mse_val_np.std(ddof=0)), float(Tbase),
    )

    # Load TTNN
    ttnn_path = Path(args.ttnn_ckpt)
    ttnn = load_ttnn_checkpoint(str(ttnn_path), device=device)
    logger.info("Loaded TTNN checkpoint: %s", ttnn_path.resolve())

    # Threshold config
    median_kernel = int(args.median_kernel) if args.median_kernel is not None else int(cfg.train.ttnn.median_kernel)
    if median_kernel % 2 == 0:
        raise ValueError("median_kernel must be odd.")
    few_steps = int(args.few_steps) if args.few_steps is not None else None

    tcfg = ThresholdTuningConfig(
        win=int(cfg.windowing.Win),
        wout=int(cfg.windowing.Wout),
        median_kernel=median_kernel,
        few_steps=few_steps,
    )

    Tg = tune_threshold_Tg(
        ttnn=ttnn,
        Eg=mse_val_np,
        Tbase=float(Tbase),
        cfg=tcfg,
        device=device,
    )
    logger.info("Tuned threshold: Tg=%.6f (Tbase=%.6f)", float(Tg), float(Tbase))
    logger.info("Tuning knobs: median_kernel=%d few_steps=%s", median_kernel, str(few_steps))

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": str(args.config),
        "dataset": {
            "parquet_normal": str(cfg.data.processed_normal_path),
            "parquet_attack": str(cfg.data.processed_attack_path),
            "m": int(artifacts["m"]),
            "mse": int(artifacts["mse"]),
            "mac": int(artifacts["mac"]),
        },
        "windowing": {
            "Win": int(cfg.windowing.Win),
            "Wout": int(cfg.windowing.Wout),
            "H": int(cfg.windowing.H),
            "S": int(cfg.windowing.S),
            "label_agg": str(cfg.windowing.label_agg),
        },
        "detection": {
            "Wanom": int(cfg.detection.Wanom),
            "Wgrace": int(cfg.detection.Wgrace),
        },
        "checkpoints": {
            "wdnn": str(wdnn_path),
            "ttnn_section0": str(ttnn_path),
        },
        "thresholds": {
            "Tbase": float(Tbase),
            "Tg_section0": float(Tg),
            "median_kernel": int(median_kernel),
            "few_steps": few_steps,
        },
        "notes": "G=1 (single section) for now; explain sectioning choice in report.",
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[OK] Saved thresholds JSON: %s", out_path.resolve())


if __name__ == "__main__":
    main()
