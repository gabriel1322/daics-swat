from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders
from daics.eval.mse import compute_section_mse_series
from daics.eval.thresholds import ThresholdTuningConfig, compute_Tbase, load_ttnn_checkpoint, tune_threshold_Tg
from daics.train.wdnn_trainer import load_wdnn_checkpoint  # déjà chez toi (utilisé dans train_ttnn)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tune_thresholds")


def _pick_device(device_str: str) -> torch.device:
    """
    Keep it explicit and reproducible.
    - 'cpu' => cpu
    - 'cuda' => cuda if available else cpu
    - 'auto' => cuda if available else cpu
    """
    d = (device_str or "auto").lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    ap.add_argument("--wdnn_ckpt", required=True, help="runs/wdnn/best.pt (or last.pt)")
    ap.add_argument("--ttnn_ckpt", default="runs/ttnn/section_0.pt", help="runs/ttnn/section_0.pt")
    ap.add_argument("--out", default="runs/thresholds.json", help="Output JSON file with Tbase/Tg")
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

    # Dataloaders (paper split semantics already handled in make_dataloaders)
    _train_loader, val_loader, _test_loader, artifacts = make_dataloaders(
        parquet_path=cfg.data.processed_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
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
    logger.info(
        "Validation MSE series length: %d",
        int(len(mse_val_np)),
    )
    logger.info(
        "Validation MSE mean/std: %.6f / %.6f  => Tbase=%.6f",
        float(mse_val_np.mean()), float(mse_val_np.std(ddof=0)), float(Tbase),
    )

    # Load TTNN
    ttnn_path = Path(args.ttnn_ckpt)
    ttnn = load_ttnn_checkpoint(str(ttnn_path), device=device)
    logger.info("Loaded TTNN checkpoint: %s", ttnn_path.resolve())

    # Tune Tg using Algorithm 1 style logic
    tcfg = ThresholdTuningConfig(
        win=int(cfg.windowing.Win),
        wout=int(cfg.windowing.Wout),
        median_kernel=int(cfg.train.ttnn.median_kernel),  # paper default 59 in your YAML
    )
    Tg = tune_threshold_Tg(
        ttnn=ttnn,
        Eg=mse_val_np,
        Tbase=float(Tbase),
        cfg=tcfg,
        device=device,
    )
    logger.info("Tuned threshold: Tg=%.6f (Tbase=%.6f)", float(Tg), float(Tbase))

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": str(args.config),
        "dataset": {
            "parquet": str(cfg.data.processed_path),
            "m": int(artifacts["m"]),
            "mse": int(artifacts["mse"]),
            "mac": int(artifacts["mac"]),
        },
        "windowing": {
            "Win": int(cfg.windowing.Win),
            "Wout": int(cfg.windowing.Wout),
            "H": int(cfg.windowing.H),
            "S": int(cfg.windowing.S),
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
            "median_kernel": int(tcfg.median_kernel),
        },
        "notes": "G=1 (single section) for now; explain sectioning choice in report.",
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[OK] Saved thresholds JSON: %s", out_path.resolve())


if __name__ == "__main__":
    main()
