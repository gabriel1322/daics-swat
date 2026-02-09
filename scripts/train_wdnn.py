from __future__ import annotations

"""
Train WDNN (DAICS paper-aligned, SWaT)

Key points:
- Uses the processed parquet produced by preprocess_swat.py (features + 'label')
- Uses make_dataloaders(...) which yields:
    x   : (B, Win, m)
    y   : (B, Wout, mse)   [sensors only]
    lab : (B,)             [aggregated label over Wout]
- Trains unsupervised on NORMAL-only splits (train/val), as in paper.
- Saves checkpoints: runs/wdnn/best.pt and runs/wdnn/last.pt
"""

import argparse
from pathlib import Path
from typing import List

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper_strict
from daics.models.wdnn import WDNNConfig
from daics.train.wdnn_trainer import WDNNTrainConfig, train_wdnn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Build loaders (paper-like one-class split is inside dataloaders)
    train_loader, val_loader, test_loader, artifacts = make_dataloaders_paper_strict(
        normal_parquet_path=cfg.data.processed_normal_path,
        merged_parquet_path=cfg.data.processed_merged_path,
        window_cfg=cfg.windowing,
        split_cfg=cfg.splits,
        loader_cfg=cfg.loader,
        label_col=cfg.data.label_col,
        test_mode=str(cfg.eval.test_mode),
    )

    mse = int(artifacts["mse"])
    m = int(artifacts["m"])
    Win = int(cfg.windowing.Win)
    Wout = int(cfg.windowing.Wout)

    # Start simple (and stable): G=1 section contains all sensors.
    # Paper uses multiple sections (PLC-based) for scalability; we can add later.
    sensor_sections: List[List[int]] = [list(range(mse))]

    # ---- WDNN model config (paper Table 4 defaults)
    wdnn_yml = cfg.train.wdnn  # <--- IMPORTANT: nested under cfg.train now

    wdnn_cfg = WDNNConfig(
        Win=Win,
        Wout=Wout,
        m=m,
        mse=mse,
        dl1=int(wdnn_yml.dl1) if wdnn_yml.dl1 is not None else 3 * Win,   # paper: 3*Win
        dl2=int(wdnn_yml.dl2) if wdnn_yml.dl2 is not None else 3 * m,     # paper: 3*m
        dl4=int(wdnn_yml.dl4),                                            # paper: 80
        cl1_channels=int(wdnn_yml.cl1_channels),
        cl1_kernel=int(wdnn_yml.cl1_kernel),
        cl2_channels=int(wdnn_yml.cl2_channels),
        cl2_kernel=int(wdnn_yml.cl2_kernel),
        leaky_slope=float(wdnn_yml.leaky_slope),
    )

    # ---- Training config (paper: SGD + MSE)
    train_cfg = WDNNTrainConfig(
        lr=float(wdnn_yml.lr),                   # paper SWaT: 0.001
        epochs=int(wdnn_yml.epochs),             # paper SWaT: 100
        weight_decay=float(wdnn_yml.weight_decay),
        grad_clip=float(wdnn_yml.grad_clip) if wdnn_yml.grad_clip is not None else None,
        seed=int(cfg.train.seed),
        out_dir=str(wdnn_yml.out_dir),
        save_best=True,
        save_last=True,
    )

    print("[INFO] WDNN training config")
    print(f"  Device        : {cfg.train.device}")
    print(f"  Data parquet  : {cfg.data.processed_normal_path}")
    print(f"  Win/Wout/H/S  : {cfg.windowing.Win}/{cfg.windowing.Wout}/{cfg.windowing.H}/{cfg.windowing.S}")
    print(f"  Wanom/Wgrace  : {cfg.detection.Wanom}/{cfg.detection.Wgrace}")
    print(f"  m/mse/mac     : {artifacts['m']}/{artifacts['mse']}/{artifacts['mac']}")
    print(f"  Sections (G)  : {len(sensor_sections)}  -> sizes {[len(s) for s in sensor_sections]}")
    print(f"  Optim         : SGD(lr={train_cfg.lr})")
    print(f"  Epochs        : {train_cfg.epochs}")
    print(f"  Out dir       : {train_cfg.out_dir}")

    _model, art = train_wdnn(
        train_loader=train_loader,
        val_loader=val_loader,
        wdnn_cfg=wdnn_cfg,
        train_cfg=train_cfg,
        sensor_sections=sensor_sections,
        device=str(cfg.train.device),
    )

    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.txt").write_text(
        "WDNN training finished\n"
        f"best_val={art['best_val']}\n"
        f"best_epoch={art['best_epoch']}\n"
        f"config={args.config}\n"
    )

    print(f"[OK] WDNN done. Checkpoints in: {out_dir.resolve()}")
    print(f"     - best: { (out_dir / 'best.pt').resolve() }")
    print(f"     - last: { (out_dir / 'last.pt').resolve() }")


if __name__ == "__main__":
    main()
