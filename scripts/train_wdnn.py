from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from daics.config import load_config
from daics.data.dataloaders import make_dataloaders_paper
from daics.models.wdnn import WDNNConfig
from daics.train.wdnn_trainer import WDNNTrainConfig, train_wdnn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    train_loader, val_loader, _test_loader, artifacts = make_dataloaders_paper(cfg)

    mse = int(artifacts["mse"])
    m = int(artifacts["m"])

    # Start simple: G=1 (all sensors in one output section).
    # Later we can encode PLC grouping to match paper modularity.
    sensor_sections: List[List[int]] = [list(range(mse))]

    wdnn_cfg = WDNNConfig(
        Win=int(cfg.windowing.Win),
        Wout=int(cfg.windowing.Wout),
        H=int(cfg.windowing.H),  # kept in cfg but not used inside model directly
        m=m,
        mse=mse,
        dl1=int(getattr(cfg.wdnn, "dl1", 3 * int(cfg.windowing.Win))),
        dl2=int(getattr(cfg.wdnn, "dl2", 3 * m)),
        dl4=int(getattr(cfg.wdnn, "dl4", 80)),
        cl1_channels=int(getattr(cfg.wdnn, "cl1_channels", 64)),
        cl1_kernel=int(getattr(cfg.wdnn, "cl1_kernel", 2)),
        cl2_channels=int(getattr(cfg.wdnn, "cl2_channels", 128)),
        cl2_kernel=int(getattr(cfg.wdnn, "cl2_kernel", 2)),
        leaky_slope=float(getattr(cfg.wdnn, "leaky_slope", 0.01)),
    )

    train_cfg = WDNNTrainConfig(
        lr=float(cfg.wdnn.lr),
        epochs=int(cfg.wdnn.epochs),
        weight_decay=float(getattr(cfg.wdnn, "weight_decay", 0.0)),
        grad_clip=float(getattr(cfg.wdnn, "grad_clip", 1.0)),
        seed=int(cfg.train.seed),
        out_dir=str(cfg.wdnn.out_dir),
    )

    print("[INFO] WDNN training config loaded.")
    print(f"  Win={wdnn_cfg.Win} Wout={wdnn_cfg.Wout} H={cfg.windowing.H} m={wdnn_cfg.m} mse={wdnn_cfg.mse}")
    print(f"  lr={train_cfg.lr} epochs={train_cfg.epochs} out_dir={train_cfg.out_dir}")

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

    print(f"[OK] Done. Checkpoints saved in: {out_dir}")


if __name__ == "__main__":
    main()
