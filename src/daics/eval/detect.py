from __future__ import annotations

"""
Detection utilities (paper-aligned)

Paper detection ideas:
- WDNN produces predictions -> compute prediction error MSE_{g,t} (per window sample)
- Threshold Tg decides whether a window is anomalous
- Wanom: "anomaly waiting time" => trigger only after Wanom consecutive anomalies
- Wgrace: optional grace/cooldown => avoid re-triggering multiple alarms too close

We implement:
- raw_pred[t] = 1 if mse[t] > Tg else 0
- alarm[t] is raised only when we have Wanom consecutive raw_pred=1
- If Wgrace>0, after an alarm is raised we won't raise another one for Wgrace steps
  (cooldown), which is a pragmatic implementation of the paper's grace time idea.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """
    Paper parameters:
      Wanom: consecutive windows required to raise an alarm
      Wgrace: cooldown steps after an alarm to avoid spamming alarms
    """
    Wanom: int = 30
    Wgrace: int = 0


def threshold_scores(scores: np.ndarray, Tg: float) -> np.ndarray:
    """
    Convert continuous anomaly scores into raw binary predictions.

    Args:
      scores: (N,) array of MSE scores (one per window sample)
      Tg: threshold

    Returns:
      raw_pred: (N,) int64 in {0,1}
    """
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    raw = (s > float(Tg)).astype(np.int64)
    return raw


def apply_wanom(raw_pred: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """
    Apply Wanom (+ Wgrace) logic to raw predictions.

    Wanom logic:
      - We count consecutive raw_pred==1
      - Raise alarm only when count >= Wanom
      - If raw_pred==0, reset count

    Wgrace logic (cooldown):
      - After an alarm is raised, start a cooldown counter of length Wgrace
      - During cooldown, we suppress *new* alarms (we still update consecutive count)
      - This avoids many alarms in a short burst

    Returns:
      alarm: (N,) int64 in {0,1}
    """
    raw = np.asarray(raw_pred, dtype=np.int64).reshape(-1)

    Wanom = int(cfg.Wanom)
    if Wanom <= 0:
        raise ValueError("Wanom must be >= 1")

    Wgrace = int(cfg.Wgrace)
    if Wgrace < 0:
        raise ValueError("Wgrace must be >= 0")

    alarm = np.zeros_like(raw, dtype=np.int64)

    consec = 0
    cooldown = 0

    for t in range(len(raw)):
        if raw[t] == 1:
            consec += 1
        else:
            consec = 0

        # cooldown decreases every step
        if cooldown > 0:
            cooldown -= 1

        # raise alarm only if:
        #  - consecutive anomalies reached Wanom
        #  - and we're not in cooldown
        if consec >= Wanom and cooldown == 0:
            alarm[t] = 1
            if Wgrace > 0:
                cooldown = Wgrace

    return alarm


def detect_from_scores(scores: np.ndarray, Tg: float, cfg: DetectionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience:
      scores -> raw_pred -> alarm

    Returns:
      raw_pred, alarm (both (N,) int64)
    """
    raw = threshold_scores(scores, Tg)
    alarm = apply_wanom(raw, cfg)
    return raw, alarm
