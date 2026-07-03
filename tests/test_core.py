from __future__ import annotations

from pathlib import Path

import numpy as np

from rs_xbd_agent.data.xbd_parser import summarize_damage_mask
from rs_xbd_agent.eval.metrics import confusion_matrix, compute_metrics_from_hist


def test_summarize_damage_mask():
    mask = np.array([[0, 1, 1], [2, 3, 4]], dtype=np.uint8)
    summary = summarize_damage_mask(mask)
    assert summary["total_building_pixels"] == 5
    assert summary["severe_damage_pixels"] == 2


def test_metrics_perfect():
    gt = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    pred = gt.copy()
    hist = confusion_matrix(pred, gt, num_classes=5)
    metrics = compute_metrics_from_hist(hist)
    assert metrics["OA"] > 0.99
