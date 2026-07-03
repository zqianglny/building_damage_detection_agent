from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """计算多类别混淆矩阵。"""
    pred = pred.astype(np.int64).ravel()
    gt = gt.astype(np.int64).ravel()
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(num_classes * gt[valid] + pred[valid], minlength=num_classes ** 2)
    return hist.reshape(num_classes, num_classes)


def compute_metrics_from_hist(hist: np.ndarray) -> Dict[str, object]:
    eps = 1e-7
    tp = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)

    iou = tp / (gt_sum + pred_sum - tp + eps)
    precision = tp / (pred_sum + eps)
    recall = tp / (gt_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    oa = tp.sum() / (hist.sum() + eps)

    return {
        "OA": float(oa),
        "mIoU_all": float(np.nanmean(iou)),
        "mF1_all": float(np.nanmean(f1)),
        "IoU_per_class": {str(i): float(v) for i, v in enumerate(iou)},
        "F1_per_class": {str(i): float(v) for i, v in enumerate(f1)},
    }


def evaluate_pairs(pred_paths: Iterable[str | Path], gt_paths: Iterable[str | Path], num_classes: int = 5) -> Dict[str, object]:
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"shape 不一致：{pred_path} {pred.shape} vs {gt_path} {gt.shape}")
        hist += confusion_matrix(pred, gt, num_classes=num_classes)
    metrics = compute_metrics_from_hist(hist)
    metrics["confusion_matrix"] = hist.tolist()
    return metrics


def sorted_pngs(path: str | Path) -> List[Path]:
    return sorted(Path(path).glob("*.png"))
