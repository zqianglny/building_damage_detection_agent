from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


DAMAGE_COLORS = {
    0: (0, 0, 0),          # 背景
    1: (0, 255, 0),        # 未损毁
    2: (255, 255, 0),      # 轻微损毁
    3: (255, 128, 0),      # 严重损毁
    4: (255, 0, 0),        # 完全损毁
}


def read_rgb(path: str | Path) -> np.ndarray:
    """读取 RGB 图像，返回 HWC uint8。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def read_mask(path: str | Path) -> np.ndarray:
    """读取单通道 mask。"""
    mask = Image.open(path).convert("L")
    return np.asarray(mask)


def save_rgb(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) / (hi - lo) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def image_absdiff(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """计算灾前/灾后图像的灰度差分热力图。"""
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(pre_gray, post_gray)
    return diff


def binary_change_mask(pre: np.ndarray, post: np.ndarray, threshold: int = 35) -> np.ndarray:
    """基于图像差分的粗略变化 mask。真实项目中建议替换为训练好的变化检测模型。"""
    diff = image_absdiff(pre, post)
    mask = (diff > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def colorize_damage_mask(mask: np.ndarray) -> np.ndarray:
    """将 0~4 的损毁等级 mask 转成彩色图。"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in DAMAGE_COLORS.items():
        out[mask == cls] = color
    return out


def overlay_mask(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """将彩色 mask 叠加到原图上。"""
    image = image.astype(np.float32)
    mask_rgb = mask_rgb.astype(np.float32)
    active = mask_rgb.sum(axis=-1, keepdims=True) > 0
    blended = image * (1 - alpha) + mask_rgb * alpha
    out = np.where(active, blended, image)
    return out.clip(0, 255).astype(np.uint8)


def draw_polygons_to_mask(polygons: Iterable[list[Tuple[float, float]]], size: Tuple[int, int], value: int) -> np.ndarray:
    """将多边形填充为 mask。size 为 (width, height)。"""
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in poly], fill=int(value))
    return np.asarray(img, dtype=np.uint8)


def connected_components_bboxes(mask: np.ndarray, min_area: int = 64) -> list[dict]:
    """提取二值 mask 中连通域 bbox。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append({"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h), "area": int(area)})
    return boxes
