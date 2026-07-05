from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


DAMAGE_MAPPING = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
    "background": 0,
}

DAMAGE_NAME_ZH = {
    0: "背景/未标注",
    1: "未损毁",
    2: "轻微损毁",
    3: "严重损毁",
    4: "完全损毁",
}


@dataclass
class XBDObject:
    """xBD 单个建筑物对象。"""

    uid: str
    subtype: str
    polygon: List[Tuple[float, float]]

    @property
    def damage_class(self) -> int:
        return DAMAGE_MAPPING.get(self.subtype, 0)


def _parse_wkt_polygon(wkt: str) -> List[Tuple[float, float]]:
    """解析 xBD 中常见的 WKT Polygon 字段。

    示例：POLYGON ((1 2, 3 4, 5 6, 1 2))
    """
    if not wkt:
        return []
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", wkt)
    vals = [float(x) for x in nums]
    pts = []
    for i in range(0, len(vals) - 1, 2):
        pts.append((vals[i], vals[i + 1]))
    return pts


def _parse_bounds_imcoords(bounds: str) -> List[Tuple[float, float]]:
    """解析 bounds_imcoords 字段。

    xBD 的不同版本中可能使用 "x1,y1,x2,y2,..." 或 WKT。
    """
    if not bounds:
        return []
    if "POLYGON" in bounds.upper():
        return _parse_wkt_polygon(bounds)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", bounds)
    vals = [float(x) for x in nums]
    pts = []
    for i in range(0, len(vals) - 1, 2):
        pts.append((vals[i], vals[i + 1]))
    return pts


def load_xbd_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_xbd_objects(json_path: str | Path) -> List[XBDObject]:
    """解析 xBD JSON 中的建筑物对象。

    兼容字段：features.xy[*].wkt、features.xy[*].properties.bounds_imcoords。
    """
    data = load_xbd_json(json_path)
    features = data.get("features", {})
    xy_features = features.get("xy", []) if isinstance(features, dict) else []

    objects: List[XBDObject] = []
    for idx, feat in enumerate(xy_features):
        props = feat.get("properties", {}) or {}
        uid = str(props.get("uid", props.get("feature_id", idx)))
        subtype = str(props.get("subtype", "no-damage"))

        poly = []
        if "wkt" in feat:
            poly = _parse_wkt_polygon(feat.get("wkt", ""))
        if not poly:
            poly = _parse_bounds_imcoords(props.get("bounds_imcoords", ""))
        if not poly:
            poly = _parse_bounds_imcoords(feat.get("bounds_imcoords", ""))

        if len(poly) >= 3:
            objects.append(XBDObject(uid=uid, subtype=subtype, polygon=poly))
    return objects


def infer_image_size_from_json(json_path: str | Path, default_size: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    """尽量从 metadata 中推断图像尺寸，失败则返回 1024x1024。"""
    data = load_xbd_json(json_path)
    metadata = data.get("metadata", {}) or {}
    width = metadata.get("width") or metadata.get("img_width") or default_size[0]
    height = metadata.get("height") or metadata.get("img_height") or default_size[1]
    return int(width), int(height)


def objects_to_masks(objects: List[XBDObject], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """将建筑物对象转为定位 mask 和损毁等级 mask。

    返回：
    - localization_mask：0 背景，1 建筑物
    - damage_mask：0 背景，1 未损毁，2 轻微，3 严重，4 完全损毁
    """
    loc_img = Image.new("L", size, 0)
    dmg_img = Image.new("L", size, 0)
    loc_draw = ImageDraw.Draw(loc_img)
    dmg_draw = ImageDraw.Draw(dmg_img)

    # 先画低等级，再画高等级。若多边形重叠，高损毁等级覆盖低等级。
    sorted_objects = sorted(objects, key=lambda obj: obj.damage_class)
    for obj in sorted_objects:
        pts = [(float(x), float(y)) for x, y in obj.polygon]
        loc_draw.polygon(pts, fill=1)
        dmg_draw.polygon(pts, fill=int(obj.damage_class))

    return np.asarray(loc_img, dtype=np.uint8), np.asarray(dmg_img, dtype=np.uint8)


def json_to_masks(json_path: str | Path, image_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, List[XBDObject]]:
    objects = parse_xbd_objects(json_path)
    size = image_size or infer_image_size_from_json(json_path)
    loc, dmg = objects_to_masks(objects, size=size)
    return loc, dmg, objects


def summarize_damage_mask(mask: np.ndarray) -> Dict[str, Any]:
    """统计损毁等级像素数量与比例。"""
    total_building = int((mask > 0).sum())
    stats = {}
    for cls, name in DAMAGE_NAME_ZH.items():
        if cls == 0:
            continue
        count = int((mask == cls).sum())
        ratio = count / total_building if total_building > 0 else 0.0
        stats[str(cls)] = {"name": name, "pixel_count": count, "ratio": ratio}
    severe_pixels = int(((mask == 3) | (mask == 4)).sum())
    return {
        "total_building_pixels": total_building,
        "class_stats": stats,
        "severe_damage_pixels": severe_pixels,
        "severe_damage_ratio": severe_pixels / total_building if total_building > 0 else 0.0,
    }
