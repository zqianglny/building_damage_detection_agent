from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from rs_xbd_agent.data.xbd_parser import DAMAGE_NAME_ZH
from rs_xbd_agent.utils.image import read_mask


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(items: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _damage_sentence(class_stats: Dict[str, dict]) -> str:
    parts = []
    for cls in [1, 2, 3, 4]:
        info = class_stats.get(str(cls), {})
        ratio = info.get("ratio", 0.0)
        if ratio > 0:
            parts.append(f"{DAMAGE_NAME_ZH[cls]}约占{ratio:.1%}")
    return "，".join(parts) if parts else "未检测到明确建筑物损毁标注"


def _connected_components(mask: np.ndarray) -> Dict[str, Any]:
    """统计二值 mask 的简单连通域信息，避免依赖 scipy/opencv。"""
    coords = np.argwhere(mask.astype(bool))
    total = int(coords.shape[0])
    if total == 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
        }

    foreground = {tuple(coord) for coord in coords.tolist()}
    component_sizes: List[int] = []
    while foreground:
        seed = foreground.pop()
        stack = [seed]
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if ny == cy and nx == cx:
                        continue
                    neighbor = (ny, nx)
                    if neighbor in foreground:
                        foreground.remove(neighbor)
                        stack.append(neighbor)
        component_sizes.append(size)

    component_sizes.sort(reverse=True)
    largest = int(component_sizes[0]) if component_sizes else 0
    return {
        "component_count": len(component_sizes),
        "largest_component_pixels": largest,
        "largest_component_ratio": largest / total if total else 0.0,
    }


def _damage_stats(post_damage_mask: np.ndarray) -> Dict[str, Any]:
    total = int((post_damage_mask > 0).sum())
    stats = {}
    for cls in [1, 2, 3, 4]:
        count = int((post_damage_mask == cls).sum())
        stats[str(cls)] = {
            "name": DAMAGE_NAME_ZH[cls],
            "ratio": count / total if total else 0.0,
            "pixel_count": count,
        }
    severe_mask = (post_damage_mask == 3) | (post_damage_mask == 4)
    severe_ratio = stats["3"]["ratio"] + stats["4"]["ratio"]
    destroyed_ratio = stats["4"]["ratio"]
    risk = "高" if severe_ratio > 0.25 else "中" if severe_ratio > 0.05 else "低"
    return {
        "total_building_pixels": total,
        "damage_stats": stats,
        "severe_damage_ratio": severe_ratio,
        "destroyed_ratio": destroyed_ratio,
        "risk_level": risk,
        "severe_components": _connected_components(severe_mask),
    }


def _class_line(stats: Dict[str, dict]) -> str:
    parts = []
    for cls in [1, 2, 3, 4]:
        info = stats[str(cls)]
        parts.append(f"{info['name']} {info['pixel_count']} 像素，占 {info['ratio']:.1%}")
    return "；".join(parts)


def _review_focus(meta: Dict[str, Any]) -> str:
    severe_ratio = meta["severe_damage_ratio"]
    destroyed_ratio = meta["destroyed_ratio"]
    components = meta["severe_components"]
    largest_ratio = components["largest_component_ratio"]

    if severe_ratio > 0.25:
        if largest_ratio > 0.5:
            return "严重或完全损毁像素占比较高，且最大连通区域占严重损毁区域比例较大，应优先复核该集中区域是否与灾后影像中的坍塌、屋顶缺失或碎片纹理一致。"
        return "严重或完全损毁像素占比较高，但严重区域较分散，应逐块复核是否存在小目标误标或边界偏移。"
    if severe_ratio > 0.05:
        return "存在一定比例严重或完全损毁区域，建议重点检查这些区域与周边未损毁建筑之间的边界是否合理。"
    if destroyed_ratio > 0:
        return "总体严重损毁占比较低，但仍存在完全损毁标注，建议人工抽查完全损毁区域，避免小面积高等级损毁被报告低估。"
    return "严重与完全损毁占比较低，复核重点应放在轻微损毁与未损毁之间的边界，以及是否存在漏标建筑物。"


def _build_basic_answer(meta: Dict[str, Any]) -> str:
    damage_text = _damage_sentence(meta["damage_stats"])
    return (
        f"根据灾后建筑物损毁标注统计，该区域建筑物损毁风险为{meta['risk_level']}。"
        f"损毁组成方面，{damage_text}。"
        "判断依据包括灾前灾后建筑物区域变化、灾后损毁等级分布以及严重损毁区域的空间集中程度。"
    )


def _build_review_answer(meta: Dict[str, Any]) -> str:
    severe = meta["severe_damage_ratio"]
    components = meta["severe_components"]
    return (
        "【复核结论】\n"
        f"该样本的建筑物损毁风险评估为{meta['risk_level']}。该结论主要依据 xBD 灾后建筑物损毁标注统计得到，"
        "多模态模型在此任务中应作为解释与复核助手，而不是替代像素级建筑物检测或损毁分割模型。\n\n"
        "【主要依据】\n"
        f"建筑物标注区域约 {meta['total_building_pixels']} 像素；损毁等级分布为：{_class_line(meta['damage_stats'])}。"
        f"其中严重损毁与完全损毁合计占 {severe:.1%}，严重损毁连通区域数量为 {components['component_count']}，"
        f"最大严重损毁连通区域占严重损毁区域 {components['largest_component_ratio']:.1%}。"
        "这些统计可用于判断损毁范围、损毁强度以及高等级损毁是否集中。\n\n"
        "【人工复核建议】\n"
        f"{_review_focus(meta)}复核时应同时查看灾前图像、灾后图像和损毁叠加结果；"
        "若图像纹理变化与 mask 等级明显不一致，应以专门分割模型输出或人工标注为准，并记录为疑似误检、漏检或等级混淆样本。"
    )


def build_instruction_item(sample: dict, image_dir: str | Path, style: str = "basic") -> dict:
    """根据一个 xBD 样本索引构造多模态指令样本。"""
    image_dir = Path(image_dir)
    post_damage_mask = read_mask(sample["post_damage_mask"])
    meta = _damage_stats(post_damage_mask)

    if style == "review":
        user_text = (
            "请作为遥感灾害评估复核助手，对这组灾前和灾后遥感图像进行建筑物损毁复核。"
            "请结合图像内容和已给出的损毁统计，输出复核结论、主要依据和人工复核建议。"
        )
        assistant_text = _build_review_answer(meta)
        task = "xbd_damage_review"
    else:
        user_text = "请对这组灾前和灾后遥感图像进行建筑物损毁评估，并说明主要依据。"
        assistant_text = _build_basic_answer(meta)
        task = "xbd_damage_assessment"

    pre_img = str(image_dir / sample["pre_image_name"])
    post_img = str(image_dir / sample["post_image_name"])

    return {
        "id": sample["sample_id"],
        "images": [pre_img, post_img],
        "conversations": [
            {
                "from": "human",
                "value": user_text,
            },
            {
                "from": "gpt",
                "value": assistant_text,
            },
        ],
        "meta": {
            "task": task,
            **meta,
        },
    }


def build_instruction_dataset(mask_index: str | Path, image_dir: str | Path, out_file: str | Path, style: str = "basic") -> None:
    items: List[dict] = []
    for sample in read_jsonl(mask_index):
        # 只有灾后标注包含损毁等级 mask，适合构造双时相损毁评估样本。
        if sample.get("pre_image_name") and sample.get("post_image_name") and sample.get("post_damage_mask"):
            items.append(build_instruction_item(sample, image_dir=image_dir, style=style))
    write_jsonl(items, out_file)
