from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def build_markdown_report(summary: Dict[str, Any], mllm_text: str, artifacts: Dict[str, str] | None = None) -> str:
    """生成 Markdown 报告。"""
    artifacts = artifacts or {}
    damage = summary.get("damage", {})
    class_stats = damage.get("class_stats", {})

    lines = [
        "# xBD 灾害建筑物损毁评估报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 一、总体结论",
        "",
        f"- 风险等级：**{summary.get('risk_level', '未知')}**",
        f"- 灾前/灾后粗略变化比例：**{summary.get('change_ratio', 0.0):.2%}**",
        f"- 严重及完全损毁比例：**{damage.get('severe_damage_ratio', 0.0):.2%}**",
        "",
        "## 二、损毁等级统计",
        "",
        "| 等级 | 像素数量 | 占建筑物区域比例 |",
        "|---|---:|---:|",
    ]

    for cls in ["1", "2", "3", "4"]:
        info = class_stats.get(cls, {})
        lines.append(f"| {info.get('name', cls)} | {info.get('pixel_count', 0)} | {info.get('ratio', 0.0):.2%} |")

    lines.extend([
        "",
        "## 三、多模态模型分析",
        "",
        mllm_text.strip(),
        "",
        "## 四、主要输出文件",
        "",
    ])

    for name, path in artifacts.items():
        lines.append(f"- {name}: `{path}`")

    boxes = summary.get("damage_bboxes", [])
    if boxes:
        lines.extend(["", "## 五、候选损毁区域", "", "| 序号 | x1 | y1 | x2 | y2 | 面积 |", "|---:|---:|---:|---:|---:|---:|"])
        for i, b in enumerate(boxes[:20], start=1):
            lines.append(f"| {i} | {b['x1']} | {b['y1']} | {b['x2']} | {b['y2']} | {b['area']} |")

    lines.extend([
        "",
        "## 六、注意事项",
        "",
        "本报告由自动化模型生成，适合用于灾情初筛与辅助判读。对于救援、保险定损或官方灾情发布等高风险决策，应结合人工判读、高分辨率影像和现场信息进行复核。",
        "",
    ])
    return "\n".join(lines)
