from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_text_only_user_text(row: Dict[str, Any]) -> str:
    convs = row["conversations"]
    user_text = convs[0]["value"]
    meta = row.get("meta", {})
    image_paths = row.get("images", [])
    return (
        f"{user_text}\n\n"
        "你无法直接读取图像，请仅基于下面的结构化视觉分析结果生成回答。\n"
        f"图像路径：{json.dumps(image_paths, ensure_ascii=False)}\n"
        f"结构化视觉分析结果：\n{json.dumps(meta, ensure_ascii=False, indent=2)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="将项目指令数据导出为 LLaMA-Factory 格式。")
    parser.add_argument("--input", required=True, help="输入 jsonl")
    parser.add_argument("--output", required=True, help="输出 json")
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="导出文本 SFT 格式，不包含 images 字段；仅在使用纯文本 Qwen 后端时开启。",
    )
    args = parser.parse_args()

    items = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            convs = row["conversations"]
            assistant_text = convs[1]["value"]
            if args.text_only:
                item = {
                    "messages": [
                        {"role": "user", "content": build_text_only_user_text(row)},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
            else:
                item = {
                    "messages": [
                        {"role": "user", "content": "<image><image>" + convs[0]["value"]},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "images": row.get("images", []),
                }
            items.append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(items)} 条样本至 {out}")


if __name__ == "__main__":
    main()
