from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from rs_xbd_agent.data.xbd_parser import json_to_masks
from rs_xbd_agent.utils.image import save_mask


def guess_pair_names(label_name: str) -> tuple[str | None, str | None, str]:
    """根据 xBD 文件名推断灾前/灾后图片名与样本 id。"""
    stem = Path(label_name).stem
    sample_id = stem.replace("_pre_disaster", "").replace("_post_disaster", "")
    pre_image_name = f"{sample_id}_pre_disaster.png"
    post_image_name = f"{sample_id}_post_disaster.png"
    return pre_image_name, post_image_name, sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description="将 xBD JSON 标注转换为 mask。")
    parser.add_argument("--labels_dir", required=True, help="xBD labels 目录")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--image_size", nargs=2, type=int, default=None, help="图像宽高，例如 1024 1024")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    loc_dir = out_dir / "localization"
    dmg_dir = out_dir / "damage"
    loc_dir.mkdir(parents=True, exist_ok=True)
    dmg_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("*.json"))
    index_items = []

    for json_path in tqdm(json_files, desc="building masks"):
        image_size = tuple(args.image_size) if args.image_size else None
        loc, dmg, objects = json_to_masks(json_path, image_size=image_size)
        loc_path = loc_dir / f"{json_path.stem}_localization.png"
        dmg_path = dmg_dir / f"{json_path.stem}_damage.png"
        save_mask(loc, loc_path)
        save_mask(dmg, dmg_path)

        pre_img, post_img, sample_id = guess_pair_names(json_path.name)
        index_items.append({
            "sample_id": sample_id,
            "label_name": json_path.name,
            "pre_image_name": pre_img,
            "post_image_name": post_img,
            "localization_mask": str(loc_path),
            "post_damage_mask": str(dmg_path) if "post_disaster" in json_path.stem else None,
            "num_objects": len(objects),
        })

    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for item in index_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成：{len(index_items)} 个标注文件，索引保存至 {index_path}")


if __name__ == "__main__":
    main()
