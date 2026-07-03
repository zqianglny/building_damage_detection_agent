from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description="生成一组用于跑通流程的玩具样例。")
    parser.add_argument("--out_dir", default="outputs/toy_sample")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = args.size

    pre = Image.new("RGB", (size, size), (80, 100, 90))
    post = Image.new("RGB", (size, size), (80, 100, 90))
    pre_draw = ImageDraw.Draw(pre)
    post_draw = ImageDraw.Draw(post)

    pre_mask = Image.new("L", (size, size), 0)
    dmg_mask = Image.new("L", (size, size), 0)
    pre_mask_draw = ImageDraw.Draw(pre_mask)
    dmg_draw = ImageDraw.Draw(dmg_mask)

    buildings = [
        ((60, 80, 140, 160), 1),
        ((190, 70, 270, 150), 2),
        ((320, 100, 410, 190), 3),
        ((120, 260, 220, 360), 4),
        ((300, 300, 420, 410), 1),
    ]

    for box, cls in buildings:
        pre_draw.rectangle(box, fill=(190, 190, 180), outline=(40, 40, 40))
        pre_mask_draw.rectangle(box, fill=1)
        if cls == 1:
            color = (188, 188, 178)
        elif cls == 2:
            color = (170, 150, 110)
        elif cls == 3:
            color = (120, 80, 60)
        else:
            color = (60, 45, 40)
        post_draw.rectangle(box, fill=color, outline=(30, 30, 30))
        dmg_draw.rectangle(box, fill=cls)

    # 增加一些灾后碎片纹理
    rng = np.random.default_rng(42)
    post_arr = np.asarray(post).copy()
    for _ in range(1200):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        post_arr[y, x] = rng.integers(30, 200, size=3)
    post = Image.fromarray(post_arr.astype(np.uint8))

    pre.save(out_dir / "pre.png")
    post.save(out_dir / "post.png")
    pre_mask.save(out_dir / "pre_building_mask.png")
    dmg_mask.save(out_dir / "post_damage_mask.png")
    print(f"玩具样例已保存至：{out_dir}")


if __name__ == "__main__":
    main()
