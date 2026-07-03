from __future__ import annotations

import argparse

from rs_xbd_agent.data.instruction_builder import build_instruction_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 xBD 多模态指令数据。")
    parser.add_argument("--image_dir", required=True, help="xBD images 目录")
    parser.add_argument("--mask_index", required=True, help="build_xbd_masks.py 生成的 index.jsonl")
    parser.add_argument("--out_file", required=True, help="输出 jsonl 文件")
    parser.add_argument(
        "--style",
        choices=["basic", "review"],
        default="basic",
        help="指令数据风格：basic 为原始统计报告，review 为解释与复核三段式数据。",
    )
    args = parser.parse_args()

    build_instruction_dataset(
        mask_index=args.mask_index,
        image_dir=args.image_dir,
        out_file=args.out_file,
        style=args.style,
    )
    print(f"指令数据已保存至：{args.out_file}")


if __name__ == "__main__":
    main()
