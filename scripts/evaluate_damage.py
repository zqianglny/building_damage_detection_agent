from __future__ import annotations

import argparse
import json
from pathlib import Path

from rs_xbd_agent.eval.metrics import evaluate_pairs, sorted_pngs


def main() -> None:
    parser = argparse.ArgumentParser(description="评估损毁等级 mask。")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--out_file", default=None)
    args = parser.parse_args()

    pred_paths = sorted_pngs(args.pred_dir)
    gt_paths = sorted_pngs(args.gt_dir)
    if len(pred_paths) != len(gt_paths):
        raise ValueError(f"预测和标注数量不一致：{len(pred_paths)} vs {len(gt_paths)}")

    metrics = evaluate_pairs(pred_paths, gt_paths, num_classes=args.num_classes)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.out_file:
        Path(args.out_file).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
