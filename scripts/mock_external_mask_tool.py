from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock external mask predictor for Agent tool integration tests.")
    parser.add_argument("--pre", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["localization", "damage"], required=True)
    args = parser.parse_args()

    pre = Image.open(args.pre).convert("RGB")
    post = Image.open(args.post).convert("RGB")
    pre_arr = np.asarray(pre, dtype=np.int16)
    post_arr = np.asarray(post, dtype=np.int16)
    diff = np.abs(pre_arr - post_arr).mean(axis=2)

    if args.mode == "localization":
        mask = (diff > 10).astype(np.uint8)
    else:
        mask = np.zeros(diff.shape, dtype=np.uint8)
        mask[diff > 15] = 2
        mask[diff > 35] = 3
        mask[diff > 65] = 4

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)


if __name__ == "__main__":
    main()
