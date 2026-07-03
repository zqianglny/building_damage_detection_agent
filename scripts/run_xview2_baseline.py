from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official xView2 baseline inference and emit project-compatible masks.")
    parser.add_argument("--repo", required=True, help="Path to DIUx-xView/xView2_baseline repository")
    parser.add_argument("--pre", required=True, help="Pre-disaster image")
    parser.add_argument("--post", required=True, help="Post-disaster image")
    parser.add_argument("--localization-weights", required=True, help="Localization model weights .h5")
    parser.add_argument("--classification-weights", required=True, help="Damage classification model weights .hdf5")
    parser.add_argument("--output", required=True, help="Output mask path")
    parser.add_argument("--mode", choices=["localization", "damage"], required=True)
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw xView2 damage output beside output")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    pre = Path(args.pre).expanduser().resolve()
    post = Path(args.post).expanduser().resolve()
    loc_weights = Path(args.localization_weights).expanduser().resolve()
    cls_weights = Path(args.classification_weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    inference_sh = repo / "utils" / "inference.sh"
    _require_file(inference_sh, "xView2 inference.sh")
    _require_file(pre, "pre image")
    _require_file(post, "post image")
    _require_file(loc_weights, "localization weights")
    _require_file(cls_weights, "classification weights")

    raw_output = output if args.mode == "damage" else output.with_name(output.stem + "_xview2_raw_damage.png")
    cmd = [
        "bash",
        str(inference_sh),
        "-x",
        str(repo),
        "-i",
        str(pre),
        "-p",
        str(post),
        "-l",
        str(loc_weights),
        "-c",
        str(cls_weights),
        "-o",
        str(raw_output),
        "-y",
    ]
    completed = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=args.timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "xView2 baseline inference failed\n"
            f"returncode={completed.returncode}\n"
            f"stdout={completed.stdout[-4000:]}\n"
            f"stderr={completed.stderr[-4000:]}"
        )
    _require_file(raw_output, "xView2 raw output mask")

    arr = np.asarray(Image.open(raw_output))
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if args.mode == "localization":
        loc = (arr > 0).astype(np.uint8)
        Image.fromarray(loc).save(output)
        if not args.keep_raw and raw_output != output and raw_output.exists():
            raw_output.unlink()
    else:
        dmg = np.clip(arr, 0, 4).astype(np.uint8)
        Image.fromarray(dmg).save(output)

    print(f"mode={args.mode}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
