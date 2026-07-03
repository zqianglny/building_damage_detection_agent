from __future__ import annotations

import shlex
from pathlib import Path


def make_xview2_baseline_command(
    *,
    mode: str,
    repo: str,
    localization_weights: str,
    classification_weights: str,
    timeout: int = 1800,
    conda_env: str = "",
) -> str:
    project_root = Path(__file__).resolve().parents[3]
    script = project_root / "scripts" / "run_xview2_baseline.py"
    executable = ["python", shlex.quote(str(script))]
    if conda_env:
        executable = ["conda", "run", "-n", shlex.quote(conda_env), "python", shlex.quote(str(script))]
    return " ".join(
        [
            *executable,
            "--repo",
            shlex.quote(repo),
            "--pre",
            "{pre_image}",
            "--post",
            "{post_image}",
            "--localization-weights",
            shlex.quote(localization_weights),
            "--classification-weights",
            shlex.quote(classification_weights),
            "--output",
            "{output_mask}",
            "--mode",
            shlex.quote(mode),
            "--timeout",
            shlex.quote(str(timeout)),
        ]
    )
