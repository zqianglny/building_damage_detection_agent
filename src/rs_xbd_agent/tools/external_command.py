from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

from rs_xbd_agent.agent.state import AgentState


def render_command(command_template: str, values: Dict[str, Any]) -> tuple[list[str], str]:
    """Render a command template safely enough for shell=False execution.

    Placeholder values are shell-quoted before shlex parsing so paths with spaces
    remain one argument. Do not add extra quotes around placeholders in config.
    """

    quoted_values = {key: shlex.quote(str(value)) for key, value in values.items()}
    rendered = command_template.format(**quoted_values)
    return shlex.split(rendered), rendered


def run_external_mask_command(
    *,
    command_template: str,
    state: AgentState,
    output_mask: str | Path,
    timeout: int = 1800,
) -> dict[str, Any]:
    output_mask = Path(output_mask)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    values = {
        "pre_image": state.pre_image,
        "post_image": state.post_image,
        "pre_mask": state.pre_mask or "",
        "post_damage_mask": state.post_damage_mask or "",
        "out_dir": str(output_mask.parent),
        "output_mask": str(output_mask),
    }
    argv, rendered = render_command(command_template, values)
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)

    return {
        "command": rendered,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output_mask": str(output_mask),
        "output_exists": output_mask.exists(),
    }
