from __future__ import annotations

from rs_xbd_agent.tools.damage import (
    DiffFallbackDamageTool,
    ExternalCommandDamageTool,
    PlannedExternalDamageTool,
    ProvidedDamageMaskTool,
)
from rs_xbd_agent.tools.xview2_command import make_xview2_baseline_command
from rs_xbd_agent.tools.localization import (
    ExternalCommandLocalizationTool,
    PlannedExternalLocalizationTool,
    ProvidedMaskLocalizationTool,
)


LOCALIZATION_BACKENDS = {
    "provided_mask",
    "gt_mask",
    "external_mask",
    "external_command",
    "xview2_baseline",
}

DAMAGE_BACKENDS = {
    "provided_mask",
    "gt_mask",
    "external_mask",
    "diff_fallback",
    "external_command",
    "xview2_baseline",
    "xview2_strong_baseline",
}


def build_localization_tool(
    backend: str,
    command_template: str = "",
    timeout: int = 1800,
    xview2_repo: str = "",
    xview2_localization_weights: str = "",
    xview2_classification_weights: str = "",
    xview2_conda_env: str = "",
):
    backend = (backend or "provided_mask").strip()
    if backend in {"provided_mask", "gt_mask", "external_mask"}:
        return ProvidedMaskLocalizationTool()
    if backend in {"external_command", "xview2_baseline"}:
        if (
            backend == "xview2_baseline"
            and not command_template
            and xview2_repo
            and xview2_localization_weights
            and xview2_classification_weights
        ):
            command_template = make_xview2_baseline_command(
                mode="localization",
                repo=xview2_repo,
                localization_weights=xview2_localization_weights,
                classification_weights=xview2_classification_weights,
                timeout=timeout,
                conda_env=xview2_conda_env,
            )
        if command_template:
            return ExternalCommandLocalizationTool(
                backend_name=backend, command_template=command_template, timeout=timeout
            )
        return PlannedExternalLocalizationTool(backend_name=backend)
    raise ValueError(f"Unsupported localization backend: {backend}")


def build_damage_tool(
    backend: str,
    command_template: str = "",
    timeout: int = 1800,
    xview2_repo: str = "",
    xview2_localization_weights: str = "",
    xview2_classification_weights: str = "",
    xview2_conda_env: str = "",
):
    backend = (backend or "provided_mask").strip()
    if backend in {"provided_mask", "gt_mask", "external_mask"}:
        return ProvidedDamageMaskTool()
    if backend == "diff_fallback":
        return DiffFallbackDamageTool()
    if backend in {"external_command", "xview2_baseline", "xview2_strong_baseline"}:
        if (
            backend == "xview2_baseline"
            and not command_template
            and xview2_repo
            and xview2_localization_weights
            and xview2_classification_weights
        ):
            command_template = make_xview2_baseline_command(
                mode="damage",
                repo=xview2_repo,
                localization_weights=xview2_localization_weights,
                classification_weights=xview2_classification_weights,
                timeout=timeout,
                conda_env=xview2_conda_env,
            )
        if command_template:
            return ExternalCommandDamageTool(backend_name=backend, command_template=command_template, timeout=timeout)
        return PlannedExternalDamageTool(backend_name=backend)
    raise ValueError(f"Unsupported damage backend: {backend}")
