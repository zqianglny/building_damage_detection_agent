from __future__ import annotations

from pathlib import Path

from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.tools.base import ToolResult
from rs_xbd_agent.tools.external_command import run_external_mask_command


class ProvidedMaskLocalizationTool:
    """Use a provided or precomputed building localization mask."""

    name = "provided_mask_localization"

    def run(self, state: AgentState) -> ToolResult:
        if state.pre_mask:
            mask_path = Path(state.pre_mask)
            if mask_path.exists():
                return ToolResult(
                    name=self.name,
                    status="ok",
                    outputs={"pre_mask": str(mask_path)},
                    message="使用输入的灾前建筑物 mask 作为定位结果。",
                )
            return ToolResult(
                name=self.name,
                status="missing",
                outputs={"pre_mask": state.pre_mask},
                message="配置了灾前建筑物 mask，但文件不存在。",
            )
        return ToolResult(
            name=self.name,
            status="skipped",
            outputs={},
            message="未提供灾前建筑物 mask，后续将退化为图像差分分析。",
        )


class ExternalCommandLocalizationTool:
    """Run an external localization command and read its output mask."""

    name = "external_command_localization"

    def __init__(self, backend_name: str, command_template: str, timeout: int = 1800):
        self.backend_name = backend_name
        self.command_template = command_template
        self.timeout = timeout

    def run(self, state: AgentState) -> ToolResult:
        output_mask = Path(state.output_dir) / "tool_outputs" / "pre_building_mask.png"
        try:
            result = run_external_mask_command(
                command_template=self.command_template,
                state=state,
                output_mask=output_mask,
                timeout=self.timeout,
            )
        except Exception as exc:
            return ToolResult(
                name=self.name,
                status="failed",
                outputs={"backend": self.backend_name, "error": str(exc)},
                message=f"{self.backend_name} 建筑物定位命令执行失败。",
            )

        outputs = {"backend": self.backend_name, **result}
        if result["returncode"] == 0 and result["output_exists"]:
            outputs["pre_mask"] = result["output_mask"]
            return ToolResult(
                name=self.name,
                status="ok",
                outputs=outputs,
                message=f"{self.backend_name} 已生成建筑物定位 mask。",
            )
        return ToolResult(
            name=self.name,
            status="failed",
            outputs=outputs,
            message=f"{self.backend_name} 未成功生成建筑物定位 mask。",
        )


class PlannedExternalLocalizationTool:
    """Integration placeholder for an external xBD/xView2 localization model."""

    name = "external_localization_placeholder"

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def run(self, state: AgentState) -> ToolResult:
        return ToolResult(
            name=self.name,
            status="not_configured",
            outputs={"backend": self.backend_name},
            message=(
                f"{self.backend_name} 建筑物定位工具尚未配置推理命令；"
                "请配置 LOCALIZATION_COMMAND，或先提供该工具输出的 pre_mask。"
            ),
        )
