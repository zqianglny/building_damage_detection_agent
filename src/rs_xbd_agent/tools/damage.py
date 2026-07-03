from __future__ import annotations

from pathlib import Path

from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.tools.base import ToolResult
from rs_xbd_agent.tools.external_command import run_external_mask_command


class ProvidedDamageMaskTool:
    """Use a provided or precomputed post-disaster damage mask."""

    name = "provided_damage_mask"

    def run(self, state: AgentState) -> ToolResult:
        if state.post_damage_mask:
            mask_path = Path(state.post_damage_mask)
            if mask_path.exists():
                return ToolResult(
                    name=self.name,
                    status="ok",
                    outputs={"post_damage_mask": str(mask_path)},
                    message="使用输入的灾后损毁等级 mask 作为损毁评估结果。",
                )
            return ToolResult(
                name=self.name,
                status="missing",
                outputs={"post_damage_mask": state.post_damage_mask},
                message="配置了灾后损毁等级 mask，但文件不存在。",
            )
        return ToolResult(
            name=self.name,
            status="skipped",
            outputs={},
            message="未提供灾后损毁等级 mask，后续将基于建筑物 mask 与变化区域生成粗略疑似损毁。",
        )


class DiffFallbackDamageTool:
    """Explicit fallback that leaves damage estimation to XBDDamageAnalyzer."""

    name = "diff_fallback_damage"

    def run(self, state: AgentState) -> ToolResult:
        return ToolResult(
            name=self.name,
            status="fallback",
            outputs={},
            message="未调用专门损毁模型，使用现有差分/建筑物区域交集逻辑作为兜底。",
        )


class ExternalCommandDamageTool:
    """Run an external damage command and read its output mask."""

    name = "external_command_damage"

    def __init__(self, backend_name: str, command_template: str, timeout: int = 1800):
        self.backend_name = backend_name
        self.command_template = command_template
        self.timeout = timeout

    def run(self, state: AgentState) -> ToolResult:
        output_mask = Path(state.output_dir) / "tool_outputs" / "post_damage_mask.png"
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
                message=f"{self.backend_name} 损毁评估命令执行失败。",
            )

        outputs = {"backend": self.backend_name, **result}
        if result["returncode"] == 0 and result["output_exists"]:
            outputs["post_damage_mask"] = result["output_mask"]
            return ToolResult(
                name=self.name,
                status="ok",
                outputs=outputs,
                message=f"{self.backend_name} 已生成灾后损毁等级 mask。",
            )
        return ToolResult(
            name=self.name,
            status="failed",
            outputs=outputs,
            message=f"{self.backend_name} 未成功生成灾后损毁等级 mask。",
        )


class PlannedExternalDamageTool:
    """Integration placeholder for an external xBD/xView2 damage model."""

    name = "external_damage_placeholder"

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def run(self, state: AgentState) -> ToolResult:
        return ToolResult(
            name=self.name,
            status="not_configured",
            outputs={"backend": self.backend_name},
            message=(
                f"{self.backend_name} 损毁评估工具尚未配置推理命令；"
                "请配置 DAMAGE_COMMAND，或先提供该工具输出的 post_damage_mask。"
            ),
        )
