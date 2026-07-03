from __future__ import annotations

import json
from pathlib import Path

from rs_xbd_agent.agent.prompts import REPORT_PROMPT
from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.models.damage_analyzer import XBDDamageAnalyzer
from rs_xbd_agent.models.mllm_base import VisionLanguageModel
from rs_xbd_agent.report.report_builder import build_markdown_report
from rs_xbd_agent.tools.base import AgentTool
from rs_xbd_agent.tools.registry import build_damage_tool, build_localization_tool


class DisasterAssessmentAgent:
    """灾害损毁评估 Agent。

    运行流程：
    1. 根据输入与配置选择建筑物定位工具和损毁评估工具。
    2. 将工具输出的 mask 交给统计分析器生成变化区域、损毁统计和 bbox。
    3. 调用多模态大模型/规则模型生成复核意见。
    4. 生成 Markdown 报告与 JSON 结果。
    """

    def __init__(
        self,
        analyzer: XBDDamageAnalyzer,
        vlm: VisionLanguageModel,
        localization_tool: AgentTool | None = None,
        damage_tool: AgentTool | None = None,
    ):
        self.analyzer = analyzer
        self.vlm = vlm
        self.localization_tool = localization_tool or build_localization_tool("provided_mask")
        self.damage_tool = damage_tool or build_damage_tool("provided_mask")

    def _run_tool(self, state: AgentState, role: str, tool: AgentTool) -> None:
        result = tool.run(state)
        state.selected_tools[role] = result.name
        state.tool_outputs[role] = {
            "status": result.status,
            "outputs": result.outputs,
            "message": result.message,
        }
        state.add_trace(
            f"tool:{role}",
            {"name": result.name, "status": result.status, "message": result.message, "outputs": result.outputs},
        )

        if role == "localization" and result.outputs.get("pre_mask"):
            state.pre_mask = result.outputs["pre_mask"]
        if role == "damage" and result.outputs.get("post_damage_mask"):
            state.post_damage_mask = result.outputs["post_damage_mask"]

    def run(self, state: AgentState) -> AgentState:
        out_dir = Path(state.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        state.add_trace("start", {"pre_image": state.pre_image, "post_image": state.post_image})
        self._run_tool(state, "localization", self.localization_tool)
        self._run_tool(state, "damage", self.damage_tool)

        analysis = self.analyzer.analyze(
            pre_image=state.pre_image,
            post_image=state.post_image,
            pre_mask=state.pre_mask,
            post_damage_mask=state.post_damage_mask,
        )
        state.summary = analysis.summary
        state.add_trace("visual_analysis", {"risk_level": state.summary.get("risk_level")})

        state.artifacts = self.analyzer.save_artifacts(analysis, out_dir=out_dir)
        state.add_trace("save_artifacts", state.artifacts)

        vlm_response = self.vlm.generate(
            prompt=REPORT_PROMPT,
            image_paths=[state.pre_image, state.post_image],
            context={
                "summary": state.summary,
                "artifacts": state.artifacts,
                "selected_tools": state.selected_tools,
                "tool_outputs": state.tool_outputs,
            },
        )
        state.mllm_text = vlm_response.text
        state.add_trace("mllm_generate", {"text_length": len(state.mllm_text)})

        state.report = build_markdown_report(state.summary, state.mllm_text, state.artifacts)
        report_path = out_dir / "report.md"
        report_path.write_text(state.report, encoding="utf-8")

        result_path = out_dir / "result.json"
        state.add_trace("finish", {"report_path": str(report_path), "result_path": str(result_path)})
        result = {
            "summary": state.summary,
            "artifacts": state.artifacts,
            "selected_tools": state.selected_tools,
            "tool_outputs": state.tool_outputs,
            "report_path": str(report_path),
            "trace": state.trace,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return state
