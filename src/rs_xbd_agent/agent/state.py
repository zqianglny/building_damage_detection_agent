from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    """Agent 状态对象，记录每一步的输入、工具输出和最终报告。"""

    pre_image: str
    post_image: str
    pre_mask: str | None = None
    post_damage_mask: str | None = None
    output_dir: str = "outputs/report"
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    selected_tools: Dict[str, str] = field(default_factory=dict)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    mllm_text: str = ""
    report: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_trace(self, step: str, detail: Dict[str, Any] | None = None) -> None:
        self.trace.append({"step": step, "detail": detail or {}})
