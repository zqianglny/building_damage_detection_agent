from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from rs_xbd_agent.agent.state import AgentState


@dataclass
class ToolResult:
    """Structured output returned by an Agent tool."""

    name: str
    status: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class AgentTool(Protocol):
    """Minimal protocol shared by all Agent tools."""

    name: str

    def run(self, state: AgentState) -> ToolResult:
        ...
