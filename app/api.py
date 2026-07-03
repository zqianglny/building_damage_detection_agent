from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from rs_xbd_agent.agent.planner import DisasterAssessmentAgent
from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.config import load_config
from rs_xbd_agent.models.damage_analyzer import XBDDamageAnalyzer
from rs_xbd_agent.models.factory import build_vlm
from rs_xbd_agent.tools.registry import build_damage_tool, build_localization_tool

app = FastAPI(title="xBD Remote Sensing MLLM Agent", version="0.1.0")


def _save_upload(file: UploadFile | None, dst_dir: Path, name: str) -> str | None:
    if file is None:
        return None
    suffix = Path(file.filename or name).suffix or ".png"
    path = dst_dir / f"{name}{suffix}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(path)


@app.get("/")
def root():
    return {"message": "xBD 灾害损毁评估 Agent 已启动。请调用 POST /analyze。"}


@app.post("/analyze")
def analyze(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    pre_mask: UploadFile | None = File(None),
    post_damage_mask: UploadFile | None = File(None),
):
    cfg = load_config()
    work_dir = Path(tempfile.mkdtemp(prefix="xbd_agent_"))
    out_dir = work_dir / "report"

    pre_path = _save_upload(pre_image, work_dir, "pre_image")
    post_path = _save_upload(post_image, work_dir, "post_image")
    pre_mask_path = _save_upload(pre_mask, work_dir, "pre_mask")
    post_damage_path = _save_upload(post_damage_mask, work_dir, "post_damage_mask")

    analyzer = XBDDamageAnalyzer(diff_threshold=cfg.diff_threshold, min_component_area=cfg.min_component_area)
    agent = DisasterAssessmentAgent(
        analyzer=analyzer,
        vlm=build_vlm(cfg),
        localization_tool=build_localization_tool(
            cfg.localization_backend,
            command_template=cfg.localization_command,
            timeout=cfg.external_tool_timeout,
            xview2_repo=cfg.xview2_repo,
            xview2_localization_weights=cfg.xview2_localization_weights,
            xview2_classification_weights=cfg.xview2_classification_weights,
            xview2_conda_env=cfg.xview2_conda_env,
        ),
        damage_tool=build_damage_tool(
            cfg.damage_backend,
            command_template=cfg.damage_command,
            timeout=cfg.external_tool_timeout,
            xview2_repo=cfg.xview2_repo,
            xview2_localization_weights=cfg.xview2_localization_weights,
            xview2_classification_weights=cfg.xview2_classification_weights,
            xview2_conda_env=cfg.xview2_conda_env,
        ),
    )
    state = AgentState(
        pre_image=pre_path,
        post_image=post_path,
        pre_mask=pre_mask_path,
        post_damage_mask=post_damage_path,
        output_dir=str(out_dir),
    )
    state = agent.run(state)
    return {
        "summary": state.summary,
        "artifacts": state.artifacts,
        "report": state.report,
        "selected_tools": state.selected_tools,
        "tool_outputs": state.tool_outputs,
        "trace": state.trace,
    }
