from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rs_xbd_agent.agent.planner import DisasterAssessmentAgent
from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.config import load_config
from rs_xbd_agent.models.damage_analyzer import XBDDamageAnalyzer
from rs_xbd_agent.models.factory import build_vlm
from rs_xbd_agent.tools.registry import build_damage_tool, build_localization_tool


st.set_page_config(page_title="xBD 灾害损毁评估 Agent", layout="wide")
st.title("基于 Qwen3.5-4B 多模态模型的 xBD 灾害损毁评估 Agent")
st.write("上传灾前/灾后遥感图像，可选上传 xBD mask，系统会生成损毁统计和 Markdown 报告。")

pre_file = st.file_uploader("灾前图像", type=["png", "jpg", "jpeg"])
post_file = st.file_uploader("灾后图像", type=["png", "jpg", "jpeg"])
pre_mask_file = st.file_uploader("灾前建筑物 mask，可选", type=["png"])
post_damage_file = st.file_uploader("灾后损毁等级 mask，可选", type=["png"])


def save_uploaded(file, path: Path) -> str | None:
    if file is None:
        return None
    path.write_bytes(file.read())
    return str(path)


if st.button("开始分析", disabled=not (pre_file and post_file)):
    cfg = load_config()
    work_dir = Path(tempfile.mkdtemp(prefix="streamlit_xbd_"))
    pre_path = save_uploaded(pre_file, work_dir / "pre.png")
    post_path = save_uploaded(post_file, work_dir / "post.png")
    pre_mask_path = save_uploaded(pre_mask_file, work_dir / "pre_mask.png")
    post_damage_path = save_uploaded(post_damage_file, work_dir / "post_damage.png")

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
        output_dir=str(work_dir / "report"),
    )
    state = agent.run(state)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pre_path, caption="灾前图像")
    with col2:
        st.image(post_path, caption="灾后图像")

    st.subheader("分析摘要")
    st.json(state.summary)

    st.subheader("Agent 工具轨迹")
    st.json({"selected_tools": state.selected_tools, "tool_outputs": state.tool_outputs})

    if "damage_overlay" in state.artifacts:
        st.image(state.artifacts["damage_overlay"], caption="损毁等级叠加图")
    if "change_heatmap" in state.artifacts:
        st.image(state.artifacts["change_heatmap"], caption="变化热力图")

    st.subheader("Markdown 报告")
    st.markdown(state.report)
