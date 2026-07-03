from __future__ import annotations

import argparse
from pathlib import Path

from rs_xbd_agent.agent.planner import DisasterAssessmentAgent
from rs_xbd_agent.agent.state import AgentState
from rs_xbd_agent.config import load_config
from rs_xbd_agent.models.damage_analyzer import XBDDamageAnalyzer
from rs_xbd_agent.models.factory import build_vlm
from rs_xbd_agent.tools.registry import build_damage_tool, build_localization_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 xBD 灾害损毁分析 Agent。")
    parser.add_argument("--pre_image", required=True)
    parser.add_argument("--post_image", required=True)
    parser.add_argument("--pre_mask", default=None)
    parser.add_argument("--post_damage_mask", default=None)
    parser.add_argument("--out_dir", default="outputs/report")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--localization_backend", default=None)
    parser.add_argument("--damage_backend", default=None)
    parser.add_argument("--localization_command", default=None)
    parser.add_argument("--damage_command", default=None)
    parser.add_argument("--external_tool_timeout", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    analyzer = XBDDamageAnalyzer(diff_threshold=cfg.diff_threshold, min_component_area=cfg.min_component_area)
    localization_backend = args.localization_backend or cfg.localization_backend
    damage_backend = args.damage_backend or cfg.damage_backend
    localization_command = args.localization_command if args.localization_command is not None else cfg.localization_command
    damage_command = args.damage_command if args.damage_command is not None else cfg.damage_command
    external_tool_timeout = args.external_tool_timeout or cfg.external_tool_timeout
    agent = DisasterAssessmentAgent(
        analyzer=analyzer,
        vlm=build_vlm(cfg),
        localization_tool=build_localization_tool(
            localization_backend,
            command_template=localization_command,
            timeout=external_tool_timeout,
            xview2_repo=cfg.xview2_repo,
            xview2_localization_weights=cfg.xview2_localization_weights,
            xview2_classification_weights=cfg.xview2_classification_weights,
            xview2_conda_env=cfg.xview2_conda_env,
        ),
        damage_tool=build_damage_tool(
            damage_backend,
            command_template=damage_command,
            timeout=external_tool_timeout,
            xview2_repo=cfg.xview2_repo,
            xview2_localization_weights=cfg.xview2_localization_weights,
            xview2_classification_weights=cfg.xview2_classification_weights,
            xview2_conda_env=cfg.xview2_conda_env,
        ),
    )
    state = AgentState(
        pre_image=args.pre_image,
        post_image=args.post_image,
        pre_mask=args.pre_mask,
        post_damage_mask=args.post_damage_mask,
        output_dir=args.out_dir,
    )
    state = agent.run(state)
    print(state.report)
    print(f"\n报告已保存至：{Path(args.out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
