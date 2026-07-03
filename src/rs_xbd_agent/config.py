from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """项目运行配置。"""

    output_dir: str = "outputs"
    mllm_backend: str = "mock"
    mllm_model_name: str = "Qwen/Qwen3.5-4B"
    mllm_adapter_path: str = ""
    mllm_device: str = "auto"
    max_new_tokens: int = 512
    localization_backend: str = "provided_mask"
    damage_backend: str = "provided_mask"
    localization_command: str = ""
    damage_command: str = ""
    external_tool_timeout: int = 1800
    xview2_repo: str = ""
    xview2_localization_weights: str = ""
    xview2_classification_weights: str = ""
    xview2_conda_env: str = ""
    diff_threshold: int = 35
    min_component_area: int = 64


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """读取 yaml 与 .env，环境变量优先级更高。"""
    load_dotenv()
    cfg = load_yaml(config_path)

    project_cfg = cfg.get("project", {})
    model_cfg = cfg.get("model", {})
    tool_cfg = cfg.get("tools", {})
    analysis_cfg = cfg.get("analysis", {})

    return AppConfig(
        output_dir=os.getenv("OUTPUT_DIR", project_cfg.get("output_dir", "outputs")),
        mllm_backend=os.getenv("VLM_BACKEND", os.getenv("MLLM_BACKEND", model_cfg.get("backend", "mock"))),
        mllm_model_name=os.getenv("MLLM_MODEL_NAME", model_cfg.get("model_name", "Qwen/Qwen3.5-4B")),
        mllm_adapter_path=os.getenv("MLLM_ADAPTER_PATH", model_cfg.get("adapter_path", "")),
        mllm_device=os.getenv("MLLM_DEVICE", model_cfg.get("device", "auto")),
        max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
        localization_backend=os.getenv(
            "LOCALIZATION_BACKEND",
            os.getenv("PREDICTOR_BACKEND", tool_cfg.get("localization_backend", "provided_mask")),
        ),
        damage_backend=os.getenv(
            "DAMAGE_BACKEND",
            os.getenv("PREDICTOR_BACKEND", tool_cfg.get("damage_backend", "provided_mask")),
        ),
        localization_command=os.getenv("LOCALIZATION_COMMAND", tool_cfg.get("localization_command", "")),
        damage_command=os.getenv("DAMAGE_COMMAND", tool_cfg.get("damage_command", "")),
        external_tool_timeout=int(os.getenv("EXTERNAL_TOOL_TIMEOUT", tool_cfg.get("external_tool_timeout", 1800))),
        xview2_repo=os.getenv("XVIEW2_REPO", tool_cfg.get("xview2_repo", "")),
        xview2_localization_weights=os.getenv(
            "XVIEW2_LOCALIZATION_WEIGHTS", tool_cfg.get("xview2_localization_weights", "")
        ),
        xview2_classification_weights=os.getenv(
            "XVIEW2_CLASSIFICATION_WEIGHTS", tool_cfg.get("xview2_classification_weights", "")
        ),
        xview2_conda_env=os.getenv("XVIEW2_CONDA_ENV", tool_cfg.get("xview2_conda_env", "")),
        diff_threshold=int(analysis_cfg.get("diff_threshold", 35)),
        min_component_area=int(analysis_cfg.get("min_component_area", 64)),
    )
