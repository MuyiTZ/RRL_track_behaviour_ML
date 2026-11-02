from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _derive_acquisition(feature_config: dict[str, Any], acquisition: dict[str, Any] | None = None) -> None:
    source = acquisition or feature_config
    pixel_size = float(source.get("pixel_size_um_per_pixel", source.get("pixel_size_um")))
    if "frame_interval_ms" not in source:
        raise ValueError("Acquisition settings must define the canonical frame_interval_ms value")
    frame_interval_ms = float(source["frame_interval_ms"])
    if pixel_size <= 0 or frame_interval_ms <= 0:
        raise ValueError("Pixel size and frame interval must both be positive")
    frame_interval_seconds = frame_interval_ms / 1000.0
    feature_config["pixel_size_um"] = pixel_size
    feature_config["frame_interval_ms"] = frame_interval_ms
    feature_config["frame_interval_seconds"] = frame_interval_seconds
    feature_config["frame_rate_fps"] = 1.0 / frame_interval_seconds


def load_config(path: str | Path, user_settings_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_dir"] = str(config_path.parent)
    _derive_acquisition(config["feature_calculation"])
    if user_settings_path is not None:
        settings_path = Path(user_settings_path).expanduser().resolve()
        with settings_path.open("r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}
        acquisition = settings.get("acquisition", {})
        _derive_acquisition(config["feature_calculation"], acquisition)
        config["visualization"].update(settings.get("visualization", {}))
        config["output"].update(settings.get("output", {}))
        config["_user_settings_path"] = str(settings_path)
    output = Path(config["paths"]["output_dir"])
    if not output.is_absolute():
        output = config_path.parent / output
    config["paths"]["output_dir"] = str(output.resolve())
    return config


def resolve_path(value: str, project_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_dir) / path
    return path.resolve()
