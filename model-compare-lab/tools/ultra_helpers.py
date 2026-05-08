from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    path_obj = Path(path).resolve()
    with path_obj.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path_obj}")
    return data


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    path_obj = Path(path).resolve()
    with path_obj.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a mapping: {path_obj}")
    return data


def resolve_path(base: Path, value: str | Path) -> Path:
    path_obj = Path(value)
    if path_obj.is_absolute():
        return path_obj
    return (base / path_obj).resolve()


def build_model(weights_or_cfg: str):
    from ultralytics import RTDETR, YOLO

    lowered = weights_or_cfg.lower()
    if "rtdetr" in lowered:
        return RTDETR(weights_or_cfg)
    return YOLO(weights_or_cfg)

