from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AlignmentConfig(BaseModel):
    anchor_preference: list[str] = Field(
        default_factory=lambda: ["mission_item_reached", "auto_mode_entry"]
    )
    resample_hz: int = 10


class RejectionConfig(BaseModel):
    min_armed_seconds: float = 10.0
    max_gap_pct: float = 5.0


class SimConfig(BaseModel):
    mot_pwm_max: int = 2000


class RuntimeConfig(BaseModel):
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    rejection: RejectionConfig = Field(default_factory=RejectionConfig)
    sim: SimConfig = Field(default_factory=SimConfig)


def load_config(path: str | Path) -> RuntimeConfig:
    raw = _load_yaml(path)
    return RuntimeConfig.model_validate(raw)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping at top-level")
    return payload
