from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GeometryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roi: tuple[float, float, float, float]
    entry_line: float
    exit_line: float
    direction: Literal["LEFT_TO_RIGHT", "RIGHT_TO_LEFT"] = "LEFT_TO_RIGHT"
    frame_aspect_ratio: float | None = Field(default=None, gt=0.01, lt=100)


class RunStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["asset", "simulation"]
    asset_id: str | None = None
    scenario: str | None = None
    mode: Literal["SIMULATION", "REAL"]
    model_id: str | None = None
    config_id: str = Field(default="default", min_length=1, max_length=80)
    geometry: GeometryRequest | None = None
