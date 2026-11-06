from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from typing import Any

from bottle_qc.domain.config import InspectionConfig


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BottleQC"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    root_dir: Path = _PROJECT_ROOT
    data_dir: Path = _LOCAL_APP_DATA
    models_dir: Path = _PROJECT_ROOT / "models"
    configs_dir: Path = _PROJECT_ROOT / "configs"

    @classmethod
    def from_environment(cls) -> Settings:
        root = Path(os.environ.get("BOTTLEQC_PROJECT_DIR", _PROJECT_ROOT)).resolve()
        data = Path(os.environ.get("BOTTLEQC_DATA_DIR", _LOCAL_APP_DATA)).resolve()
        models = Path(os.environ.get("BOTTLEQC_MODELS_DIR", root / "models")).resolve()
        configs = Path(os.environ.get("BOTTLEQC_CONFIGS_DIR", root / "configs")).resolve()
        return cls(root_dir=root, data_dir=data, models_dir=models, configs_dir=configs)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "bottleqc.sqlite3"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    def load_config(self, config_id: str) -> tuple[dict[str, Any], str, Path]:
        if not _ID.fullmatch(config_id):
            raise ConfigurationError("Invalid config_id")
        path = (self.configs_dir / f"{config_id}.toml").resolve()
        if path.parent != self.configs_dir.resolve() or not path.is_file():
            raise ConfigurationError(f"Configuration '{config_id}' is unavailable")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Configuration '{config_id}' cannot be read: {exc}") from exc
        required = {
            "minimum_confidence": (0.0, 1.0),
            "minimum_observations": (1, 1000),
            "maximum_missed_frames": (0, 1000),
            "maximum_tracks": (1, 100000),
            "maximum_frames_per_run": (1, 10000000),
            "minimum_region_edge_variance": (0.0, 1e9),
            "tracker_iou_threshold": (0.0, 1.0),
            "tracker_ambiguity_margin": (0.0, 1.0),
            "max_upload_bytes": (1, 2**40),
        }
        for name, (low, high) in required.items():
            value = data.get(name)
            valid_type = type(value) is int if name == "maximum_missed_frames" else type(value) in (int, float)
            if not valid_type or not low <= value <= high:
                raise ConfigurationError(f"Configuration '{config_id}' has invalid {name}")
        try:
            InspectionConfig.from_mapping(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Configuration '{config_id}' has invalid inspection policy: {exc}") from exc
        return data, _sha256(path.read_bytes()), path

    def list_configs(self) -> list[dict[str, str]]:
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for path in sorted(self.configs_dir.glob("*.toml")):
            try:
                config, digest, _ = self.load_config(path.stem)
            except ConfigurationError:
                continue
            items.append({"config_id": path.stem, "sha256": digest, "schema_version": str(config["schema_version"])})
        return items


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
