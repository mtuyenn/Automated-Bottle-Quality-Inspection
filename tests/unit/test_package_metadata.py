from __future__ import annotations

import tomllib
import re
from pathlib import Path


def test_package_declares_runtime_and_operator_entry_points() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    scripts = project["scripts"]

    assert project["requires-python"] == ">=3.13,<3.14"
    assert set(scripts) >= {
        "bottleqc-api",
        "bottleqc-ui",
        "bottleqc-demo",
        "bottleqc-train",
        "bottleqc-eval-system",
        "bottleqc-eval-detector",
    }
    dependency_names = {
        re.split(r"[\s<=>!~;\[]", dependency, maxsplit=1)[0].lower().replace("_", "-")
        for dependency in project["dependencies"]
    }
    assert "fastapi" in dependency_names
