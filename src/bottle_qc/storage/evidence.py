from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import uuid

from PIL import Image


class EvidenceStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, event_id: str, image_bytes: bytes) -> tuple[str, str]:
        run_dir = self._run_dir(run_id)
        event_id = _validated_uuid(event_id, "event")
        run_dir.mkdir(parents=True, exist_ok=True)
        destination = run_dir / f"{event_id}.jpg"
        # Keep the temporary basename short so deeply nested Windows data paths
        # remain below legacy MAX_PATH when Pillow opens the file.
        temporary = run_dir / f".{uuid.uuid4().hex}.tmp"
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(temporary, format="JPEG", quality=88, optimize=True)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
            with Image.open(destination) as saved:
                saved.verify()
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return f"{run_id}/{event_id}.jpg", digest

    def path_for(self, run_id: str, event_id: str) -> Path:
        run_dir = self._run_dir(run_id)
        event_id = _validated_uuid(event_id, "event")
        path = (run_dir / f"{event_id}.jpg").resolve()
        if path.parent != run_dir:
            raise ValueError("Invalid event evidence path")
        return path if path.is_file() else None

    def _run_dir(self, run_id: str) -> Path:
        run_id = _validated_uuid(run_id, "run")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise ValueError("Invalid run evidence path")
        return path


def _validated_uuid(value: str, identity: str) -> str:
    try:
        canonical = str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {identity} identity") from exc
    if canonical != value.casefold():
        raise ValueError(f"Invalid {identity} identity")
    return canonical
