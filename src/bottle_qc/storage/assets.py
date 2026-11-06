from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import re
import uuid

import cv2
from PIL import Image


_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_IMAGE_FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tif", "WEBP": ".webp"}


class AssetError(ValueError):
    pass


class AssetStore:
    """Backend-owned media files with IDs and paths independent of client names."""

    def __init__(self, root: Path, *, max_upload_bytes: int):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes

    def save(self, data: bytes, original_name: str) -> dict[str, object]:
        if not data:
            raise AssetError("Uploaded file is empty")
        if len(data) > self.max_upload_bytes:
            raise AssetError(f"Upload exceeds configured limit of {self.max_upload_bytes} bytes")
        display_name = _safe_display_name(original_name)
        kind, suffix, dimensions = _detect_image(data)
        if kind is None:
            requested_suffix = Path(display_name).suffix.casefold()
            suffix = requested_suffix if requested_suffix in _VIDEO_SUFFIXES else ".mp4"
            kind = "video"
        asset_id = str(uuid.uuid4())
        stored_name = f"{asset_id}{suffix}"
        destination = self.root / stored_name
        temporary = self.root / f".{asset_id}.upload{suffix}"
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if kind == "video":
                capture = cv2.VideoCapture(str(temporary))
                try:
                    ok, frame = capture.read()
                    if not capture.isOpened() or not ok:
                        raise AssetError("Video cannot be decoded; upload a supported video file")
                    dimensions = (int(frame.shape[1]), int(frame.shape[0]))
                finally:
                    capture.release()
            temporary.replace(destination)
        except AssetError:
            temporary.unlink(missing_ok=True)
            raise
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "asset_id": asset_id,
            "sha256": hashlib.sha256(data).hexdigest(),
            "kind": kind,
            "original_name": display_name,
            "stored_name": stored_name,
            "size_bytes": len(data),
            "width": dimensions[0],
            "height": dimensions[1],
            "path": destination,
        }

    def path_for(self, asset: dict[str, object]) -> Path:
        name = asset.get("stored_name")
        if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f-]{36}\.(jpg|png|bmp|tif|webp|mp4|mov|avi|mkv|webm)", name):
            raise AssetError("Stored asset identity is invalid")
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_file():
            raise AssetError("Stored asset file is unavailable")
        return path

    def remove_unreferenced(self, stored_name: str) -> None:
        path = (self.root / stored_name).resolve()
        if path.parent == self.root and path.is_file():
            path.unlink()


def _detect_image(data: bytes) -> tuple[str | None, str | None, tuple[int, int] | None]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            dimensions = image.size
            image.verify()
        suffix = _IMAGE_FORMAT_SUFFIX.get(image_format)
        if suffix is None:
            raise AssetError(f"Unsupported image format: {image_format or 'unknown'}")
        return "image", suffix, dimensions
    except AssetError:
        raise
    except Exception:
        return None, None, None


def _safe_display_name(value: str) -> str:
    name = str(value or "upload").replace("\\", "/").split("/")[-1]
    name = "".join(char for char in name if char.isprintable() and char not in "\r\n\0").strip()
    return (name or "upload")[:255]
