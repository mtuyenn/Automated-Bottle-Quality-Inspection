from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


class APIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Thin HTTP client. It owns no model, media decoder, database, or run worker."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 8.0):
        self.base_url = (base_url or os.environ.get("BOTTLEQC_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def readiness(self) -> dict[str, Any]:
        return self._json("GET", "/readiness")

    def configs(self) -> list[dict[str, Any]]:
        return self._json("GET", "/configs")

    def models(self) -> list[dict[str, Any]]:
        return self._json("GET", "/models")

    def upload_asset(self, filename: str, content: bytes) -> dict[str, Any]:
        response = self._client.post("/assets", files={"file": (Path(filename).name, content, "application/octet-stream")})
        return self._checked_json(response)

    def start_run(self, body: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return self._json("POST", "/runs", json=body, headers={"Idempotency-Key": idempotency_key})

    def run(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{run_id}")

    def runs(self) -> dict[str, Any]:
        return self._json("GET", "/runs")

    def stop_run(self, run_id: str) -> dict[str, Any]:
        return self._json("POST", f"/runs/{run_id}/stop")

    def events(self, run_id: str) -> dict[str, Any]:
        return self._json("GET", f"/runs/{run_id}/events")

    def preview(self, run_id: str) -> bytes:
        response = self._client.get(f"/runs/{run_id}/preview")
        self._raise(response)
        return response.content

    def evidence(self, run_id: str, event_id: str) -> bytes:
        response = self._client.get(f"/runs/{run_id}/events/{event_id}/evidence")
        self._raise(response)
        return response.content

    def report(self, run_id: str) -> bytes:
        response = self._client.get(f"/runs/{run_id}/report")
        self._raise(response)
        return response.content

    def _json(self, method: str, path: str, **kwargs) -> Any:
        response = self._client.request(method, path, **kwargs)
        return self._checked_json(response)

    @staticmethod
    def _checked_json(response: httpx.Response) -> Any:
        APIClient._raise(response)
        return response.json()

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise APIError(response.status_code, str(detail))
