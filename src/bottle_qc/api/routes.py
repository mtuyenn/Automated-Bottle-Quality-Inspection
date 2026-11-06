from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from bottle_qc.api.schemas import RunStartRequest
from bottle_qc.domain.config import InspectionConfig, frame_aspect_ratio_matches
from bottle_qc.domain.entities import Box
from bottle_qc.settings import ConfigurationError, Settings
from bottle_qc.storage.assets import AssetError, AssetStore
from bottle_qc.vision.model_manifest import ModelUnavailableError, sha256_file


router = APIRouter()
_TERMINAL_STATUSES = {"COMPLETED", "STOPPED", "FAILED", "INTERRUPTED"}


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _repo(request: Request):
    return request.app.state.repository


def _asset_store(settings: Settings) -> AssetStore:
    values, _, _ = settings.load_config("default")
    return AssetStore(settings.assets_dir, max_upload_bytes=int(values["max_upload_bytes"]))


def _run_view(request: Request, run: dict[str, Any]) -> dict[str, Any]:
    repository = _repo(request)
    progress = request.app.state.worker.progress(run["run_id"])
    view = {
        "run_id": run["run_id"],
        "source_type": run["source_type"],
        "asset_id": run["asset_id"],
        "scenario": run["scenario"],
        "kind": run["kind"],
        "mode": run["mode"],
        "model_id": run["model_id"],
        "model_hash": run["model_hash"],
        "config_id": run["config_id"],
        "config_hash": run["config_hash"],
        "status": run["status"],
        "created_at": run["created_at"],
        "finished_at": run["finished_at"],
        "error": run["error"],
        "counters": repository.summary(run["run_id"]),
        "progress": progress,
    }
    return view


def _effective_config(settings: Settings, config_id: str, geometry) -> tuple[dict[str, Any], str, InspectionConfig]:
    try:
        values, _, _ = settings.load_config(config_id)
    except ConfigurationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if geometry is not None:
        values = dict(values)
        values.update({
            "roi": list(geometry.roi),
            "entry_line": geometry.entry_line,
            "exit_line": geometry.exit_line,
            "direction": geometry.direction,
        })
        if geometry.frame_aspect_ratio is not None:
            values["frame_aspect_ratio"] = geometry.frame_aspect_ratio
    try:
        config = InspectionConfig.from_mapping(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid inspection geometry/configuration: {exc}") from exc
    snapshot = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return values, hashlib.sha256(snapshot.encode("utf-8")).hexdigest(), config


def _validate_source_geometry(path: Path, kind: str, config: InspectionConfig, asset_id: str) -> None:
    if kind == "image":
        from bottle_qc.pipeline.sources import ImageFileSource

        source = ImageFileSource(path, source_item_id=asset_id)
    else:
        from bottle_qc.pipeline.sources import VideoFileSource

        source = VideoFileSource(path, source_item_id=asset_id)
    try:
        frame = source.read()
        if frame is None:
            raise HTTPException(status_code=422, detail="Source contains no decodable frames")
        width, height = frame.image.size
        ratio = width / height
        if not frame_aspect_ratio_matches(width, height, config.frame_aspect_ratio):
            raise HTTPException(
                status_code=422,
                detail=f"Source aspect ratio {ratio:.5f} differs from configured {config.frame_aspect_ratio:.5f}; set geometry for this source explicitly.",
            )
        frame.image.close()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        source.close()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readiness")
def readiness(request: Request) -> dict[str, Any]:
    settings = _settings(request)
    try:
        request.app.state.repository.db.fetchone("SELECT 1")
        storage = {
            "database": "ready",
            "assets": "ready" if settings.assets_dir.is_dir() else "unavailable",
            "evidence": "ready" if settings.evidence_dir.is_dir() else "unavailable",
        }
        ready = all(value == "ready" for value in storage.values())
    except Exception as exc:
        storage, ready = {"error": f"{type(exc).__name__}: {exc}"}, False
    models = request.app.state.model_registry.list()
    return {
        "status": "ready" if ready else "not_ready",
        "storage": storage,
        "models": [{"model_id": item["model_id"], "status": item["status"], "reason": item.get("reason")} for item in models],
        "simulation_available": True,
        "active_run": request.app.state.repository.active_run() is not None,
    }


@router.post("/assets", status_code=201)
async def upload_asset(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    settings = _settings(request)
    values, _, _ = settings.load_config("default")
    maximum = int(values["max_upload_bytes"])
    data = await file.read(maximum + 1)
    original_name = file.filename or "upload"
    await file.close()
    store = AssetStore(settings.assets_dir, max_upload_bytes=maximum)
    try:
        stored = store.save(data, original_name)
    except AssetError as exc:
        status = 413 if "exceeds configured limit" in str(exc) else 415
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    try:
        asset = _repo(request).create_asset(
            sha256=stored["sha256"],
            kind=stored["kind"],
            original_name=stored["original_name"],
            stored_name=stored["stored_name"],
            size_bytes=stored["size_bytes"],
            width=stored["width"],
            height=stored["height"],
        )
    except Exception as exc:
        store.remove_unreferenced(str(stored["stored_name"]))
        raise HTTPException(status_code=500, detail=f"Asset metadata was not committed: {type(exc).__name__}") from exc
    return {key: asset[key] for key in ("asset_id", "sha256", "kind", "original_name", "size_bytes", "width", "height", "created_at")}


@router.get("/assets/{asset_id}")
def get_asset(request: Request, asset_id: str) -> dict[str, Any]:
    asset = _repo(request).get_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {key: asset[key] for key in ("asset_id", "sha256", "kind", "original_name", "size_bytes", "width", "height", "created_at")}


@router.get("/configs")
def list_configs(request: Request) -> list[dict[str, str]]:
    return _settings(request).list_configs()


@router.get("/models")
def list_models(request: Request) -> list[dict[str, Any]]:
    response = []
    for model in request.app.state.model_registry.list():
        response.append({key: value for key, value in model.items() if key != "manifest_path"})
    return response


@router.post("/runs", status_code=202)
def start_run(
    request: Request,
    body: RunStartRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    if not idempotency_key or not idempotency_key.strip() or len(idempotency_key) > 200:
        raise HTTPException(status_code=422, detail="A nonempty Idempotency-Key of at most 200 characters is required")
    settings, repository = _settings(request), _repo(request)
    payload = body.model_dump(mode="json", exclude_none=True)
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    existing = repository.find_idempotent_run(idempotency_key)
    if existing is not None:
        if existing["request_hash"] != request_hash:
            raise HTTPException(status_code=409, detail="Idempotency-Key was already used with a different request")
        response.status_code = 200
        return {**_run_view(request, existing), "idempotent_replay": True}
    if repository.active_run() is not None:
        raise HTTPException(status_code=409, detail="One inspection run is already active; stop it or wait for it to finish")

    values, config_hash, config = _effective_config(settings, body.config_id, body.geometry)
    model_id = model_hash = asset_id = scenario = None
    source_type, kind = body.source_type, "video"
    if source_type == "simulation":
        if body.mode != "SIMULATION" or body.asset_id is not None or body.model_id is not None:
            raise HTTPException(status_code=422, detail="Simulation runs require mode=SIMULATION and a named scenario, with no asset/model")
        if body.scenario != "phase-2-four-bottle-demo":
            raise HTTPException(status_code=422, detail="Unknown deterministic simulation scenario")
        scenario = body.scenario
    else:
        if body.mode != "REAL" or not body.asset_id:
            raise HTTPException(status_code=422, detail="Asset runs require mode=REAL and an uploaded asset_id")
        if not body.model_id:
            raise HTTPException(status_code=422, detail="REAL inspection requires a registered bottle/cap/label model; simulation is never substituted")
        model = request.app.state.model_registry.get(body.model_id)
        if model is None or model.get("status") != "AVAILABLE":
            reason = model.get("reason") if model else "No compatible registered weights are available"
            raise HTTPException(
                status_code=422,
                detail=f"Model capability unavailable: {reason}; simulation is never substituted for real inspection",
            )
        model_id, model_hash = body.model_id, model.get("weights_sha256")
        asset = repository.get_asset(body.asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="Uploaded asset not found")
        store = _asset_store(settings)
        try:
            asset_path = store.path_for(asset)
        except AssetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        kind, asset_id = asset["kind"], asset["asset_id"]
        _validate_source_geometry(asset_path, kind, config, asset_id)

    config_snapshot = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        run = repository.create_run({
            "source_type": source_type,
            "asset_id": asset_id,
            "scenario": scenario,
            "kind": kind,
            "mode": body.mode,
            "model_id": model_id,
            "model_hash": model_hash,
            "config_id": body.config_id,
            "config_hash": config_hash,
            "config_snapshot": config_snapshot,
            "idempotency_key": idempotency_key,
            "request_hash": request_hash,
        })
    except sqlite3.IntegrityError as exc:
        replay = repository.find_idempotent_run(idempotency_key)
        if replay is not None:
            if replay["request_hash"] != request_hash:
                raise HTTPException(status_code=409, detail="Idempotency-Key was already used with a different request") from exc
            response.status_code = 200
            return {**_run_view(request, replay), "idempotent_replay": True}
        raise HTTPException(status_code=409, detail="One inspection run is already active") from exc
    try:
        request.app.state.worker.start(run)
    except Exception as exc:
        repository.set_run_status(run["run_id"], "FAILED", f"Worker could not start: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Inspection worker could not start") from exc
    return {**_run_view(request, repository.get_run(run["run_id"])), "idempotent_replay": False}


@router.get("/runs")
def list_runs(request: Request, limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    runs = _repo(request).list_runs(limit=limit, offset=offset)
    return {"items": [_run_view(request, item) for item in runs], "limit": limit, "offset": offset}


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict[str, Any]:
    run = _repo(request).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_view(request, run)


@router.post("/runs/{run_id}/stop")
def stop_run(request: Request, run_id: str) -> dict[str, Any]:
    repository = _repo(request)
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] == "RUNNING":
        stopped = request.app.state.worker.request_stop(run_id)
        if not stopped:
            repository.set_run_status(run_id, "INTERRUPTED", "Run had no active worker during stop reconciliation")
            run = repository.get_run(run_id)
        else:
            return {**_run_view(request, run), "stop_requested": True}
    return {**_run_view(request, run), "stop_requested": False, "idempotent_stop": run["status"] in _TERMINAL_STATUSES}


@router.get("/runs/{run_id}/events")
def get_events(
    request: Request,
    run_id: str,
    outcome: str | None = Query(default=None, pattern="^(PASS|FAIL|UNINSPECTED)$"),
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    repository = _repo(request)
    if repository.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    items = repository.list_events(run_id, limit=limit, offset=offset, outcome=outcome)
    return {"items": items, "limit": limit, "offset": offset, "summary": repository.summary(run_id)}


@router.get("/runs/{run_id}/preview")
def get_preview(request: Request, run_id: str) -> Response:
    if _repo(request).get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    image = request.app.state.worker.preview(run_id)
    if image is None:
        raise HTTPException(status_code=404, detail="No decoded preview frame is available yet")
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/runs/{run_id}/events/{event_id}/evidence")
def get_evidence(request: Request, run_id: str, event_id: str) -> FileResponse:
    repository = _repo(request)
    event = repository.get_event(run_id, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event["evidence_state"] != "AVAILABLE" or not event["evidence_path"]:
        raise HTTPException(status_code=410, detail={"state": event["evidence_state"], "error": event["evidence_error"]})
    path = request.app.state.evidence_store.path_for(run_id, event_id)
    if path is None or sha256_file(path) != event["evidence_sha256"]:
        repository.update_evidence(event_id, state="FAILED", error="Evidence file is missing or checksum changed")
        raise HTTPException(status_code=410, detail="Evidence verification failed; stored event remains available")
    return FileResponse(path, media_type="image/jpeg", filename=f"{event_id}.jpg")


@router.get("/runs/{run_id}/report")
def export_report(request: Request, run_id: str) -> Response:
    repository = _repo(request)
    if repository.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    fields = (
        "event_id", "run_id", "source_item_id", "track_id", "outcome", "reasons", "source_index",
        "source_timestamp_ms", "mode", "model_id", "model_hash", "config_id", "config_hash",
        "first_seen_at", "finalized_at", "valid_observation_count", "observation_duration",
        "cap_support", "label_support", "extra", "evidence_state", "evidence_path",
        "evidence_sha256", "evidence_error", "created_at",
    )

    def rows():
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        offset = 0
        while True:
            events = repository.list_events(run_id, limit=1000, offset=offset)
            if not events:
                break
            for event in events:
                row = {field: event.get(field) for field in fields}
                row["reasons"] = ";".join(event["reasons"])
                row["extra"] = json.dumps(event["extra"], sort_keys=True, separators=(",", ":"))
                writer.writerow(row)
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)
            offset += len(events)

    return StreamingResponse(
        rows(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="bottleqc-{run_id}.csv"'},
    )
