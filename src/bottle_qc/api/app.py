from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from bottle_qc.api.routes import router
from bottle_qc.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    selected_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected_settings.data_dir.mkdir(parents=True, exist_ok=True)
        selected_settings.assets_dir.mkdir(parents=True, exist_ok=True)
        selected_settings.evidence_dir.mkdir(parents=True, exist_ok=True)
        values, _, _ = selected_settings.load_config("default")
        from bottle_qc.pipeline.worker import RunWorker
        from bottle_qc.storage.evidence import EvidenceStore
        from bottle_qc.storage.repository import Repository
        from bottle_qc.vision.model_registry import ModelRegistry

        repository = Repository(selected_settings.database_path)
        evidence_store = EvidenceStore(selected_settings.evidence_dir)
        repository.reconcile(selected_settings.evidence_dir)
        registry = ModelRegistry(
            selected_settings.models_dir,
            minimum_confidence=float(values["minimum_confidence"]),
            minimum_region_edge_variance=float(values["minimum_region_edge_variance"]),
        )
        registry.refresh()
        worker = RunWorker(selected_settings, repository, evidence_store, registry)
        app.state.settings = selected_settings
        app.state.repository = repository
        app.state.evidence_store = evidence_store
        app.state.model_registry = registry
        app.state.worker = worker
        try:
            yield
        finally:
            worker.shutdown()
            repository.close()

    app = FastAPI(title="BottleQC local API", version="1.0.0", lifespan=lifespan)
    app.state.settings = selected_settings
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)


if __name__ == "__main__":
    main()
