# Local operation

## Install and launch

Use the locked Python 3.13 setup and the start/stop commands in the [root README](../README.md). Start one API process and one UI process. The API listens on 127.0.0.1:8000 and the Streamlit client on 127.0.0.1:8501. The API has no network authentication; keep it loopback-only.

The local screen is a prototype, not the final Windows desktop .exe. A successful health check only means the API responds; readiness reports storage and model availability separately.

## Runtime directories

By default, database files, uploads, evidence, process records and logs live under %LOCALAPPDATA%\BottleQC. Set BOTTLEQC_DATA_DIR before launch to choose another root. BOTTLEQC_MODELS_DIR and BOTTLEQC_CONFIGS_DIR can point to local model and configuration directories.

Large training data, frame collections and checkpoints belong on Colab/Drive. Do not mirror a full dataset to the Windows repo. See [data rules](data-and-annotation.md).

## Operator sequence

1. Check health and readiness.
2. Use SIMULATION only for a labelled software demonstration.
3. For REAL, select an admitted model, upload an image or file video, choose a configuration and supply source geometry when required.
4. Start one run and use its run ID and persisted status as the source of truth.
5. Review counters, events, preview and per-product evidence. Treat UNINSPECTED as requiring review.
6. Stop through the UI or the guarded stop script.

No compatible weights are installed in the local models directory, so the local prototype cannot currently perform REAL inspection.

## Recovery

The stop script only stops a recorded BottleQC process whose executable and command line still match. If it reports a PID mismatch, inspect the process and record; do not kill an unrelated process.

After an API restart, an interrupted RUNNING run becomes INTERRUPTED. Evidence writes interrupted after an event are reconciled from the stored image or marked FAILED with a reason. SQLite events are not recreated from screenshots. Do not delete or reset SQLite to repair a run.

## Geometry and API limits

Decoded source dimensions must match the selected frame_aspect_ratio within max(1e-6, abs(configured_ratio) * 0.005). Other aspect ratios require explicit source geometry; this does not calibrate a physical line.

The API exposes health/readiness, configs/models, uploads, run start/stop/status, events, preview, report and evidence endpoints. Run reports page through all events; event-list pagination is bounded. See src/bottle_qc/api/routes.py and the request schemas for exact endpoint contracts.