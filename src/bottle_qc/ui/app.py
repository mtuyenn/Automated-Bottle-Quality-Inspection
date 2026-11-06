"""Streamlit operator client for the locally running BottleQC HTTP API."""

from __future__ import annotations

import uuid

from bottle_qc.ui.api_client import APIClient, APIError


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="BottleQC", page_icon="🧴", layout="wide")
    st.title("BottleQC local inspection")
    st.caption("The API process owns inspection, model execution, tracking, storage and results.")

    try:
        with APIClient() as api:
            health = api.health()
            readiness = api.readiness()
            st.sidebar.caption(f"API: {health.get('status', 'unknown')} · {readiness.get('status', 'unknown')}")
            configs = api.configs()
            models = api.models()
            run_items = api.runs().get("items", [])

            if readiness.get("status") != "ready":
                st.error(f"API storage is not ready: {readiness.get('storage', {})}")
            else:
                st.sidebar.success("Local API ready")

            config_ids = [item["config_id"] for item in configs]
            if not config_ids:
                st.error("No valid inspection configuration is available from the API.")
                return
            selected_config = st.sidebar.selectbox("Inspection configuration", config_ids)
            mode = st.sidebar.radio("Inspection mode", ["SIMULATION", "REAL"], horizontal=True)

            available_models = [item for item in models if item.get("status") == "AVAILABLE"]
            selected_model = None
            if mode == "REAL":
                if available_models:
                    selected_model = st.sidebar.selectbox("Admitted model", available_models, format_func=lambda item: item["model_id"])
                else:
                    st.sidebar.warning("No compatible model passed manifest, hash, class and inference admission. Real inspection is unavailable.")
                unavailable = [item for item in models if item.get("status") != "AVAILABLE"]
                for item in unavailable:
                    st.sidebar.caption(f"{item.get('model_id')}: {item.get('reason', 'unavailable')}")

            source_kind = "simulation"
            selected_file = None
            geometry = None
            asset_id = st.session_state.get("bottleqc_asset_id")
            if mode == "REAL":
                source_kind = "asset"
                selected_file = st.file_uploader(
                    "Choose a local image or video",
                    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "mp4", "avi", "mov", "mkv", "m4v"],
                    key="bottleqc_media_upload",
                    help="The file is sent to the local API. Frames are decoded and processed in source order.",
                )
                with st.expander("Source geometry", expanded=False):
                    override = st.checkbox("Set geometry for this source", value=False)
                    if override:
                        col1, col2 = st.columns(2)
                        with col1:
                            roi_left = st.number_input("ROI left", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
                            roi_top = st.number_input("ROI top", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
                            entry_line = st.number_input("Entry line", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
                        with col2:
                            roi_right = st.number_input("ROI right", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
                            roi_bottom = st.number_input("ROI bottom", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
                            exit_line = st.number_input("Exit line", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
                        direction = st.selectbox("Travel direction", ["LEFT_TO_RIGHT", "RIGHT_TO_LEFT"])
                        aspect_ratio = st.number_input("Frame width / height", min_value=0.01, max_value=100.0, value=16 / 9, step=0.01, format="%.5f")
                        geometry = {
                            "roi": [roi_left, roi_top, roi_right, roi_bottom],
                            "entry_line": entry_line,
                            "exit_line": exit_line,
                            "direction": direction,
                            "frame_aspect_ratio": aspect_ratio,
                        }

                if selected_file is not None:
                    st.caption(f"Selected: {selected_file.name} ({selected_file.size:,} bytes)")
                    if st.button("Upload selected file to local API", key="bottleqc_upload"):
                        try:
                            uploaded = api.upload_asset(selected_file.name, selected_file.getvalue())
                            st.session_state["bottleqc_asset_id"] = uploaded["asset_id"]
                            st.session_state["bottleqc_asset_name"] = uploaded["original_name"]
                            st.success(f"Uploaded {uploaded['original_name']} · {uploaded['kind']} · SHA-256 {uploaded['sha256']}")
                            asset_id = uploaded["asset_id"]
                        except APIError as exc:
                            st.error(f"Upload failed ({exc.status_code}): {exc}")

                if asset_id:
                    st.caption(f"API asset: {st.session_state.get('bottleqc_asset_name', 'selected file')} · {asset_id}")
            else:
                st.info("Deterministic fixture mode runs the labelled four-bottle simulation through the inspection engine.")

            active_run = bool(readiness.get("active_run"))
            can_start = not active_run and (mode == "SIMULATION" or (asset_id is not None and selected_model is not None))
            if active_run:
                st.warning("A run is already active. Stop or wait for it to finish before starting another.")
            if mode == "REAL" and selected_model is None:
                st.info("No simulation fallback is used for a real request. Provide compatible local weights and pass model admission first.")

            if st.button("Start inspection", type="primary", disabled=not can_start, key="bottleqc_start"):
                key = st.session_state.setdefault("bottleqc_idempotency_key", str(uuid.uuid4()))
                body = {"source_type": source_kind, "mode": mode, "config_id": selected_config}
                if source_kind == "simulation":
                    body["scenario"] = "phase-2-four-bottle-demo"
                else:
                    body["asset_id"] = asset_id
                    body["model_id"] = selected_model["model_id"]
                    if geometry is not None:
                        body["geometry"] = geometry
                try:
                    started = api.start_run(body, idempotency_key=key)
                    st.session_state["bottleqc_run_id"] = started["run_id"]
                    st.session_state["bottleqc_idempotency_key"] = str(uuid.uuid4())
                    st.success(f"Run {started['run_id']} · {started['status']}")
                except APIError as exc:
                    st.error(f"Start failed ({exc.status_code}): {exc}")

            current_run_id = st.session_state.get("bottleqc_run_id")
            history_ids = [item["run_id"] for item in run_items]
            choices = list(dict.fromkeys(([current_run_id] if current_run_id else []) + history_ids))
            if choices:
                default_index = choices.index(current_run_id) if current_run_id in choices else 0
                selected_run_id = st.selectbox("Run history", choices, index=default_index)
                if st.button("Refresh authoritative run status", key="bottleqc_refresh"):
                    st.rerun()
                run = api.run(selected_run_id)
                st.subheader(f"Run {run['run_id']}")
                st.write({
                    "status": run["status"],
                    "mode": run["mode"],
                    "source": run["source_type"],
                    "model": run.get("model_id"),
                    "config": run["config_id"],
                    "progress": run.get("progress"),
                    "error": run.get("error"),
                })
                counters = run["counters"]
                metrics = st.columns(5)
                for column, label in zip(metrics, ("PASS", "FAIL", "UNINSPECTED", "PENDING", "total")):
                    column.metric(label, counters.get(label, 0))

                if run["status"] == "RUNNING" and st.button("Stop this run", key=f"bottleqc_stop_{selected_run_id}"):
                    try:
                        stopped = api.stop_run(selected_run_id)
                        st.session_state["bottleqc_last_stop"] = stopped
                        st.info("Stop requested; the API finalizes active identities before marking the run stopped.")
                    except APIError as exc:
                        st.error(f"Stop failed ({exc.status_code}): {exc}")

                try:
                    st.image(api.preview(selected_run_id), caption="Latest API preview", use_container_width=True)
                except APIError as exc:
                    if exc.status_code != 404:
                        st.caption(f"Preview unavailable: {exc}")

                event_data = api.events(selected_run_id)
                events = event_data.get("items", [])
                if events:
                    st.subheader("Per-bottle results")
                    st.dataframe(events, use_container_width=True, hide_index=True)
                    for event in events:
                        title = f"Track {event['track_id']} · {event['outcome']} · {event.get('source_timestamp_ms')} ms"
                        with st.expander(title):
                            st.write({key: event.get(key) for key in ("event_id", "source_item_id", "source_index", "reasons", "mode", "model_id", "model_hash", "config_hash", "evidence_state", "evidence_error")})
                            if event.get("evidence_state") == "AVAILABLE":
                                try:
                                    evidence = api.evidence(selected_run_id, event["event_id"])
                                    st.image(evidence, caption=f"{event['mode']} evidence", use_container_width=True)
                                    st.download_button(
                                        "Download evidence",
                                        data=evidence,
                                        file_name=f"bottleqc-{event['event_id']}.jpg",
                                        mime="image/jpeg",
                                        key=f"evidence-{event['event_id']}",
                                    )
                                except APIError as exc:
                                    st.error(f"Evidence unavailable ({exc.status_code}): {exc}")
                    try:
                        report = api.report(selected_run_id)
                        st.download_button(
                            "Export run results as CSV",
                            data=report,
                            file_name=f"bottleqc-{selected_run_id}.csv",
                            mime="text/csv",
                            key=f"report-{selected_run_id}",
                        )
                    except APIError as exc:
                        st.error(f"CSV export unavailable ({exc.status_code}): {exc}")
                else:
                    st.caption("No product events have been finalized for this run yet.")
            else:
                st.caption("No run history is available yet.")
    except APIError as exc:
        st.error(f"BottleQC API request failed ({exc.status_code}): {exc}")
    except Exception as exc:
        st.error(f"BottleQC API is unreachable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
