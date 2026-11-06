# Product and architecture

## Product direction and current state

The target operator product is a Windows desktop application packaged as an .exe. The current worktree is a local Python prototype with a FastAPI service and a Streamlit HTTP client. The Streamlit screen is not the final product interface; there is no packaged desktop app yet.

Current supported inputs are independent still images and uploaded file video. File video is decoded sequentially in source order. Live camera/RTSP, PLC/rejector control, remote deployment, multi-camera and multi-SKU operation are not implemented.

## Runtime ownership

FastAPI owns uploaded assets, one-active-run admission, model/tracker/source lifecycle, the sequential worker, SQLite, evidence and restart reconciliation. There is no run queue. Streamlit sends HTTP requests and renders responses; it does not own decoding, inference, tracking, persistence or the processing loop. Both processes bind to loopback by default.

~~~text
Operator screen -> FastAPI -> source adapter -> detector/tracker
                                  |-> bottle/component association
                                  |-> inspection domain engine
                                  |-> SQLite and verified evidence
~~~

| Package | Owner |
| --- | --- |
| api | HTTP validation, admission, run lifecycle, history and export |
| pipeline | Image/video sources, sequential worker and frame processing |
| vision | Simulation, model admission, YOLO/ByteTrack and overlays |
| domain | Bottle identity, component association, observation eligibility and final outcomes |
| storage | SQLite, uploaded assets and checked evidence |
| ui | Streamlit HTTP client only |
| training | Dataset validation, grouped splits and training CLI |
| evaluation | Detector metrics and product-level event evaluation |

## Inspection contract

The detector uses whole-frame object detection with exactly 0=bottle, 1=cap and 2=label. Components are associated one-to-one with bottles. Still images receive fresh image-local identities; file video uses a run-scoped tracker. A track ID does not identify a product across runs.

A missing cap or label is a product inspection result, not a new detector class. The domain engine can report FAIL only when enough eligible observations support the missing-part conclusion. Unobservable regions, uncertain identity or association, occlusion, early termination and insufficient observations remain UNINSPECTED. Each product identity finalizes at most once per run.

Simulation uses the accepted domain engine and persistence path, but is always labelled SIMULATION. It is software-flow evidence, never model inference or accuracy evidence.

## Persistence and model admission

SQLite is authoritative for run status, counters and events. Evidence has a verified digest and an explicit AVAILABLE or FAILED state. Uploads and model paths are resolved beneath configured roots. A model manifest binds the detection task, exact classes, weights SHA-256, framework version and provenance. Real mode is refused if the weights or capability check fails; the application does not fall back to simulation.

The broader inspection limits and current evidence are summarized in [project status](project-status.md) and [evaluation](evaluation.md).