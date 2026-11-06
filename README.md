# BottleQC

BottleQC is intended to become a Windows desktop application packaged as an .exe. The current code is a local prototype: FastAPI runs the inspection pipeline and a Streamlit client talks to it over loopback. It is not yet a desktop package and does not support a live camera or PLC/rejector control.

The agreed workflow is to edit in VS Code on Windows, prepare and train YOLO detection on Google Colab, keep full datasets and checkpoints on Drive, and bring back only a selected model with its metadata. The detector classes stay 0=bottle, 1=cap, 2=label.

## Run the current local prototype

Use Python 3.13 and the checked-in lock files:

~~~powershell
.\scripts\install.ps1 -Python 'C:\Program Files\Python313\python.exe'
~~~

Start the API and UI in separate PowerShell terminals from the repo root:

~~~powershell
.\scripts\start-api.ps1
.\scripts\start-ui.ps1
~~~

The current operator screen opens at http://127.0.0.1:8501; the API binds to 127.0.0.1:8000. Stop only recorded BottleQC services:

~~~powershell
.\scripts\stop-local.ps1 -Service All
~~~

The deterministic demo is visibly marked SIMULATION and does not measure detector quality:

~~~powershell
.\.venv\Scripts\python.exe -B scripts/demo_domain.py
~~~

## Checks

Run from the Python 3.13 environment:

~~~powershell
python -B -m pytest -q -p no:cacheprovider
~~~

Passing local checks establish software behavior only. They do not establish model accuracy, line performance or pilot readiness.

## Project map

Start at [docs/README.md](docs/README.md). It links the current [project status](docs/project-status.md), [architecture](docs/architecture.md), [data and annotation rules](docs/data-and-annotation.md), [active v2 training plan](docs/training-v2-plan.md), [operations](docs/operations.md), and [evaluation](docs/evaluation.md).