param([string]$PythonPath)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python environment not found: $PythonPath. Run scripts/install.ps1 first." }
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$dataRoot = if ($env:BOTTLEQC_DATA_DIR) { $env:BOTTLEQC_DATA_DIR } else { Join-Path $env:LOCALAPPDATA "BottleQC" }
$processDir = Join-Path $dataRoot "processes"
$logDir = Join-Path $dataRoot "logs"
New-Item -ItemType Directory -Force -Path $processDir, $logDir | Out-Null
$recordPath = Join-Path $processDir "ui.json"
if (Test-Path -LiteralPath $recordPath) { throw "UI process record already exists: $recordPath. Run scripts/stop-local.ps1 -Service UI and inspect its result." }
$stdout = Join-Path $logDir "ui.stdout.log"
$stderr = Join-Path $logDir "ui.stderr.log"
$uiPath = Join-Path $projectRoot "src\bottle_qc\ui\app.py"
$arguments = @("-B", "-m", "streamlit", "run", $uiPath, "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false")
$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
@{ service = "ui"; pid = $process.Id; executable = $PythonPath; marker = "bottle_qc\ui\app.py" } |
    ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding ascii
Write-Output "UI_PID=$($process.Id)"
Write-Output "UI_URL=http://127.0.0.1:8501"
Write-Output "UI_LOG=$stderr"
