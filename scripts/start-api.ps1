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
$recordPath = Join-Path $processDir "api.json"
if (Test-Path -LiteralPath $recordPath) { throw "API process record already exists: $recordPath. Run scripts/stop-local.ps1 -Service API and inspect its result." }
$stdout = Join-Path $logDir "api.stdout.log"
$stderr = Join-Path $logDir "api.stderr.log"
$arguments = @("-B", "-m", "bottle_qc.api.app")
$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
@{ service = "api"; pid = $process.Id; executable = $PythonPath; marker = "bottle_qc.api.app" } |
    ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding ascii
Write-Output "API_PID=$($process.Id)"
Write-Output "API_URL=http://127.0.0.1:8000"
Write-Output "API_LOG=$stderr"
