param([ValidateSet("API", "UI", "All")][string]$Service = "All")

$ErrorActionPreference = "Stop"
$dataRoot = if ($env:BOTTLEQC_DATA_DIR) { $env:BOTTLEQC_DATA_DIR } else { Join-Path $env:LOCALAPPDATA "BottleQC" }
$processDir = Join-Path $dataRoot "processes"
$services = if ($Service -eq "All") { @("api", "ui") } else { @($Service.ToLowerInvariant()) }
foreach ($name in $services) {
    $recordPath = Join-Path $processDir "$name.json"
    if (-not (Test-Path -LiteralPath $recordPath)) {
        Write-Output "$($name.ToUpperInvariant())_NOT_RUNNING"
        continue
    }
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $processInfo = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
    if (-not $processInfo) {
        Remove-Item -LiteralPath $recordPath
        Write-Output "$($name.ToUpperInvariant())_ALREADY_EXITED"
        continue
    }
    if ($processInfo.ExecutablePath -ne $record.executable -or $processInfo.CommandLine -notlike "*$($record.marker)*") {
        throw "Refusing to stop PID $($record.pid): its executable or command line no longer matches the BottleQC $name record."
    }
    Stop-Process -Id ([int]$record.pid) -Force
    Wait-Process -Id ([int]$record.pid) -Timeout 5 -ErrorAction SilentlyContinue
    if (Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue) {
        throw "BottleQC $name process $($record.pid) did not stop within 5 seconds."
    }
    Remove-Item -LiteralPath $recordPath
    Write-Output "$($name.ToUpperInvariant())_STOPPED PID=$($record.pid)"
}
