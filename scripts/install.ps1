param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $Python -PathType Leaf) {
    $pythonExecutable = (Resolve-Path -LiteralPath $Python).Path
} else {
    throw "Python executable not found: $Python. Install Python 3.13 and pass its executable path with -Python."
}

$versionText = & $pythonExecutable --version
if ($LASTEXITCODE -ne 0 -or $versionText -notmatch '^Python 3\.13\.') {
    throw "BottleQC requires Python 3.13.x; selected interpreter reported: $versionText"
}

$resolvedVenv = if ([System.IO.Path]::IsPathRooted($VenvPath)) { $VenvPath } else { Join-Path $projectRoot $VenvPath }
if (Test-Path -LiteralPath $resolvedVenv) {
    throw "Refusing to overwrite existing environment: $resolvedVenv. Choose a new -VenvPath."
}

& $pythonExecutable -m venv $resolvedVenv
if ($LASTEXITCODE -ne 0) { throw "venv creation failed with exit code $LASTEXITCODE" }
$venvPython = Join-Path $resolvedVenv "Scripts\python.exe"
& $venvPython -m pip install --requirement (Join-Path $projectRoot "requirements.lock")
if ($LASTEXITCODE -ne 0) { throw "Locked dependency installation failed with exit code $LASTEXITCODE" }
& $venvPython -m pip install --no-build-isolation --no-deps --editable $projectRoot
if ($LASTEXITCODE -ne 0) { throw "Editable BottleQC package installation failed with exit code $LASTEXITCODE" }
& $venvPython --version
