param([switch]$BrowserTests)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $projectRoot
$uvVersion = '0.13.5'
$pythonVersion = '3.11.17'
$uvPath = Join-Path $projectRoot '.tools\bin\uv.exe'
if (-not (Test-Path -LiteralPath $uvPath)) {
    python -m pip install --index-url https://pypi.org/simple --disable-pip-version-check --no-input --target .tools "uv==$uvVersion"
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the environment tool. An existing Python with pip is required.' }
}
& $uvPath python install $pythonVersion --install-dir .python --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Could not obtain the dedicated Python runtime.' }
$runtime = Join-Path $projectRoot ".python\cpython-$pythonVersion-windows-x86_64-none\python.exe"
& $uvPath venv .venv-paper --python $runtime --no-python-downloads
if ($LASTEXITCODE -ne 0) { throw 'Could not create the dedicated paper environment.' }
& $uvPath pip sync requirements-paper.lock --python .venv-paper\Scripts\python.exe --require-hashes --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
$priorBuildMode = $env:SETUPTOOLS_USE_DISTUTILS
$priorTemp = $env:TEMP
$priorTmp = $env:TMP
try {
    # Scoped compatibility for the CPython 3.11 standalone build backend.
    $env:SETUPTOOLS_USE_DISTUTILS = 'stdlib'
    $taskTemp = Join-Path $projectRoot '.tmp'
    New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    & $uvPath pip install --python .venv-paper\Scripts\python.exe --no-deps --no-build-isolation -e . --cache-dir .uv-cache
    if ($LASTEXITCODE -ne 0) { throw 'The paper package could not be installed.' }
} finally {
    $env:SETUPTOOLS_USE_DISTUTILS = $priorBuildMode
    $env:TEMP = $priorTemp
    $env:TMP = $priorTmp
}
if ($BrowserTests) {
    & .\.venv-paper\Scripts\python.exe -m playwright install chromium
    if ($LASTEXITCODE -ne 0) { throw 'Browser test runtime installation failed.' }
}
& .\.venv-paper\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3,11); import numpy,databento,websockets; print('Dedicated paper environment ready.')"
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
Write-Output 'Existing venv and paper history were preserved. Configure data access before starting the cross-asset application.'
