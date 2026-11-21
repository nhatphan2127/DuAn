param([string]$EnvironmentDirectory = '.venv-data')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$environmentTarget = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $EnvironmentDirectory))
if (-not $environmentTarget.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Environment must be inside this project'
}
if (Test-Path -LiteralPath $environmentTarget) { throw 'Choose a new environment directory; existing environments are preserved' }
$archiveRecord = Get-Content -LiteralPath 'reports/data_freeze_20261016/python_runtime_archive.json' -Raw | ConvertFrom-Json
$archiveHash = (Get-FileHash -LiteralPath 'data/tools/python-base.zip' -Algorithm SHA256).Hash.ToLowerInvariant()
if ($archiveHash -ne $archiveRecord.archive.sha256) { throw 'Base Python archive checksum differs' }
$baseFolder = Join-Path $projectRoot 'data/tools/python-base'
if (-not (Test-Path -LiteralPath $baseFolder)) {
    Expand-Archive -LiteralPath 'data/tools/python-base.zip' -DestinationPath $baseFolder
}
$basePython = Join-Path $baseFolder 'python.exe'
& $basePython -m venv --without-pip $environmentTarget
if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated environment' }
$pipWheel = @(Get-ChildItem -LiteralPath 'data/tools/wheelhouse' -Filter 'pip-*.whl')
if ($pipWheel.Count -ne 1) { throw 'Expected exactly one pinned pip wheel' }
$bootstrapFolder = Join-Path $projectRoot 'data/tools/pip-bootstrap'
& $basePython -c "import sys,zipfile;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $pipWheel[0].FullName $bootstrapFolder
if ($LASTEXITCODE -ne 0) { throw 'Could not unpack the pinned pip wheel' }
$bootstrapCode = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__',alter_sys=True)"
& $basePython -c $bootstrapCode $bootstrapFolder --python $environmentTarget install --no-index --find-links data/tools/wheelhouse --require-hashes -r data_pipeline/requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Offline hash-checked install failed' }
& (Join-Path $environmentTarget 'Scripts/python.exe') data_pipeline/check_environment.py --output environment-reproduction.json
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
