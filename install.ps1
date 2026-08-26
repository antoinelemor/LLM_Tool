<#
.SYNOPSIS
    LLM Tool - one-command installer for Windows.

.DESCRIPTION
    Windows counterpart of install.sh. It finds a suitable Python interpreter,
    creates a local virtual environment in .venv, installs LLM Tool and its
    dependencies, configures VS Code, and verifies the result.

    Everything lands inside the project folder: nothing is written to the system
    Python, and uninstalling is just deleting the folder.

.PARAMETER Preset
    Which dependency set to install:
      core - the pipeline only (annotation + training)
      all  - core plus every optional wheel-installable feature (default)
      dev  - core plus the test/lint toolchain
      full - all + dev

.PARAMETER Cuda
    Install the CUDA build of PyTorch instead of the CPU-only build that PyPI
    ships for Windows. Takes a tag understood by download.pytorch.org, such as
    cu126 or cu128. Requires an NVIDIA GPU and a matching driver.

.PARAMETER Python
    Full path to a specific python.exe to build the virtual environment with.
    Skips auto-detection.

.PARAMETER Recreate
    Delete an existing .venv and start over without asking.

.PARAMETER SkipVerify
    Do not run verify_installation.py at the end.

.EXAMPLE
    .\install.ps1
    Installs the recommended preset (all) with CPU PyTorch.

.EXAMPLE
    .\install.ps1 -Preset all -Cuda cu126
    Installs everything with the CUDA 12.6 build of PyTorch.

.NOTES
    Author: Antoine Lemor
    If PowerShell refuses to run this file, double-click install.bat instead, or:
        powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

[CmdletBinding()]
param(
    [ValidateSet('core', 'all', 'dev', 'full')]
    [string]$Preset = 'all',

    [ValidatePattern('^(cpu|cu\d{2,4})$')]
    [string]$Cuda,

    [string]$Python,

    [switch]$Recreate,

    [switch]$SkipVerify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# PowerShell 7.4+ turns a non-zero exit from a native command into a terminating
# error when ErrorActionPreference is Stop. This script checks $LASTEXITCODE
# itself so it can print an actionable message instead of a stack trace.
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# --- Console setup -----------------------------------------------------------
# The CLI prints accented text and box-drawing characters. Without this the
# legacy console decodes them as the ANSI code page and mangles the output.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Redirected or non-interactive host: nothing to configure.
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

$MinPythonMajor = 3
$MinPythonMinor = 11

# Newest first. Every version listed here has prebuilt Windows wheels for the
# whole dependency set, so nothing needs a C++ compiler.
$PreferredVersions = @('3.13', '3.12', '3.11')

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

# --- Output helpers ----------------------------------------------------------
function Write-Step {
    param([string]$Text)
    Write-Host ''
    Write-Host ('-' * 78) -ForegroundColor DarkGray
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('-' * 78) -ForegroundColor DarkGray
}
function Write-Ok   { param([string]$T) Write-Host "  [OK] $T" -ForegroundColor Green }
function Write-Warn { param([string]$T) Write-Host "  [!]  $T" -ForegroundColor Yellow }
function Write-Err  { param([string]$T) Write-Host "  [x]  $T" -ForegroundColor Red }
function Write-Info { param([string]$T) Write-Host "       $T" -ForegroundColor Gray }

function Stop-WithHelp {
    param([string]$Message, [string[]]$Hints = @())
    Write-Host ''
    Write-Err $Message
    if ($Hints.Count -gt 0) {
        Write-Host ''
        foreach ($h in $Hints) { Write-Info $h }
    }
    Write-Host ''
    Write-Info 'Full Windows guide: docs/WINDOWS.md'
    exit 1
}

Write-Host ''
Write-Host '  ##      ##      ##    ##     ######  ####   ####  ##    ' -ForegroundColor Blue
Write-Host '  ##      ##      ####  ####      ##  ##  ## ##  ## ##    ' -ForegroundColor Blue
Write-Host '  ##      ##      ##  ##  ##      ##  ##  ## ##  ## ##    ' -ForegroundColor Blue
Write-Host '  ######  ######  ##      ##      ##   ####   ####  ######' -ForegroundColor Blue
Write-Host ''
Write-Host '  AI-Powered Annotation & ML Training Pipeline - Windows installer' -ForegroundColor White
$torchLabel = if ($Cuda) { $Cuda } else { 'CPU' }
Write-Host "  Preset: $Preset  |  PyTorch: $torchLabel" -ForegroundColor DarkGray
Write-Host ''

# --- Step 1: locate a usable Python ------------------------------------------
Write-Step 'Step 1/7  Locating Python 3.11+'

function Test-PythonCandidate {
    <#
      Returns Path/Version/Major/Minor for a working interpreter, or $null.
      Store stubs are rejected here: Windows ships a zero-byte python.exe under
      WindowsApps that only opens the Microsoft Store, and it cannot build venvs.
    #>
    param([string]$Exe, [string[]]$Prefix = @())

    $probe = 'import sys; print("%d.%d.%d" % sys.version_info[:3]); print(sys.executable)'
    try {
        $argv = @($Prefix) + @('-c', $probe)
        $out = & $Exe @argv 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    } catch {
        return $null
    }

    $lines = @($out) | Where-Object { $_ -and $_.ToString().Trim() }
    if ($lines.Count -lt 2) { return $null }

    $version = $lines[0].ToString().Trim()
    $realExe = $lines[1].ToString().Trim()

    if ($realExe -like '*\WindowsApps\*') { return $null }   # Store alias stub
    if (-not (Test-Path -LiteralPath $realExe)) { return $null }

    $parts = $version.Split('.')
    if ($parts.Count -lt 2) { return $null }

    return [PSCustomObject]@{
        Path    = $realExe
        Version = $version
        Major   = [int]$parts[0]
        Minor   = [int]$parts[1]
    }
}

$candidates = New-Object System.Collections.Generic.List[object]

if ($Python) {
    if (-not (Test-Path -LiteralPath $Python)) {
        Stop-WithHelp "The interpreter given with -Python does not exist: $Python"
    }
    $c = Test-PythonCandidate -Exe $Python
    if (-not $c) { Stop-WithHelp "The interpreter given with -Python is not usable: $Python" }
    $candidates.Add($c)
} else {
    # The py launcher is the reliable way to enumerate installed versions.
    if (Get-Command 'py' -ErrorAction SilentlyContinue) {
        foreach ($v in $PreferredVersions) {
            $c = Test-PythonCandidate -Exe 'py' -Prefix @("-$v")
            if ($c) { $candidates.Add($c) }
        }
    }
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $c = Test-PythonCandidate -Exe $cmd.Source
            if ($c) { $candidates.Add($c) }
        }
    }
}

$usable = @($candidates | Where-Object {
    $_.Major -gt $MinPythonMajor -or ($_.Major -eq $MinPythonMajor -and $_.Minor -ge $MinPythonMinor)
})

if ($usable.Count -eq 0) {
    $found = if ($candidates.Count -gt 0) {
        'Found only: ' + (($candidates | ForEach-Object { "$($_.Version) at $($_.Path)" }) -join '; ')
    } else {
        'No Python interpreter was found on PATH.'
    }
    Stop-WithHelp "LLM Tool requires Python $MinPythonMajor.$MinPythonMinor or newer. $found" @(
        'Install Python 3.12 (recommended) with either:',
        '  winget install -e --id Python.Python.3.12',
        '  https://www.python.org/downloads/windows/',
        '',
        'With the python.org installer, TICK "Add python.exe to PATH" on the',
        'first screen, then open a NEW terminal and re-run this script.',
        '',
        'If typing "python" opens the Microsoft Store, turn the alias off in',
        'Settings > Apps > Advanced app settings > App execution aliases.'
    )
}

# Prefer the newest interpreter we can actually use.
$selected = $usable | Sort-Object -Property Major, Minor -Descending | Select-Object -First 1
Write-Ok "Python $($selected.Version)"
Write-Info $selected.Path

# 64-bit is required: PyTorch publishes no 32-bit Windows wheels.
$archOut = @(& $selected.Path -c 'import platform, struct; print(platform.machine()); print(struct.calcsize("P") * 8)')
if ($archOut.Count -ge 2) {
    if ($archOut[1].ToString().Trim() -ne '64') {
        Stop-WithHelp 'A 32-bit Python was found. PyTorch only ships 64-bit Windows wheels.' @(
            'Install "Windows installer (64-bit)" from python.org and re-run.'
        )
    }
    if ($archOut[0].ToString().Trim() -eq 'ARM64') {
        Write-Warn 'ARM64 Python detected. PyTorch has no official win_arm64 wheels.'
        Write-Info 'Install the x64 build of Python instead (it runs fine under emulation).'
    }
}

# --- Step 2: environment sanity checks ---------------------------------------
Write-Step 'Step 2/7  Checking the environment'

# Windows rejects paths over 260 characters unless long paths are enabled, and
# HuggingFace checkpoint directories nest deeply enough to hit that.
$longPaths = 0
try {
    $longPaths = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
                                   -Name 'LongPathsEnabled' -ErrorAction Stop).LongPathsEnabled
} catch {
    $longPaths = 0
}
if ($longPaths -eq 1) {
    Write-Ok 'Long path support is enabled'
} else {
    Write-Warn 'Long path support is OFF (260-character limit)'
    Write-Info 'Model checkpoints nest deeply and can exceed it. Enable it once from'
    Write-Info 'an ADMINISTRATOR PowerShell, then reboot:'
    Write-Info '  Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -Type DWord'
}

# A project inside OneDrive gets its files virtualised, which breaks training
# checkpoints mid-run in ways that are hard to diagnose.
if ($ProjectRoot -like '*OneDrive*') {
    Write-Warn 'This folder is inside OneDrive'
    Write-Info 'OneDrive can lock or dehydrate files while a model is training.'
    Write-Info 'Move the project somewhere local, e.g. C:\Dev\LLM_Tool.'
}

try {
    $driveName = (Split-Path -Qualifier $ProjectRoot).TrimEnd(':')
    $free = (Get-PSDrive -Name $driveName -ErrorAction Stop).Free
    if ($null -ne $free) {
        $freeGb = [math]::Round($free / 1GB, 1)
        if ($freeGb -lt 12) {
            Write-Warn "Only $freeGb GB free on drive $driveName - the full install needs about 10 GB"
        } else {
            Write-Ok "$freeGb GB free on drive $driveName"
        }
    }
} catch {
    # Network share or an unusual provider: skip the check rather than fail.
}

# --- Step 3: create the virtual environment ----------------------------------
Write-Step 'Step 3/7  Creating the virtual environment (.venv)'

$VenvDir    = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

if (Test-Path -LiteralPath $VenvDir) {
    $doRecreate = [bool]$Recreate
    if (-not $doRecreate) {
        Write-Warn '.venv already exists'
        $answer = Read-Host '       Remove and recreate it? [y/N]'
        $doRecreate = $answer -match '^[Yy]'
    }
    if ($doRecreate) {
        try {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
            Write-Ok 'Removed the existing virtual environment'
        } catch {
            Stop-WithHelp 'Could not delete .venv - a file in it is still in use.' @(
                'Close every terminal, VS Code window and Python process using this',
                'project (deactivate first), then re-run this script.'
            )
        }
    } else {
        Write-Info 'Reusing the existing virtual environment'
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $selected.Path -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
        Stop-WithHelp 'Could not create the virtual environment.' @(
            'If the error mentions "ensurepip", reinstall Python with the "pip"',
            'component ticked in the optional features screen.'
        )
    }
    Write-Ok "Created .venv with Python $($selected.Version)"
}

# --- Step 4: configure VS Code ------------------------------------------------
Write-Step 'Step 4/7  Configuring VS Code'

$VsCodeDir = Join-Path $ProjectRoot '.vscode'
New-Item -ItemType Directory -Path $VsCodeDir -Force | Out-Null

# defaultInterpreterPath holds a single value, so it has to name this platform's
# layout: Windows venvs put the interpreter in Scripts\, not bin/.
$settings = @'
{
    "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.env.windows": {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
    },
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
'@
Set-Content -LiteralPath (Join-Path $VsCodeDir 'settings.json') -Value $settings -Encoding UTF8
Write-Ok 'VS Code will use .venv\Scripts\python.exe'

# --- Step 5: upgrade the build tooling ---------------------------------------
Write-Step 'Step 5/7  Upgrading pip'

& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) { Stop-WithHelp 'Could not upgrade pip inside the virtual environment.' }
Write-Ok ((& $VenvPython -m pip --version | Select-Object -First 1) -replace ' from .*', '')

# --- Step 6: install LLM Tool -------------------------------------------------
Write-Step 'Step 6/7  Installing LLM Tool'

if ($Cuda -and $Cuda -ne 'cpu') {
    # PyPI's torch wheels for Windows are CPU-only, so the CUDA build has to come
    # from PyTorch's own index. Installing it first means the editable install
    # below sees the requirement as satisfied and leaves it alone.
    Write-Info "Installing the CUDA ($Cuda) build of PyTorch first..."
    & $VenvPython -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp "Could not install the $Cuda build of PyTorch." @(
            'Check that the tag exists at https://download.pytorch.org/whl/ and',
            'that your NVIDIA driver is recent enough for it.',
            'Re-running without -Cuda installs the CPU build instead.'
        )
    }
    Write-Ok "PyTorch ($Cuda) installed"
}

$target = switch ($Preset) {
    'core' { '.' }
    'all'  { '.[all]' }
    'dev'  { '.[dev]' }
    'full' { '.[all,dev]' }
}

Write-Info "pip install -e `"$target`"   (downloads 3-6 GB; expect 5-20 minutes)"
& $VenvPython -m pip install -e $target
if ($LASTEXITCODE -ne 0) {
    Stop-WithHelp 'Dependency installation failed.' @(
        'If the log says "Microsoft Visual C++ 14.0 or greater is required", a',
        'package fell back to a source build. Install the C++ build tools:',
        '  winget install -e --id Microsoft.VisualStudio.2022.BuildTools',
        '(tick "Desktop development with C++"), then re-run this script.',
        '',
        'If one optional package is the culprit, install the core preset:',
        '  .\install.ps1 -Preset core'
    )
}
Write-Ok 'LLM Tool installed'

# --- Step 7: verify -----------------------------------------------------------
Write-Step 'Step 7/7  Verifying the installation'

if ($SkipVerify) {
    Write-Info 'Skipped (-SkipVerify)'
} else {
    & $VenvPython (Join-Path $ProjectRoot 'verify_installation.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Warn 'Some verification checks failed - see the report above.'
    }
}

# --- Done ---------------------------------------------------------------------
Write-Host ''
Write-Host ('=' * 78) -ForegroundColor Green
Write-Host '  INSTALLATION COMPLETE' -ForegroundColor Green
Write-Host ('=' * 78) -ForegroundColor Green
Write-Host ''
Write-Host '  Next steps' -ForegroundColor White
Write-Host ''
Write-Host '    1. Activate the environment in this terminal:' -ForegroundColor Gray
Write-Host '         .\.venv\Scripts\Activate.ps1      (PowerShell)' -ForegroundColor Cyan
Write-Host '         .venv\Scripts\activate.bat        (Command Prompt)' -ForegroundColor Cyan
Write-Host ''
Write-Host '    2. Launch it:' -ForegroundColor Gray
Write-Host '         llm-tool' -ForegroundColor Cyan
Write-Host ''
Write-Host '    3. Optional - local LLMs with Ollama:' -ForegroundColor Gray
Write-Host '         winget install -e --id Ollama.Ollama' -ForegroundColor Cyan
Write-Host ''
Write-Host '    4. Optional - cloud LLM keys (persisted for future terminals):' -ForegroundColor Gray
Write-Host '         setx OPENAI_API_KEY "sk-..."' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Documentation: README.md  |  Windows guide: docs\WINDOWS.md' -ForegroundColor DarkGray
Write-Host ''

if (-not $Cuda) {
    if (Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue) {
        Write-Warn 'An NVIDIA GPU was detected but PyTorch was installed CPU-only.'
        Write-Info 'For GPU training, re-run:  .\install.ps1 -Preset all -Cuda cu126'
        Write-Host ''
    }
}
