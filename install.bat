@echo off
REM ============================================================================
REM  LLM Tool - Windows installer (double-click me)
REM
REM  This is a thin wrapper around install.ps1. It exists because PowerShell
REM  refuses to run downloaded .ps1 files under the default execution policy,
REM  which is the single most common reason a Windows install stalls before it
REM  starts. Launching PowerShell with -ExecutionPolicy Bypass applies only to
REM  this one process and changes nothing on the machine.
REM
REM  Usage:
REM    install.bat                  installs the recommended set (all features)
REM    install.bat -Preset core     installs the pipeline only
REM    install.bat -Preset all -Cuda cu126    installs with GPU PyTorch
REM
REM  Any argument is forwarded verbatim to install.ps1.
REM ============================================================================

setlocal

REM Run from the folder this file lives in, so double-clicking works no matter
REM what the current directory happens to be.
pushd "%~dp0"

REM Prefer PowerShell 7+ when it is installed; fall back to Windows PowerShell.
where /q pwsh.exe
if %ERRORLEVEL% EQU 0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
)

set "INSTALL_EXIT=%ERRORLEVEL%"

popd

REM Keep the window open when the script was double-clicked from Explorer,
REM otherwise the result would flash past unread.
echo(
if %INSTALL_EXIT% NEQ 0 (
    echo Installation failed with exit code %INSTALL_EXIT%.
    echo See docs\WINDOWS.md for troubleshooting.
)
echo(
pause

exit /b %INSTALL_EXIT%
