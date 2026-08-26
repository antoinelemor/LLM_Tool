@echo off
REM ============================================================================
REM  LLM Tool - developer shortcuts for Windows
REM
REM  Windows has no `make`, so this file mirrors the Makefile target for target:
REM  whatever CONTRIBUTING.md tells you to run as `make <target>`, run here as
REM  `make.bat <target>`.
REM
REM    make.bat help
REM    make.bat install | install-dev | install-all
REM    make.bat run | test | lint | format | check-build
REM    make.bat clean | clean-all
REM ============================================================================

setlocal enabledelayedexpansion
pushd "%~dp0"

REM Prefer the project's virtual environment; fall back to the launcher.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3"

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=help"

if /I "%TARGET%"=="help"        goto :help
if /I "%TARGET%"=="venv"        goto :venv
if /I "%TARGET%"=="install"     goto :install
if /I "%TARGET%"=="install-dev" goto :install_dev
if /I "%TARGET%"=="install-all" goto :install_all
if /I "%TARGET%"=="run"         goto :run
if /I "%TARGET%"=="test"        goto :test
if /I "%TARGET%"=="lint"        goto :lint
if /I "%TARGET%"=="format"      goto :format
if /I "%TARGET%"=="check-build" goto :check_build
if /I "%TARGET%"=="clean"       goto :clean
if /I "%TARGET%"=="clean-all"   goto :clean_all

echo Unknown target: %TARGET%
echo(
goto :help

:help
echo(
echo   LLM TOOL - make.bat targets
echo(
echo   Setup
echo     make.bat venv           Create the virtual environment
echo     make.bat install        Install with core dependencies
echo     make.bat install-dev    Install with development dependencies
echo     make.bat install-all    Install with all dependencies
echo(
echo   Development
echo     make.bat run            Launch the LLM Tool CLI
echo     make.bat test           Run the test suite
echo     make.bat lint           Run flake8
echo     make.bat format         Run black and isort
echo     make.bat check-build    Validate the package metadata
echo(
echo   Cleanup
echo     make.bat clean          Remove build artifacts and caches
echo     make.bat clean-all      Also remove .venv
echo(
echo   First-time install: run install.bat instead.
echo(
goto :done

:venv
echo Creating virtual environment...
py -3 -m venv .venv
echo Activate with: .\.venv\Scripts\Activate.ps1
goto :done

:install
"%PY%" -m pip install -e .
goto :done

:install_dev
"%PY%" -m pip install -e ".[dev]"
goto :done

:install_all
"%PY%" -m pip install -e ".[all]"
goto :done

:run
"%PY%" -m llm_tool
goto :done

:test
"%PY%" -m pytest tests/ -v
goto :done

:lint
"%PY%" -m flake8 llm_tool/ --max-line-length=120 --ignore=E203,W503
goto :done

:format
"%PY%" -m black llm_tool/
"%PY%" -m isort llm_tool/
goto :done

:check_build
"%PY%" -m pip install --quiet --upgrade build twine
"%PY%" -m build --sdist --wheel --outdir dist/
"%PY%" -m twine check dist/*
goto :done

:clean
echo Cleaning build artifacts and cache...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%D in (*.egg-info) do rmdir /s /q "%%D"
REM Skip .venv: its caches belong to the installed packages, not to this project.
for /d /r %%D in (__pycache__) do (
    echo %%D | find /i "\.venv\" >nul || if exist "%%D" rmdir /s /q "%%D"
)
for /d /r %%D in (.pytest_cache) do (
    echo %%D | find /i "\.venv\" >nul || if exist "%%D" rmdir /s /q "%%D"
)
echo Cleanup complete.
goto :done

:clean_all
call "%~f0" clean
echo Removing virtual environment...
if exist .venv rmdir /s /q .venv
echo Deep cleanup complete.
goto :done

:done
popd
endlocal
