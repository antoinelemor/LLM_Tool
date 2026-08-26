# VS Code Configuration

This directory contains pre-configured settings for Visual Studio Code to ensure optimal development experience with LLM Tool.

## Automatic Configuration

When you run the installer for your platform — `install.bat` on Windows,
`install.sh` on macOS and Linux — VS Code is automatically configured to:

- ✅ Use the `.venv` virtual environment as the default Python interpreter
  (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere)
- ✅ Automatically activate the virtual environment in terminals
- ✅ Force UTF-8 in Windows integrated terminals, so the CLI's emoji, box-drawing
  characters and accented text render instead of raising `UnicodeEncodeError`
- ✅ Enable pytest for testing
- ✅ Format code on save with Black and organize imports

The committed `settings.json` deliberately leaves `python.defaultInterpreterPath`
unset: it holds a single value, and the interpreter lives at a different path on
Windows than on macOS/Linux. The Python extension auto-detects a `.venv` in the
workspace root on every platform, and each installer writes the exact path for
the machine it runs on.

## Manual Configuration

If you need to manually select the Python interpreter:

1. Open Command Palette: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: "Python: Select Interpreter"
3. Choose the one inside this project's `.venv`:
   - **Windows**: `.\.venv\Scripts\python.exe`
   - **macOS/Linux**: `./.venv/bin/python`

## Customization

You can customize these settings by editing `.vscode/settings.json`. The file is version-controlled to ensure consistent development experience across the team.

If you want workspace-specific settings that won't be committed, create a `.vscode/settings.local.json` file (this pattern is not currently implemented but can be added to `.gitignore` if needed).

## Recommended Extensions

For the best development experience, install these VS Code extensions:

- **Python** (ms-python.python) - Required for Python support
- **Pylance** (ms-python.vscode-pylance) - Fast Python language server
- **Black Formatter** (ms-python.black-formatter) - Code formatting
- **Flake8** (ms-python.flake8) - Linting
- **Jupyter** (ms-toolsai.jupyter) - Notebook support
- **GitLens** (eamodio.gitlens) - Enhanced Git features

## Troubleshooting

### VS Code doesn't recognize the virtual environment

1. Reload the VS Code window: `Cmd+Shift+P` → "Developer: Reload Window"
2. Close and reopen VS Code
3. Manually select the interpreter as described above

### Terminal doesn't activate the virtual environment

1. Check that `python.terminal.activateEnvironment` is set to `true` in settings
2. Close all terminal instances and open a new one
3. Manually activate:
   - **Windows PowerShell**: `.\.venv\Scripts\Activate.ps1`
   - **Windows Command Prompt**: `.venv\Scripts\activate.bat`
   - **macOS/Linux**: `source .venv/bin/activate`

### Windows: "running scripts is disabled on this system"

PowerShell blocks `Activate.ps1` under the default execution policy. Either allow
local scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

…or set the integrated terminal's default profile to **Command Prompt** and use
`.venv\Scripts\activate.bat`.

### Linting/Formatting not working

Ensure development dependencies are installed:

```powershell
.\.venv\Scripts\Activate.ps1     # Windows
```
```bash
source .venv/bin/activate        # macOS / Linux
```
```bash
pip install -e ".[dev]"
```
