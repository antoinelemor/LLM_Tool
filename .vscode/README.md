# VS Code Configuration

This directory contains pre-configured settings for Visual Studio Code to ensure optimal development experience with LLM Tool.

## Automatic Configuration

When you run `install.sh`, VS Code is automatically configured to:

- ✅ Use the `.venv` virtual environment as the default Python interpreter
- ✅ Automatically activate the virtual environment in terminals
- ✅ Enable pytest for testing
- ✅ Configure Black for code formatting
- ✅ Enable Flake8 and MyPy for linting
- ✅ Format code on save
- ✅ Organize imports automatically

## Manual Configuration

If you need to manually select the Python interpreter:

1. Open Command Palette: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: "Python: Select Interpreter"
3. Choose: `./.venv/bin/python`

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
3. Manually activate: `source .venv/bin/activate`

### Linting/Formatting not working

Ensure development dependencies are installed:
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```
