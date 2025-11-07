# Auto-activate the workspace virtual environment for VS Code integrated terminals.
# This file is loaded because VS Code sets ZDOTDIR to this directory.

# Resolve workspace root from the location of this file (even when sourced).
_vscode_zsh_file="${(%):-%N}"
_workspace_root="$(cd "$(dirname "$_vscode_zsh_file")/.." && pwd -P)"
_venv_path="${WORKSPACE_VENV:-${_workspace_root}/.venv}"

if [ -f "${_venv_path}/bin/activate" ]; then
    source "${_venv_path}/bin/activate"
else
    echo "[LLM Tool] No virtual environment found at ${_venv_path}."
    echo "[LLM Tool] Run ./install.sh --all to create it."
fi

# Load the user's regular zsh configuration to preserve custom aliases and prompts.
if [ -f "${HOME}/.zshrc" ]; then
    source "${HOME}/.zshrc"
fi

unset _vscode_zsh_file _workspace_root _venv_path
