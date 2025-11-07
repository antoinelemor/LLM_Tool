# Auto-activate the workspace virtual environment for VS Code integrated terminals.

setopt PROMPT_SUBST

_vscode_zsh_file="${(%):-%N}"
_workspace_root="$(cd "$(dirname "${_vscode_zsh_file}")/.." && pwd -P)"
_venv_path="${WORKSPACE_VENV:-${_workspace_root}/.venv}"

if [ -f "${HOME}/.zshrc" ]; then
  source "${HOME}/.zshrc"
fi

if [ -f "${_venv_path}/bin/activate" ]; then
  source "${_venv_path}/bin/activate"
else
  echo "[LLM Tool] No virtual environment found at ${_venv_path}"
  echo "[LLM Tool] Run ./install.sh --all to create it."
fi

unset _workspace_root _vscode_zsh_file _venv_path
