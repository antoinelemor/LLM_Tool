#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
main_cli.py

MAIN OBJECTIVE:
---------------
Fallback CLI that displays a warning message when required dependencies are
missing. This CLI is used when the advanced CLI cannot be loaded.

Because it runs precisely when the installation is broken, it uses nothing
beyond the standard library, prints plain ASCII, and gives instructions for the
platform it is actually running on.

Dependencies:
-------------
- os
- sys

Author:
-------
Antoine Lemor
"""

import os
import sys

WIDTH = 74


def _supports_colour() -> bool:
    """
    Whether ANSI colour codes will be rendered rather than printed literally.

    Returns
    -------
    bool
        False when output is redirected, when NO_COLOR is set, or on a Windows
        console that has not negotiated virtual-terminal processing -- where the
        escapes would show up as literal garbage such as ``<-[0;31m``.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # llm_tool.platform_compat turns VT processing on at import, but this module
    # can be reached before that or with it having failed, so check the markers
    # that modern Windows terminals set for themselves.
    return bool(
        os.environ.get("WT_SESSION")
        or os.environ.get("ConEmuANSI") == "ON"
        or os.environ.get("TERM_PROGRAM", "").lower() == "vscode"
    )


class LLMToolCLI:
    """Fallback CLI that displays dependency warning"""

    def __init__(self):
        """Initialize the fallback CLI"""
        self._colour = _supports_colour()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _c(self, text: str, code: str) -> str:
        """Wrap `text` in an ANSI colour `code`, or return it unchanged."""
        return f"\033[{code}m{text}\033[0m" if self._colour else text

    def _rule(self, char: str = "=") -> None:
        print(char * WIDTH)

    def _heading(self, text: str) -> None:
        print()
        self._rule()
        print(f"  {self._c(text, '1;37')}")
        self._rule()

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def display_warning(self):
        """Display a warning message about missing dependencies."""
        windows = os.name == "nt"

        print()
        self._rule()
        print(self._c("  LLM TOOL - REQUIRED DEPENDENCIES ARE NOT INSTALLED", "1;31"))
        self._rule()
        print()
        print("  The interactive interface could not start because essential")
        print("  packages are missing from this Python environment.")
        print()
        print(f"  Interpreter: {sys.executable}")
        print(f"  Python:      {sys.version.split()[0]}")
        print()

        self._heading("HOW TO FIX IT")
        print()

        if windows:
            print(f"  {self._c('Option 1', '0;36')}  Run the installer (recommended)")
            print("      install.bat")
            print()
            print(f"  {self._c('Option 2', '0;36')}  Install into the active environment by hand")
            print("      .\\.venv\\Scripts\\Activate.ps1")
            print('      pip install -e ".[all]"')
            print()
            print(f"  {self._c('Option 3', '0;36')}  Start over from a clean environment")
            print("      install.bat -Recreate")
        else:
            print(f"  {self._c('Option 1', '0;36')}  Run the installer (recommended)")
            print("      ./install.sh --all")
            print()
            print(f"  {self._c('Option 2', '0;36')}  Install into the active environment by hand")
            print("      source .venv/bin/activate")
            print('      pip install -e ".[all]"')
            print()
            print(f"  {self._c('Option 3', '0;36')}  Check what is missing")
            print("      python verify_installation.py")

        print()

        self._heading("MOST LIKELY CAUSE")
        print()
        print("  The virtual environment is not active in this terminal.")
        if windows:
            print("  Activate it, then try again:")
            print()
            print("      .\\.venv\\Scripts\\Activate.ps1     (PowerShell)")
            print("      .venv\\Scripts\\activate.bat       (Command Prompt)")
        else:
            print("  Activate it, then try again:")
            print()
            print("      source .venv/bin/activate")
        print()
        print("  Your prompt shows (.venv) when it is active.")
        print()

        self._heading("DOCUMENTATION")
        print()
        print("  README.md")
        if windows:
            print("  docs\\WINDOWS.md          Windows install and troubleshooting")
        print("  https://github.com/antoinelemor/LLM_Tool/issues")
        print()

    def run(self):
        """Display warning and exit"""
        self.display_warning()
        self._rule("-")
        print("  LLM Tool cannot start without its dependencies.")
        self._rule("-")
        print()
        sys.exit(1)


def main():
    """Entry point for the fallback CLI"""
    cli = LLMToolCLI()
    cli.run()


if __name__ == "__main__":
    main()
