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
Fallback CLI that displays a warning message when required dependencies are missing.
This CLI is used when the advanced CLI cannot be loaded.

Author:
-------
Antoine Lemor
"""

import sys


class LLMToolCLI:
    """Fallback CLI that displays dependency warning"""

    def __init__(self):
        """Initialize the fallback CLI"""
        pass

    def display_warning(self):
        """Display a large warning message about missing dependencies"""
        # Colors
        RED = '\033[0;31m'
        YELLOW = '\033[1;33m'
        CYAN = '\033[0;36m'
        WHITE = '\033[1;37m'
        NC = '\033[0m'  # No Color

        print()
        print(f"{CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗{NC}")
        print(f"{CYAN}║                                                                               ║{NC}")
        print(f"{CYAN}║{WHITE}██╗     ██╗     ███╗   ███╗    ████████╗ ██████╗  ██████╗ ██╗.    {CYAN}║{NC}")
        print(f"{CYAN}║{WHITE}██║     ██║     ████╗ ████║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     {CYAN}║{NC}")
        print(f"{CYAN}║{WHITE}██║     ██║     ██╔████╔██║       ██║   ██║   ██║██║   ██║██║     {CYAN}║{NC}")
        print(f"{CYAN}║{WHITE}██║     ██║     ██║╚██╔╝██║       ██║   ██║   ██║██║   ██║██║.    {CYAN}║{NC}")
        print(f"{CYAN}║{WHITE}███████╗███████╗██║ ╚═╝ ██║       ██║   ╚██████╔╝╚██████╔╝███████╗{CYAN}║{NC}")
        print(f"{CYAN}║{WHITE}╚══════╝╚══════╝╚═╝     ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝{CYAN}║{NC}")
        print(f"{CYAN}║                                                                               ║{NC}")
        print(f"{CYAN}╚═══════════════════════════════════════════════════════════════════════════════╝{NC}")
        print()

        print(f"{RED}╔═══════════════════════════════════════════════════════════════════════════════╗{NC}")
        print(f"{RED}║                                                                               ║{NC}")
        print(f"{RED}║                           {YELLOW}⚠️  WARNING  ⚠️{RED}                                      ║{NC}")
        print(f"{RED}║                                                                               ║{NC}")
        print(f"{RED}║{NC}  {WHITE}REQUIRED DEPENDENCIES NOT INSTALLED{RED}                                         ║{NC}")
        print(f"{RED}║                                                                               ║{NC}")
        print(f"{RED}║{NC}  The advanced CLI interface is not available because essential            {RED}║{NC}")
        print(f"{RED}║{NC}  dependencies are missing from your installation.                         {RED}║{NC}")
        print(f"{RED}║                                                                               ║{NC}")
        print(f"{RED}║{NC}  This usually happens when LLM Tool was not properly installed.           {RED}║{NC}")
        print(f"{RED}║                                                                               ║{NC}")
        print(f"{RED}╚═══════════════════════════════════════════════════════════════════════════════╝{NC}")
        print()

        print(f"{YELLOW}┌─────────────────────────────────────────────────────────────────────────────┐{NC}")
        print(f"{YELLOW}│{NC}  {WHITE}HOW TO FIX THIS:{NC}                                                           {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}                                                                             {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}  {CYAN}Option 1:{NC} Run the automated installation script                           {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}    {WHITE}$ ./install.sh --all{NC}                                                     {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}                                                                             {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}  {CYAN}Option 2:{NC} Manually install all dependencies                                {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}    {WHITE}$ pip install -e \".[all]\"{NC}                                                {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}                                                                             {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}  {CYAN}Option 3:{NC} Run the cryptography fix script (if already installed)          {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}    {WHITE}$ ./fix_cryptography.sh{NC}                                                  {YELLOW}│{NC}")
        print(f"{YELLOW}│{NC}                                                                             {YELLOW}│{NC}")
        print(f"{YELLOW}└─────────────────────────────────────────────────────────────────────────────┘{NC}")
        print()

        print(f"{CYAN}┌─────────────────────────────────────────────────────────────────────────────┐{NC}")
        print(f"{CYAN}│{NC}  {WHITE}WHAT YOU'RE MISSING:{NC}                                                        {CYAN}│{NC}")
        print(f"{CYAN}│{NC}                                                                             {CYAN}│{NC}")
        print(f"{CYAN}│{NC}  ❌ Advanced Rich CLI with beautiful menus and progress bars              {CYAN}│{NC}")
        print(f"{CYAN}│{NC}  ❌ Interactive annotation workflow                                        {CYAN}│{NC}")
        print(f"{CYAN}│{NC}  ❌ Model training and benchmarking                                        {CYAN}│{NC}")
        print(f"{CYAN}│{NC}  ❌ Validation and quality control tools                                   {CYAN}│{NC}")
        print(f"{CYAN}│{NC}  ❌ Complete pipeline automation                                           {CYAN}│{NC}")
        print(f"{CYAN}│{NC}                                                                             {CYAN}│{NC}")
        print(f"{CYAN}└─────────────────────────────────────────────────────────────────────────────┘{NC}")
        print()

        print(f"{WHITE}For help and support:{NC}")
        print(f"  • Documentation: {CYAN}README.md{NC}")
        print(f"  • Examples: {CYAN}examples/{NC}")
        print(f"  • Issues: {CYAN}https://github.com/antoine-lemor/LLMTool/issues{NC}")
        print()

    def run(self):
        """Display warning and exit"""
        self.display_warning()

        print(f"\033[1;33m{'─' * 79}\033[0m")
        print(f"\033[1;37mLLM Tool cannot start without required dependencies.\033[0m")
        print(f"\033[1;37mPlease install them using one of the methods above.\033[0m")
        print(f"\033[1;33m{'─' * 79}\033[0m")
        print()

        sys.exit(1)


def main():
    """Entry point for the fallback CLI"""
    cli = LLMToolCLI()
    cli.run()


if __name__ == "__main__":
    main()
