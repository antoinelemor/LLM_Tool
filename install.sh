#!/bin/bash
#
# LLM Tool - Quick Installation Script
#
# This script automates the installation process for LLM Tool on macOS and Linux.
# It creates a virtual environment, installs dependencies, and verifies the installation.
#
# Windows users: run install.bat instead (or install.ps1 from PowerShell).
# See docs/WINDOWS.md.
#
# Usage:
#   ./install.sh              # Install core features
#   ./install.sh --all        # Install all features (recommended)
#   ./install.sh --dev        # Install with development tools
#
# Author: Antoine Lemor

set -e  # Exit on error

# Windows has its own installer: this script's venv activation, chmod and
# read -p all assume a POSIX shell, and under Git Bash the interpreter would
# land in .venv/Scripts rather than .venv/bin.
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*)
        echo ""
        echo "This is the macOS/Linux installer."
        echo ""
        echo "On Windows, run this instead (from PowerShell or Explorer):"
        echo "    ./install.bat"
        echo ""
        echo "Full guide: docs/WINDOWS.md"
        echo ""
        exit 1
        ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ASCII Banner
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║     ██╗     ██╗     ███╗   ███╗    ████████╗ ██████╗  ██████╗ ██╗             ║"
echo "║     ██║     ██║     ████╗ ████║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║             ║"
echo "║     ██║     ██║     ██╔████╔██║       ██║   ██║   ██║██║   ██║██║             ║"
echo "║     ██║     ██║     ██║╚██╔╝██║       ██║   ██║   ██║██║   ██║██║             ║"
echo "║     ███████╗███████╗██║ ╚═╝ ██║       ██║   ╚██████╔╝╚██████╔╝███████╗        ║"
echo "║     ╚══════╝╚══════╝╚═╝     ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝        ║"
echo "║                                                                               ║"
echo "║               🤖 AI-Powered Annotation & ML Training Pipeline                 ║"
echo "║               Transform Text into Trained Models in 45 Minutes                ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
INSTALL_TYPE="core"
if [ "$1" == "--all" ]; then
    INSTALL_TYPE="all"
    echo -e "${BLUE}Installing with ALL features...${NC}"
elif [ "$1" == "--dev" ]; then
    INSTALL_TYPE="dev"
    echo -e "${BLUE}Installing with DEVELOPMENT tools...${NC}"
else
    echo -e "${BLUE}Installing with CORE features...${NC}"
    echo -e "${YELLOW}Tip: Use './install.sh --all' for all features${NC}"
fi
echo ""

# Check and select Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Detecting and selecting Python interpreter..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Display diagnostic information
echo "Python diagnostics:"
if command -v python &> /dev/null; then
    echo "  which python:  $(which python 2>/dev/null || echo 'not found')"
    echo "  python --version: $(python --version 2>&1 || echo 'not found')"
else
    echo "  'python' command: not found"
fi

if command -v python3 &> /dev/null; then
    echo "  which python3: $(which python3 2>/dev/null || echo 'not found')"
    echo "  python3 --version: $(python3 --version 2>&1 || echo 'not found')"
else
    echo "  'python3' command: not found"
fi

if command -v python3.11 &> /dev/null; then
    echo "  which python3.11: $(which python3.11 2>/dev/null || echo 'not found')"
    echo "  python3.11 --version: $(python3.11 --version 2>&1 || echo 'not found')"
else
    echo "  'python3.11' command: not found"
fi
echo ""

# Function to check if a Python version is >= 3.11
check_python_version() {
    local python_bin=$1
    if ! command -v "$python_bin" &> /dev/null; then
        return 1
    fi

    local version=$($python_bin --version 2>&1 | cut -d' ' -f2)
    local major=$(echo $version | cut -d'.' -f1)
    local minor=$(echo $version | cut -d'.' -f2)

    if [ "$major" -gt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -ge 11 ]); then
        echo "$python_bin:$version"
        return 0
    fi
    return 1
}

# Try to find a suitable Python interpreter (priority order)
PYTHON_BIN=""
PYTHON_VERSION=""

echo "Searching for Python ≥3.11..."
echo ""

# 1. Try /opt/homebrew/bin/python3.11 (Homebrew on Apple Silicon)
if [ -z "$PYTHON_BIN" ] && [ -f "/opt/homebrew/bin/python3.11" ]; then
    result=$(check_python_version "/opt/homebrew/bin/python3.11")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found /opt/homebrew/bin/python3.11 version $PYTHON_VERSION"
    fi
fi

# 2. Try python3.11 in PATH
if [ -z "$PYTHON_BIN" ]; then
    result=$(check_python_version "python3.11")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found python3.11 in PATH version $PYTHON_VERSION"
    fi
fi

# 3. Try python3.12 in PATH
if [ -z "$PYTHON_BIN" ]; then
    result=$(check_python_version "python3.12")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found python3.12 in PATH version $PYTHON_VERSION"
    fi
fi

# 4. Try python3.13 in PATH
if [ -z "$PYTHON_BIN" ]; then
    result=$(check_python_version "python3.13")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found python3.13 in PATH version $PYTHON_VERSION"
    fi
fi

# 5. Try generic python3 (only if version >= 3.11)
if [ -z "$PYTHON_BIN" ]; then
    result=$(check_python_version "python3")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found python3 version $PYTHON_VERSION"
    fi
fi

# 6. Try generic python (only if version >= 3.11)
if [ -z "$PYTHON_BIN" ]; then
    result=$(check_python_version "python")
    if [ $? -eq 0 ]; then
        PYTHON_BIN=$(echo $result | cut -d':' -f1)
        PYTHON_VERSION=$(echo $result | cut -d':' -f2)
        echo "  ✓ Found python version $PYTHON_VERSION"
    fi
fi

# If no suitable Python was found, exit with error
if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo -e "${RED}✗ No compatible Python interpreter found${NC}"
    echo ""
    echo "LLM Tool requires Python 3.11 or higher."
    echo ""
    echo "Please install Python 3.11+ from one of these sources:"
    echo "  • Official Python: https://www.python.org/downloads/"
    echo "  • Homebrew (macOS): brew install python@3.11"
    echo "  • Conda: conda install python=3.11"
    echo ""
    echo "After installation, make sure the Python binary is in your PATH."
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Selected Python interpreter:${NC}"
echo -e "  Binary: ${BLUE}$PYTHON_BIN${NC}"
echo -e "  Version: ${BLUE}$PYTHON_VERSION${NC}"
echo -e "  Location: ${BLUE}$(which $PYTHON_BIN)${NC}"
echo ""

# Create virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Creating virtual environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists at .venv/${NC}"
    read -p "  Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        echo -e "${GREEN}✓ Removed existing virtual environment${NC}"
    else
        echo -e "${BLUE}→ Using existing virtual environment${NC}"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment using: $PYTHON_BIN"
    $PYTHON_BIN -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created at .venv/ using $PYTHON_BIN${NC}"
fi
echo ""

# Configure VS Code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Configuring VS Code..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p .vscode

# The interpreter path is written per-platform: this one is the POSIX layout.
# install.ps1 writes the .venv\Scripts\python.exe form on Windows.
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
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
    "terminal.integrated.env.windows": {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
    },
    "files.eol": "\n",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
EOF

echo -e "${GREEN}✓ VS Code configured to use .venv as default Python interpreter${NC}"
echo ""

# Activate virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Activating virtual environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Upgrading pip..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Invoked through the interpreter so the upgrade targets this venv's pip even
# if the shell's `pip` still resolves to another environment.
python -m pip install --upgrade pip setuptools wheel > /dev/null
echo -e "${GREEN}✓ pip upgraded to latest version${NC}"
echo ""

# Install LLM Tool
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Installing LLM Tool..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$INSTALL_TYPE" == "all" ]; then
    python -m pip install -e ".[all]"
elif [ "$INSTALL_TYPE" == "dev" ]; then
    python -m pip install -e ".[dev]"
else
    python -m pip install -e .
fi

echo -e "${GREEN}✓ LLM Tool installed successfully${NC}"
echo ""

# Verify installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Verifying installation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python verify_installation.py

# Success message
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║                  🎉 INSTALLATION COMPLETE! 🎉            ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo -e "     ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "  2. Launch LLM Tool:"
echo -e "     ${BLUE}llm-tool${NC}"
echo ""
echo "  3. Read the documentation:"
echo -e "     ${BLUE}README.md${NC}  (features and the five modes)"
echo -e "     ${BLUE}docs/modes_reference.md${NC}"
echo ""
echo "  4. VS Code users:"
echo -e "     ${GREEN}✓ VS Code is already configured to use .venv${NC}"
echo -e "     ${GREEN}  Just open/reload the workspace in VS Code!${NC}"
echo ""
echo "Optional setup:"
echo ""
echo "  • Install Ollama for local LLMs:"
echo -e "    ${YELLOW}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
echo ""
echo "  • Configure API keys (if using cloud LLMs):"
echo -e "    ${YELLOW}export OPENAI_API_KEY='sk-...'${NC}"
echo ""
echo "For help and support:"
echo "  • README: README.md"
echo "  • Docs: docs/"
echo "  • Issues: https://github.com/antoinelemor/LLM_Tool/issues"
echo ""
echo "Happy annotating! 🚀"
echo ""
