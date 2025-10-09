#!/bin/bash
#
# LLM Tool - Quick Installation Script
#
# This script automates the installation process for LLM Tool.
# It creates a virtual environment, installs dependencies, and verifies the installation.
#
# Usage:
#   ./install.sh              # Install core features
#   ./install.sh --all        # Install all features (recommended)
#   ./install.sh --dev        # Install with development tools
#
# Author: Antoine Lemor

set -e  # Exit on error

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

# Check Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python version..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.9 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}✗ Python $PYTHON_VERSION is too old${NC}"
    echo "  Required: Python 3.9 or higher"
    echo "  Found: Python $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
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
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created at .venv/${NC}"
fi
echo ""

# Configure VS Code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Configuring VS Code..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p .vscode

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
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
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

pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded to latest version${NC}"
echo ""

# Install LLM Tool
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Installing LLM Tool..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$INSTALL_TYPE" == "all" ]; then
    pip install -e ".[all]"
elif [ "$INSTALL_TYPE" == "dev" ]; then
    pip install -e ".[dev]"
else
    pip install -e .
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
echo "  3. Try the examples:"
echo -e "     ${BLUE}python examples/quickstart_annotation.py${NC}"
echo ""
echo "  4. Read the documentation:"
echo -e "     ${BLUE}cat README.md | less${NC}"
echo ""
echo "  5. VS Code users:"
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
echo "  • Examples: examples/"
echo "  • Docs: docs/"
echo ""
echo "Happy annotating! 🚀"
echo ""
