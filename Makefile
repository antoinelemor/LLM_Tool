.PHONY: help install install-dev install-all clean test lint format run check-setup venv

help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║              LLM TOOL - Makefile Commands                ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make venv          Create virtual environment"
	@echo "  make install       Install package with core dependencies"
	@echo "  make install-dev   Install package with dev dependencies"
	@echo "  make install-all   Install package with all dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  make run           Launch LLM Tool CLI"
	@echo "  make test          Run test suite"
	@echo "  make lint          Run linting checks (flake8)"
	@echo "  make format        Format code with black and isort"
	@echo "  make check-setup   Verify setup.py configuration"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  make clean         Remove build artifacts and cache"
	@echo "  make clean-all     Remove build artifacts, cache, and venv"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "✓ Virtual environment created at .venv/"
	@echo "Activate with: source .venv/bin/activate (macOS/Linux)"
	@echo "             or .venv\\Scripts\\activate (Windows)"

install:
	@echo "Installing LLM Tool with core dependencies..."
	pip install -e .
	@echo "✓ Installation complete!"

install-dev:
	@echo "Installing LLM Tool with development dependencies..."
	pip install -e ".[dev]"
	@echo "✓ Installation complete!"

install-all:
	@echo "Installing LLM Tool with all dependencies..."
	pip install -e ".[all]"
	@echo "✓ Installation complete!"

run:
	@echo "Launching LLM Tool CLI..."
	llm-tool

test:
	@echo "Running test suite..."
	pytest tests/ -v

lint:
	@echo "Running linting checks..."
	flake8 llm_tool/ --max-line-length=120 --ignore=E203,W503

format:
	@echo "Formatting code with black..."
	black llm_tool/
	@echo "Sorting imports with isort..."
	isort llm_tool/
	@echo "✓ Code formatting complete!"

check-setup:
	@echo "Verifying setup.py configuration..."
	python setup.py check

clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf .venv/
	@echo "✓ Deep cleanup complete!"

# Quick start for new users
quickstart: venv
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║            LLM TOOL - Quick Start Guide                  ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate virtual environment:"
	@echo "     source .venv/bin/activate"
	@echo ""
	@echo "  2. Install LLM Tool:"
	@echo "     make install          (core features)"
	@echo "     make install-all      (all features)"
	@echo ""
	@echo "  3. Launch the CLI:"
	@echo "     make run"
	@echo ""
	@echo "  4. Read the docs:"
	@echo "     cat README.md | less"
	@echo ""
