# Contributing to LLM Tool

Thank you for your interest in contributing to LLM Tool! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/LLM_Tool.git
   cd LLM_Tool
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/antoinelemor/LLM_Tool.git
   ```

## Development Setup

### Prerequisites

- **Python 3.11 or higher** (3.12 recommended), 64-bit
- Git

### Setup Instructions

**The fastest route on any platform is the installer**, which creates `.venv`,
installs the dev toolchain and configures VS Code:

```powershell
.\install.bat -Preset full      # Windows
```
```bash
./install.sh --dev              # macOS / Linux
```

<details>
<summary>Or set it up by hand</summary>

1. **Create and activate a virtual environment**:
   ```bash
   py -3.12 -m venv .venv                # Windows
   python3.12 -m venv .venv              # macOS / Linux

   .\.venv\Scripts\Activate.ps1          # Windows PowerShell
   .venv\Scripts\activate.bat            # Windows Command Prompt
   source .venv/Scripts/activate         # Windows Git Bash
   source .venv/bin/activate             # macOS / Linux
   ```

2. **Install in development mode with all dependencies**:
   ```bash
   pip install -e ".[all,dev]"
   ```

3. **Verify installation**:
   ```bash
   python verify_installation.py
   llm-tool --version
   ```

</details>

### Task shortcuts

`make` is not available on Windows, so every target has a `make.bat` twin with
the same name:

| Task | macOS / Linux | Windows |
|------|---------------|---------|
| Install (all) | `make install-all` | `make.bat install-all` |
| Run the CLI | `make run` | `make.bat run` |
| Tests | `make test` | `make.bat test` |
| Lint | `make lint` | `make.bat lint` |
| Format | `make format` | `make.bat format` |
| Validate packaging | `make check-build` | `make.bat check-build` |
| Clean | `make clean` | `make.bat clean` |

### Cross-platform expectations

This project is developed on macOS and must keep working on Windows and Linux.
CI (`.github/workflows/install.yml`) installs and launches on all three across
Python 3.11–3.13 on every change to `llm_tool/` or the packaging files. When you
touch anything that talks to the filesystem, the console or a subprocess:

- **Open text files with `encoding='utf-8'`.** Python defaults to the ANSI code
  page on Windows; this codebase handles French, Arabic and Chinese.
- **Add `newline=''`** to any `open()` feeding a `csv.reader`/`csv.writer`.
- **Never hardcode `/tmp`, `/usr`, or `~/.config`.** Use `tempfile.gettempdir()`,
  `Path.home()`, or `llm_tool.platform_compat.writable_dir()`.
- **Build filenames with `platform_compat.sanitize_path_component()`** — Windows
  rejects `: * ? " < > | \ /` and reserves `CON`, `NUL`, `AUX`, `COM1`…
- **Use `platform_compat.replace_path()`** instead of `os.rename`/`shutil.move`
  onto a destination that may exist: `os.rename` raises on Windows where POSIX
  overwrites.
- **Keep paths short.** Windows caps a path at 260 characters by default;
  checkpoint trees already nest deeply.
- **Guard POSIX-only APIs** (`select` on stdin, `fcntl`, `os.fork`, `SIGKILL`)
  and provide a Windows branch, as `llm_tool/utils/interactive_skip.py` does.
- **Prefer `torch.cuda`/`torch.backends.mps` checks** over assuming a backend;
  `mps` does not exist off macOS.
- **New dependencies must have a prebuilt wheel** for Windows, macOS and Linux
  on CPython 3.11–3.13. If one needs a compiler, put it in its own opt-in extra
  and keep it out of `[all]`.

## Making Changes

### Branch Naming Convention

Create a descriptive branch name:

- Feature: `feature/add-new-model-support`
- Bug fix: `fix/annotation-validation-error`
- Documentation: `docs/update-readme`
- Refactor: `refactor/simplify-training-loop`

### Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(trainers): add support for ELECTRA models

- Implement ELECTRA discriminator training
- Add model configuration for ELECTRA variants
- Update model catalog with ELECTRA entries

Closes #123
```

```
fix(validators): handle empty annotation batches

Previously, the validator would crash when encountering
empty annotation batches. This fix adds proper validation
and error handling.

Fixes #456
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **String quotes**: Double quotes preferred
- **Imports**: Organized with isort
- **Formatting**: Automated with Black

### Formatting Tools

Format your code before committing:

```bash
make format        # macOS / Linux
make.bat format    # Windows
```

This runs:
- **Black** for code formatting
- **isort** for import sorting

### Linting

Check code quality:

```bash
make lint          # macOS / Linux
make.bat lint      # Windows
```

This runs:
- **flake8** for style violations

### Type Hints

While not strictly enforced, type hints are encouraged:

```python
def annotate_texts(
    texts: list[str],
    model: str,
    schema: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Annotate texts using specified model.

    Args:
        texts: List of text strings to annotate
        model: Model identifier (e.g., "gpt-4")
        schema: Annotation schema definition

    Returns:
        List of annotation dictionaries
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    dataset_path: str,
    model_name: str,
    epochs: int = 10
) -> dict:
    """
    Train a BERT model on provided dataset.

    This function handles the complete training pipeline including
    data loading, model initialization, training loop, and evaluation.

    Args:
        dataset_path: Path to training dataset (CSV/JSON/JSONL)
        model_name: HuggingFace model identifier
        epochs: Number of training epochs (default: 10)

    Returns:
        Dictionary containing training metrics and model path:
        {
            'f1_score': float,
            'accuracy': float,
            'model_path': str,
            'training_time': float
        }

    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If model_name is not supported

    Example:
        >>> results = train_model(
        ...     dataset_path="data/train.jsonl",
        ...     model_name="bert-base-uncased",
        ...     epochs=5
        ... )
        >>> print(f"F1 Score: {results['f1_score']:.3f}")
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_annotators.py -v

# Run with coverage
pytest tests/ --cov=llm_tool --cov-report=html
```

### Writing Tests

Place tests in the `tests/` directory with naming convention `test_*.py`:

```python
# tests/test_annotators.py

import pytest
from llm_tool.annotators import LLMAnnotator

class TestLLMAnnotator:
    """Test suite for LLM annotation functionality."""

    def test_annotation_schema_validation(self):
        """Test that annotation schema is properly validated."""
        schema = {
            "label": {"type": "string", "enum": ["positive", "negative"]}
        }
        annotator = LLMAnnotator(schema=schema)
        assert annotator.schema == schema

    def test_invalid_model_raises_error(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            LLMAnnotator(model="invalid-model-name")
```

### Test Markers

Use pytest markers for test categorization:

```python
@pytest.mark.slow
def test_full_training_pipeline():
    """Integration test for complete training workflow."""
    pass

@pytest.mark.integration
def test_ollama_connection():
    """Test Ollama API connection."""
    pass
```

Run specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests
```

## Submitting Changes

### Before Submitting

1. **Format code**:
   ```bash
   make format
   ```

2. **Run linting**:
   ```bash
   make lint
   ```

3. **Run tests**:
   ```bash
   make test
   ```

4. **Update documentation** if needed (README, docstrings, etc.)

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(scope): descriptive message"
   ```

### Pull Request Process

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title following commit message convention
   - Detailed description of changes
   - Link to related issues (e.g., "Closes #123")
   - Screenshots/examples if applicable

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of what this PR does.

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] All tests pass locally
   - [ ] Added new tests for this change
   - [ ] Updated documentation

   ## Related Issues
   Closes #123

   ## Screenshots (if applicable)
   ```

4. **Address Review Comments**:
   - Respond to feedback constructively
   - Make requested changes
   - Push updates to the same branch

5. **Squash Commits** (if requested):
   ```bash
   git rebase -i HEAD~3  # Squash last 3 commits
   git push --force-with-lease
   ```

## Development Workflow Example

```bash
# 1. Sync with upstream
git checkout main
git fetch upstream
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/add-new-model

# 3. Make changes
# ... edit files ...

# 4. Format and test  (Windows: make.bat format / lint / test)
make format
make lint
make test

# 5. Commit changes
git add .
git commit -m "feat(models): add support for new BERT variant"

# 6. Push to fork
git push origin feature/add-new-model

# 7. Create PR on GitHub
```

## Areas for Contribution

Looking for ideas? Here are some areas that need help:

### High Priority
- [ ] Add more SOTA model support (e.g., newer DeBERTa variants)
- [ ] Improve error handling and user feedback
- [ ] Add comprehensive test coverage
- [ ] Performance optimization for large datasets
- [ ] Better memory management during training

### Documentation
- [ ] Tutorial videos or screencasts
- [ ] More usage examples
- [ ] API documentation
- [ ] Translation to other languages

### Features
- [ ] Web UI for annotation
- [ ] REST API mode
- [ ] Docker support
- [ ] Cloud deployment guides
- [ ] Active learning integration

### Bug Fixes
- Check [Issues](https://github.com/antoinelemor/LLM_Tool/issues) labeled "bug"

## Getting Help

- **Questions?** Open a [Discussion](https://github.com/antoinelemor/LLM_Tool/discussions)
- **Found a bug?** Open an [Issue](https://github.com/antoinelemor/LLM_Tool/issues)
- **Want to chat?** Join our community (Discord/Slack link)

## License

By contributing to LLM Tool, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🎉**
