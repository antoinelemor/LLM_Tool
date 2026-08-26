#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
verify_installation.py

MAIN OBJECTIVE:
---------------
Verification script to ensure LLM Tool is correctly installed with all
dependencies available and CLI commands accessible, on Windows, macOS and
Linux alike.

Dependencies:
-------------
- sys
- os
- shutil
- importlib.util
- importlib.metadata
- subprocess

MAIN FEATURES:
--------------
1) Python version verification (matches pyproject's requires-python)
2) LLM Tool package verification
3) Core dependencies check (pandas, numpy, rich, matplotlib, etc.)
4) LLM dependencies check (openai, ollama, transformers, torch)
5) Training dependencies check (datasets, sklearn, scipy, nltk, peft)
6) Optional dependencies check (anthropic, google.genai, etc.)
7) GPU support detection (CUDA, MPS) with Windows-specific guidance
8) Console encoding check (UnicodeEncodeError is a Windows-only failure mode)
9) CLI command verification, resolved through the venv's script directory
10) Comprehensive summary report

Author:
-------
Antoine Lemor
"""

import os
import shutil
import subprocess
import sys
import importlib.util
from importlib import metadata as importlib_metadata

# Keep this in step with `requires-python` in pyproject.toml. Reporting a
# laxer minimum here would green-light an environment that pip has already
# refused to install into.
MIN_PYTHON = (3, 11)

IS_WINDOWS = os.name == "nt"


def _can_print_unicode() -> bool:
    """Report whether stdout can encode the tick and cross glyphs below."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# On a Windows console still on cp1252/cp850 -- a redirected stream, an old
# conhost, a CI runner -- printing U+2713 raises UnicodeEncodeError and the
# verification dies before reporting anything. Degrade the glyphs instead.
if _can_print_unicode():
    OK, BAD, SKIP = "✓", "✗", "-"
else:
    OK, BAD, SKIP = "[OK]", "[X]", "[-]"


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    version = sys.version_info
    minimum = ".".join(str(p) for p in MIN_PYTHON)
    current = f"{version.major}.{version.minor}.{version.micro}"

    if version[:2] >= MIN_PYTHON:
        print(f"  {OK} Python {current} (OK)")
        return True

    print(f"  {BAD} Python {current} (FAILED)")
    print(f"    Required: Python {minimum} or higher")
    if IS_WINDOWS:
        print("    Install it with:  winget install -e --id Python.Python.3.12")
        print("    Then re-run:      .\\install.bat -Recreate")
    return False


def _candidate_distribution_names(module):
    """Yield possible distribution names for a module (best-effort)."""
    names = [
        getattr(module, "__package__", None),
        getattr(module, "__name__", None),
    ]

    module_name = getattr(module, "__name__", "")
    if module_name:
        names.append(module_name.split(".")[0])

    seen = set()
    for name in filter(None, names):
        variants = {name}
        if "." in name:
            variants.add(name.replace(".", "-"))
        if "_" in name:
            variants.add(name.replace("_", "-"))
        for variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                yield variant


def _resolve_module_version(module):
    """Return the best-effort version string for a module."""
    for name in _candidate_distribution_names(module):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue

    version = getattr(module, "__version__", None)
    if version and version != "unknown":
        return version

    return "unknown"


def check_module(module_name, display_name=None, optional=False):
    """Check if a module is installed."""
    if display_name is None:
        display_name = module_name

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.import_module(module_name)
            version = _resolve_module_version(module)
            print(f"  {OK} {display_name:30s} version {version}")
            return True
        if optional:
            print(f"  {SKIP} {display_name:30s} (optional, not installed)")
            return True
        print(f"  {BAD} {display_name:30s} (MISSING)")
        return False
    except ImportError as e:
        if optional:
            print(f"  {SKIP} {display_name:30s} (optional, not installed)")
            return True
        print(f"  {BAD} {display_name:30s} (MISSING: {e})")
        return False


def check_llm_tool():
    """Check if llm_tool package is installed."""
    print("\nChecking LLM Tool installation...")
    try:
        import llm_tool
        version = getattr(llm_tool, "__version__", "1.0.0")
        print(f"  {OK} llm-tool version {version}")
        return True
    except ImportError as e:
        print(f"  {BAD} llm-tool not found: {e}")
        print("    Run: pip install -e .")
        return False


def check_core_dependencies():
    """Check core dependencies."""
    print("\nChecking core dependencies...")
    deps = [
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("rich", "Rich"),
        ("tqdm", "tqdm"),
        ("pydantic", "Pydantic"),
        ("sqlalchemy", "SQLAlchemy"),
        ("loguru", "Loguru"),
        ("click", "Click"),
        # Imported unconditionally by the annotation and training displays.
        ("matplotlib", "Matplotlib"),
        ("psutil", "psutil"),
        ("tabulate", "tabulate"),
    ]
    return all(check_module(mod, name) for mod, name in deps)


def check_llm_dependencies():
    """Check LLM-related dependencies."""
    print("\nChecking LLM dependencies...")
    deps = [
        ("openai", "OpenAI SDK"),
        ("ollama", "Ollama SDK"),
        ("transformers", "HuggingFace Transformers"),
        ("torch", "PyTorch"),
    ]
    return all(check_module(mod, name) for mod, name in deps)


def check_training_dependencies():
    """Check training-related dependencies."""
    print("\nChecking training dependencies...")
    deps = [
        ("datasets", "HuggingFace Datasets"),
        ("sklearn", "scikit-learn"),
        ("scipy", "SciPy"),
        ("nltk", "NLTK"),
        ("peft", "PEFT (LoRA/DoRA adapters)"),
    ]
    return all(check_module(mod, name) for mod, name in deps)


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    deps = [
        ("anthropic", "Anthropic SDK", True),
        ("google.genai", "Google GenAI SDK (Gemini)", True),
        ("langdetect", "langdetect", True),
        ("fastapi", "FastAPI (--api mode)", True),
        ("tensorboard", "TensorBoard", True),
    ]
    all(check_module(mod, name, optional) for mod, name, optional in deps)
    return True  # Optional deps don't fail verification


def check_console_encoding():
    """
    Check that the console can carry the CLI's Unicode output.

    This is the failure mode that makes LLM Tool look broken on Windows: the
    interactive menus are built from emoji and box-drawing characters, and a
    console left on the ANSI code page raises UnicodeEncodeError on the first
    frame. ``llm_tool`` reconfigures the streams at import, so a failure here
    means something is redirecting output past that fix.
    """
    print("\nChecking console encoding...")
    encoding = (getattr(sys.stdout, "encoding", None) or "unknown").lower()

    try:
        "🚀 ─ é 中".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        print(f"  {BAD} stdout encoding is '{encoding}' and cannot carry the CLI's output")
        print("    Fix it for this session:  set PYTHONUTF8=1        (Command Prompt)")
        print("                              $env:PYTHONUTF8 = '1'   (PowerShell)")
        print("    Or permanently:           setx PYTHONUTF8 1")
        return False

    print(f"  {OK} stdout encoding: {encoding}")
    if IS_WINDOWS:
        print("    Tip: use Windows Terminal for correct emoji and table borders.")
    return True


def check_gpu_support():
    """Check GPU availability."""
    print("\nChecking GPU support...")
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"  {OK} CUDA available: {device_count} device(s)")
            print(f"    Primary GPU: {device_name}")
            return True

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  {OK} MPS (Apple Silicon) available")
            return True

        print(f"  {SKIP} No GPU detected (CPU only)")
        print("    Training will use CPU (slower)")
        # The PyPI build of PyTorch for Windows is CPU-only, so an NVIDIA box
        # lands here unless the CUDA build was installed from PyTorch's index.
        if IS_WINDOWS and shutil.which("nvidia-smi"):
            print("    An NVIDIA GPU was detected but this PyTorch build is CPU-only.")
            print("    Install the CUDA build:")
            print("      pip install --force-reinstall torch \\")
            print("        --index-url https://download.pytorch.org/whl/cu126")
        return True

    except Exception as e:
        print(f"  {BAD} Error checking GPU: {e}")
        return False


def _script_dir_candidates():
    """
    Yield the directories a console script for this interpreter can live in.

    Relying on PATH alone reports a false failure whenever the virtual
    environment has not been activated in the shell that launched this script,
    which is exactly the situation the installer runs it in.
    """
    exe_dir = os.path.dirname(sys.executable)
    yield exe_dir                                     # Windows: <venv>\Scripts
    yield os.path.join(os.path.dirname(exe_dir), "bin")   # POSIX: <venv>/bin
    try:
        import sysconfig
        scripts = sysconfig.get_path("scripts")
        if scripts:
            yield scripts
    except Exception:
        pass


def _find_console_script(name):
    """Return the full path to a console script, or None."""
    exts = [".exe", ".cmd", ".bat", ""] if IS_WINDOWS else [""]
    for directory in _script_dir_candidates():
        for ext in exts:
            candidate = os.path.join(directory, name + ext)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which(name)


def check_cli_commands():
    """Check if CLI commands are available."""
    print("\nChecking CLI commands...")

    success = True
    for cmd in ("llm-tool", "llmtool"):
        path = _find_console_script(cmd)
        if not path:
            print(f"  {BAD} '{cmd}' command not found")
            success = False
            continue

        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                # Without this, Windows decodes the output with the OEM code
                # page and a non-ASCII byte would raise instead of reporting.
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print(f"  {BAD} '{cmd}' timed out (heavy imports on a cold cache?)")
            success = False
            continue
        except OSError as e:
            print(f"  {BAD} '{cmd}' error: {e}")
            success = False
            continue

        if result.returncode == 0:
            print(f"  {OK} '{cmd}' command available")
        else:
            print(f"  {BAD} '{cmd}' exited with code {result.returncode}")
            if result.stderr:
                print(f"      {result.stderr.strip().splitlines()[-1][:120]}")
            success = False

    if not success and IS_WINDOWS:
        print("    On Windows the scripts live in .venv\\Scripts and need the")
        print("    environment activated:  .\\.venv\\Scripts\\Activate.ps1")

    return success


def main():
    print("=" * 70)
    print("LLM TOOL - Installation Verification")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("LLM Tool Package", check_llm_tool),
        ("Core Dependencies", check_core_dependencies),
        ("LLM Dependencies", check_llm_dependencies),
        ("Training Dependencies", check_training_dependencies),
        ("Optional Dependencies", check_optional_dependencies),
        ("Console Encoding", check_console_encoding),
        ("GPU Support", check_gpu_support),
        ("CLI Commands", check_cli_commands),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  {BAD} Unexpected error during {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    non_optional_results = [r for r in results if r[0] != "Optional Dependencies"]
    passed = sum(1 for _, result in non_optional_results if result)
    total = len(non_optional_results)

    for name, result in non_optional_results:
        status = f"{OK} PASS" if result else f"{BAD} FAIL"
        print(f"  {status:8s} {name}")

    print("=" * 70)

    if passed == total:
        print(f"{OK} ALL CHECKS PASSED")
        print()
        print("LLM Tool is correctly installed and ready to use!")
        print()
        print("Next steps:")
        print("  1. Run the CLI: llm-tool")
        print("  2. Read the docs: README.md")
        if IS_WINDOWS:
            print("  3. Windows-specific guide: docs\\WINDOWS.md")
        print()
        return 0

    print(f"{BAD} {total - passed} CHECK(S) FAILED")
    print()
    print("Please fix the failed checks before using LLM Tool.")
    print()
    print("Common fixes:")
    print('  - Missing dependencies: pip install -e ".[all]"')
    print("  - CLI not found: pip install -e .")
    print("  - Import errors: check the virtual environment is activated")
    if IS_WINDOWS:
        print("  - Anything else: docs\\WINDOWS.md has a troubleshooting section")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
