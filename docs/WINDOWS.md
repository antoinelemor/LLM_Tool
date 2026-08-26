# LLM Tool on Windows

Complete installation and troubleshooting guide for Windows 10 and Windows 11.

Everything below is written for people who do not program. You can copy and
paste every command exactly as it appears.

---

## Table of contents

- [The short version](#the-short-version)
- [1. Before you start](#1-before-you-start)
- [2. Install](#2-install)
- [3. Launch LLM Tool](#3-launch-llm-tool)
- [4. Use a GPU (NVIDIA)](#4-use-a-gpu-nvidia)
- [5. Run local LLMs with Ollama](#5-run-local-llms-with-ollama)
- [6. Set your API keys](#6-set-your-api-keys)
- [7. Recommended Windows setup](#7-recommended-windows-setup)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Update and uninstall](#9-update-and-uninstall)
- [10. Known Windows limitations](#10-known-windows-limitations)

---

## The short version

If you already have Python 3.11+ and Git:

```powershell
git clone https://github.com/antoinelemor/LLM_Tool.git
cd LLM_Tool
.\install.bat
```

Then, in the same window:

```powershell
.\.venv\Scripts\Activate.ps1
llm-tool
```

That is the whole installation. Everything else on this page is either
background or a fix for a specific error message.

---

## 1. Before you start

### 1.1 Python 3.11 or newer (required)

**Python 3.12 is the recommended version on Windows.** Every dependency ships a
prebuilt Windows package for 3.11, 3.12 and 3.13, so nothing has to be compiled.

Open **PowerShell** (press `Win`, type `powershell`, press `Enter`) and check:

```powershell
py --version
```

If that prints `Python 3.11.x` or newer, you are done with this step.

If it prints an error, or a version older than 3.11, install Python:

**Option A — winget (fastest, no clicking):**

```powershell
winget install -e --id Python.Python.3.12
```

**Option B — the installer:**

1. Download the **Windows installer (64-bit)** from
   [python.org/downloads/windows](https://www.python.org/downloads/windows/).
2. On the very first screen of the installer, **tick "Add python.exe to PATH"**.
   This one checkbox is the single most common cause of a failed install.
3. Click **Install Now**.

After either option, **close PowerShell and open a new window** so it picks up
the new PATH.

> **64-bit only.** PyTorch does not publish 32-bit or ARM64 Windows packages.
> On an ARM device (Surface Pro X, Snapdragon X Elite), install the **x64**
> build of Python — it runs fine under emulation.

### 1.2 Git (recommended)

```powershell
winget install -e --id Git.Git
```

Git makes updating a one-line command later. If you would rather not install it,
download the repository as a ZIP from GitHub (green **Code** button →
**Download ZIP**) and extract it somewhere like `C:\Dev\LLM_Tool`.

### 1.3 Optional extras

| What | When you need it | Command |
|------|------------------|---------|
| **Ollama** | Running LLMs locally, free and offline | `winget install -e --id Ollama.Ollama` |
| **NVIDIA driver** | GPU-accelerated training | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| **VS Code** | A friendlier editor and terminal | `winget install -e --id Microsoft.VisualStudioCode` |
| **Windows Terminal** | Correct emoji and colours (preinstalled on Win 11) | `winget install -e --id Microsoft.WindowsTerminal` |

You do **not** need Visual Studio, C++ build tools, CMake, Rust or Anaconda.
The installer only uses prebuilt packages.

### 1.4 Where to put the project

Pick a **short, local path** such as `C:\Dev\LLM_Tool`.

Two folders to avoid:

- **OneDrive-synced folders** (usually `Documents` and `Desktop`). OneDrive can
  lock or offload files while a model is training, which corrupts checkpoints
  mid-run.
- **`C:\Program Files`.** It is read-only for normal accounts.

---

## 2. Install

### Method A — the installer (recommended)

Open PowerShell, go to where you want the project, and run:

```powershell
cd C:\Dev
git clone https://github.com/antoinelemor/LLM_Tool.git
cd LLM_Tool
.\install.bat
```

If you downloaded the ZIP instead, just `cd` into the extracted folder and run
`.\install.bat` — or simply **double-click `install.bat` in File Explorer**.

The installer:

1. finds the newest usable Python (3.13 → 3.12 → 3.11) and refuses the
   Microsoft Store placeholder,
2. warns you about long paths, OneDrive and low disk space,
3. creates a private virtual environment in `.venv`,
4. points VS Code at it,
5. installs LLM Tool and every dependency,
6. runs `verify_installation.py` and prints a report.

Expect **5 to 20 minutes** and **3–6 GB** of downloads, mostly PyTorch.

**Installer options:**

```powershell
.\install.bat -Preset all              # everything (default)
.\install.bat -Preset core             # annotation + training only, ~2 GB smaller
.\install.bat -Preset dev              # core + pytest/black/mypy
.\install.bat -Preset full             # all + dev
.\install.bat -Preset all -Cuda cu126  # everything, with GPU PyTorch
.\install.bat -Recreate                # wipe .venv and start over
```

> **Why `install.bat` and not `install.ps1`?** PowerShell blocks unsigned script
> files by default, so running `.\install.ps1` on a fresh machine fails with
> *"running scripts is disabled on this system"*. `install.bat` launches the
> PowerShell script with that restriction lifted **for that one process only** —
> nothing on your machine is changed. If you have already allowed local scripts,
> `.\install.ps1` works identically.

### Method B — manual

Useful if you want to see each step, or if you are scripting a lab machine.

**PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
python verify_installation.py
```

**Command Prompt (`cmd.exe`):**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
python verify_installation.py
```

**Git Bash:**

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
python verify_installation.py
```

Only the activation line differs between the three shells. Note the quotes
around `".[all]"` — without them PowerShell treats the brackets as a wildcard
and pip installs the core package only.

### What you get

| Preset | Size on disk | Contents |
|--------|-------------|----------|
| `core` | ~4 GB | The full pipeline: annotation, training, benchmarking, validation |
| `all`  | ~6 GB | `core` + Label Studio, MLflow, Weights & Biases, TensorBoard, Optuna, Gradio, Ray, sentence-transformers |
| `dev`  | ~4 GB | `core` + pytest, black, flake8, mypy, isort, Jupyter |
| `full` | ~6 GB | `all` + `dev` |

Two optional extras are **not** part of `all`, because they have no prebuilt
Windows package and would try to compile C++ or Rust on your machine:

```powershell
pip install -e ".[llamacpp]"    # llama.cpp GGUF inference; needs CMake + MSVC
pip install -e ".[fasttext]"    # fastText language ID; needs MSVC build tools
```

Neither is required: Ollama covers local inference, and language detection uses
`lingua`, which does ship a Windows package.

---

## 3. Launch LLM Tool

Every new terminal needs the environment activated first:

```powershell
cd C:\Dev\LLM_Tool
.\.venv\Scripts\Activate.ps1
llm-tool
```

You will know it worked when your prompt starts with `(.venv)`.

Both `llm-tool` and `llmtool` work, and so does `python -m llm_tool`.

**In VS Code:** open the `LLM_Tool` folder, press `Ctrl+Shift+P`, run
**Python: Select Interpreter**, and pick `.venv\Scripts\python.exe`. New
terminals (`` Ctrl+` ``) then activate it automatically.

---

## 4. Use a GPU (NVIDIA)

The Windows packages on PyPI are **CPU-only**. This is a PyTorch packaging
decision, not a limitation of this project — the CUDA build lives on PyTorch's
own server.

Check whether your current install sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False` and you have an NVIDIA card, install the CUDA build:

```powershell
.\install.bat -Preset all -Cuda cu126
```

or, into an environment you already have:

```powershell
.\.venv\Scripts\Activate.ps1
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Pick the tag that matches your driver — `cu126`, `cu128`, and so on. The list of
available tags is at [download.pytorch.org/whl](https://download.pytorch.org/whl/).
`nvidia-smi` reports the highest CUDA version your driver supports, in the top
right of its output.

There is **no GPU acceleration for AMD or Intel GPUs on Windows**: ROCm is
Linux-only. Those machines train on CPU, which works but is slow — a few hours
instead of a few minutes for a typical BERT fine-tune.

---

## 5. Run local LLMs with Ollama

Ollama lets you annotate with no API key, no cost and no data leaving your
machine.

```powershell
winget install -e --id Ollama.Ollama
```

Ollama installs as a background service and starts with Windows. After
installing, open a **new** terminal and pull a model:

```powershell
ollama pull llama3.2
ollama list
```

LLM Tool finds Ollama at `http://localhost:11434` automatically. To point it
somewhere else — another machine on your network, or Ollama Cloud:

```powershell
llm-tool --ollama-host http://192.168.1.10:11434
llm-tool --ollama-cloud --ollama-api-key "your-key"
```

> The `curl -fsSL https://ollama.ai/install.sh | sh` command you will find in
> most tutorials is the **Linux** installer. On Windows use winget or the
> installer from [ollama.com/download](https://ollama.com/download).

---

## 6. Set your API keys

Skip this if you only use Ollama.

**Easiest — let the CLI store them encrypted:** launch `llm-tool`, open
**Documentation & Help → API Key Management**, and paste your keys. They are
encrypted at rest under your user profile. See
[API_KEY_MANAGEMENT.md](API_KEY_MANAGEMENT.md).

**Or use environment variables.** On Windows there are two forms, and mixing
them up is a common source of confusion:

```powershell
# This terminal only, forgotten when you close the window:
$env:OPENAI_API_KEY = "sk-..."

# Permanent, applies to every NEW terminal (not the current one):
setx OPENAI_API_KEY "sk-..."
```

`export OPENAI_API_KEY=...` is the macOS and Linux form; it does nothing in
PowerShell.

Recognised variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`OLLAMA_API_KEY`, `OLLAMA_HOST`.

---

## 7. Recommended Windows setup

None of these are required, but each removes a class of problem.

### Use Windows Terminal

The CLI draws tables, panels and emoji. **Windows Terminal** renders them
correctly; the old `conhost` console window (what you get from `cmd.exe` on
Windows 10) drops emoji and misaligns table borders. Windows 11 ships it by
default; on Windows 10:

```powershell
winget install -e --id Microsoft.WindowsTerminal
```

LLM Tool forces UTF-8 output on itself at startup, so text is never *corrupted*
either way — only the glyphs the font lacks look wrong.

### Enable long paths

Windows rejects file paths over 260 characters unless you turn this on. Training
checkpoints nest deeply (`models/<session>/<model>/checkpoint-1500/...`) and can
cross that line on a long project name.

In an **Administrator** PowerShell, once, then reboot:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -Type DWord
```

### Exclude the project from Defender

Real-time scanning inspects every file PyTorch writes, which can double
installation time and slow down training noticeably. In an **Administrator**
PowerShell:

```powershell
Add-MpPreference -ExclusionPath "C:\Dev\LLM_Tool"
```

### Allow local PowerShell scripts

Optional. Lets you run `.\install.ps1` and `Activate.ps1` without the `.bat`
wrapper. Scripts you wrote or downloaded locally are allowed; scripts from the
internet still need to be signed.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 8. Troubleshooting

### "running scripts is disabled on this system"

```
.\install.ps1 : File ... cannot be loaded because running scripts is disabled
on this system.
```

PowerShell's default execution policy. Use the wrapper, which bypasses it for
one process:

```powershell
.\install.bat
```

Or allow local scripts permanently — see [section 7](#allow-local-powershell-scripts).

### Typing `python` opens the Microsoft Store

Windows ships a placeholder `python.exe` that does nothing but advertise the
Store. The installer detects and skips it, but if you are installing manually:

- use `py` instead of `python` (the launcher is never a placeholder), **or**
- turn the alias off: **Settings → Apps → Advanced app settings →
  App execution aliases**, switch off both `python.exe` and `python3.exe`.

### "Python was not found" after installing it

You installed without ticking **Add python.exe to PATH**, or you are reusing a
terminal opened before the install.

1. Close every terminal window and open a new one.
2. Still broken? Re-run the python.org installer, choose **Modify**, and enable
   **Add Python to environment variables**.
3. Or point the installer at the interpreter directly:
   `.\install.bat -Python "C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe"`

### "Microsoft Visual C++ 14.0 or greater is required"

A package tried to compile itself instead of using a prebuilt one. With the
supported Python versions this should not happen for `core`, `all` or `dev`.

- Check your Python version: `py --version`. On 3.14 or newer, some packages
  have no Windows build yet — install 3.12 alongside and re-run
  `.\install.bat -Recreate`.
- If you asked for the `llamacpp` or `fasttext` extras, they genuinely need a
  compiler:
  `winget install -e --id Microsoft.VisualStudio.2022.BuildTools`, then select
  **Desktop development with C++** in the installer that opens.

### `UnicodeEncodeError: 'charmap' codec can't encode character`

The console is using a legacy code page. LLM Tool configures UTF-8 for itself at
startup, so this should not appear — if it does, force it for the whole session:

```powershell
$env:PYTHONUTF8 = "1"
llm-tool
```

Make it permanent with `setx PYTHONUTF8 1`, then open a new terminal.

### Boxes, question marks or `?` instead of emoji

A font problem, not an encoding problem — the characters arrived intact, the
font has no glyph for them. Use Windows Terminal, or switch the console font to
**Cascadia Mono** (right-click the title bar → Properties → Font).

### `[Errno 2] No such file or directory` with a very long path

The 260-character limit. Enable long paths
([section 7](#enable-long-paths)) and move the project closer to the drive root,
e.g. `C:\Dev\LLM_Tool`.

### "Access is denied" or `PermissionError` while training

Usually one of:

- **OneDrive** is syncing the folder. Move the project to `C:\Dev\`.
- **Antivirus** locked a file mid-write. Add an exclusion
  ([section 7](#exclude-the-project-from-defender)).
- Another process still has the file open — VS Code's file watcher, an Explorer
  preview pane, or a previous run that did not exit. Close them and retry.

### `llm-tool` is not recognised

The virtual environment is not active. Every new terminal needs:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation itself is blocked, use `.\.venv\Scripts\python.exe -m llm_tool`,
which needs no activation at all.

### Cannot delete `.venv` — "file is in use"

Windows will not delete an open file. Close VS Code, every terminal with
`(.venv)` in the prompt, and any running `python.exe` (check Task Manager), then
retry `.\install.bat -Recreate`.

### `torch.cuda.is_available()` returns False

See [section 4](#4-use-a-gpu-nvidia) — the PyPI package for Windows is CPU-only
and has to be replaced with the CUDA build.

### Training is very slow

On CPU that is expected. Also check:

- `python -c "import torch; print(torch.cuda.is_available())"` — should be
  `True` on an NVIDIA machine.
- Defender exclusion added ([section 7](#exclude-the-project-from-defender)).
- Project not inside OneDrive.

### The installation still fails

Capture the full log and open an issue with it:

```powershell
.\install.bat -Preset all *> install-log.txt
```

Include the output of `py --version`, `python -c "import platform; print(platform.platform())"`,
and the last 50 lines of `install-log.txt`.

---

## 9. Update and uninstall

**Update:**

```powershell
cd C:\Dev\LLM_Tool
git pull
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]" --upgrade
```

**Uninstall:** delete the project folder. Everything — the virtual environment,
the dependencies, the models — lives inside it. Two things live elsewhere and
can be removed separately:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.llm_tool"       # settings and stored API keys
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"  # downloaded base models
```

---

## 10. Known Windows limitations

| Limitation | Detail | Workaround |
|-----------|--------|------------|
| No AMD/Intel GPU training | ROCm is Linux-only | Train on CPU, or use CUDA on NVIDIA |
| No ARM64 PyTorch | No official `win_arm64` package | Install x64 Python; it runs under emulation |
| `llama-cpp-python` needs a compiler | No prebuilt Windows package upstream | Use Ollama instead |
| `fastText` needs a compiler | No prebuilt Windows package for Python 3.11+ | `lingua` is the default detector and needs nothing |
| Slower dataloading | Windows has no `fork`, so worker processes are more expensive to start | The trainer already accounts for this and uses fewer workers |
| Distributed training uses `gloo` | `nccl` is Linux-only | Single-GPU and CPU training are unaffected |

---

## See also

- [../README.md](../README.md) — features, modes, and the full user guide
- [modes_reference.md](modes_reference.md) — what each of the five modes does
- [API_KEY_MANAGEMENT.md](API_KEY_MANAGEMENT.md) — how keys are stored
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — developing on Windows
