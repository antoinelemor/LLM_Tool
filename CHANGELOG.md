# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Google Gemini support

- Gemini is now a first-class annotation provider, offered in the model picker
  next to OpenAI and Ollama, and usable headlessly.
- Switched from `google-generativeai` to `google-genai`. The declared package was
  the deprecated SDK (end-of-life, provides `google.generativeai`) while the code
  imported the maintained one (`google.genai`) -- so `HAS_GOOGLE` was always
  False after a documented install and Gemini silently never loaded.
- New `--provider {ollama,openai,anthropic,google}`, inferred from the model name
  when omitted. `--annotate data.csv --model gemini-3.6-flash` previously fell
  through to Ollama and died trying to pull "gemini-3.6-flash" from the Ollama
  registry; there was no way to reach a cloud provider headlessly at all.
- `GEMINI_API_KEY` is accepted as an alias for `GOOGLE_API_KEY`, since that is the
  name Google's own quickstarts and the SDK itself use.
- `GoogleClient` rewritten for robustness: exponential-backoff retry on 429/5xx,
  a `GoogleFatalError` that stops the run immediately on a rejected key or a
  retired model instead of replaying the failure for every row, native
  `response_schema` JSON mode, system instructions, token-usage capture, and a
  minimum output budget (Gemini's hidden reasoning otherwise consumes the whole
  allowance and returns empty text).
- 29 tests, 25 of which run without a key.

#### First-class Windows support

- `install.bat` and `install.ps1`: a one-command Windows installer mirroring
  `install.sh`. It finds the newest usable Python (3.13 → 3.12 → 3.11), rejects the
  Microsoft Store placeholder `python.exe`, refuses 32-bit interpreters, warns about
  long paths / OneDrive / low disk space, creates `.venv`, configures VS Code, installs
  and verifies. `install.bat` launches it with the execution policy bypassed for that
  one process, so a fresh machine needs no PowerShell configuration.
  `-Cuda cu126` installs the GPU build of PyTorch from PyTorch's own index.
- `docs/WINDOWS.md`: complete Windows install, GPU, Ollama and troubleshooting guide
- `make.bat`: Windows twin of every `Makefile` target
- `.github/workflows/install.yml`: CI installing and launching on
  windows-latest / ubuntu-latest / macos-latest across Python 3.11–3.13, plus a job that
  runs the real installer scripts end to end
- `.gitattributes`: normalises line endings so `install.sh` is not checked out with CRLF
  on Windows (which makes its shebang unusable)
- `llm_tool/platform_compat.py`: shared cross-platform helpers —
  `configure_console()` (UTF-8 standard streams and ANSI escapes on Windows),
  `sanitize_path_component()`, `replace_path()`, `writable_dir()`, `supports_unicode()`
- Windows-aware `verify_installation.py`: checks console encoding, resolves console
  scripts through the venv's `Scripts` directory, degrades its ✓/✗ glyphs on legacy code
  pages, and points at the CUDA index when an NVIDIA GPU is present but PyTorch is CPU-only

- Ollama Cloud support (`https://ollama.com`) alongside the local daemon, selectable in
  Mode 1, Mode 2, the Resume Center and Agent mode
- Configurable Ollama endpoint via `OLLAMA_HOST` / `OLLAMA_API_KEY`, or a key stored under
  the `ollama` provider
- Reachability test from the model picker (endpoint, credential, model availability and a
  live one-token generation) before an annotation run starts
- Free-text model entry for Ollama, so any model name can be used without waiting for a
  catalogue update
- Recent annotation-oriented models in the catalogue: Gemma 4, Qwen 3.5, GLM-5.1/5.2,
  Kimi K2.6/K3, DeepSeek V4 Flash/Pro, MiniMax M2.7/M3, Nemotron 3, Mistral Large 3, GPT-OSS
- Initial public release preparation
- Comprehensive README with installation and usage instructions
- CONTRIBUTING.md for contributor guidelines
- Makefile for common development tasks
- pyproject.toml for modern Python packaging

### Changed

#### Dependencies

- `matplotlib` and `psutil` are now **core** dependencies. Both are imported
  unconditionally on the CLI path, so `pip install -e .` produced an installation that
  could not launch.
- Dropped two declared-but-unused dependencies: `inquirer` (never imported; it pulled in
  `blessed`, `readchar`, `wcwidth` and `editor`) and `colorama`. The colorama import in
  `bert_base` also called `init(autoreset=True)`, which **replaces `sys.stdout` and
  `sys.stderr` on Windows** — undoing the UTF-8 setup and reintroducing
  `UnicodeEncodeError` on the training dashboard, for symbols the module never used.
- The `[all]` extra no longer pulls `llama-cpp-python` or `fasttext`. Neither publishes a
  Windows wheel, so the documented recommended install tried to compile C++ and failed on
  any machine without Visual Studio build tools. Both moved to opt-in extras
  (`[llamacpp]`, `[fasttext]`) and are documented as such; Ollama and `lingua` cover the
  same ground with no compiler.
- `[all]` also dropped `label-studio`, `ray`, `dask`, `gradio`, `redis`, `pymongo`,
  `sentence-transformers`, `imbalanced-learn` and `seaborn` — several GB of packages that
  no module in this codebase imports. (`seaborn` was never needed: the charts use
  matplotlib's built-in `seaborn-v0_8-whitegrid` style.) Experiment trackers moved to an
  `[mlops]` extra, the HTTP server to `[api]`, the extra providers to `[providers]`, all
  three still included in `[all]`.
- Removed the `pyarrow<19` cap, which silently pinned `datasets` to exactly 4.0.0 and sent
  pip through a long backtracking search.
- `requirements.txt` no longer holds a macOS `pip freeze` — it contained an editable
  `git+https` self-reference that needed `git.exe` on PATH, `numpy==1.26.4` (no wheel for
  Python 3.13) and ~55 unrelated transitive pins. It now defers to `pyproject.toml`.
- The minimum Python version is **3.11** everywhere. The README badge, the README text,
  `CONTRIBUTING.md` and `verify_installation.py` all claimed 3.9, which pip had been
  refusing since `requires-python` was set.

### Fixed

#### Windows correctness

- The console entry point created and opened log files **inside the installed package** at
  import time, so any non-editable install into a read-only tree (`C:\Program Files`, a
  system `site-packages`) raised `PermissionError` before the CLI could start. Logs now
  follow the working directory, with a fallback to the user profile and then the temp
  directory. `llm_tool.transcription` had the same bug and the same fix.
- The CLI's emoji, box-drawing characters and multilingual text raised
  `UnicodeEncodeError` on the first banner, because Python opens stdout with the ANSI code
  page on Windows. The standard streams are now reconfigured to UTF-8 at package import,
  and ANSI escape handling is enabled for the legacy console.
- ~40 file reads and writes omitted `encoding=`, so JSON configs, API-key stores, session
  metadata, annotation output and log handlers were read and written in the locale
  encoding — corrupting accented, Arabic and CJK text on Windows and raising on write.
- CSV readers now pass `newline=''`, without which the `csv` module mis-parses newlines
  embedded in quoted fields.
- Model checkpoint promotion used `shutil.rmtree` followed by `shutil.move`/`os.rename`
  onto the same name. On Windows `os.rename` raises `FileExistsError` where POSIX
  overwrites, and `shutil.move` onto an existing directory moves *into* it, burying the
  checkpoint a level deeper on every rerun. Both now go through `replace_path()`.
- Annotation and training-metrics output paths repeated the dataset stem as both a
  directory and a filename component, with nothing bounding their length; real paths in
  this repository already reach 297 characters, past Windows' 260-character limit.
  Components are now bounded and no longer duplicated.
- Hybrid parallel training hardcoded `force_device='mps'`, which does not exist off macOS
  and raised on every Windows and Linux box. Device selection and cache clearing are now
  resolved from the accelerator actually present.
- `MemoryMonitor.trigger_cleanup()` tested `hasattr(torch.mps, 'empty_cache')`, which is
  true on every platform — so the CUDA branch was unreachable and an out-of-memory retry
  retried into the same full GPU.
- The interactive "press `s` to skip" listener called `select.select()` on the stdin file
  descriptor. On Windows `select` only accepts sockets, so the listener thread died
  silently and skipping never worked; it now uses `msvcrt` there.
- DataLoader worker counts were tuned for `fork`. Windows has no `fork`, so each worker
  re-imported torch and transformers; workers are now capped there and kept alive between
  epochs on every device.
- System resource detection fell through to `/proc/meminfo` on Windows and reported 0 GB
  of RAM, collapsing the batch-size and worker heuristics that read it. It now uses
  `GlobalMemoryStatusEx`. CPU and GPU detection no longer shell out to `wmic`, which
  Windows 11 24H2 removed.
- The prompt wizard's "edit manually" launched `nano` with no fallback and no exception
  handler, raising `FileNotFoundError` on Windows. It now resolves `$EDITOR`/`$VISUAL`,
  falls back per platform, and reports cleanly when no editor exists.
- `_clear_terminal_buffer` called `Control("erase", target)`, which is not the Rich API and
  raised `KeyError` on every platform, and wrote a raw ANSI sequence that a legacy Windows
  console prints literally.
- The missing-dependency fallback CLI printed raw ANSI escapes, misaligned box art, and
  instructions naming a `fix_cryptography.sh` and an `examples/` directory that do not
  exist. It now prints plain ASCII with per-platform instructions.
- Six hardcoded writes to `/tmp/llmtool_debug.log`, one inside the per-row annotation loop,
  replaced by an opt-in trace behind `LLM_TOOL_TRACE_PROGRESS`.
- Ollama recovery advice, GGUF model search paths and Ollama model names used as filenames
  are all platform-aware.

- Sampling settings configured in the wizard (temperature, top_p, top_k, seed, max tokens)
  reached the local model client but were then discarded; every Ollama annotation silently
  ran at the defaults
- Ollama lifecycle calls shelled out to the `ollama` CLI, so a machine without the binary
  could not drive a reachable server; they now go over the HTTP API
- A transient network error while annotating aborted the whole sequential run instead of
  failing the affected row
- Rejected credentials and missing models triggered the full retry-and-recover path instead
  of failing fast
- `NameError` when opening the model parameters step of the database annotator
- Training Arena session persistence issues (all sessions now recallable)
- AttributeError in resume/relaunch training flow
- Metadata saving now mandatory (no more lost sessions)

## [1.0.0] - 2025-01-XX

### Added

#### Core Features
- **The Annotator**: Zero-shot LLM annotation with multi-provider support
  - OpenAI (GPT-4, GPT-4o, o1, o3)
  - Anthropic (Claude 3, Claude 3.5)
  - Google (Gemini 1.5)
  - Ollama (local LLMs)
  - LlamaCPP (GGUF models)
- **The Annotator Factory**: End-to-end annotation → training pipeline
- **Training Arena**: Model training with 70+ transformer models
  - BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT variants
  - Language-specific models (CamemBERT, AraBERT, etc.)
  - Long-document models (Longformer, BigBird, LED)
  - Multilingual models (XLM-RoBERTa, mBERT, mDeBERTa)
- **BERT Annotation Studio**: High-throughput inference with trained models
- **Validation Lab**: Quality assurance and annotation validation

#### Language Support
- Automatic language detection with 96%+ accuracy (lingua-based)
- 75+ language support with specialized models for 15+ languages
- Per-document language tagging
- Language-specific model recommendations

#### Training Features
- Multi-label classification support
- Reinforcement learning for class imbalance
- Automatic early stopping and checkpointing
- Comprehensive benchmarking across multiple models
- Training session persistence with resume/relaunch capability
- Live metrics tracking (F1, accuracy, precision, recall)

#### Data Processing
- Multiple format support: CSV, Excel, JSON/JSONL, Parquet, RData/RDS
- PostgreSQL database integration
- Automatic data splitting and stratification
- Class balancing options

#### CLI & UX
- Rich interactive CLI with progress tracking
- System resource monitoring (GPU/CPU/RAM)
- Profile management for saved configurations
- Secure API key storage with encryption
- Session management and history

#### Validation & Export
- Inter-annotator agreement (Cohen's Kappa)
- Quality scoring and confidence analysis
- Export to Doccano/Label Studio formats
- Schema validation with Pydantic
- Stratified sampling for quality review

### Documentation
- Comprehensive README with quick start guide
- Installation instructions for VSCode
- CLI mode descriptions and workflows
- Architecture overview
- Troubleshooting section
- Performance benchmarks

### Technical
- Python 3.9+ support (tested with 3.9-3.13)
- GPU acceleration (CUDA, MPS)
- Multi-processing for CPU inference
- Incremental saving and resume capability
- JSON repair with 5-retry mechanism
- Parallel processing with thread/process pools

### Dependencies
- PyTorch 2.0+ for deep learning
- HuggingFace Transformers 4.35+ for model support
- Rich 14.0+ for CLI rendering
- Lingua 2.0+ for language detection
- Pydantic 2.0+ for validation
- SQLAlchemy 2.0+ for database support

---

## Release Notes

### Version 1.0.0 Highlights

This is the initial public release of LLM Tool, a comprehensive package for LLM-powered annotation and BERT model training. The package provides:

1. **Complete Annotation Pipeline**: From raw text to labeled datasets using state-of-the-art LLMs
2. **Production-Ready Training**: Train custom BERT models with automatic optimization
3. **Multi-Language Support**: 75+ languages with specialized models
4. **Professional CLI**: Rich, interactive interface with real-time progress tracking
5. **Quality Assurance**: Built-in validation and quality scoring tools

### Breaking Changes

N/A (initial release)

### Deprecations

N/A (initial release)

### Known Issues

- Large datasets (>100K documents) may require significant RAM during annotation
- Some LongFormer variants may have memory issues on GPUs <8GB
- Windows support is experimental (primarily tested on macOS/Linux)
  — *superseded in [Unreleased]: Windows is now a first-class, CI-tested platform*

### Upgrade Notes

N/A (initial release)

---

## Future Roadmap

### Planned for 1.1.0
- [ ] Web UI for annotation workflow
- [ ] Docker containerization
- [ ] Improved multi-GPU training support

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
