# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
