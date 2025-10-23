# LLM_Tool Architecture Documentation Index

## Documents Overview

This package includes comprehensive architecture analysis documentation:

### 1. **ARCHITECTURE_QUICK_SUMMARY.md** (10 KB)
Quick reference guide covering:
- Executive overview
- Core capabilities overview
- Module structure table
- Key statistics
- Technology stack
- Quick usage patterns
- File locations

**Best for:** Quick reference, getting started, understanding the big picture

### 2. **ARCHITECTURE_ANALYSIS.md** (42 KB)
Comprehensive detailed analysis covering:
- Complete directory structure
- All 8 main modules with detailed descriptions
- 65+ classes with full documentation
- All 500+ methods
- 6 model providers with examples
- All 7 validation features with locations
- All 7 text detection capabilities
- Complete Training Arena documentation
- Benchmark capabilities with examples
- Full CLI feature breakdown
- Data processing and export details
- Statistical analysis features
- Technology stack with versions
- File locations with line numbers

**Best for:** In-depth understanding, development reference, feature research

---

## Quick Navigation

### By Topic

#### Annotation & LLM Integration
- **Main Class:** `LLMAnnotator` - llm_tool/annotators/llm_annotator.py
- **API Providers:** OpenAI, Anthropic, Google, Ollama, Llama.cpp
- **Features:** Multi-prompt, batch, parallel, cost estimation
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Core Modules > llm_tool/annotators/"

#### Model Training (50+ Models)
- **Base Class:** `BertBase` - llm_tool/trainers/bert_base.py
- **SOTA Models:** DeBERTa, RoBERTa, ELECTRA, ALBERT, BigBird, Longformer
- **Language Models:** 13 language-specific variants
- **French Models:** CamemBERT, FlauBERT, BARThez, FrALBERT, etc.
- **Features:** Early stopping, cross-validation, multi-GPU, FP16
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Core Modules > llm_tool/trainers/"

#### Validation & Quality Control
- **Main Class:** `AnnotationValidator` - llm_tool/validators/annotation_validator.py
- **Methods:** 7 validation features (schema, label, data quality, agreement, sampling, export, cost)
- **Formats:** Doccano JSONL export
- **Metrics:** Cohen's Kappa, confusion matrix, accuracy
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Validation Features"

#### Text Detection & Analysis
- **Language Detection:** LanguageDetector - llm_tool/utils/language_detector.py (96%+ accuracy, 75+ languages)
- **Text Analysis:** Text length, token analysis, column detection
- **Label Detection:** Automatic categorical column identification
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Text Detection Capabilities"

#### CLI & User Interface
- **Main CLI:** AdvancedCLI - llm_tool/cli/advanced_cli.py
- **Modes:** 3 major modes (Annotator Factory, Training Arena, Validation Lab)
- **Features:** Auto-detection, smart suggestions, progress tracking, resume
- **See:** ARCHITECTURE_ANALYSIS.md - Section "CLI Features"

#### Training Orchestration
- **Main Class:** `TrainingArena` - llm_tool/cli/training_arena_integrated.py
- **Features:** 18-step interactive wizard, per-category/language training
- **Dataset Building:** TrainingDatasetBuilder with auto-balancing
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Training Arena Functionality"

#### Benchmarking
- **Main Class:** `BenchmarkRunner` - llm_tool/trainers/benchmarking.py
- **Features:** Multi-model evaluation, per-category/language training, result aggregation
- **Metrics:** Accuracy, F1, precision, recall, macro/micro/weighted averages
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Benchmark Capabilities"

#### Configuration Management
- **Main Class:** `Settings` - llm_tool/config/settings.py
- **Components:** APIConfig, LocalModelConfig, TrainingConfig, LanguageConfig, etc.
- **Key Manager:** APIKeyManager for secure credential storage
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Configuration Classes"

#### Data Processing
- **Input Formats:** CSV, Excel, Parquet, JSON, JSONL, RData, PostgreSQL
- **Output Formats:** Same + Doccano JSONL
- **Main Classes:** FileHandler, PostgreSQLHandler, DataDetector
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Data Processing & Export"

#### System Resource Detection
- **Main Class:** `SystemResourceDetector` - llm_tool/utils/system_resources.py
- **Features:** GPU detection, CPU info, memory monitoring, storage analysis
- **Recommendations:** Automatic configuration based on hardware
- **See:** ARCHITECTURE_ANALYSIS.md - Section "System Resource Classes"

#### Statistical Analysis
- **Features:** 10 major statistical capabilities
- **Includes:** Label distribution, text statistics, language analysis, metrics, agreement
- **See:** ARCHITECTURE_ANALYSIS.md - Section "Statistical Analysis"

---

## File Structure Quick Reference

```
llm_tool/
├── __init__.py                          # Package initialization
├── __main__.py                          # CLI entry point
│
├── annotators/                          # LLM Annotation (3,500+ lines)
│   ├── llm_annotator.py                # Main annotation engine
│   ├── api_clients.py                  # OpenAI, Anthropic, Google clients
│   ├── local_models.py                 # Ollama, Llama.cpp clients
│   ├── prompt_manager.py               # Prompt loading and management
│   ├── prompt_wizard.py                # Interactive prompt builder
│   ├── json_cleaner.py                 # JSON repair and validation
│   └── __init__.py
│
├── cli/                                 # Command-Line Interface (25,000+ lines)
│   ├── advanced_cli.py                 # Main advanced CLI (10,000+ lines)
│   ├── annotation_workflow.py           # Annotation mode workflow
│   ├── training_arena_integrated.py     # Training mode (6,000+ lines)
│   ├── validation_lab.py                # Validation mode (3,000+ lines)
│   ├── bert_annotation_studio.py        # BERT training interface
│   ├── banners.py                      # ASCII art and visualization
│   └── __init__.py
│
├── config/                              # Configuration Management (700+ lines)
│   ├── settings.py                     # Main Settings class with 7 config dataclasses
│   ├── api_key_manager.py              # Secure credential management
│   └── __init__.py
│
├── database/                            # Data I/O (1,000+ lines)
│   ├── file_handlers.py                # CSV, Excel, Parquet, RData support
│   ├── postgresql_handler.py           # PostgreSQL support
│   └── __init__.py
│
├── exporters/                           # Export functionality
│   └── __init__.py
│
├── pipelines/                           # Orchestration (1,500+ lines)
│   ├── pipeline_controller.py          # Main orchestration engine
│   ├── enhanced_pipeline_wrapper.py     # Enhancement layer
│   └── __init__.py
│
├── trainers/                            # Model Training (15,000+ lines)
│   ├── bert_abc.py                     # Abstract base class
│   ├── bert_base.py                    # Base BERT implementation (4,000+ lines)
│   ├── models.py                       # 13 language-specific models
│   ├── sota_models.py                  # 30+ SOTA models
│   ├── model_trainer.py                # Training orchestration
│   ├── benchmarking.py                 # Multi-model benchmarking
│   ├── model_selector.py               # Automatic model selection
│   ├── multilingual_selector.py        # Multilingual strategy
│   ├── multi_label_trainer.py          # Multi-label classification
│   ├── data_utils.py                   # Training data utilities
│   ├── data_splitter.py                # Dataset splitting
│   ├── training_data_builder.py        # Dataset building
│   ├── benchmark_dataset_builder.py    # Benchmark dataset creation
│   ├── parallel_inference.py           # Batch inference
│   ├── cli.py                          # Training CLI
│   └── __init__.py
│
├── utils/                               # Utilities (20,000+ lines)
│   ├── language_detector.py            # Language detection (96%+)
│   ├── system_resources.py             # Hardware detection
│   ├── data_detector.py                # Dataset discovery and analysis
│   ├── cost_estimator.py               # Cost calculation
│   ├── token_analysis.py               # Token counting
│   ├── metadata_manager.py             # Session metadata
│   ├── annotation_to_training.py       # Workflow conversion
│   ├── training_data_utils.py          # Training data management
│   ├── annotation_session_manager.py   # Session tracking
│   ├── session_summary.py              # Session summaries
│   ├── training_summary_generator.py   # Summary reports
│   ├── logging_utils.py                # Logging infrastructure
│   ├── data_filter_logger.py           # Filter statistics
│   ├── benchmark_utils.py              # Benchmark utilities
│   ├── benchmark_helpers.py            # Benchmark helpers
│   ├── language_filtering.py           # Language distribution
│   ├── language_normalizer.py          # Language code normalization
│   ├── training_paths.py               # Path management
│   ├── resource_display.py             # CLI resource visualization
│   ├── model_display.py                # Model recommendation display
│   ├── rich_progress_manager.py        # Progress tracking
│   └── __init__.py
│
└── validators/                          # Validation & QC (1,000+ lines)
    ├── annotation_validator.py         # Main validator class
    ├── doccano_exporter.py             # Doccano format export
    └── __init__.py
```

---

## Key Numbers at a Glance

| Metric | Count |
|--------|-------|
| Total Python Files | 72 |
| Total Lines of Code | ~79,268 |
| Total Classes | 65+ |
| Total Methods | 500+ |
| Modules | 9 |
| Language Models | 13 |
| SOTA Models | 30+ |
| LLM Providers | 5 |
| Validation Methods | 7 |
| Text Detection Capabilities | 7 |
| CLI Modes | 3 |
| Data Formats Supported | 5+ |
| Languages Supported | 75+ |
| Documentation Files | 3 |

---

## Getting Started

### For Quick Understanding
1. Read **ARCHITECTURE_QUICK_SUMMARY.md**
2. Navigate to relevant section in this index
3. Check file locations for source code

### For Development
1. Start with **ARCHITECTURE_QUICK_SUMMARY.md**
2. Read relevant section in **ARCHITECTURE_ANALYSIS.md**
3. Navigate to actual source files using locations provided
4. Check class/method documentation in analysis

### For Integration
1. Check "Usage Pattern Example" in QUICK_SUMMARY.md
2. Review relevant module section in ARCHITECTURE_ANALYSIS.md
3. Access source files at provided absolute paths
4. Consult pyproject.toml for dependencies

---

## Core Dependencies Summary

### Essential (included)
- transformers, torch, pandas, numpy
- openai, anthropic, ollama
- rich, inquirer (CLI)
- lingua-language-detector
- sqlalchemy, psycopg2 (database)
- pydantic (validation)

### Optional (advanced features)
- google-generativeai (Gemini)
- label-studio-sdk (annotation platform)
- mlflow, wandb (experiment tracking)
- fastapi, gradio (web serving)

See pyproject.toml for full dependency list with versions.

---

## Documentation Links

**Generated:** October 22, 2025
**LLM_Tool Version:** 1.0.0
**Author:** Antoine Lemor

### Files Created
- `/Users/antoine/Documents/GitHub/LLM_Tool/ARCHITECTURE_INDEX.md` (this file)
- `/Users/antoine/Documents/GitHub/LLM_Tool/ARCHITECTURE_QUICK_SUMMARY.md` (10 KB)
- `/Users/antoine/Documents/GitHub/LLM_Tool/ARCHITECTURE_ANALYSIS.md` (42 KB)

All documents use absolute file paths for easy reference and navigation.
