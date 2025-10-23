# LLM_Tool Architecture - Quick Summary

## Executive Overview

**LLM_Tool** is a comprehensive Python package (79,268 lines of code across 72 files) that provides:
- State-of-the-art LLM-powered text annotation
- BERT model training and benchmarking (50+ models)
- Multilingual text processing and validation
- Professional-grade CLI with Rich visualization

---

## Core Capabilities at a Glance

### 1. ANNOTATION (LLMAnnotator Class)
- **5 Model Providers:** OpenAI, Anthropic, Google, Ollama, Llama.cpp
- **Parallel Execution:** ProcessPoolExecutor with process pooling
- **Batch Operations:** OpenAI Batch API for cost-effective large-scale annotation
- **Data Formats:** CSV, Excel, Parquet, JSON, JSONL, RData, PostgreSQL
- **Features:** Multi-prompt support, JSON repair, schema validation, cost estimation

### 2. TRAINING ARENA (50+ Models)
**Language-Specific Models (13):**
- BERT, Camembert (FR), GermanBert, SpanishBert, ItalianBert
- PortugueseBert, ArabicBert, ChineseBert, RussianBert, HindiBert, SwedishBert
- MultiBERT, XLMRoberta

**SOTA Models (30+):**
- DeBERTa v3 (3 sizes), RoBERTa (3 sizes), ELECTRA (3 sizes), ALBERT (3 sizes)
- BigBird, Longformer, mDeBERTa, XLM-RoBERTa variants
- French models: CamembertaV2, FlauBERT, BARThez, FrALBERT, FrELECTRA

**Features:**
- Early stopping with patience
- Cross-validation support
- Multi-GPU training
- Mixed precision (FP16) support
- Multi-label classification
- Auto model selection based on task complexity

### 3. VALIDATION (7 Methods)
1. **Schema Validation** - Pydantic-based field validation
2. **Label Validation** - Distribution and consistency analysis
3. **Data Quality** - Missing values, duplicates, type checking
4. **Inter-Annotator Agreement** - Cohen's Kappa, confusion matrix
5. **Annotation Sampling** - Stratified sampling for 95% CI human review
6. **Doccano Export** - JSONL format for interactive review
7. **Cost Validation** - Token counting and budget tracking

### 4. TEXT DETECTION (7 Capabilities)
1. **Language Detection** - 75+ languages, 96%+ accuracy (lingua-language-detector)
2. **Text Column Detection** - Automatic identification with scoring
3. **Text Length Analysis** - Character, word, sentence statistics + percentiles
4. **Token Analysis** - Prompt/completion token estimation per-text
5. **Label Column Detection** - Categorical column identification
6. **Identifier Column Detection** - ID uniqueness checking
7. **Text Classification Scoring** - Entropy, cardinality, JSON field detection

### 5. CLI FEATURES (3 Major Modes)
**Mode 1: Annotator Factory**
- End-to-end annotation workflow
- Model/API selection with auto-detection
- Prompt configuration and management
- Cost estimation before execution
- Post-annotation training trigger

**Mode 2: Training Arena**
- Dataset loading and exploration
- Column selection (text, label, identifier)
- Language detection and filtering
- Model selection with complexity assessment
- 18-step interactive training wizard
- Benchmarking of multiple models
- Per-category and per-language training

**Mode 3: Validation Lab**
- Quality control and validation
- Consensus building from multiple annotators
- Inter-annotator agreement metrics
- Doccano export for human review

**Additional Features:**
- System resource detection and display
- Auto-detection of available models
- Smart suggestions based on context
- Resume center for interrupted workflows
- Execution history tracking
- Configuration profiles

### 6. BENCHMARKING
- **BenchmarkRunner** - Multi-model evaluation orchestration
- **Per-Category Training** - Category-specific model training
- **Per-Language Training** - Language-specific optimization
- **Result Aggregation** - CSV/JSON consolidation
- **Ranking & Comparison** - Cross-model performance analysis
- **Metrics:** Accuracy, Precision, Recall, F1, Macro/Micro/Weighted averages

### 7. CONFIGURATION MANAGEMENT
**Settings Classes:**
- APIConfig - Provider settings (OpenAI, Anthropic, Google, etc.)
- LocalModelConfig - Ollama/Llama.cpp configuration
- TrainingConfig - Batch size, learning rate, epochs, early stopping
- LanguageConfig - Language detection settings
- DataConfig - Format, chunk size, workers
- PathConfig - Directory management
- LoggingConfig - Log level, file/console output

**APIKeyManager:**
- Secure key storage
- Environment variable fallback
- Multi-provider support
- Credential management

### 8. DATA PROCESSING & EXPORT
**Input:** CSV, Excel, Parquet, JSON, JSONL, RData, PostgreSQL
**Output:** CSV, Excel, Parquet, JSON, JSONL, PostgreSQL, Doccano
**Features:**
- Incremental file appending
- Unicode sanitization
- Batch processing
- Database transactions

### 9. STATISTICAL ANALYSIS
- Label distribution analysis
- Text statistics (length, word count, etc.)
- Language distribution analysis
- Model performance metrics
- Confidence analysis
- Inter-annotator agreement
- Cost statistics
- Resource utilization analysis
- Training metrics aggregation
- Benchmark comparison

### 10. SYSTEM RESOURCE DETECTION
- GPU detection (CUDA, MPS, CPU)
- CPU information (cores, frequency, usage)
- Memory monitoring
- Storage analysis
- Automatic configuration recommendations
- Mac and Windows support

---

## Module Structure

| Module | Purpose | Size | Key Classes |
|--------|---------|------|-------------|
| `annotators/` | LLM annotation | 3,500+ lines | LLMAnnotator, OpenAIClient, AnthropicClient, GoogleClient, OllamaClient, LlamaCPPClient |
| `cli/` | Command-line interface | 25,000+ lines | AdvancedCLI, AnnotationWorkflow, TrainingArena, ValidationLab |
| `config/` | Configuration management | 700+ lines | Settings, APIConfig, APIKeyManager |
| `database/` | Data I/O | 1,000+ lines | FileHandler, PostgreSQLHandler, IncrementalFileWriter |
| `trainers/` | Model training | 15,000+ lines | BertBase, ModelTrainer, BenchmarkRunner, 50+ model wrappers |
| `utils/` | Utilities | 20,000+ lines | LanguageDetector, SystemResources, DataDetector, CostEstimator, etc. |
| `validators/` | Validation & QC | 1,000+ lines | AnnotationValidator, DoccanoExporter |
| `pipelines/` | Orchestration | 1,500+ lines | PipelineController |

---

## Key Statistics

- **72 Python files**
- **~79,268 lines of code**
- **65+ classes**
- **500+ methods**
- **75+ languages supported**
- **50+ transformer models**
- **5 LLM API providers**
- **7 text detection methods**
- **7 validation features**
- **3 CLI modes**
- **5+ data format support**

---

## Technology Stack

### Core Dependencies
- `transformers>=4.35.0` - Model implementations
- `torch>=2.0.0` - PyTorch training
- `pandas>=2.0.0` - Data manipulation
- `openai>=1.0.0` - OpenAI API
- `anthropic>=0.18.0` - Anthropic Claude API
- `ollama>=0.6.0` - Local Ollama models
- `rich>=14.0.0` - CLI visualization
- `lingua-language-detector>=2.0.0` - Language detection
- `sqlalchemy>=2.0.0` - Database ORM
- `pydantic>=2.0.0` - Data validation

### Optional Dependencies
- google-generativeai - Google Gemini
- label-studio-sdk - Annotation platform integration
- mlflow, wandb - Experiment tracking
- fastapi, gradio - Web serving
- sentence-transformers - Advanced embeddings

---

## File Locations (Absolute Paths)

All files are located in `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/`

### Key File Locations

**Annotation:**
- LLMAnnotator: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/llm_annotator.py`
- API Clients: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/api_clients.py`
- Local Models: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/local_models.py`
- Prompt Wizard: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/prompt_wizard.py`

**Training:**
- BertBase: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/bert_base.py`
- SOTA Models: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/sota_models.py`
- Model Trainer: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/model_trainer.py`
- Benchmarking: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py`

**CLI:**
- Advanced CLI: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py`
- Annotation Workflow: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/annotation_workflow.py`
- Training Arena: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/training_arena_integrated.py`
- Validation Lab: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/validation_lab.py`

**Utilities:**
- Language Detector: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/language_detector.py`
- System Resources: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/system_resources.py`
- Data Detector: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/data_detector.py`
- Cost Estimator: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/cost_estimator.py`

**Validation:**
- Annotation Validator: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py`
- Doccano Exporter: `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/doccano_exporter.py`

---

## Usage Pattern Example

```python
# 1. Annotation
from llm_tool.annotators import LLMAnnotator
annotator = LLMAnnotator()
results = annotator.annotate({
    'input_file': 'data.csv',
    'provider': 'openai',
    'model': 'gpt-4',
    'prompt': 'Classify this text...',
    'text_columns': ['text'],
    'output_file': 'annotated.csv'
})

# 2. Training
from llm_tool.trainers import ModelTrainer
trainer = ModelTrainer()
results = trainer.train_and_benchmark({
    'dataset': 'training_data.csv',
    'model': 'DeBERTaV3Base',
    'epochs': 5
})

# 3. Validation
from llm_tool.validators import AnnotationValidator
validator = AnnotationValidator()
validation_results = validator.validate(annotated_data)

# 4. CLI (Interactive)
from llm_tool.cli import AdvancedCLI
cli = AdvancedCLI()
cli.run()  # Starts interactive menu
```

---

## Entry Point

The package has a main CLI entry point defined in `pyproject.toml`:
```
[project.scripts]
llm-tool = "llm_tool.__main__:main"
llmtool = "llm_tool.__main__:main"
```

Run with: `llm-tool` or `llmtool`

---

## Complete Documentation

A full detailed analysis is available in: `/Users/antoine/Documents/GitHub/LLM_Tool/ARCHITECTURE_ANALYSIS.md`

This quick summary covers the major features. The full report includes:
- Detailed class documentation
- Method signatures
- Data flow diagrams
- Configuration examples
- Advanced features
- Performance tuning recommendations

