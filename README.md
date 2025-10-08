```
██╗     ██╗     ███╗   ███╗    ████████╗ ██████╗  ██████╗ ██╗
██║     ██║     ████╗ ████║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║
██║     ██║     ██╔████╔██║       ██║   ██║   ██║██║   ██║██║
██║     ██║     ██║╚██╔╝██║       ██║   ██║   ██║██║   ██║██║
███████╗███████╗██║ ╚═╝ ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚══════╝╚══════╝╚═╝     ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
```

**State-of-the-Art LLM-Powered Annotation & BERT Training Pipeline**

> 🚀 Zero-shot annotation with GPT/Claude/Ollama → Training data generation → Fine-tuned BERT models → Production inference
> 🌍 75+ languages • 70+ transformer models • Multi-label classification • Benchmarking & validation

---

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage in VSCode](#-usage-in-vscode)
- [CLI Modes](#-cli-modes)
- [Architecture](#-architecture)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## ✨ Features

### 🎨 **The Annotator** - Zero-Shot LLM Annotation
- Annotate datasets using OpenAI (GPT-4, o1, o3), Claude, Gemini, or local Ollama/LlamaCPP models
- Multi-prompt fusion with JSON validation and auto-repair (5-retry mechanism)
- Parallel processing with incremental saves and resume capability
- Export to Label Studio/Doccano for human review
- Statistical sample size calculation (95% confidence intervals)

### 🏭 **The Annotator Factory** - End-to-End Pipeline
- LLM annotation → Training data preparation → Model fine-tuning (one-click workflow)
- Automatic language detection (96%+ accuracy with lingua)
- Smart class balancing and stratified splitting
- PostgreSQL, CSV, Excel, Parquet, JSON/JSONL, RData/RDS support

### 🎮 **Training Arena** - Model Training & Benchmarking
- Train 70+ pre-trained models: BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT, XLM-RoBERTa, CamemBERT, etc.
- Automatic model selection based on detected languages
- Multi-label classification with reinforcement learning
- Comprehensive benchmarking across multiple models
- Training session persistence with resume/relaunch capability
- Live metrics tracking (F1, accuracy, precision, recall, confusion matrix)

### 🤖 **BERT Annotation Studio** - Production Inference
- High-throughput parallel inference (GPU/CPU)
- Batch processing with progress tracking
- Export annotations in multiple formats

### 🔍 **Validation Lab** - Quality Assurance
- Annotation quality scoring
- Inter-annotator agreement (Cohen's Kappa)
- Stratified sampling for review
- Schema validation with Pydantic

---

## 🔧 Requirements

### Python Version
- **Python 3.9 or higher** (tested with 3.9, 3.10, 3.11, 3.12, 3.13)
- Python 3.11+ recommended for optimal performance

### Operating System
- **macOS** (Apple Silicon MPS and Intel)
- **Linux** (CUDA/ROCm support)
- **Windows** (CPU/CUDA support)

### Hardware
- **Minimum**: 8 GB RAM, 4 CPU cores
- **Recommended**: 16+ GB RAM, 8+ CPU cores, GPU (NVIDIA/Apple MPS)
- **Optimal**: 32+ GB RAM, 16+ CPU cores, GPU with 8+ GB VRAM

### External Dependencies (Optional)
- **Ollama**: For local LLM inference (install from https://ollama.ai)
- **PostgreSQL**: For database-backed datasets (optional)

---

## 📦 Installation

### Method 1: Install from Source (Recommended for Development)

#### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/LLM_Tool.git
cd LLM_Tool
```

#### Step 2: Create a Virtual Environment
```bash
# Using venv (built-in)
python3 -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

#### Step 3: Install LLM Tool
```bash
# Install with core dependencies
pip install -e .

# OR install with advanced features (Anthropic, Google, LlamaCPP, Label Studio, MLOps tools)
pip install -e ".[advanced]"

# OR install with development tools
pip install -e ".[dev]"

# OR install everything
pip install -e ".[all]"
```

### Method 2: Install from PyPI (When Published)
```bash
pip install llm-tool

# With advanced features
pip install llm-tool[advanced]
```

### Verify Installation
```bash
llm-tool --version
```

---

## 🚀 Quick Start

### 1. Configure API Keys (If Using Cloud LLMs)

LLM Tool stores API keys securely with encryption. Run the interactive CLI to set up:

```bash
llm-tool
```

Navigate to **Profile Manager → API Key Configuration** and add your keys:
- OpenAI API Key (for GPT-4, o1, o3 models)
- Anthropic API Key (for Claude models)
- Google API Key (for Gemini models)

**OR** use environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

### 2. Launch the Interactive CLI
```bash
llm-tool
```

You'll see the main menu:
```
╭─────────────────────────────────── Main Menu ────────────────────────────────────╮
│ 1  🎨 The Annotator - LLM Tool annotates, you decide                             │
│ 2  🏭 The Annotator Factory - Clone The Annotator into ML Models                 │
│ 3  🎮 Training Arena - Train Your Own Models                                     │
│ 4  🤖 BERT Annotation Studio - Annotate with Trained Models                      │
│ 5  🔍 Validation Lab - Quality Assurance Tools                                   │
│ 6  💾 Profile Manager - Save & Load Configurations                               │
│ 7  📚 Documentation & Help                                                       │
│ 0  ❌ Exit                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

### 3. Quick Annotation Example (Using Ollama - 100% Local)

#### Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Llama 3.2)
ollama pull llama3.2
```

#### Run Annotation
1. Launch `llm-tool`
2. Select **1 - The Annotator**
3. Choose your dataset (CSV/JSON/Excel/PostgreSQL)
4. Select text column and configure annotation schema
5. Choose **Ollama** as LLM provider → select `llama3.2`
6. Start annotation → Monitor progress → Export to Doccano/Label Studio

### 4. Train Your First Model

1. Launch `llm-tool`
2. Select **3 - Training Arena**
3. Select **New Training Session**
4. Choose your annotated dataset
5. System auto-detects languages and recommends models
6. Select model (e.g., `bert-base-uncased` for English)
7. Configure epochs (default: 10)
8. Start training → Monitor live metrics → Best model saved automatically

---

## 💻 Usage in VSCode

### Step 1: Open Project in VSCode
```bash
cd LLM_Tool
code .
```

### Step 2: Select Python Interpreter

1. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Python: Select Interpreter"
3. Choose the virtual environment: `.venv/bin/python`

### Step 3: Configure VSCode Terminal

Ensure your integrated terminal uses the virtual environment:

**File → Preferences → Settings** (or `Cmd+,`)

Search for `python.terminal.activateEnvironment` and ensure it's **checked**.

### Step 4: Run LLM Tool from VSCode Terminal

Open integrated terminal (`Ctrl+` ` or **View → Terminal**):

```bash
# Terminal should show (.venv) prefix
llm-tool
```

### Step 5: Debug Mode (Optional)

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "LLM Tool CLI",
      "type": "python",
      "request": "launch",
      "module": "llm_tool",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

Press `F5` to launch in debug mode.

### Step 6: Recommended VSCode Extensions

- **Python** (ms-python.python) - Python language support
- **Pylance** (ms-python.vscode-pylance) - Fast Python IntelliSense
- **Jupyter** (ms-toolsai.jupyter) - For notebook-based workflows
- **Rainbow CSV** (mechatroner.rainbow-csv) - CSV visualization

---

## 🎯 CLI Modes

### 🎨 Mode 1: The Annotator
**Use when**: You have raw text data that needs labeling

**Workflow**:
```
Load Data → Define Schema → Configure LLM → Run Annotation → Validate → Export
```

**Example Use Cases**:
- Sentiment analysis annotation (positive/negative/neutral)
- Topic classification (politics, sports, tech, etc.)
- Named entity recognition (NER)
- Multi-label document tagging

**Supported LLM Providers**:
- **OpenAI**: gpt-4, gpt-4o, gpt-4-turbo, o1-preview, o1-mini, o3-mini
- **Anthropic**: claude-3-opus, claude-3.5-sonnet, claude-3-haiku
- **Google**: gemini-1.5-pro, gemini-1.5-flash
- **Ollama**: llama3.2, mistral, phi, gemma, qwen (local)
- **LlamaCPP**: Any GGUF model file (local)

### 🏭 Mode 2: The Annotator Factory
**Use when**: You want end-to-end automation (LLM annotation → trained model)

**Workflow**:
```
Load Data → LLM Annotation → Training Data Prep → Model Training → Benchmarking → Model Export
```

**Features**:
- Automatic language detection
- Smart data splitting (train/val/test)
- Class balancing options
- Model recommendations based on language
- One-click execution

### 🎮 Mode 3: Training Arena
**Use when**: You have labeled training data and want to train custom models

**Workflow**:
```
Load Training Data → Detect Languages → Select Models → Configure Training → Train → Benchmark → Export
```

**Supported Models** (70+ total):

| Category | Models |
|----------|--------|
| **English** | BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT, DistilBERT |
| **French** | CamemBERT, FlauBERT, BARThez |
| **German** | GBERT, GermanBERT |
| **Spanish** | BETO, RoBERTuito |
| **Chinese** | BERT-Chinese, RoBERTa-Chinese |
| **Arabic** | AraBERT, mARBERTv2 |
| **Multilingual** | XLM-RoBERTa (100+ languages), mBERT, mDeBERTa |
| **Long Documents** | Longformer (4096 tokens), BigBird, LED (16384 tokens) |

**Training Features**:
- Automatic early stopping with best checkpoint saving
- Reinforcement learning for class imbalance
- Multi-label classification support
- Cross-validation and hyperparameter tuning
- Resume/relaunch interrupted sessions

### 🤖 Mode 4: BERT Annotation Studio
**Use when**: You have trained models and need to annotate new data at scale

**Workflow**:
```
Load Trained Model → Load New Data → Configure Inference → Run Parallel Annotation → Export
```

**Features**:
- GPU-accelerated inference (CUDA/MPS)
- CPU multi-processing fallback
- Batch processing with dynamic batch sizing
- Progress tracking with ETA
- Export to CSV/JSON/JSONL/Doccano

### 🔍 Mode 5: Validation Lab
**Use when**: You need quality control on annotations

**Workflow**:
```
Load Annotations → Quality Scoring → Agreement Analysis → Stratified Sampling → Export Issues
```

**Metrics**:
- Inter-annotator agreement (Cohen's Kappa)
- Confidence score distributions
- Label imbalance detection
- Anomaly identification
- Schema compliance validation

---

## 🏗️ Architecture

```
LLM_Tool/
├── llm_tool/
│   ├── annotators/          # LLM annotation engines
│   │   ├── llm_annotator.py         # Core annotation orchestrator
│   │   ├── api_clients.py           # OpenAI/Anthropic/Google clients
│   │   ├── local_models.py          # Ollama/LlamaCPP integration
│   │   ├── prompt_wizard.py         # Interactive prompt creation
│   │   └── json_cleaner.py          # JSON repair & validation
│   │
│   ├── trainers/            # Model training & benchmarking
│   │   ├── model_trainer.py         # Training orchestration
│   │   ├── bert_base.py             # Base BERT implementation
│   │   ├── multi_label_trainer.py   # Multi-label classification
│   │   ├── benchmarking.py          # Model comparison
│   │   ├── models.py                # Standard model catalog
│   │   └── sota_models.py           # 50+ SOTA models
│   │
│   ├── cli/                 # Command-line interfaces
│   │   ├── advanced_cli.py          # Rich interactive CLI
│   │   ├── main_cli.py              # Simple CLI
│   │   └── bert_annotation_studio.py
│   │
│   ├── validators/          # Quality control
│   │   ├── annotation_validator.py  # Quality metrics
│   │   └── doccano_exporter.py      # Export utilities
│   │
│   ├── utils/               # Utilities
│   │   ├── language_detector.py     # 96%+ accuracy detection
│   │   ├── system_resources.py      # GPU/CPU monitoring
│   │   ├── metadata_manager.py      # Session persistence
│   │   └── training_data_utils.py   # Data preparation
│   │
│   ├── pipelines/           # Pipeline orchestration
│   ├── config/              # Configuration & API key management
│   └── database/            # Data handlers (PostgreSQL, files)
│
├── data/                    # Data storage
├── models/                  # Trained model storage
├── prompts/                 # Prompt templates
└── docs/                    # Documentation
```

---

## 🔬 Advanced Features

### Multi-Label Classification
Train models that predict multiple labels per document:
```python
# Example: Document can be tagged with ["politics", "international", "economics"]
Training Arena → Multi-label strategy → Automatic threshold optimization
```

### Reinforcement Learning
Automatically handles class imbalance with adaptive retraining:
- F1 threshold monitoring
- Minority class oversampling
- Adaptive learning rate adjustment
- Loss weight balancing

### Language Detection
Automatic detection of 75+ languages using lingua (96%+ accuracy):
- Per-document language tagging
- Language-specific model recommendations
- Mixed-language dataset support
- Separate models per language option

### Prompt Engineering Wizard
Interactive wizard for creating effective annotation prompts:
- Context configuration
- Few-shot examples
- Output schema definition
- Multi-prompt strategies

### Session Management
All training sessions are persisted and recallable:
- Resume interrupted training
- Relaunch with same parameters
- Session history browsing
- Metadata tracking (model, dataset, hyperparameters)

### Benchmarking Mode
Compare multiple models before committing to full training:
- Quick evaluation (3-5 epochs)
- Performance comparison table
- Class-wise F1 scores
- Confusion matrices
- Automatic best model selection

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'llm_tool'"
**Solution**: Ensure virtual environment is activated and package is installed:
```bash
source .venv/bin/activate  # macOS/Linux
pip install -e .
```

### Issue: "CUDA out of memory" during training
**Solution**: Reduce batch size in Training Arena settings:
- Try batch size: 8 → 4 → 2
- Use CPU-only mode if GPU memory is limited
- Close other GPU-intensive applications

### Issue: Ollama connection refused
**Solution**: Ensure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve
```

### Issue: MPS backend errors (macOS Apple Silicon)
**Solution**: Fall back to CPU:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
llm-tool
```

### Issue: API rate limits (OpenAI/Anthropic)
**Solution**: Configure rate limiting in Advanced CLI settings:
- Reduce concurrent requests
- Add delay between batches
- Use batch API endpoints (OpenAI)

### Issue: "Training session not found" when resuming
**Solution**: Ensure metadata files exist:
```bash
ls logs/training_arena/
# Should show training_session_YYYYMMDD_HHMMSS/ directories with training_metadata.json
```

---

## 📊 Performance Benchmarks

**Annotation Speed** (Ollama Llama 3.2 on M2 Max):
- 15-30 documents/minute (depends on prompt complexity)
- Parallel processing: 50-100 documents/minute (4 workers)

**Training Speed** (BERT-base, 5K documents):
- Apple M2 Max (MPS): ~8 min/epoch
- NVIDIA RTX 3090: ~3 min/epoch
- CPU (16 cores): ~25 min/epoch

**Inference Speed** (Trained BERT, batch size 32):
- Apple M2 Max (MPS): ~500 docs/second
- NVIDIA RTX 3090: ~1200 docs/second
- CPU (16 cores): ~150 docs/second

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **HuggingFace** for the Transformers library
- **Ollama** for local LLM inference
- **OpenAI**, **Anthropic**, **Google** for LLM APIs
- **Rich** for beautiful CLI rendering

---

## 📬 Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review examples in `examples/`

---

**Made with ❤️ by Antoine Lemor**

*Transform your text data into production-ready ML models in minutes, not months.*
