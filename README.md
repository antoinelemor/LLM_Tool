```
██╗     ██╗     ███╗   ███╗    ████████╗ ██████╗  ██████╗ ██╗
██║     ██║     ████╗ ████║    ╚══██╔══╝██╔═══██╗██╔═══██╗██║
██║     ██║     ██╔████╔██║       ██║   ██║   ██║██║   ██║██║
██║     ██║     ██║╚██╔╝██║       ██║   ██║   ██║██║   ██║██║
███████╗███████╗██║ ╚═╝ ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚══════╝╚══════╝╚═╝     ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
```

**Transform Your Research Data into Machine Learning Models — No Coding Required**

> 🎓 Built for Social Science Researchers • 🌍 75+ Languages • 🤖 Zero-Shot Annotation with AI • 📊 Automated Model Training

---

## 🎯 What is LLM Tool?

**LLM Tool is an all-in-one platform that turns raw text data into trained machine learning models**, designed specifically for researchers who want to leverage AI without becoming programmers.

### The Problem It Solves

As a social science researcher, you might have:

- **Thousands of survey responses** that need to be categorized by theme
- **Social media posts** that need sentiment analysis or topic classification
- **Interview transcripts** that require coding into analytical categories
- **News articles** that need to be classified by topic, stance, or framing
- **Historical documents** that require systematic categorization

**Traditional approaches are painful:**
- ❌ **Manual annotation**: Extremely time-consuming (weeks/months for large datasets)
- ❌ **Hiring annotators**: Expensive, requires extensive training and quality control
- ❌ **Existing ML tools**: Require programming skills, complex setup, and technical expertise
- ❌ **Commercial APIs**: Costly at scale, data privacy concerns, vendor lock-in

### Why LLM Tool is Different

**LLM Tool makes state-of-the-art AI accessible through a simple, interactive interface:**

✅ **No Coding Required** — Beautiful, interactive command-line interface guides you through every step
✅ **Start with AI, End with Your Own Models** — Use GPT/Claude to annotate initial data, then train custom models that you own
✅ **100% Local Option** — Works entirely on your computer with Ollama (no cloud APIs, no costs, no data sharing)
✅ **Research-Oriented** — Built for social science workflows: inter-annotator agreement, stratified sampling, quality metrics
✅ **Multilingual from Day One** — Automatic language detection and 75+ language support
✅ **Production-Ready** — Generate thousands of annotations per hour, train models in minutes

### Who Should Use This?

- **Social Scientists** analyzing qualitative or mixed-method data
- **Political Scientists** coding news articles, speeches, or social media
- **Sociologists** categorizing interview responses or ethnographic notes
- **Communication Researchers** analyzing media content or discourse
- **Historians** classifying historical documents or archives
- **Digital Humanities** scholars working with large text corpora
- **Anyone** with text data that needs systematic categorization

**You don't need to know Python. You don't need to understand transformers. You just need data and research questions.**

---

## 📋 Table of Contents

- [What is LLM Tool?](#-what-is-llm-tool)
- [How It Works: The Workflow](#-how-it-works-the-workflow)
- [Installation](#-installation)
  - [Step-by-Step Installation (VSCode)](#step-by-step-installation-vscode)
  - [Step-by-Step Installation (Terminal Only)](#step-by-step-installation-terminal-only)
- [The 5 Modes Explained](#-the-5-modes-explained)
  - [Mode 1: The Annotator](#mode-1-the-annotator)
  - [Mode 2: The Annotator Factory](#mode-2-the-annotator-factory)
  - [Mode 3: Training Arena](#mode-3-training-arena)
  - [Mode 4: BERT Annotation Studio](#mode-4-bert-annotation-studio)
  - [Mode 5: Validation Lab](#mode-5-validation-lab)
- [Complete Example: From Raw Data to Trained Model](#-complete-example-from-raw-data-to-trained-model)
- [FAQ](#-faq)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🔄 How It Works: The Workflow

LLM Tool offers **two main pathways** depending on your needs:

### **Pathway 1: AI-Assisted Annotation → Export for Human Review**

```
Your Data (CSV/Excel) → AI Annotation → Quality Check → Export to Label Studio/Doccano → Human Review
```

**Use when**: You want AI assistance but need human validation for publication-quality data.

### **Pathway 2: AI Annotation → Train Your Own Model → Automated Classification**

```
Your Data → AI Annotation → Train BERT Model → Use Model for New Data → Thousands of Classifications
```

**Use when**: You need to classify large volumes of data and can validate a sample for quality.

### **Pathway 3: Start from Scratch (You Already Have Labeled Data)**

```
Your Labeled Data → Train Multiple Models → Benchmark Performance → Deploy Best Model
```

**Use when**: You already have annotated training data and want the best custom model.

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

**Two installation methods**: VSCode (recommended for beginners) or Terminal only (for advanced users).

### Step-by-Step Installation (VSCode)

**Perfect for researchers new to programming. VSCode provides a visual interface for managing your project.**

#### 1. Install Prerequisites

**Install Python 3.11 or higher:**
- **macOS**: Download from [python.org](https://www.python.org/downloads/) or `brew install python@3.11`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt install python3.11` (Ubuntu/Debian) or `sudo dnf install python3.11` (Fedora)

**Install Visual Studio Code:**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install the **Python extension** (search "Python" in VSCode Extensions panel)

#### 2. Download LLM Tool

**Option A: Download ZIP** (easiest for beginners)
1. Download this repository as ZIP
2. Extract it to a folder (e.g., `Documents/LLM_Tool`)

**Option B: Use Git** (if you have it installed)
```bash
git clone https://github.com/YOUR-USERNAME/LLM_Tool.git
cd LLM_Tool
```

#### 3. Open in VSCode

1. Launch VSCode
2. **File → Open Folder**
3. Select the `LLM_Tool` folder

You should see this structure:
```
LLM_Tool/
├── llm_tool/          ← Main code
├── examples/          ← Example scripts
├── README.md          ← This file
├── setup.py           ← Installation config
└── install.sh         ← Automated installer
```

#### 4. Run the Installation Script

Open the integrated terminal in VSCode (**View → Terminal** or `` Ctrl+` ``).

**macOS/Linux:**
```bash
chmod +x install.sh
./install.sh --all
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[all]"
```

The installer will:
- ✅ Check your Python version
- ✅ Create a virtual environment (`.venv`)
- ✅ Install all dependencies
- ✅ Verify everything works

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║              LLM Tool - Installation Script             ║
╚══════════════════════════════════════════════════════════╝

Step 1: Checking Python version...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Python 3.11.0 found

Step 2: Creating virtual environment...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Virtual environment created at .venv/

...

╔══════════════════════════════════════════════════════════╗
║                  🎉 INSTALLATION COMPLETE! 🎉            ║
╚══════════════════════════════════════════════════════════╝
```

#### 5. Select Python Interpreter in VSCode

1. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: **Python: Select Interpreter**
3. Choose: **`.venv/bin/python`** (should show first)

You'll see `(.venv)` in the bottom-left corner of VSCode.

#### 6. Launch LLM Tool

In the VSCode terminal:
```bash
llm-tool
```

You should see the main menu:
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

Select option [0/1/2/3/4/5/6/7] (1):
```

**🎉 Success! LLM Tool is now running.**

---

### Step-by-Step Installation (Terminal Only)

**For users comfortable with the command line.**

#### 1. Install Python 3.11+

Verify you have Python 3.11 or higher:
```bash
python3 --version
# Should show: Python 3.11.x or higher
```

If not, install from [python.org](https://www.python.org/downloads/).

#### 2. Clone or Download

```bash
git clone https://github.com/YOUR-USERNAME/LLM_Tool.git
cd LLM_Tool
```

#### 3. Run Automated Installer

**macOS/Linux:**
```bash
chmod +x install.sh
./install.sh --all
```

**Or manually:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install
pip install -e ".[all]"

# Verify
python verify_installation.py
```

#### 4. Launch

```bash
llm-tool
```

---

### Installation Options

- **Core only** (minimal dependencies): `pip install -e .`
- **All features** (recommended): `pip install -e ".[all]"`
- **Development** (with testing tools): `pip install -e ".[dev]"`

---

### What Gets Installed?

**Core Dependencies** (~500 MB):
- PyTorch for deep learning
- HuggingFace Transformers for BERT models
- OpenAI SDK for GPT access
- Ollama SDK for local LLMs
- Rich for beautiful CLI
- Pandas/NumPy for data handling

**Optional Dependencies** (with `[all]`):
- Anthropic SDK (Claude)
- Google GenAI SDK (Gemini)
- Label Studio SDK (annotation platform)
- MLOps tools (MLflow, Weights & Biases)

---

### Verify Installation

Run the verification script:
```bash
python verify_installation.py
```

**Expected output:**
```
═══════════════════════════════════════════════════════════
LLM TOOL - Installation Verification
═══════════════════════════════════════════════════════════
Checking Python version...
  ✓ Python 3.11.0 (OK)

Checking LLM Tool installation...
  ✓ llm-tool version 1.0.0

Checking core dependencies...
  ✓ Pandas                         version 2.3.3
  ✓ PyTorch                        version 2.8.0
  ✓ Transformers                   version 4.56.2
  ...

Checking GPU support...
  ✓ MPS (Apple Silicon) available
  # or: ✓ CUDA available: 1 device(s)
  # or: - No GPU detected (CPU only)

═══════════════════════════════════════════════════════════
✓ ALL CHECKS PASSED

LLM Tool is correctly installed and ready to use!
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

## 🎯 The 5 Modes Explained

LLM Tool has **5 specialized modes** for different stages of your research workflow. Each mode has a dedicated, interactive interface that guides you step-by-step.

---

### Mode 1: The Annotator

**🎨 Zero-Shot Annotation with Large Language Models**

**What it does**: Uses AI (GPT, Claude, or local Ollama models) to automatically annotate your text data based on a schema you define.

**Perfect for**:
- Creating initial training data from scratch
- Exploring what categories emerge from your data
- Getting a "first pass" on large datasets before human review
- Testing classification schemes quickly

**Real-World Example**:
> You have 5,000 social media posts about climate change. You want to classify each post's **stance** (supporter/denier/neutral) and **emotion** (angry/hopeful/fearful/neutral). Instead of manually coding all 5,000, you use The Annotator with GPT-4 or a free local Ollama model to auto-annotate them in ~30 minutes.

#### What You'll See (Step-by-Step)

**Step 1: Select The Annotator**
```
Select option [0/1/2/3/4/5/6/7] (1): 1
```

**Step 2: Load Your Data**
```
📊 Select Your Data Source

╭─────┬──────────────────────────────────────────────────────╮
│ #   │ Data Source                                          │
├─────┼──────────────────────────────────────────────────────┤
│ 1   │ CSV File                                             │
│ 2   │ Excel File (.xlsx)                                   │
│ 3   │ JSON/JSONL File                                      │
│ 4   │ PostgreSQL Database                                  │
│ 5   │ Parquet File                                         │
│ 6   │ R Data File (.RData, .RDS)                           │
╰─────┴──────────────────────────────────────────────────────╯

Choose data source [1/2/3/4/5/6] (1): 1

📂 Enter path to CSV file: data/climate_posts.csv

✓ Loaded 5,243 rows from climate_posts.csv

Detected columns: ['post_id', 'text', 'author', 'date', 'likes']

Which column contains the text to annotate? text
```

**Step 3: Define Your Annotation Schema**
```
🏷️  Define Annotation Schema

You can:
  1. Use the Prompt Wizard (interactive, recommended)
  2. Load existing schema from file
  3. Define manually

Select option [1/2/3] (1): 1

╔════════════════════════════════════════════════════════════╗
║          Welcome to the Annotation Wizard! 🧙              ║
╚════════════════════════════════════════════════════════════╝

Let's design your annotation schema step by step.

What are you trying to annotate? (e.g., "sentiment", "topic", "stance"): stance

What categories should we use?
  Enter category 1: supporter
  Enter category 2: denier
  Enter category 3: neutral
  Enter category 4: [leave blank to finish]

✓ Schema created: 3 categories (supporter, denier, neutral)

Would you like to add additional fields? [y/n] (y): y

Additional field name: emotion
Type:
  1. Single choice (one category)
  2. Multiple choice (can select multiple)
  3. Free text

Select type [1/2/3] (1): 1

Emotion categories:
  Enter category 1: angry
  Enter category 2: hopeful
  Enter category 3: fearful
  Enter category 4: neutral
  Enter category 5: [leave blank to finish]

✓ Complete schema:
  - stance: single choice (supporter, denier, neutral)
  - emotion: single choice (angry, hopeful, fearful, neutral)
```

**Step 4: Choose Your LLM**
```
🤖 Select LLM Provider

╭─────┬────────────────────┬──────────────┬───────────────╮
│ #   │ Provider           │ Cost         │ Privacy       │
├─────┼────────────────────┼──────────────┼───────────────┤
│ 1   │ Ollama (Local)     │ FREE         │ 100% Private  │
│ 2   │ OpenAI (GPT-4)     │ ~$0.01/doc   │ Cloud-based   │
│ 3   │ Anthropic (Claude) │ ~$0.008/doc  │ Cloud-based   │
│ 4   │ Google (Gemini)    │ ~$0.001/doc  │ Cloud-based   │
╰─────┴────────────────────┴──────────────┴───────────────╯

Select provider [1/2/3/4] (1): 1

🦙 Available Ollama Models:
  1. llama3.2 (recommended - 3B params, fast)
  2. mistral (7B params, balanced)
  3. phi3 (3.8B params, efficient)

Select model [1/2/3] (1): 1

✓ Using ollama:llama3.2 (100% local, no API costs)
```

**Step 5: Configure and Run**
```
⚙️  Annotation Configuration

╭─────────────────────────┬──────────────────────────────╮
│ Parameter               │ Value                        │
├─────────────────────────┼──────────────────────────────┤
│ Total documents         │ 5,243                        │
│ Sample size (95% CI)    │ 357                          │
│ Full dataset            │ 5,243                        │
│ Parallel workers        │ 4                            │
│ Estimated time          │ ~30 minutes (full)           │
│                         │ ~2 minutes (sample)          │
╰─────────────────────────┴──────────────────────────────╯

Annotate full dataset or statistical sample? [full/sample] (sample): full

🚀 Starting annotation...

Progress: ████████████████████░░░░░░░░░ 2,547/5,243 (48.6%)
Speed: 43 docs/min | ETA: 12:34 | Errors: 3 (0.1%)

[Real-time progress bar with live updates]
```

**Step 6: Review and Export**
```
✓ Annotation complete!

📊 Summary Statistics:
  Total annotated: 5,240 (3 failed)

  Stance distribution:
    - supporter: 2,341 (44.7%)
    - denier: 1,523 (29.1%)
    - neutral: 1,376 (26.3%)

  Emotion distribution:
    - angry: 1,873 (35.7%)
    - hopeful: 1,201 (22.9%)
    - fearful: 1,432 (27.3%)
    - neutral: 734 (14.0%)

📤 Export Options:
  1. CSV (for analysis)
  2. JSONL (for training)
  3. Doccano format (for human review)
  4. Label Studio format (for human review)
  5. All formats

Select export [1/2/3/4/5] (5): 5

✓ Exported to:
  - data/annotations/climate_posts_annotated.csv
  - data/annotations/climate_posts_annotated.jsonl
  - data/annotations/climate_posts_doccano.jsonl
  - data/annotations/climate_posts_labelstudio.json
```

**Key Features:**
- **100% Free Option**: Use Ollama (local) at no cost
- **Multi-Prompt Fusion**: Optionally use multiple prompts and merge results for higher accuracy
- **Incremental Saving**: Stops and resumes automatically if interrupted
- **Quality Metrics**: See confidence scores, detect anomalies
- **Statistical Sampling**: Annotate representative samples for pilot studies

---

### Mode 2: The Annotator Factory

**🏭 Complete Pipeline: LLM Annotation → Trained Model (One Click)**

**What it does**: Combines The Annotator + Training Arena into one seamless workflow. You provide raw data, it outputs a trained model ready to use.

**Perfect for**:
- You need a custom classifier but don't have labeled data
- You want the fastest path from raw data to deployed model
- You're okay with AI-generated training data (with validation)

**Real-World Example**:
> You have 10,000 news articles. You want a model that classifies them by topic (politics/sports/tech/health). The Factory uses GPT-4 to annotate 1,000 articles, splits them into train/val/test, trains 3 different BERT models, benchmarks them, and gives you the best one — all automatically.

#### What You'll See (Abbreviated Flow)

```
🏭 THE ANNOTATOR FACTORY

This mode runs the complete pipeline:
  Step 1: Load your data
  Step 2: LLM annotation
  Step 3: Data preparation
  Step 4: Model training
  Step 5: Benchmarking
  Step 6: Export best model

Estimated total time: 45-90 minutes

Continue? [y/n] (y): y

[Goes through same data loading as Mode 1]
[Goes through same annotation as Mode 1]

✓ Annotation complete: 1,000 documents annotated

🔧 Preparing training data...
  - Train set: 700 documents (70%)
  - Validation set: 150 documents (15%)
  - Test set: 150 documents (15%)
  - Class balance: ✓ Stratified

🌍 Detected languages: EN (100%)

🤖 Recommended models for English:
  1. bert-base-uncased (fast, good baseline)
  2. roberta-base (better performance)
  3. deberta-v3-base (best performance, slower)

Select models to benchmark [1,2,3 or all] (all): all

🎯 Training 3 models with 5-fold cross-validation...

Model 1/3: bert-base-uncased
  Epoch 1/5: ████████████ 100% | F1: 0.7234
  Epoch 2/5: ████████████ 100% | F1: 0.7891 ⬆
  [...]
  ✓ Best F1: 0.8123 (epoch 4)

Model 2/3: roberta-base
  [...]
  ✓ Best F1: 0.8456 (epoch 3)

Model 3/3: deberta-v3-base
  [...]
  ✓ Best F1: 0.8621 (epoch 4)

📊 Benchmark Results:
╭──────────────────┬──────────┬──────────┬──────────┬──────────╮
│ Model            │ F1 Score │ Accuracy │ Precision│ Recall   │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ deberta-v3-base  │ 0.8621   │ 0.8667   │ 0.8598   │ 0.8645   │
│ roberta-base     │ 0.8456   │ 0.8467   │ 0.8401   │ 0.8512   │
│ bert-base-uncased│ 0.8123   │ 0.8133   │ 0.8089   │ 0.8156   │
╰──────────────────┴──────────┴──────────┴──────────┴──────────╯

🏆 Best model: deberta-v3-base

✓ Exported to: models/news_classifier_deberta/
  - Model files: pytorch_model.bin, config.json
  - Tokenizer: tokenizer files
  - Metrics: benchmark_results.json
  - Training log: training.log

Next steps:
  1. Use this model in Mode 4 (BERT Annotation Studio)
  2. Or load programmatically: ModelTrainer.load('models/news_classifier_deberta')
```

**What Makes This Special:**
- **Zero Manual Steps**: Fully automated from raw data to model
- **Multiple Model Comparison**: Always trains 2-3 models to find the best
- **Quality Checks**: Validates training data quality before training
- **Reproducible**: Saves all configurations for replication

---

### Mode 3: Training Arena

**🎮 Train & Benchmark Custom BERT Models (You Provide Labeled Data)**

**What it does**: Takes your already-labeled data and trains state-of-the-art transformer models. Supports 70+ model architectures in 15+ languages.

**Perfect for**:
- You already have labeled training data (from humans or LLM annotation)
- You want to compare multiple models to find the best one
- You need a model you can deploy and own (no API costs)
- You want to fine-tune for your specific domain

**Real-World Example**:
> You manually labeled 2,000 tweets for hate speech detection (hate/not hate). You want to train a model to classify millions more. Training Arena lets you train BERT, RoBERTa, and DeBERTa simultaneously, shows you which performs best, and gives you the trained model to use.

#### What You'll See

**Step 1: Choose Training Mode**
```
🎮 TRAINING ARENA

╭─────┬────────────────────────────────────────────────╮
│ #   │ Training Mode                                  │
├─────┼────────────────────────────────────────────────┤
│ 1   │ 🔄 Resume/Relaunch Training                    │
│     │    Load saved parameters from previous session │
│ 2   │ 🆕 New Training Session                        │
│     │    Start fresh with dataset selection          │
╰─────┴────────────────────────────────────────────────╯

Select option [1/2] (2): 2
```

**Step 2: Load Training Data**
```
📂 Load Training Dataset

Format requirements:
  - Must have a text column
  - Must have a label column
  - Supported formats: CSV, JSON, JSONL, Excel, Parquet

Enter path to dataset: data/hate_speech_labeled.csv

✓ Loaded 2,000 rows

Detected columns: ['tweet', 'label', 'annotator']

Text column: tweet
Label column: label

Label distribution:
  - hate: 734 (36.7%)
  - not_hate: 1,266 (63.3%)

⚠️  Class imbalance detected (36.7% vs 63.3%)
   → Recommendation: Enable reinforcement learning
```

**Step 3: Language Detection & Model Selection**
```
🌍 Language Detection

Analyzing text column...

Detected languages:
  - English (EN): 1,987 (99.3%)
  - Other: 13 (0.7%)

Primary language: English

🤖 Top 10 Recommended Models for EN:

╭───┬─────────────────────────┬─────────┬───────────┬──────────────╮
│ # │ Model ID                │ Size    │ Max Tokens│ Description  │
├───┼─────────────────────────┼─────────┼───────────┼──────────────┤
│ 1 │ bert-base-uncased       │ Base    │ 512       │ Solid        │
│   │                         │         │           │ baseline     │
│ 2 │ roberta-base            │ Base    │ 512       │ Better than  │
│   │                         │         │           │ BERT         │
│ 3 │ microsoft/deberta-v3-ba │ Base    │ 512       │ Best         │
│   │ se                      │         │           │ performance  │
│ 4 │ google/electra-base     │ Base    │ 512       │ Efficient    │
│ 5 │ distilbert-base-uncased │ Small   │ 512       │ Fast &       │
│   │                         │         │           │ lightweight  │
╰───┴─────────────────────────┴─────────┴───────────┴──────────────╯

Select model (enter number or model name): 3

✓ Selected: microsoft/deberta-v3-base
```

**Step 4: Training Configuration**
```
⚙️  Training Configuration

📏 Token Length Strategy

Your data stats:
  - Mean tokens: 42
  - Max tokens: 256
  - 95th percentile: 89

✓ All documents fit within 512 tokens → No special handling needed

🎓 Reinforced Learning for Class Imbalance?

What is it?
  - Automatically retrains if F1 < threshold
  - Oversamples minority class (hate: 36.7%)
  - Adjusts loss weights

Recommended for imbalanced datasets like yours.

Enable? [y/n] (y): y

⏱️  Training Epochs

Recommendation: 10 epochs (system auto-saves best checkpoint)

Number of epochs (10): 10

📊 Batch Size

Available GPU memory: 96 GB (Apple MPS)
Recommended batch size: 16

Batch size (16): 16

✅ Configuration Summary:
  Model: microsoft/deberta-v3-base
  Epochs: 10
  Batch size: 16
  Reinforcement: Enabled
  Early stopping: Enabled

Start training? [y/n] (y): y
```

**Step 5: Training Progress**
```
🚀 Training Started

Session ID: training_session_20250108_143045
Metadata will be automatically saved for resume capability

Epoch 1/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 125/125 | 2:34

Train Loss: 0.4521 | Val Loss: 0.3891
Val F1: 0.7234 | Val Acc: 0.7467

Epoch 2/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 125/125 | 2:31

Train Loss: 0.2891 | Val Loss: 0.2645
Val F1: 0.8123 ⬆ (NEW BEST! Checkpoint saved)

[...]

Epoch 7/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 125/125 | 2:28

Train Loss: 0.0891 | Val Loss: 0.1823
Val F1: 0.8876 ⬆ (NEW BEST! Checkpoint saved)

Epoch 8/10
Train Loss: 0.0756 | Val Loss: 0.1891
Val F1: 0.8845 (no improvement)

⚠️  Early stopping triggered (patience: 2 epochs)

✓ Training complete!

📊 Final Results:
  Best epoch: 7
  Best validation F1: 0.8876
  Best validation accuracy: 0.8934
  Training time: 18 minutes 34 seconds

Model saved to: models/hate_speech_classifier/best_model/

Evaluate on test set? [y/n] (y): y

📈 Test Set Evaluation:
  F1 Score: 0.8821
  Accuracy: 0.8900
  Precision: 0.8756
  Recall: 0.8889

Confusion Matrix:
                Predicted
              hate  not_hate
Actual hate     147      18
     not_hate    22     253
```

**Key Features:**
- **70+ Models**: BERT, RoBERTa, DeBERTa, ELECTRA, CamemBERT, AraBERT, XLM-RoBERTa, etc.
- **Automatic Best Checkpoint**: Saves only the best-performing epoch
- **Resume Capability**: All sessions are saved and can be restarted
- **Multilingual**: Automatic model recommendation based on detected language
- **Benchmarking**: Compare multiple models in one run

---

### Mode 4: BERT Annotation Studio

**🤖 High-Throughput Inference with Your Trained Models**

**What it does**: Uses your trained BERT models to annotate large datasets at scale (thousands per hour).

**Perfect for**:
- You trained a model and now have new data to classify
- You need to process tens of thousands of documents
- You want fast, parallelized inference (GPU/CPU)

**Real-World Example**:
> You trained a hate speech detector (Mode 3). Now you have 50,000 new tweets to classify. Instead of manually reviewing them, you use BERT Annotation Studio with your trained model to classify all 50,000 in ~15 minutes.

#### What You'll See

```
🤖 BERT ANNOTATION STUDIO

Step 1: Load Trained Model

Available models:
  1. models/hate_speech_classifier/
  2. models/news_classifier_deberta/
  3. models/sentiment_analyzer/

Select model [1/2/3] (1): 1

✓ Loaded: hate_speech_classifier
  Architecture: microsoft/deberta-v3-base
  Classes: ['hate', 'not_hate']
  Device: MPS (Apple Silicon GPU)

Step 2: Load Data to Annotate

Enter path: data/new_tweets.csv

✓ Loaded 50,000 rows

Text column: tweet_text

Step 3: Configure Inference

Batch size (auto-optimized for GPU): 32
Parallel workers: 4

Estimated time: ~12 minutes

Start inference? [y/n] (y): y

🚀 Running inference...

Progress: ████████████████████████ 42,341/50,000 (84.7%)
Speed: 3,821 docs/min | ETA: 2:01

✓ Complete!

Results:
  - hate: 4,523 (9.0%)
  - not_hate: 45,477 (91.0%)

Export to: data/new_tweets_classified.csv

✓ Exported with columns:
  - Original: [tweet_text, date, user_id, ...]
  - Added: [predicted_label, confidence_score]
```

**Key Features:**
- **GPU Acceleration**: Uses CUDA/MPS if available, falls back to CPU
- **Parallel Processing**: Multi-core CPU or multi-GPU
- **Confidence Scores**: Know how certain the model is
- **Batch Optimization**: Automatically finds optimal batch size

---

### Mode 5: Validation Lab

**🔍 Quality Assurance & Inter-Annotator Agreement**

**What it does**: Analyzes annotation quality, computes inter-annotator agreement, and helps you identify problematic examples.

**Perfect for**:
- Checking LLM annotation quality before using it for training
- Comparing human annotators' agreement
- Finding edge cases or ambiguous examples
- Preparing data for publication (need reliability metrics)

**Real-World Example**:
> You used GPT-4 to annotate 1,000 documents. Before training a model on this data, you want to validate quality. You also had 2 human annotators code a random sample of 100. Validation Lab computes Cohen's Kappa, identifies low-confidence annotations, and flags disagreements.

#### What You'll See

```
🔍 VALIDATION LAB

Step 1: Load Annotations

Primary annotations: data/gpt4_annotations.jsonl
✓ Loaded 1,000 annotations

Compare with second annotator? [y/n] (y): y
Secondary annotations: data/human_annotations.jsonl
✓ Loaded 100 annotations (overlapping sample)

Step 2: Quality Metrics

📊 Primary Annotator (GPT-4) Quality:
  - Mean confidence: 0.87
  - Low confidence (<0.5): 23 (2.3%)
  - Label distribution:
      positive: 456 (45.6%)
      negative: 321 (32.1%)
      neutral: 223 (22.3%)

📊 Inter-Annotator Agreement (100 overlapping):
  - Percent agreement: 89.0%
  - Cohen's Kappa: 0.832 (substantial agreement)
  - Krippendorff's Alpha: 0.829

  Per-category agreement:
    positive: 94% (47/50)
    negative: 86% (31/36)
    neutral: 86% (12/14)

⚠️  Disagreement Analysis:
  Found 11 disagreements

  Top disagreement pattern:
    GPT-4: neutral → Human: positive (6 cases)

  Export disagreements for review? [y/n] (y): y

  ✓ Exported to: data/validation/disagreements.csv

Step 3: Quality Issues Detection

🔍 Scanning for potential issues...

Low-confidence annotations (confidence < 0.6):
  - 23 found
  - Median confidence: 0.52
  - Preview: [shows first 5]

Possible mislabels (model uncertainty):
  - 14 found (high entropy)

Class imbalance:
  ✓ No severe imbalance detected

Export quality report? [y/n] (y): y

✓ Report saved to: data/validation/quality_report.html
```

**Key Features:**
- **Inter-Annotator Metrics**: Cohen's Kappa, Krippendorff's Alpha
- **Confidence Analysis**: Identify uncertain predictions
- **Disagreement Patterns**: Find systematic differences
- **Stratified Sampling**: Generate balanced review samples
- **HTML Reports**: Beautiful visualizations for presentations

---

## 📖 Complete Example: From Raw Data to Trained Model

**Scenario**: You're a political scientist studying online political discourse. You have 10,000 tweets about a recent election and want to classify them by **sentiment** (positive/negative/neutral) and **topic** (economy/immigration/healthcare/other).

### Step 1: Use The Annotator (Mode 1)

```bash
llm-tool
```

1. Select **1 - The Annotator**
2. Load data: `data/election_tweets.csv` (10,000 tweets)
3. Use **Prompt Wizard** to define schema:
   - Field 1: `sentiment` → Categories: positive, negative, neutral
   - Field 2: `topic` → Categories: economy, immigration, healthcare, other
4. Choose **Ollama (local)** to keep it free and private
5. Annotate statistical sample (95% CI): **370 tweets** in ~5 minutes
6. Review results, export to `data/election_tweets_annotated.jsonl`

**Result**: 370 annotated tweets, ready for human review or training.

### Step 2: Validate Quality (Mode 5)

Since you used AI annotation, check quality first:

1. Select **5 - Validation Lab**
2. Load your annotations: `data/election_tweets_annotated.jsonl`
3. Optionally compare with 50 human-coded tweets (if you have them)
4. Review metrics:
   - Mean confidence: 0.84 (good)
   - Cohen's Kappa (vs human): 0.78 (substantial agreement)
   - Low confidence: 23 tweets (6.2%) → Flag for manual review
5. Export disagreements and low-confidence cases to CSV

**Decision**: Quality is good enough to proceed with training.

### Step 3: Train a Model (Mode 3)

1. Select **3 - Training Arena**
2. Load annotated data: `data/election_tweets_annotated.jsonl`
3. System detects English, recommends models
4. Select **benchmark mode** to compare 3 models:
   - bert-base-uncased
   - roberta-base
   - microsoft/deberta-v3-base
5. Enable **reinforcement learning** (for any class imbalance)
6. Train for 10 epochs (~20 minutes total)
7. Review benchmark results:
   ```
   Best model: microsoft/deberta-v3-base
   Test F1: 0.86 | Accuracy: 0.87
   ```
8. Model saved to `models/election_sentiment_classifier/`

**Result**: Production-ready model, trained and validated.

### Step 4: Classify Remaining Data (Mode 4)

Now apply your model to the full 10,000 tweets:

1. Select **4 - BERT Annotation Studio**
2. Load model: `models/election_sentiment_classifier/`
3. Load remaining data: `data/election_tweets_full.csv` (9,630 unannotated)
4. Run inference: ~6 minutes on GPU
5. Export results: `data/election_tweets_classified.csv`

**Result**: All 10,000 tweets classified with sentiment and topic.

### Step 5: Analyze Results

Open `election_tweets_classified.csv` in Excel/R/Python:
- Each tweet has `predicted_sentiment`, `predicted_topic`, `confidence_score`
- Filter by confidence > 0.8 for high-quality subset
- Analyze distributions, trends over time, correlations
- Publish findings with methodology: "LLM annotation (Llama 3.2) + BERT classification (DeBERTa-v3-base, F1=0.86)"

**Total Time**: ~45 minutes (from raw data to 10,000 classified tweets)

**Total Cost**: $0 (using Ollama locally)

---

## ❓ FAQ

### General Questions

**Q: Do I need to know how to code?**
A: No. The interactive CLI guides you through every step with menus and prompts. However, if you want to automate workflows, you can use LLM Tool programmatically (see `examples/`).

**Q: Is this free to use?**
A: The software is free (MIT license). Using Ollama (local LLMs) is also free. Cloud APIs (OpenAI, Claude) have costs (~$0.001-$0.01 per document).

**Q: Can I use this offline?**
A: Yes, with Ollama. Everything runs on your computer: no internet, no data sharing.

**Q: What languages are supported?**
A: 75+ languages with automatic detection. Specialized models for: English, French, Spanish, German, Chinese, Arabic, Russian, Japanese, Hindi, Portuguese, Italian, Polish, Dutch, Swedish, and more.

### Technical Questions

**Q: What hardware do I need?**
A: Minimum: 8 GB RAM, any CPU. Recommended: 16 GB RAM, GPU (NVIDIA/Apple Silicon). Works on macOS, Linux, Windows.

**Q: How long does training take?**
A: Depends on dataset size and hardware:
- 1,000 documents on M2 Max (MPS): ~8 minutes
- 1,000 documents on CPU (16 cores): ~25 minutes
- 10,000 documents on RTX 3090: ~15 minutes

**Q: Can I use my own BERT model from HuggingFace?**
A: Yes! Training Arena accepts any HuggingFace model ID. Just type the model name when prompted.

**Q: How accurate is LLM annotation?**
A: Depends on the task and model:
- Simple tasks (sentiment): 80-90% accuracy (comparable to humans)
- Complex tasks (nuanced framing): 65-80% accuracy
- Always validate with Mode 5 (Validation Lab) before trusting fully

**Q: Can I export for human review?**
A: Yes. Mode 1 exports to **Doccano** and **Label Studio** formats, which are popular open-source annotation platforms.

### Data & Privacy Questions

**Q: Where is my data stored?**
A: Locally on your computer in the `data/` directory. If you use Ollama, everything stays on your machine.

**Q: If I use OpenAI/Claude, where does my data go?**
A: Data is sent to their APIs for processing. Check their privacy policies. For sensitive data, use Ollama (100% local).

**Q: Can I delete my data?**
A: Yes, it's all in your `LLM_Tool/data/` folder. Delete it anytime.

**Q: What about GDPR/IRB compliance?**
A: Using Ollama (local) = no data sharing = easier compliance. For cloud APIs, check your institution's policies.

### Workflow Questions

**Q: Should I annotate my full dataset or a sample?**
A: For pilot studies: annotate a sample (Mode 1 calculates sample size). For final analysis: either annotate full dataset with LLM, or use LLM to annotate a sample, train a model (Mode 3), then classify the full dataset (Mode 4).

**Q: Can I combine human and AI annotations?**
A: Yes! Common workflow:
1. LLM annotates full dataset (Mode 1)
2. Humans review a sample (export to Label Studio)
3. Train on combined data (Mode 3)
4. Validate quality (Mode 5)

**Q: What if my categories change mid-project?**
A: You'll need to re-annotate. Save your prompts/schemas (Mode 6 - Profile Manager) for reproducibility.

**Q: Can I fine-tune GPT or Claude?**
A: No, but you can fine-tune BERT/RoBERTa/DeBERTa models (Mode 3), which you fully own and can deploy anywhere.

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

## 📚 Citation

If you use LLM Tool in your research, please cite:

```bibtex
@software{lemor2025llmtool,
  author = {Lemor, Antoine},
  title = {LLM Tool: State-of-the-Art LLM-Powered Annotation and BERT Training Pipeline},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/YOUR-USERNAME/LLM_Tool}
}
```

**In-text citation (APA):**
> Lemor, A. (2025). LLM Tool: State-of-the-Art LLM-Powered Annotation and BERT Training Pipeline (Version 1.0.0) [Computer software]. https://github.com/YOUR-USERNAME/LLM_Tool

**Methodology description for papers:**
> "Text classification was performed using LLM Tool (Lemor, 2025), an open-source pipeline for LLM-assisted annotation and BERT model training. Documents were initially annotated using [GPT-4/Claude/Ollama Llama 3.2] with a custom annotation schema. A subset of [N] documents was manually validated, achieving Cohen's Kappa of [X.XX]. The final classifier was trained using [model name] transformer architecture, achieving F1-score of [X.XX] on held-out test data."

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this software for any purpose, including commercial applications.

See [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

LLM Tool builds on the work of many amazing open-source projects:

- **HuggingFace Transformers** — The foundation for all BERT/RoBERTa/DeBERTa models
- **Ollama** — Making local LLM inference accessible and easy
- **PyTorch** — Deep learning framework powering model training
- **Rich** — Beautiful terminal UI rendering
- **OpenAI, Anthropic, Google** — LLM APIs for zero-shot annotation
- **The open-source ML community** — For pre-trained models and research

Special thanks to all contributors and early adopters who provided feedback.

---

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug reports
- ✨ Feature requests
- 📖 Documentation improvements
- 🧪 New model integrations
- 🌍 Translations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📬 Support & Community

**Need help?**
- 📖 **Read the docs**: Check `docs/` folder for detailed guides
- 💬 **Ask questions**: Open a GitHub Discussion
- 🐛 **Report bugs**: Open a GitHub Issue
- 💡 **Request features**: Open a GitHub Issue with the "enhancement" label

**Stay updated:**
- ⭐ Star this repository to get notifications
- 👀 Watch for new releases and features
- 🔔 Check [CHANGELOG.md](CHANGELOG.md) for version history

---

## 🌟 Why LLM Tool Matters

**Traditional annotation is broken.** Manually labeling thousands of documents takes months and costs thousands of dollars. Hiring annotators requires extensive training and quality control. Off-the-shelf solutions lack flexibility for custom research questions.

**LLM Tool changes this.** It combines the power of large language models (GPT, Claude, local Ollama) with the precision of fine-tuned BERT models, giving researchers:

✅ **Speed**: Annotate 1,000 documents in minutes, not weeks
✅ **Cost**: Use free local models (Ollama) or affordable cloud APIs
✅ **Quality**: Validate with inter-annotator agreement metrics
✅ **Ownership**: Train models you control and can deploy anywhere
✅ **Flexibility**: Support for any language, any classification scheme
✅ **Transparency**: Full control over methodology, reproducible workflows

**Built for social scientists, by social scientists.** LLM Tool was designed from the ground up for researchers who need rigorous, reproducible, and publication-ready text classification — without requiring a PhD in computer science.

---

**Made with ❤️ for researchers by [Antoine Lemor](https://github.com/YOUR-USERNAME)**

*Transform your text data into production-ready ML models in 45 minutes, not 45 days.*
