#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
setup.py

MAIN OBJECTIVE:
---------------
This script provides the complete setup configuration for the LLMTool package,
including all dependencies, entry points, and metadata.

Dependencies:
-------------
- setuptools
- sys

MAIN FEATURES:
--------------
1) Complete package metadata
2) All dependencies from both annotation and training modules
3) Entry points for CLI execution
4) Optional dependencies for advanced features
5) Development dependencies

Author:
-------
Antoine Lemor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

# Version
__version__ = "1.0.0"

# Core dependencies that are always needed
CORE_DEPENDENCIES = [
    # Data manipulation
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "openpyxl>=3.0.0",  # Excel support
    "pyarrow>=10.0.0",  # Parquet support
    "pyreadr>=0.4.0",   # RData/RDS support

    # Database
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",

    # JSON and data validation
    "pydantic>=2.0.0",
    "jsonschema>=4.0.0",

    # CLI and interface
    "rich>=13.0.0",
    "tqdm>=4.65.0",
    "inquirer>=3.0.0",

    # Language detection
    "langdetect>=1.0.9",
    "langid>=1.1.6",

    # Logging
    "loguru>=0.7.0",

    # Utilities
    "python-dotenv>=1.0.0",
    "click>=8.0.0",
    "requests>=2.28.0",
]

# LLM and AI dependencies
LLM_DEPENDENCIES = [
    # OpenAI API
    "openai>=1.0.0",

    # Anthropic API
    "anthropic>=0.18.0",

    # Google API
    "google-generativeai>=0.3.0",

    # Local LLM
    "ollama>=0.1.7",
    "llama-cpp-python>=0.2.0",

    # Transformers for local models
    "transformers>=4.35.0",
    "torch>=2.0.0",
    "accelerate>=0.24.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "safetensors>=0.4.0",
]

# Training dependencies
TRAINING_DEPENDENCIES = [
    # Deep learning frameworks
    "torch>=2.0.0",
    "tensorflow>=2.13.0",

    # Transformers and models
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "tokenizers>=0.14.0",
    "sentence-transformers>=2.2.0",

    # Training utilities
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "seaborn>=0.12.0",
    "matplotlib>=3.6.0",

    # Model evaluation
    "evaluate>=0.4.0",
    "rouge-score>=0.1.2",
    "bert-score>=0.3.13",

    # Optimization
    "optuna>=3.3.0",
    "ray[tune]>=2.6.0",

    # MLOps
    "mlflow>=2.8.0",
    "wandb>=0.15.0",
    "tensorboard>=2.13.0",

    # Additional utilities
    "imbalanced-learn>=0.11.0",
    "category_encoders>=2.6.0",
]

# Optional advanced dependencies
ADVANCED_DEPENDENCIES = [
    # FastText for language detection
    "fasttext>=0.9.2",
    "fasttext-wheel>=0.9.2",

    # Distributed computing
    "dask[complete]>=2023.8.0",
    "ray>=2.6.0",

    # Advanced NLP
    "spacy>=3.6.0",
    "nltk>=3.8.0",

    # Web serving
    "fastapi>=0.103.0",
    "uvicorn>=0.23.0",
    "gradio>=3.45.0",

    # Database optimization
    "redis>=5.0.0",
    "pymongo>=4.5.0",
]

# Development dependencies
DEV_DEPENDENCIES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.7.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "isort>=5.12.0",
    "pre-commit>=3.3.0",
    "ipykernel>=6.25.0",
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
]

# Combine all required dependencies
ALL_DEPENDENCIES = (
    CORE_DEPENDENCIES +
    LLM_DEPENDENCIES +
    TRAINING_DEPENDENCIES
)

setup(
    name="llm-tool",
    version=__version__,
    author="Antoine Lemor",
    author_email="antoine.lemor@example.com",
    description="State-of-the-Art Python Package for LLM Annotation, Training, and Large-Scale Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antoine-lemor/LLMTool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=ALL_DEPENDENCIES,
    extras_require={
        "advanced": ADVANCED_DEPENDENCIES,
        "dev": DEV_DEPENDENCIES,
        "all": ADVANCED_DEPENDENCIES + DEV_DEPENDENCIES,
    },
    entry_points={
        "console_scripts": [
            "llm-tool=llm_tool.__main__:main",
            "llmtool=llm_tool.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_tool": [
            "data/*.json",
            "prompts/*.txt",
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "annotation",
        "machine-learning",
        "natural-language-processing",
        "transformers",
        "bert",
        "gpt",
        "training",
        "benchmarking",
        "multilingual",
        "ai",
        "deep-learning",
        "data-annotation",
        "model-training",
    ],
    project_urls={
        "Bug Reports": "https://github.com/antoine-lemor/LLMTool/issues",
        "Source": "https://github.com/antoine-lemor/LLMTool",
        "Documentation": "https://github.com/antoine-lemor/LLMTool/wiki",
    },
)