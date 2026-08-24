"""
Tool definitions for the LLM agent.

Each tool maps to an existing headless LLMTool operation.
Schemas follow the JSON Schema format used by both OpenAI and Anthropic.
"""

from typing import Any, Dict, List


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return all tool definitions for the agent."""
    return [
        {
            "name": "list_available_resources",
            "description": (
                "List available resources in the project: data files, prompt files, "
                "trained models, Ollama models, or resumable sessions. "
                "Call this first to discover what is available before starting any operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": [
                            "data_files",
                            "prompts",
                            "trained_models",
                            "ollama_models",
                            "sessions",
                            "all",
                        ],
                        "description": "Type of resources to list",
                    }
                },
                "required": ["resource_type"],
            },
        },
        {
            "name": "preview_data",
            "description": (
                "Preview the first rows and column names of a data file "
                "(CSV, Excel, JSON, Parquet). Use this before annotation or training "
                "to identify the correct text column, label column, and data structure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file",
                    },
                    "num_rows": {
                        "type": "integer",
                        "description": "Number of rows to preview (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "run_annotation",
            "description": (
                "Run LLM annotation on a dataset. Annotates text using an LLM "
                "(OpenAI, Anthropic, Google, or Ollama) according to a prompt. "
                "This is a long-running operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to input data file",
                    },
                    "data_format": {
                        "type": "string",
                        "enum": ["csv", "excel", "json", "jsonl", "parquet"],
                        "description": "Format of the input data",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Name of the column containing text to annotate",
                    },
                    "annotation_provider": {
                        "type": "string",
                        "enum": ["ollama", "openai", "anthropic", "google"],
                        "description": "LLM provider for annotation",
                    },
                    "annotation_model": {
                        "type": "string",
                        "description": (
                            "Model name (e.g., 'llama3.2' local or 'gemma4:31b' cloud for "
                            "ollama, 'gpt-5-2025-08-07', 'claude-sonnet-4-20250514')"
                        ),
                    },
                    "prompt_path": {
                        "type": "string",
                        "description": "Path to the prompt file (.txt or directory)",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key for cloud providers (optional if already configured)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (auto-generated if omitted)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for processing (default: 100)",
                        "default": 100,
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Number of parallel workers (default: 4)",
                        "default": 4,
                    },
                    "identifier_column": {
                        "type": "string",
                        "description": "Column to use as row identifier (optional)",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume from previous incomplete annotation",
                        "default": False,
                    },
                },
                "required": [
                    "file_path",
                    "data_format",
                    "text_column",
                    "annotation_provider",
                    "annotation_model",
                    "prompt_path",
                ],
            },
        },
        {
            "name": "run_training",
            "description": (
                "Train a BERT/transformer model on annotated data for text classification. "
                "Supports single-label and multi-label training, benchmarking multiple models, "
                "and automatic best model selection. This is a long-running operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to training data (CSV with text and label columns)",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text (default: 'text')",
                        "default": "text",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Column containing labels (default: 'label')",
                        "default": "label",
                    },
                    "training_strategy": {
                        "type": "string",
                        "enum": ["single-label", "multi-label"],
                        "description": "Classification strategy",
                        "default": "single-label",
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Base model (default: 'bert-base-multilingual-cased')",
                        "default": "bert-base-multilingual-cased",
                    },
                    "benchmark_mode": {
                        "type": "boolean",
                        "description": "Test multiple models and pick best",
                        "default": False,
                    },
                    "models_to_test": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model names to benchmark (if benchmark_mode)",
                    },
                    "max_epochs": {
                        "type": "integer",
                        "description": "Max training epochs (default: 10)",
                        "default": 10,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Training batch size (default: 16)",
                        "default": 16,
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate (default: 2e-5)",
                        "default": 2e-5,
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save trained model (optional)",
                    },
                },
                "required": ["input_file"],
            },
        },
        {
            "name": "run_bert_inference",
            "description": (
                "Run inference on a dataset using a previously trained BERT/transformer model. "
                "Applies the classifier to new text data with confidence scoring."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the trained model directory",
                    },
                    "input_file": {
                        "type": "string",
                        "description": "Path to input data file",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text to classify",
                    },
                    "data_format": {
                        "type": "string",
                        "enum": ["csv", "excel", "json", "jsonl", "parquet"],
                        "description": "Format of the input data",
                        "default": "csv",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for output file (optional)",
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Inference batch size (default: 32)",
                        "default": 32,
                    },
                },
                "required": ["model_path", "input_file", "text_column"],
            },
        },
        {
            "name": "run_validation",
            "description": (
                "Run validation and quality analysis on annotated data. "
                "Computes quality metrics and can export for human review."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to annotated data file",
                    },
                    "validation_sample_size": {
                        "type": "integer",
                        "description": "Number of samples for validation (default: 100)",
                        "default": 100,
                    },
                    "export_to_doccano": {
                        "type": "boolean",
                        "description": "Export to Doccano format",
                        "default": False,
                    },
                },
                "required": ["input_file"],
            },
        },
        {
            "name": "run_full_pipeline",
            "description": (
                "Run the complete end-to-end pipeline: LLM annotation followed by "
                "BERT model training. Equivalent to the Annotator Factory workflow."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to input data file",
                    },
                    "data_format": {
                        "type": "string",
                        "enum": ["csv", "excel", "json", "jsonl", "parquet"],
                        "description": "Format of the input data",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text",
                    },
                    "annotation_provider": {
                        "type": "string",
                        "enum": ["ollama", "openai", "anthropic", "google"],
                        "description": "LLM provider for annotation",
                    },
                    "annotation_model": {
                        "type": "string",
                        "description": "Model name for annotation",
                    },
                    "prompt_path": {
                        "type": "string",
                        "description": "Path to the prompt file",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key for cloud providers (optional)",
                    },
                    "training_model_type": {
                        "type": "string",
                        "description": "BERT model for training (default: bert-base-multilingual-cased)",
                        "default": "bert-base-multilingual-cased",
                    },
                    "training_strategy": {
                        "type": "string",
                        "enum": ["single-label", "multi-label"],
                        "default": "single-label",
                    },
                    "max_epochs": {
                        "type": "integer",
                        "description": "Max training epochs (default: 10)",
                        "default": 10,
                    },
                },
                "required": [
                    "file_path",
                    "data_format",
                    "text_column",
                    "annotation_provider",
                    "annotation_model",
                    "prompt_path",
                ],
            },
        },
        {
            "name": "transcribe_audio",
            "description": (
                "Transcribe an audio file using OpenAI Whisper. Supports all Whisper models "
                "(tiny to large-v3), automatic language detection for 75+ languages, "
                "optional speaker diarization, and multiple output formats "
                "(TXT, CSV, JSON, SRT subtitles, WebVTT subtitles). "
                "This is a long-running operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "audio_file": {
                        "type": "string",
                        "description": "Path to the audio file to transcribe (MP3, WAV, M4A, FLAC, etc.)",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        "description": "Whisper model size (default: large-v3). Larger = more accurate but slower.",
                        "default": "large-v3",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'fr'). Auto-detected if omitted.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["txt", "csv", "json", "srt", "vtt"],
                        "description": "Output format (default: txt)",
                        "default": "txt",
                    },
                    "diarize": {
                        "type": "boolean",
                        "description": "Enable speaker diarization (requires pyannote + HF_TOKEN)",
                        "default": False,
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for transcription results",
                    },
                },
                "required": ["audio_file"],
            },
        },
        {
            "name": "download_youtube_audio",
            "description": (
                "Download audio from a YouTube video, channel, or playlist URL. "
                "Extracts audio in WAV format for subsequent transcription. "
                "Supports browser cookies for authenticated access."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube URL (video, channel, or playlist)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for downloaded audio (default: data/audio)",
                    },
                    "cookies_browser": {
                        "type": "string",
                        "enum": ["chrome", "firefox", "safari", "edge"],
                        "description": "Browser to use for cookies (for age-restricted or private videos)",
                    },
                    "max_videos": {
                        "type": "integer",
                        "description": "Maximum number of videos to download (for channels/playlists)",
                        "default": 50,
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "download_tiktok_audio",
            "description": (
                "Download audio from a TikTok video or user profile URL. "
                "Extracts audio in WAV format for subsequent transcription."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "TikTok URL (single video or user profile)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for downloaded audio (default: data/audio)",
                    },
                    "cookies_browser": {
                        "type": "string",
                        "enum": ["chrome", "firefox", "safari", "edge"],
                        "description": "Browser to use for cookies",
                    },
                    "max_videos": {
                        "type": "integer",
                        "description": "Maximum number of videos to download (for user profiles)",
                        "default": 50,
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "process_local_audio",
            "description": (
                "Process local audio or video files for transcription. "
                "Converts files to WAV format, scans directories for media files, "
                "and prepares them for Whisper transcription. Supports: "
                "MP3, WAV, M4A, AAC, FLAC, OGG, OPUS, MP4, MKV, AVI, MOV, WEBM."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a single audio/video file or a directory to scan",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Scan directories recursively (default: true)",
                        "default": True,
                    },
                    "include_video": {
                        "type": "boolean",
                        "description": "Include video files when scanning directories (default: true)",
                        "default": True,
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for processed audio (default: data/audio)",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "run_transcription_pipeline",
            "description": (
                "Run a full transcription pipeline on multiple audio files: "
                "transcribe each file with Whisper, optionally apply speaker diarization, "
                "and export results. Use this after downloading/processing audio files. "
                "This is a long-running operation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "audio_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of audio file paths to transcribe",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        "description": "Whisper model size (default: large-v3)",
                        "default": "large-v3",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'fr'). Auto-detected if omitted.",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["txt", "csv", "json", "srt", "vtt"],
                        "description": "Output format for transcriptions (default: csv)",
                        "default": "csv",
                    },
                    "diarize": {
                        "type": "boolean",
                        "description": "Enable speaker diarization",
                        "default": False,
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for transcription results",
                    },
                },
                "required": ["audio_files"],
            },
        },
        {
            "name": "tokenize_text",
            "description": (
                "Tokenize text using SOTA NLP tools: sentence segmentation, "
                "lemmatization, POS tagging, and named entity recognition (NER). "
                "Supports 23+ languages. Input can be raw text or a document file "
                "(PDF, DOCX, HTML, EPUB, TXT, MD, RTF, ODT). "
                "Outputs structured CSV or JSON."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": (
                            "Path to a text/document file (TXT, PDF, DOCX, HTML, EPUB, MD, RTF, ODT) "
                            "or a directory of documents. Omit if providing raw text."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "Raw text to tokenize (alternative to input_path)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'fr'). Auto-detected if omitted.",
                    },
                    "segmentation_level": {
                        "type": "integer",
                        "description": (
                            "Number of sentences per segment: 0=paragraph, 1=sentence (default), "
                            "2=two sentences, 3=three sentences"
                        ),
                        "default": 1,
                    },
                    "lemmatization": {
                        "type": "boolean",
                        "description": "Enable lemmatization (default: false)",
                        "default": False,
                    },
                    "pos_tagging": {
                        "type": "boolean",
                        "description": "Enable part-of-speech tagging (default: false)",
                        "default": False,
                    },
                    "ner": {
                        "type": "boolean",
                        "description": "Enable named entity recognition (default: false)",
                        "default": False,
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["csv", "json"],
                        "description": "Output format (default: csv)",
                        "default": "csv",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for tokenized results",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "load_document",
            "description": (
                "Load and extract text from document files. Supports PDF, DOCX, "
                "HTML, EPUB, TXT, Markdown, RTF, and ODT formats. "
                "Returns the extracted text content with metadata (pages, word count, etc.)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a document file or a directory of documents",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Scan directories recursively (default: true)",
                        "default": True,
                    },
                },
                "required": ["file_path"],
            },
        },
        # ── Utility & Analysis Tools ─────────────────────────────────
        {
            "name": "detect_language",
            "description": (
                "Detect the language of a text or an entire dataset column. "
                "Supports 75+ languages with confidence scoring. "
                "Use this to determine the language before annotation or training."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A single text to detect language for",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a data file (CSV, Excel, JSON) to detect languages across a column",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column name containing text (required when using file_path)",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of rows to sample for detection (default: 100)",
                        "default": 100,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "estimate_annotation_cost",
            "description": (
                "Estimate the cost of an LLM annotation job before running it. "
                "Calculates token usage and cost based on provider pricing. "
                "Supports OpenAI, Anthropic, and Google models."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file to annotate",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text to annotate",
                    },
                    "prompt_path": {
                        "type": "string",
                        "description": "Path to the prompt file",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "anthropic", "google"],
                        "description": "LLM provider",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., gpt-5-2025-08-07, claude-sonnet-4-20250514)",
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of samples to annotate (default: all rows)",
                    },
                },
                "required": ["file_path", "text_column", "prompt_path", "provider", "model"],
            },
        },
        {
            "name": "analyze_dataset",
            "description": (
                "Perform deep analysis of a dataset: detect column types, "
                "identify text/label/ID columns, detect languages, analyze text lengths, "
                "and provide recommendations. Use this before annotation or training "
                "to understand the data structure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file (CSV, Excel, JSON, JSONL, Parquet)",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "convert_annotations_to_training",
            "description": (
                "Convert LLM annotation output (JSON annotations in a column) "
                "into training-ready format. Supports single-label and multi-label "
                "conversion strategies. Essential step between annotation and training."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to annotated CSV file",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for training files",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text (default: 'sentence')",
                        "default": "sentence",
                    },
                    "annotation_column": {
                        "type": "string",
                        "description": "Column containing JSON annotations (default: 'annotation')",
                        "default": "annotation",
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["single_label", "multi_label"],
                        "description": "Create per-key single-label datasets or one multi-label dataset",
                        "default": "single_label",
                    },
                    "label_strategy": {
                        "type": "string",
                        "enum": ["key_value", "value_only", "hybrid"],
                        "description": "How to format labels (default: 'key_value')",
                        "default": "key_value",
                    },
                    "annotation_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific annotation keys to extract (default: all keys)",
                    },
                    "id_column": {
                        "type": "string",
                        "description": "Optional identifier column name",
                    },
                    "lang_column": {
                        "type": "string",
                        "description": "Optional language column name",
                    },
                },
                "required": ["input_file", "output_dir"],
            },
        },
        {
            "name": "export_to_doccano",
            "description": (
                "Export annotated data to Doccano JSONL format for human review "
                "and validation. Supports stratified sampling and confidence filtering. "
                "Creates a sample suitable for quality assessment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to annotated data file (CSV)",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path for output JSONL file",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column containing text (default: 'text')",
                        "default": "text",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Column containing labels (default: 'label')",
                        "default": "label",
                    },
                    "sample_size": {
                        "type": "integer",
                        "description": "Number of samples to export (default: all)",
                    },
                    "stratified": {
                        "type": "boolean",
                        "description": "Use stratified sampling (default: true)",
                        "default": True,
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence score to include (0.0-1.0)",
                    },
                },
                "required": ["input_file", "output_file"],
            },
        },
        {
            "name": "detect_system_resources",
            "description": (
                "Detect available system resources: GPU (CUDA/MPS), CPU cores, "
                "RAM, storage. Returns recommendations for optimal batch size, "
                "number of workers, and device selection. "
                "Call this to optimize training and inference parameters."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Optional model name to get model-specific configuration recommendations",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "split_dataset",
            "description": (
                "Split a dataset into train/validation/test sets with stratified sampling. "
                "Supports stratification by label and/or language. "
                "Essential step between annotation and training."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input data file (CSV)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for split files",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Text column name (default: 'text')",
                        "default": "text",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Label column name (default: 'label')",
                        "default": "label",
                    },
                    "train_ratio": {
                        "type": "number",
                        "description": "Training set ratio (default: 0.8)",
                        "default": 0.8,
                    },
                    "val_ratio": {
                        "type": "number",
                        "description": "Validation set ratio (default: 0.1)",
                        "default": 0.1,
                    },
                    "test_ratio": {
                        "type": "number",
                        "description": "Test set ratio (default: 0.1)",
                        "default": 0.1,
                    },
                    "stratify": {
                        "type": "boolean",
                        "description": "Stratify by label distribution (default: true)",
                        "default": True,
                    },
                    "random_seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (default: 42)",
                        "default": 42,
                    },
                },
                "required": ["input_file", "output_dir"],
            },
        },
        {
            "name": "manage_prompts",
            "description": (
                "List available prompt files, read prompt content, and extract "
                "expected annotation keys from a prompt. Use this to discover "
                "and validate prompts before annotation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "read", "analyze"],
                        "description": (
                            "'list' to list all prompts in prompts directory, "
                            "'read' to read a prompt file content, "
                            "'analyze' to extract expected JSON keys from a prompt"
                        ),
                    },
                    "prompt_path": {
                        "type": "string",
                        "description": "Path to a specific prompt file (required for 'read' and 'analyze' actions)",
                    },
                },
                "required": ["action"],
            },
        },
    ]
