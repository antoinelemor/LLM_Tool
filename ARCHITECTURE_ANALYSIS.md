# LLM_Tool Package - Comprehensive Architecture Analysis

**Package Version:** 1.0.0
**Author:** Antoine Lemor
**Total Lines of Code:** ~79,268
**Total Python Files:** 72

---

## TABLE OF CONTENTS
1. [Package Overview](#package-overview)
2. [Directory Structure](#directory-structure)
3. [Core Modules](#core-modules)
4. [Model Providers](#model-providers)
5. [All Classes and Methods](#all-classes-and-methods)
6. [Validation Features](#validation-features)
7. [Text Detection Capabilities](#text-detection-capabilities)
8. [Training Arena Functionality](#training-arena-functionality)
9. [Benchmark Capabilities](#benchmark-capabilities)
10. [CLI Features](#cli-features)
11. [Data Processing & Export](#data-processing--export)
12. [Statistical Analysis](#statistical-analysis)

---

## PACKAGE OVERVIEW

LLM_Tool is a state-of-the-art LLM-powered annotation and BERT training pipeline for multilingual text classification. It supports:
- Multiple LLM providers (OpenAI, Anthropic, Google, Ollama, local models)
- 50+ transformer models for training
- Multi-label classification
- Comprehensive validation and benchmarking
- Interactive CLI with rich visualization
- Multilingual text processing
- Database support (PostgreSQL, CSV, Excel, Parquet, RData)

---

## DIRECTORY STRUCTURE

```
llm_tool/
├── annotators/          # LLM annotation components
├── cli/                 # Command-line interface modules
├── config/              # Configuration management
├── database/            # File and database handlers
├── exporters/           # Export functionality
├── pipelines/           # Pipeline orchestration
├── trainers/            # BERT model training
├── utils/               # Utility functions
└── validators/          # Data validation
```

---

## CORE MODULES

### 1. llm_tool/annotators/
**Purpose:** LLM-based text annotation with multiple provider support

#### Key Files:
- **llm_annotator.py** (3,500+ lines)
  - Main annotation engine with single/multi-prompt support
  - Parallel execution with ProcessPoolExecutor
  - JSON repair and schema validation
  - Support for PostgreSQL, CSV, Excel, Parquet, RData/RDS
  - Incremental saving and resume capability
  - OpenAI Batch API integration

- **api_clients.py** (400+ lines)
  - OpenAI API client (GPT-4, o1, o3 support)
  - Anthropic Claude API client
  - Google Gemini API client
  - Retry mechanisms and error handling

- **local_models.py** (500+ lines)
  - OllamaClient for local Ollama integration
  - LlamaCPPClient for llama.cpp models
  - Model discovery and management

- **prompt_manager.py** (400+ lines)
  - Load and manage annotation prompts
  - Combine multiple text columns
  - JSON merging for multi-prompt outputs
  - Interactive prompt loading

- **prompt_wizard.py** (3,000+ lines)
  - Interactive prompt creation wizard
  - Social science annotation guidance
  - Category and value definition
  - LLM-assisted definition generation
  - Example generation

- **json_cleaner.py** (500+ lines)
  - JSON repair with 5 retry attempts
  - Schema validation with Pydantic
  - Handle malformed JSON responses

### 2. llm_tool/cli/
**Purpose:** Interactive command-line interface with Rich visualization

#### Key Files:
- **advanced_cli.py** (10,000+ lines)
  - Professional-grade CLI with auto-detection
  - Model discovery (Ollama, OpenAI, Anthropic, Google)
  - Intelligent suggestions
  - Configuration profiles
  - Execution history
  - System resource recommendations
  - Dataset auto-detection

- **annotation_workflow.py** (5,000+ lines)
  - Complete annotation workflow orchestration
  - Multi-prompt support
  - Cost estimation
  - Progress tracking

- **training_arena_integrated.py** (6,000+ lines)
  - Training studio for 50+ models
  - Multi-label classification training
  - Model benchmarking
  - Interactive training configuration
  - Training session management

- **validation_lab.py** (3,000+ lines)
  - Quality control and validation
  - Inter-annotator agreement calculation
  - Annotation consensus building
  - Export to Doccano format

- **bert_annotation_studio.py** (4,000+ lines)
  - BERT training interface
  - Model selection and configuration

- **banners.py** (500+ lines)
  - ASCII art banners and visualization

### 3. llm_tool/config/
**Purpose:** Configuration and credential management

#### Key Classes:
- **Settings** (400 lines)
  - Global configuration manager
  - API settings (APIConfig)
  - Local model configuration (LocalModelConfig)
  - Training configuration (TrainingConfig)
  - Language settings (LanguageConfig)
  - Path management (PathConfig)
  - Logging configuration (LoggingConfig)

- **APIKeyManager** (300 lines)
  - Secure API key storage
  - Multi-provider support
  - Environment variable fallback

### 4. llm_tool/database/
**Purpose:** Data I/O for multiple formats

#### Key Classes:
- **FileHandler** (300 lines)
  - CSV, Excel, Parquet, RData/RDS support
  - Incremental file writing
  - Unicode sanitization

- **IncrementalFileWriter** (200 lines)
  - Batch processing support
  - Safe file appending

- **PostgreSQLHandler** (500+ lines)
  - PostgreSQL connection management
  - SQL query execution
  - Data loading and saving

### 5. llm_tool/trainers/
**Purpose:** BERT model training and benchmarking (50+ models)

#### Base Classes:
- **BertABC** (100 lines)
  - Abstract base class for model wrappers

- **BertBase** (4,000+ lines)
  - Complete BERT training implementation
  - Training arguments and callbacks
  - Evaluation and metrics
  - Early stopping support
  - Model checkpointing

#### Language-Specific Models:
- **models.py** (200+ lines)
  - Bert (bert-base-uncased)
  - ArabicBert (bert-base-arabic)
  - Camembert (French)
  - ChineseBert (Chinese)
  - GermanBert, SpanishBert, ItalianBert
  - PortugueseBert, RussianBert, HindiBert
  - SwedishBert, XLMRoberta (multilingual)

#### SOTA Models (sota_models.py - 1,000+ lines):
**DeBERTa Family:**
- DeBERTaV3Base, DeBERTaV3Large, DeBERTaV3XSmall

**RoBERTa Family:**
- RoBERTaBase, RoBERTaLarge, DistilRoBERTa

**ELECTRA Family:**
- ELECTRABase, ELECTRALarge, ELECTRASmall

**ALBERT Family:**
- ALBERTBase, ALBERTLarge, ALBERTXLarge

**Long-Context Models:**
- BigBirdBase, BigBirdLarge
- LongformerBase, LongformerLarge

**Multilingual Models:**
- MDeBERTaV3Base, XLMRobertaBase, XLMRobertaLarge

**French-Specific Models:**
- CamembertaV2Base, CamembertLarge
- FlauBERTBase, FlauBERTLarge
- BARThez, FrALBERT, DistilCamemBERT
- FrELECTRA, CamembertBioBERT

#### Training Components:
- **model_trainer.py** (4,000+ lines)
  - ModelTrainer orchestration class
  - Training and benchmarking workflows
  - Model selection
  - Hyperparameter optimization
  - Multi-GPU support

- **benchmarking.py** (2,000+ lines)
  - BenchmarkRunner for multi-model evaluation
  - BenchmarkConfig configuration
  - TrainingRunSummary for results
  - Per-category and per-language training loops
  - Result aggregation into CSV/JSON

- **multilingual_selector.py** (500+ lines)
  - MultilingualModelSelector
  - Language distribution analysis
  - ModelSize and TaskType enums
  - Ensemble strategy recommendations

- **model_selector.py** (1,000+ lines)
  - ModelSelector with automatic selection
  - TaskComplexity and ResourceProfile enums
  - Model profiling and benchmarking

- **multi_label_trainer.py** (2,000+ lines)
  - MultiLabelTrainer for multi-label classification
  - Hamming loss and Jaccard score support
  - Multi-label specific metrics

- **data_utils.py** (600+ lines)
  - DataSample and DataLoader utilities
  - PerformanceTracker
  - Safe conversion functions

- **data_splitter.py** (400+ lines)
  - DataSplitter for train/val/test splits
  - SplitConfig configuration
  - Stratified splitting

- **training_data_builder.py** (1,000+ lines)
  - TrainingDatasetBuilder
  - TrainingDataRequest and TrainingDataBundle
  - Dataset preparation and validation

- **benchmark_dataset_builder.py** (600+ lines)
  - BenchmarkDatasetBuilder
  - BenchmarkDataset
  - Balanced dataset creation

- **parallel_inference.py** (300+ lines)
  - parallel_predict for batch inference

### 6. llm_tool/utils/
**Purpose:** Utility functions for various operations

#### Key Utilities:
- **language_detector.py** (600+ lines)
  - LanguageDetector with multiple backends
  - DetectionMethod enum (LINGUA, LANGID, FASTTEXT)
  - 75+ language support
  - Confidence scoring
  - Batch processing capability
  - 96%+ accuracy with lingua-language-detector

- **system_resources.py** (600+ lines)
  - SystemResourceDetector for hardware detection
  - GPUInfo, CPUInfo, MemoryInfo, StorageInfo
  - SystemResources and SystemInfo dataclasses
  - Optimal configuration recommendations
  - Mac and Windows support

- **data_detector.py** (1,000+ lines)
  - DataDetector for dataset discovery
  - DatasetInfo dataclass
  - Recursive directory scanning
  - Format detection (CSV, JSON, JSONL, Excel, Parquet, RData)
  - Column type inference
  - Text scoring

- **cost_estimator.py** (800+ lines)
  - ModelPricing and CostEstimate dataclasses
  - OpenAI pricing database
  - Token-based cost calculation
  - Cost formatting

- **token_analysis.py** (300+ lines)
  - TokenAnalysisResult dataclass
  - analyse_text_tokens function
  - Tokenizer management

- **metadata_manager.py** (700+ lines)
  - MetadataManager for session tracking
  - Metadata persistence

- **annotation_to_training.py** (1,000+ lines)
  - AnnotationToTrainingConverter
  - Annotation workflow to training conversion

- **training_data_utils.py** (1,000+ lines)
  - TrainingDataSessionManager
  - Training data preparation

- **annotation_session_manager.py** (500+ lines)
  - AnnotationStudioSessionManager
  - SessionStep tracking

- **session_summary.py** (500+ lines)
  - SessionSummary dataclass
  - SummaryRecord for session tracking
  - Session collection utilities

- **training_summary_generator.py** (1,000+ lines)
  - TrainingSummaryGenerator
  - Summary report generation

- **logging_utils.py** (400+ lines)
  - StructuredLogger
  - JsonFormatter
  - PerformanceLogger
  - ErrorAggregator

- **data_filter_logger.py** (400+ lines)
  - DataFilterLogger for filtering statistics

- **benchmark_utils.py** (800+ lines)
  - Class imbalance analysis
  - Benchmark category selection
  - Dataset creation utilities

- **benchmark_helpers.py** (900+ lines)
  - Numeric metric extraction
  - Label sufficiency validation
  - Benchmark splitting and filtering

- **language_filtering.py** (400+ lines)
  - Language distribution analysis
  - Sample filtering for languages

- **language_normalizer.py** (300+ lines)
  - LanguageNormalizer for language code standardization

- **training_paths.py** (200+ lines)
  - Path management for training sessions
  - Session directory resolution

- **resource_display.py** (900+ lines)
  - Resource visualization functions
  - Tables and panels for CLI display

- **model_display.py** (1,000+ lines)
  - Model recommendation display
  - Language formatting
  - Relevance scoring

### 7. llm_tool/validators/
**Purpose:** Annotation validation and quality control

#### Key Classes:
- **AnnotationValidator** (500+ lines)
  - ValidationConfig dataclass
  - DoccanoAnnotation support
  - ValidationResult dataclass
  - Label distribution analysis
  - Confidence score analysis
  - Sample selection for human review

- **DoccanoExporter** (500+ lines)
  - Export to Doccano JSONL format
  - Inter-annotator agreement calculation

### 8. llm_tool/pipelines/
**Purpose:** Workflow orchestration

#### Key Classes:
- **PipelineController** (1,000+ lines)
  - PipelinePhase enum (INITIALIZATION, ANNOTATION, VALIDATION, TRAINING, BENCHMARKING, DEPLOYMENT, INFERENCE, COMPLETED)
  - PipelineState dataclass
  - Sequential phase execution
  - Progress tracking

- **EnhancedPipelineWrapper** (200+ lines)
  - AnnotationTracker
  - Pipeline enhancement layer

---

## MODEL PROVIDERS SUPPORTED

### 1. OpenAI (openai>=1.0.0)
- **Models:** GPT-4, GPT-4-turbo, GPT-3.5-turbo, o1, o3
- **Features:** 
  - Batch API support for cost-effective large-scale annotation
  - Streaming responses
  - Function calling
  - Vision capabilities (for images)
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/api_clients.py:88-170`

### 2. Anthropic Claude (anthropic>=0.18.0)
- **Models:** Claude 3 family (Opus, Sonnet, Haiku)
- **Features:**
  - Long context windows
  - Extended thinking
  - Vision support
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/api_clients.py:172-250`

### 3. Google Gemini (google-generativeai>=0.3.0)
- **Models:** Gemini Pro, Gemini Vision
- **Features:**
  - Multi-modal input
  - Streaming support
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/api_clients.py:252-330`

### 4. Ollama (ollama>=0.6.0) - Local Models
- **Models:** Llama 2, Mistral, Neural Chat, etc.
- **Features:**
  - Local GPU inference
  - No API costs
  - Privacy-preserving
  - Warm-up calls for optimal performance
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/local_models.py:50-280`

### 5. Llama.cpp (llama-cpp-python>=0.2.0) - Local Models
- **Models:** GGUF format models
- **Features:**
  - CPU inference
  - Memory efficient
  - Quantized models
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/local_models.py:282-450`

### 6. HuggingFace (transformers>=4.35.0)
- **Models:** 50+ BERT variants for fine-tuning
- **Features:**
  - Local training
  - Custom model architectures
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/`

---

## ALL CLASSES AND THEIR FUNCTIONALITIES

### Annotation Module Classes

#### 1. LLMAnnotator (llm_annotator.py:200+)
```python
class LLMAnnotator:
    def __init__(self, settings: Optional[Settings] = None)
    def annotate(config: Dict[str, Any]) -> Dict[str, Any]
    def _validate_config(config: Dict[str, Any])
    def _setup_model_client(config: Dict[str, Any])
    def _load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]
    def _load_file(file_path: str, format: str) -> pd.DataFrame
    def _load_postgresql(db_config: Dict) -> pd.DataFrame
    def _prepare_prompts(config: Dict[str, Any]) -> List[Dict]
    def _annotate_data(...)
    def _detect_text_columns(data: pd.DataFrame) -> List[str]
    def _create_unique_id(data: pd.DataFrame) -> str
    def _resolve_identifier_column(...)
    def _warmup_model()
    def _prepare_annotation_tasks(...) -> List[Dict]
    def _execute_parallel_annotation(...) -> pd.DataFrame
    def _execute_sequential_annotation(...) -> pd.DataFrame
    def _execute_openai_batch_annotation(...) -> pd.DataFrame
    def _save_results(data: pd.DataFrame, config: Dict[str, Any])
    def _save_data(data: pd.DataFrame, path: str, format: str)
    def _save_annotated_subset(data: pd.DataFrame, config: Dict[str, Any])
    def _export_to_doccano(...)
    def _export_doccano_jsonl(...)
    def _store_annotation_payload(...)
    def _append_to_csv(...)
    def _write_log_entry(log_path: str, entry: Dict[str, Any])
    def _generate_summary(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]
    def calculate_sample_size(total: int, confidence: float = 0.95) -> int
    def analyze_text_with_model(...)
```

#### 2. API Client Classes

**BaseAPIClient** (abstract)
- api_key: str
- max_retries: int
- timeout: int
- progress_manager: Optional
- generate(prompt: str) -> Optional[str]

**OpenAIClient(BaseAPIClient)**
- init(api_key: str, **kwargs)
- generate(prompt: str, **kwargs) -> Optional[str]
- generate_batch(...) -> Tuple[List, List]
- estimate_tokens(...) -> int

**AnthropicClient(BaseAPIClient)**
- init(api_key: str, **kwargs)
- generate(prompt: str, **kwargs) -> Optional[str]

**GoogleClient(BaseAPIClient)**
- init(api_key: str, **kwargs)
- generate(prompt: str, **kwargs) -> Optional[str]

#### 3. Local Model Classes

**BaseLocalClient** (abstract)
- model_name: str
- generate(prompt: str, **kwargs) -> Optional[str]
- is_available() -> bool

**OllamaClient(BaseLocalClient)**
- init(model_name: str, **kwargs)
- _check_ollama_service()
- _pull_model()
- is_available() -> bool
- list_models() -> List[str]
- generate(...) -> Optional[str]

**LlamaCPPClient(BaseLocalClient)**
- init(model_path: str, **kwargs)
- _init_model()
- is_available() -> bool
- generate(...) -> Optional[str]
- get_model_info() -> Dict[str, Any]

#### 4. JSON Cleaner Classes

**JSONCleaner**
- repair(json_str: str) -> Dict
- validate(data: Dict, schema: BaseModel) -> Tuple[bool, Optional[str]]

#### 5. Prompt Manager Classes

**PromptManager**
- init()
- load_prompt(prompt_path: str) -> Tuple[str, List[str]]
- verify_prompt_structure(base_prompt: str, expected_keys: List[str])
- get_prompts_with_prefix() -> List[Tuple[str, List[str], str]]
- build_combined_text(row: Dict, text_columns: List[str], prefixes: List[str]) -> str
- merge_json_objects(json_list: List[Dict]) -> Dict
- apply_prefix(data: Dict, prefix: str) -> Dict

#### 6. Prompt Wizard Classes

**AnnotationCategory** (dataclass)
- name: str
- description: str
- values: List[str]

**PromptSpecification** (dataclass)
- project_description: str
- research_objectives: List[str]
- data_characteristics: str
- categories: Dict[str, AnnotationCategory]
- entities: List[str]

**SocialSciencePromptWizard**
- init(llm_client=None)
- run() -> Tuple[str, List[str]]
- _display_welcome()
- _get_project_description() -> str
- [50+ more methods for interactive prompting]

### Configuration Classes

**APIConfig**
- provider: str = "openai"
- api_key: Optional[str] = None
- model_name: str = "gpt-4"
- max_tokens: int = 4096
- temperature: float = 0.7
- timeout: int = 60

**LocalModelConfig**
- provider: str = "ollama"
- model_name: str = "llama3.2"
- device: str = "auto"  # auto, cuda, cpu, mps
- quantization: Optional[str] = None

**DataConfig**
- default_format: str = "csv"
- chunk_size: int = 1000
- max_workers: int = 4
- cache_enabled: bool = True

**TrainingConfig**
- default_model: str = "bert-base-multilingual-cased"
- batch_size: int = 16
- learning_rate: float = 2e-5
- max_epochs: int = 10
- early_stopping_patience: int = 3

**LanguageConfig**
- detection_mode: str = "auto"
- confidence_threshold: float = 0.8
- supported_languages: List[str]

**PathConfig**
- base_dir: Path
- data_dir: Path
- models_dir: Path
- prompts_dir: Path
- logs_dir: Path
- validation_dir: Path
- cache_dir: Path

**LoggingConfig**
- level: str = "INFO"
- file_logging: bool = True
- console_logging: bool = True
- log_file: str = "application/llm_tool.log"

**Settings** (Main Configuration Manager)
- api: APIConfig
- local_model: LocalModelConfig
- data: DataConfig
- training: TrainingConfig
- language: LanguageConfig
- paths: PathConfig
- logging: LoggingConfig
- key_manager: APIKeyManager
- load(config_file: str)
- save(config_file: str)
- update_api_settings(settings: Dict)
- update_language_settings(settings: Dict)
- update_training_settings(settings: Dict)
- get_api_key(provider: str) -> Optional[str]
- set_api_key(provider: str, api_key: str)
- get_or_prompt_api_key(provider: str) -> Optional[str]

### Database Classes

**FileHandler**
- init(file_path: Union[str, Path], format: Optional[str] = None)
- read_file(file_path: str) -> pd.DataFrame
- write_file(data: pd.DataFrame, path: str)
- append_to_file(data: pd.DataFrame, path: str)
- detect_format(file_path: str) -> str

**IncrementalFileWriter**
- init(file_path: Path, format: str = "csv")
- append(data: pd.DataFrame)
- finalize()

**PostgreSQLHandler**
- init(connection_string: str)
- connect()
- disconnect()
- query(sql: str) -> pd.DataFrame
- execute(sql: str) -> bool
- insert(table: str, data: pd.DataFrame) -> int
- update(table: str, data: pd.DataFrame, where: str) -> int

### Training Classes

**BertBase** (4000+ lines)
- init(model_name: str, num_labels: int, ...)
- train(train_dataset, val_dataset, ...)
- evaluate(dataset) -> Dict[str, float]
- predict(texts: List[str]) -> np.ndarray
- get_model() -> PreTrainedModel
- get_tokenizer() -> PreTrainedTokenizer
- save_model(path: str)
- load_model(path: str)
- cross_validate(dataset, n_splits: int = 5)
- get_inference_pipeline()

**ModelTrainer**
- init(settings: Optional[Settings] = None)
- train_and_benchmark(config: Dict) -> TrainingResults
- benchmark_models(models: List[str], dataset: str) -> Dict
- select_best_model(results: Dict) -> str
- save_training_logs(path: str)

**MultilingualModelSelector**
- init()
- score_models(languages: List[str], task: TaskType) -> Dict[str, float]
- recommend_model(languages: List[str]) -> str
- create_ensemble(languages: List[str]) -> List[str]

**ModelSelector**
- init()
- auto_select(task_complexity: TaskComplexity, resource_profile: ResourceProfile) -> str
- benchmark_candidates(models: List[str], dataset: str) -> Dict
- get_recommendations(task_complexity: TaskComplexity) -> List[str]

**MultiLabelTrainer**
- init(config: TrainingConfig)
- train(train_data: List[MultiLabelSample], val_data: List[MultiLabelSample])
- evaluate(val_data: List[MultiLabelSample]) -> Dict
- predict(texts: List[str]) -> List[Set[str]]

**DataSplitter**
- init(config: SplitConfig)
- split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
- stratified_split(data: pd.DataFrame, label_column: str)
- language_split(data: pd.DataFrame, language_column: str)

**BenchmarkRunner**
- init(config: BenchmarkConfig)
- run() -> Dict[str, TrainingRunSummary]
- aggregate_results() -> pd.DataFrame
- generate_report(output_path: str)

**TrainingDatasetBuilder**
- init(request: TrainingDataRequest)
- build() -> TrainingDataBundle
- validate() -> bool
- balance_dataset()
- apply_filters()

### Language Detection Classes

**LanguageDetector**
- init(method: DetectionMethod = DetectionMethod.LINGUA)
- detect(text: str) -> Tuple[str, float]
- detect_batch(texts: List[str]) -> List[Tuple[str, float]]
- detect_ensemble(text: str) -> Tuple[str, float]
- set_fallback_language(lang: str)

### System Resource Classes

**GPUInfo**
- available: bool
- device_type: str  # "cpu", "cuda", "mps"
- device_count: int
- total_memory_gb: float
- available_memory_gb: float

**CPUInfo**
- cores: int
- frequency_ghz: float
- usage_percent: float
- architecture: str

**MemoryInfo**
- total_gb: float
- available_gb: float
- used_gb: float
- percent: float

**StorageInfo**
- total_gb: float
- free_gb: float
- used_gb: float
- percent: float

**SystemResources**
- gpu: GPUInfo
- cpu: CPUInfo
- memory: MemoryInfo
- storage: StorageInfo
- os_name: str
- python_version: str
- get_recommendation() -> Dict[str, Any]
- to_dict() -> Dict[str, Any]

**SystemResourceDetector**
- detect() -> SystemResources
- detect_gpu() -> GPUInfo
- detect_cpu() -> CPUInfo
- detect_memory() -> MemoryInfo
- detect_storage() -> StorageInfo

### Data Detection Classes

**DatasetInfo**
- path: Path
- format: str
- rows: Optional[int]
- columns: List[str]
- size_mb: Optional[float]
- detected_language: Optional[str]
- has_labels: bool
- label_column: Optional[str]
- column_types: Dict[str, str]
- text_scores: Dict[str, float]

**DataDetector**
- scan_directory(directory: Path) -> List[DatasetInfo]
- inspect_file(path: Path) -> DatasetInfo
- infer_schema(data: pd.DataFrame) -> Dict[str, str]
- score_text_columns(columns: List[str]) -> Dict[str, float]

### Validation Classes

**ValidationConfig**
- sample_size: int = 100
- stratified_sampling: bool = True
- confidence_threshold: float = 0.8
- export_format: str = "jsonl"
- export_to_doccano: bool = True
- check_label_consistency: bool = True

**DoccanoAnnotation**
- text: str
- label: str
- meta: Dict[str, Any]
- id: Optional[int]

**ValidationResult**
- total_samples: int
- valid_samples: int
- invalid_samples: int
- confidence_scores: Dict[str, float]
- label_distribution: Dict[str, int]
- agreement_scores: Dict[str, float]

**AnnotationValidator**
- init(config: ValidationConfig)
- validate(data: pd.DataFrame) -> ValidationResult
- check_schema_compliance(data: pd.DataFrame) -> bool
- check_label_consistency(data: pd.DataFrame) -> Dict[str, Any]
- calculate_inter_annotator_agreement(annotators: List[str]) -> float
- sample_for_human_review(data: pd.DataFrame) -> pd.DataFrame
- export_to_doccano(data: pd.DataFrame, output_path: str)

### Pipeline Classes

**PipelinePhase** (Enum)
- INITIALIZATION, ANNOTATION, VALIDATION, TRAINING, BENCHMARKING, DEPLOYMENT, INFERENCE, COMPLETED

**PipelineState** (Dataclass)
- current_phase: PipelinePhase
- phases_completed: List[PipelinePhase]
- start_time: float
- end_time: Optional[float]
- annotation_results: Optional[Dict]
- validation_results: Optional[Dict]
- training_results: Optional[Dict]
- deployment_results: Optional[Dict]
- errors: List[Dict]
- warnings: List[str]

**PipelineController**
- init(settings: Settings, progress_callback, session_id: str)
- execute() -> PipelineState
- run_phase(phase: PipelinePhase) -> Dict[str, Any]
- resume_from_phase(phase: PipelinePhase)
- get_status() -> Dict[str, Any]
- log_phase_completion(phase: PipelinePhase)

---

## VALIDATION FEATURES

### 1. Schema Validation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/llm_annotator.py:1500-1700`
- Pydantic-based dynamic schema creation
- Field type validation
- Required/optional field checking
- JSON schema compliance

### 2. Label Validation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py:200-400`
- Label distribution analysis
- Confidence score analysis
- Label consistency checking
- Imbalanced class detection

### 3. Data Quality Checks
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/data_filter_logger.py`
- Missing value detection
- Duplicate detection
- Data type validation
- Format validation

### 4. Inter-Annotator Agreement
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py:350-500`
- Cohen's Kappa score
- Accuracy calculation
- Confusion matrix analysis
- Agreement summary statistics

### 5. Annotation Sampling
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py:450-600`
- Stratified sampling (95% CI)
- Sample size calculation
- Per-label sample management
- Human review sample selection

### 6. Doccano Export for Human Validation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/doccano_exporter.py`
- JSONL format export
- Metadata preservation
- Label suggestion
- Interactive review-friendly format

### 7. Cost-Based Validation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/cost_estimator.py:100-300`
- Token counting validation
- Cost estimation
- Budget tracking

---

## TEXT DETECTION CAPABILITIES

### 1. Language Detection
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/language_detector.py`
- **Primary Method:** lingua-language-detector (96%+ accuracy)
- **Fallback Methods:** langid, fasttext
- **Supported Languages:** 75+ languages
- **Features:**
  - Batch processing
  - Confidence scoring
  - Ensemble detection
  - Deterministic results (no randomness)
  - ISO 639-1 standardization

### 2. Text Column Detection
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/llm_annotator.py:1000-1100`
- Automatic text column identification
- Heuristic-based scoring
- Multiple column handling

### 3. Text Length Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:2000-2200`
- Character count statistics
- Word count distribution
- Sentence count analysis
- Percentile analysis

### 4. Token Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/token_analysis.py`
- Tokenizer-based counting
- Prompt token estimation
- Completion token estimation
- Header token tracking

### 5. Label Column Detection
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/data_detector.py:200-400`
- Categorical column identification
- Unique value analysis
- Distribution analysis

### 6. Identifier Column Detection
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:1800-2000`
- ID column recognition
- Uniqueness checking
- Type inference

### 7. Text Classification Scoring
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/data_detector.py:300-500`
- Cardinality analysis
- Entropy calculation
- JSON field detection
- Numeric vs text determination

---

## TRAINING ARENA FUNCTIONALITY

### 1. Training Studio Interface
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/training_arena_integrated.py:1-2000`
- Interactive dataset preparation
- Model configuration wizard
- Training parameter tuning
- Progress visualization

### 2. Multi-Model Training
- **Supported Models:** 50+ transformer models
  - BERT variants (13 language-specific)
  - SOTA models (DeBERTa, RoBERTa, ELECTRA, ALBERT, etc.)
  - Long-context models (BigBird, Longformer)
  - Multilingual models (mDeBERTa, XLM-RoBERTa)
  - French-specific models (CamemBERT, FlauBERT, etc.)

### 3. Dataset Builder
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/training_data_builder.py`
- TrainingDatasetBuilder class
- Automatic dataset validation
- Data balancing for imbalanced classes
- Train/val/test splitting
- Stratification support

### 4. Training Features
- **Batch Size Optimization:** Automatic recommendation based on system resources
- **Early Stopping:** Patience-based model checkpoint management
- **Gradient Accumulation:** For memory-constrained training
- **Mixed Precision:** FP16 support for faster training
- **Multi-GPU Support:** DataParallel and DistributedDataParallel
- **Cross-Validation:** k-fold validation support
- **Learning Rate Scheduling:** Warmup steps and polynomial decay

### 5. Multi-Label Classification
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/multi_label_trainer.py`
- MultiLabelTrainer class
- Hamming loss metric
- Jaccard similarity metric
- Per-label precision/recall/F1
- Label distribution analysis

### 6. Training Monitoring
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/bert_base.py:2000-3000`
- Real-time training metrics
- Validation curve tracking
- Loss visualization
- Metric logging

### 7. Model Selection
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/model_selector.py`
- Automatic model recommendation
- Task complexity assessment (SIMPLE, MODERATE, COMPLEX, EXPERT)
- Resource profile matching (MINIMAL, STANDARD, OPTIMAL, UNRESTRICTED)
- Language coverage analysis

### 8. Multilingual Strategy
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/multilingual_selector.py`
- Language distribution analysis
- Model size selection
- Ensemble recommendations
- Deployment guidance

### 9. Session Management
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/training_data_utils.py`
- TrainingDataSessionManager
- Session persistence
- Resumable training
- Training metadata tracking

### 10. Training Artifacts
- **Checkpoints:** Best model saving
- **Logs:** Training curves and metrics
- **Metadata:** Configuration and data info
- **Reports:** Summary statistics

---

## BENCHMARK CAPABILITIES

### 1. Benchmark Runner
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py:1-2000`
- BenchmarkRunner orchestration
- BenchmarkConfig configuration
- TrainingRunSummary results
- Multi-model evaluation

### 2. Per-Category Training
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py:2000-2500`
- Category-specific model training
- Per-category metrics
- Category performance ranking

### 3. Per-Language Training
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py:2500-3000`
- Language-specific model selection
- Language performance analysis
- Language-specific metrics

### 4. Dataset Balancing
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmark_dataset_builder.py`
- BenchmarkDatasetBuilder
- Balanced sample creation
- Class distribution management
- Reinforcement rescue strategies

### 5. Results Aggregation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py:3000-3500`
- CSV metric consolidation
- JSON results aggregation
- Cross-model comparison
- Ranking and sorting

### 6. Metrics Calculation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/benchmark_helpers.py`
- Accuracy, Precision, Recall, F1
- Per-class metrics
- Macro/micro averages
- Weighted averages

### 7. Benchmark Reporting
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/benchmark_utils.py`
- Result consolidation
- CSV report generation
- JSON summary
- Performance ranking

### 8. Cost-Benefit Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/cost_estimator.py`
- Model size assessment
- Training time estimation
- Resource requirement analysis
- Cost-performance trade-offs

---

## CLI FEATURES

### 1. Main Menu System
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:500-1500`
- **Features:**
  - Smart suggestions based on context
  - Resume center for interrupted workflows
  - Quick-start wizard for new users
  - Mode selection (Annotator Factory, Training Arena, Validation Lab)

### 2. Annotator Factory Mode
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/annotation_workflow.py`
- Complete annotation pipeline:
  - Step 1: Dataset selection
  - Step 2: Model/API selection
  - Step 3: Prompt configuration
  - Step 4: Output format selection
  - Step 5: Execute annotation
  - Step 6: Post-annotation training

### 3. Training Arena Mode
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/training_arena_integrated.py`
- Complete training pipeline:
  - Step 1: Data loading and exploration
  - Step 2: Column selection (text, label, identifier)
  - Step 3: Text length analysis
  - Step 4: Language detection
  - Step 5-15: Training configuration
  - Step 16: Model selection
  - Step 17: Reinforced learning parameters
  - Step 18: Training execution

### 4. Validation Lab Mode
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/validation_lab.py`
- Quality control workflow:
  - Annotation export review
  - Consensus building
  - Inter-annotator agreement analysis
  - Human review sample selection
  - Doccano export

### 5. Auto-Detection Features
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:1500-2500`
- Automatic detection:
  - Available models (Ollama, OpenAI, Anthropic, Google)
  - Local datasets
  - Text columns
  - Language
  - Label columns
  - Identifier columns

### 6. Cost Estimation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/annotation_workflow.py:1000-1500`
- Pre-annotation cost preview
- Per-request cost estimation
- Total cost calculation
- Token analysis

### 7. System Resource Display
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/resource_display.py`
- GPU/CPU/Memory information
- Storage analysis
- Recommendations for configuration
- Rich visualization with color coding

### 8. Configuration Management
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:3000-3500`
- Execution profiles
- Execution history
- Configuration import/export
- Default settings

### 9. Progress Tracking
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/rich_progress_manager.py`
- RichProgressManager
- Real-time progress bars
- Task tracking
- ETA estimation
- Status messages

### 10. Model Display
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/model_display.py`
- All available models listing
- Model recommendations
- Language coverage display
- Parameter count information

---

## DATA PROCESSING & EXPORT

### 1. Data Input Formats
- **CSV** - with various encodings
- **Excel** (.xlsx, .xls) - sheet selection
- **Parquet** - columnar format
- **RData/RDS** - R format via pyreadr
- **JSON/JSONL** - nested structures
- **PostgreSQL** - direct database queries

### 2. Data Output Formats
- **CSV** - incremental append mode
- **Excel** (.xlsx) - formatted output
- **Parquet** - optimized storage
- **JSON/JSONL** - structured export
- **PostgreSQL** - database persistence

### 3. Annotation Export
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/annotators/llm_annotator.py:2000-2500`
- Original data + annotations
- Subset of annotated samples
- Complete result sets
- Incremental saving support

### 4. Doccano Export
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/doccano_exporter.py`
- JSONL format
- Metadata preservation
- Label suggestions
- Interactive review interface

### 5. Training Data Export
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/training_data_builder.py`
- PyTorch Dataset format
- HuggingFace datasets format
- CSV/JSON for external use
- Train/val/test splits

### 6. Benchmark Export
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/benchmarking.py:3200-3500`
- Per-category results
- Per-language results
- Aggregated results
- CSV and JSON formats

### 7. Session Export
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/session_summary.py`
- Session metadata
- Training summaries
- Validation reports
- Complete session state

---

## STATISTICAL ANALYSIS

### 1. Label Distribution Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/benchmark_utils.py:100-300`
- Class imbalance calculation
- Distribution visualization
- Stratification recommendations

### 2. Text Statistics
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py:1900-2200`
- Character count distribution
- Word count distribution
- Sentence count analysis
- Percentile analysis (min, 25%, 50%, 75%, max, mean, std)

### 3. Language Distribution
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/language_filtering.py`
- Language frequency analysis
- Diversity scoring
- Per-language sample counting
- Sufficient sample filtering

### 4. Model Performance Statistics
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/trainers/bert_base.py:3500-4000`
- Per-class precision/recall/F1
- Macro/micro averaged metrics
- Weighted averages
- Support counting

### 5. Confidence Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py:250-400`
- Confidence score distribution
- Threshold-based filtering
- Per-label confidence tracking

### 6. Inter-Annotator Agreement
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/validators/annotation_validator.py:400-600`
- Cohen's Kappa score
- Accuracy calculation
- Confusion matrix
- Agreement percentage

### 7. Cost Statistics
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/cost_estimator.py:200-500`
- Token usage breakdown
- Cost per request
- Total cost estimation
- Cost per sample

### 8. Resource Utilization Analysis
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/system_resources.py:400-600`
- GPU memory usage
- CPU usage percentage
- RAM availability
- Storage space analysis

### 9. Training Metrics Aggregation
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/training_summary_generator.py`
- Epoch-wise metrics
- Best model selection
- Convergence analysis
- Overfitting detection

### 10. Benchmark Comparison
- **Location:** `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/benchmark_helpers.py:500-800`
- Cross-model ranking
- Performance deltas
- Category-wise comparison
- Language-wise comparison

---

## KEY FEATURES SUMMARY

### Annotation Capabilities
- Single and multi-prompt annotation
- 5 API providers (OpenAI, Anthropic, Google, Ollama, Llama.cpp)
- Batch processing with OpenAI Batch API
- Parallel execution with process pooling
- JSON schema validation
- Cost estimation and tracking
- Resume capability
- Database support (PostgreSQL, CSV, Excel, Parquet, RData)

### Training Capabilities
- 50+ transformer models
- Multi-label classification
- Language-specific model selection
- Multilingual ensemble recommendations
- Hyperparameter optimization
- Early stopping and checkpointing
- Cross-validation
- Multi-GPU support
- Mixed precision training

### Validation Capabilities
- Schema compliance checking
- Label consistency validation
- Inter-annotator agreement metrics
- Confidence-based filtering
- Sample selection for human review
- Doccano export for review workflows

### CLI Features
- Interactive wizards for all major workflows
- Auto-detection of resources, models, datasets
- Cost estimation before annotation
- System resource display
- Progress tracking with real-time updates
- Resumable workflows
- Configuration profiles
- Execution history

### Export Capabilities
- Multiple data formats (CSV, Excel, Parquet, JSON)
- Doccano JSONL for annotation review
- Training-ready formats
- Benchmark reports
- Session summaries

---

## TECHNOLOGY STACK

### Core Dependencies
- **Data:** pandas>=2.0.0, numpy>=1.24.0, pyarrow>=18.1.0, openpyxl>=3.1.0
- **LLMs:** openai>=1.0.0, anthropic>=0.18.0, ollama>=0.6.0, transformers>=4.35.0
- **ML/DL:** torch>=2.0.0, transformers>=4.35.0, scikit-learn>=1.5.0
- **CLI:** rich>=14.0.0, inquirer>=3.0.0, click>=8.0.0
- **Language Detection:** lingua-language-detector>=2.0.0
- **Database:** sqlalchemy>=2.0.0, psycopg2-binary>=2.9.0
- **Utilities:** python-dotenv>=1.0.0, pydantic>=2.0.0, loguru>=0.7.0

### Optional Dependencies (Advanced)
- Google: google-generativeai>=0.3.0
- Label Studio: label-studio>=1.20.0, label-studio-sdk>=2.0.0
- Language Detection Alternatives: langdetect>=1.0.9, fasttext>=0.9.2
- ML Enhancement: sentence-transformers>=2.2.0, imbalanced-learn>=0.11.0
- Visualization: matplotlib>=3.6.0, seaborn>=0.12.0
- MLOps: mlflow>=2.8.0, wandb>=0.15.0, tensorboard>=2.13.0
- Web Serving: fastapi>=0.103.0, gradio>=3.45.0
- Databases: redis>=5.0.0, pymongo>=4.5.0
- Distributed Computing: ray>=2.6.0, dask>=2023.8.0

---

## SUMMARY STATISTICS

- **Total Python Modules:** 72 files
- **Total Lines of Code:** ~79,268
- **Total Classes:** 65+
- **Total Methods:** 500+
- **Languages Supported:** 75+ for detection
- **Models Supported:** 50+ transformers + 5 LLM API providers
- **Data Formats:** CSV, Excel, Parquet, JSON, JSONL, RData, PostgreSQL
- **CLI Modes:** 3 major modes (Annotator Factory, Training Arena, Validation Lab)
- **Validation Methods:** 7 major validation features
- **Text Detection Methods:** 7 detection capabilities
- **Export Formats:** 5+ output formats

---

