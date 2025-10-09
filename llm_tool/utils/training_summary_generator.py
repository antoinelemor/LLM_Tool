"""
PROJECT:
-------
LLMTool

TITLE:
------
training_summary_generator.py

MAIN OBJECTIVE:
---------------
Comprehensive summary file generation for Training Arena sessions.
Creates detailed CSV and JSONL summary files at the end of each training cycle
with complete metrics, dataset information, model details, and training parameters.

Dependencies:
-------------
- pandas
- numpy
- json
- csv
- pathlib
- datetime
- logging

MAIN FEATURES:
--------------
1) Generate comprehensive CSV summary with all metrics
2) Generate comprehensive JSONL summary with complete session data
3) Dataset distribution analysis with detailed statistics
4) Model performance aggregation across all training modes
5) Time tracking and duration calculations
6) Robust error handling with fallback mechanisms
7) Support for benchmark and normal training modes
8) Language-specific metrics aggregation
9) Hyperparameter tracking
10) Research reproducibility support

Author:
-------
Antoine Lemor
"""

import json
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, Counter
import traceback

logger = logging.getLogger(__name__)


class TrainingSummaryGenerator:
    """
    Generates comprehensive summary files for Training Arena sessions.
    Ensures all training information is captured for reproducibility.
    """

    def __init__(self, session_id: str, session_dir: Path):
        """
        Initialize the summary generator.

        Args:
            session_id: Training session identifier
            session_dir: Path to session directory (e.g., logs/training_arena/{session_id})
        """
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.training_metrics_dir = self.session_dir / "training_metrics"
        self.metadata_dir = self.session_dir / "training_session_metadata"

        # Output files - Use 'training_summary_final' as requested
        self.csv_summary_path = self.session_dir / "training_summary_final.csv"
        self.jsonl_summary_path = self.session_dir / "training_summary_final.jsonl"

        # Also keep versioned copies with session ID for tracking
        self.csv_versioned_path = self.session_dir / f"{session_id}_training_summary_final.csv"
        self.jsonl_versioned_path = self.session_dir / f"{session_id}_training_summary_final.jsonl"

        # Session data storage
        self.session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "dataset_info": {},
            "model_info": {},
            "training_params": {},
            "time_info": {},
            "errors": []
        }

    def generate_comprehensive_summaries(self) -> Tuple[Path, Path]:
        """
        Generate both CSV and JSONL comprehensive summary files.

        Returns:
            Tuple of (csv_path, jsonl_path) for generated files
        """
        try:
            # Step 1: Load all data
            self._load_session_metadata()
            self._load_training_metrics()
            self._load_best_models()
            self._analyze_datasets()
            self._calculate_time_statistics()

            # Step 2: Generate summaries
            csv_path = self._generate_csv_summary()
            jsonl_path = self._generate_jsonl_summary()

            # Step 3: Generate best models summary
            best_models_path = self.generate_best_models_final_csv()

            logger.info(f"Generated comprehensive summaries for session {self.session_id}")
            logger.info(f"  Standard names: training_summary_final.csv/jsonl")
            logger.info(f"  Best models: best_models_final.csv")
            logger.info(f"  Versioned names: {self.csv_versioned_path.name}, {self.jsonl_versioned_path.name}")
            logger.info(f"  Location: {self.session_dir}")

            return csv_path, jsonl_path

        except Exception as e:
            logger.error(f"Failed to generate summaries: {e}")
            logger.error(traceback.format_exc())
            # Try to generate partial summaries with available data
            return self._generate_partial_summaries()

    def _load_session_metadata(self) -> None:
        """Load session metadata from training_metadata.json."""
        metadata_path = self.metadata_dir / "training_metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Extract key information
                self.session_data["training_params"] = metadata.get("training_params", {})
                self.session_data["model_info"]["config"] = metadata.get("model_config", {})
                self.session_data["dataset_info"]["config"] = metadata.get("dataset_config", {})
                self.session_data["language_config"] = metadata.get("language_config", {})
                self.session_data["split_config"] = metadata.get("split_config", {})
                self.session_data["reinforced_learning"] = metadata.get("reinforced_learning_config", {})
                self.session_data["execution_status"] = metadata.get("execution_status", {})
                self.session_data["session_info"] = metadata.get("training_session", {})

                # Extract time information
                if "training_session" in metadata:
                    session = metadata["training_session"]
                    self.session_data["time_info"]["start_time"] = metadata.get("created_at")
                    self.session_data["time_info"]["mode"] = session.get("mode", "unknown")

            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.session_data["errors"].append(f"Metadata loading error: {str(e)}")

    def _load_training_metrics(self) -> None:
        """Load and aggregate all training metrics from CSV files."""
        try:
            # Find all training.csv and reinforced.csv files
            training_csvs = list(self.training_metrics_dir.rglob("training.csv"))
            reinforced_csvs = list(self.training_metrics_dir.rglob("reinforced.csv"))

            all_metrics = []

            # Process training metrics
            for csv_path in training_csvs:
                try:
                    df = pd.read_csv(csv_path, comment='#')

                    # Extract context from path
                    rel_path = csv_path.relative_to(self.training_metrics_dir)
                    parts = list(rel_path.parts[:-1])

                    # Add context columns
                    df['phase'] = 'normal'
                    df['source_file'] = str(csv_path.name)

                    if parts and parts[0] in ['benchmark', 'normal_training']:
                        df['mode'] = parts[0]
                        if len(parts) > 1:
                            df['category'] = parts[1]
                        if len(parts) > 2:
                            df['language'] = parts[2]

                    all_metrics.append(df)

                except Exception as e:
                    logger.warning(f"Failed to read {csv_path}: {e}")

            # Process reinforced metrics
            for csv_path in reinforced_csvs:
                try:
                    df = pd.read_csv(csv_path, comment='#')
                    df['phase'] = 'reinforced'
                    df['source_file'] = str(csv_path.name)

                    # Extract context from path
                    rel_path = csv_path.relative_to(self.training_metrics_dir)
                    parts = list(rel_path.parts[:-1])

                    if parts and parts[0] in ['benchmark', 'normal_training']:
                        df['mode'] = parts[0]
                        if len(parts) > 1:
                            df['category'] = parts[1]
                        if len(parts) > 2:
                            df['language'] = parts[2]

                    all_metrics.append(df)

                except Exception as e:
                    logger.warning(f"Failed to read {csv_path}: {e}")

            # Combine all metrics
            if all_metrics:
                # Filter out empty dataframes to avoid FutureWarning
                non_empty_metrics = [df for df in all_metrics if not df.empty]
                if non_empty_metrics:
                    combined_metrics = pd.concat(non_empty_metrics, ignore_index=True)
                else:
                    combined_metrics = pd.DataFrame()  # Empty DataFrame if all are empty

                # Calculate aggregated statistics
                if not combined_metrics.empty:
                    self.session_data["metrics"]["total_epochs"] = len(combined_metrics)
                    self.session_data["metrics"]["models_trained"] = combined_metrics['model_name'].nunique() if 'model_name' in combined_metrics else 0

                # Best overall metrics
                if 'val_f1_macro' in combined_metrics:
                    best_idx = combined_metrics['val_f1_macro'].idxmax()
                    best_row = combined_metrics.iloc[best_idx]
                    self.session_data["metrics"]["best_f1_macro"] = float(best_row['val_f1_macro'])
                    self.session_data["metrics"]["best_model"] = best_row.get('model_name', 'unknown')
                    self.session_data["metrics"]["best_accuracy"] = float(best_row.get('val_accuracy', 0))

                # Average metrics
                numeric_cols = combined_metrics.select_dtypes(include=[np.number]).columns
                avg_metrics = {}
                for col in numeric_cols:
                    if col.startswith('val_') or col.startswith('train_'):
                        avg_metrics[f"avg_{col}"] = float(combined_metrics[col].mean())

                self.session_data["metrics"]["averages"] = avg_metrics

                # Store raw dataframe for CSV generation
                self.session_data["_raw_metrics_df"] = combined_metrics

        except Exception as e:
            logger.error(f"Failed to load training metrics: {e}")
            self.session_data["errors"].append(f"Metrics loading error: {str(e)}")

    def _load_best_models(self) -> None:
        """Load best model information from best.csv files."""
        try:
            best_csvs = list(self.training_metrics_dir.rglob("best.csv"))

            if not best_csvs:
                # Try consolidated file
                consolidated_best = self.session_dir / f"{self.session_id}_best_models.csv"
                if consolidated_best.exists():
                    best_csvs = [consolidated_best]

            best_models = []

            for csv_path in best_csvs:
                try:
                    df = pd.read_csv(csv_path, comment='#')

                    # Extract context from path if not consolidated
                    if csv_path.parent != self.session_dir:
                        rel_path = csv_path.relative_to(self.training_metrics_dir)
                        parts = list(rel_path.parts[:-1])

                        if parts and parts[0] in ['benchmark', 'normal_training']:
                            df['training_mode'] = parts[0]
                            if len(parts) > 1:
                                df['category'] = parts[1]

                    best_models.append(df)

                except Exception as e:
                    logger.warning(f"Failed to read {csv_path}: {e}")

            if best_models:
                # Filter out empty dataframes to avoid FutureWarning
                non_empty_best = [df for df in best_models if not df.empty]
                if non_empty_best:
                    combined_best = pd.concat(non_empty_best, ignore_index=True)
                else:
                    combined_best = pd.DataFrame()  # Empty DataFrame if all are empty

                # Get top models by combined score
                if not combined_best.empty and 'combined_score' in combined_best:
                    # Select only columns that exist in the dataframe
                    available_cols = ['model_name', 'combined_score', 'macro_f1', 'accuracy']
                    cols_to_use = [col for col in available_cols if col in combined_best.columns]

                    if 'combined_score' in combined_best.columns and len(combined_best) > 0:
                        top_models = combined_best.nlargest(min(5, len(combined_best)), 'combined_score')
                        if cols_to_use:
                            top_models = top_models[cols_to_use]
                        self.session_data["model_info"]["top_models"] = top_models.to_dict('records')

                # Store for later use (even if empty)
                if not combined_best.empty:
                    self.session_data["_best_models_df"] = combined_best

        except Exception as e:
            logger.error(f"Failed to load best models: {e}")
            self.session_data["errors"].append(f"Best models loading error: {str(e)}")

    def _analyze_datasets(self) -> None:
        """Analyze dataset distributions and statistics."""
        try:
            dataset_info = {}

            # Load training data files if available
            training_data_dir = Path("data/training_data") / self.session_id
            if not training_data_dir.exists():
                # Try alternative location
                training_data_dir = self.session_dir / "training_data"

            if training_data_dir.exists():
                # Analyze each training file
                for data_file in training_data_dir.glob("*.jsonl"):
                    category = data_file.stem
                    dataset_info[category] = self._analyze_single_dataset(data_file)

                # Also check for CSV files
                for data_file in training_data_dir.glob("*.csv"):
                    category = data_file.stem
                    if category not in dataset_info:
                        dataset_info[category] = self._analyze_single_dataset(data_file)

            # Calculate overall statistics
            if dataset_info:
                total_samples = sum(info.get('sample_count', 0) for info in dataset_info.values())
                total_labels = sum(info.get('num_labels', 0) for info in dataset_info.values())

                # Aggregate label distributions across all categories
                all_label_counts = {}
                for category_info in dataset_info.values():
                    if 'label_distribution' in category_info:
                        for label, count in category_info['label_distribution'].items():
                            all_label_counts[label] = all_label_counts.get(label, 0) + count

                # Calculate overall class balance
                if all_label_counts:
                    counts = list(all_label_counts.values())
                    max_count = max(counts)
                    min_count = min(counts)
                    overall_imbalance = max_count / min_count if min_count > 0 else float('inf')
                else:
                    overall_imbalance = 1.0

                self.session_data["dataset_info"]["detailed"] = dataset_info
                self.session_data["dataset_info"]["summary"] = {
                    "total_samples": total_samples,
                    "total_categories": len(dataset_info),
                    "total_unique_labels": total_labels,
                    "categories": list(dataset_info.keys()),
                    "overall_label_distribution": all_label_counts,
                    "overall_imbalance_ratio": overall_imbalance,
                    "dataset_files": [info.get('file', 'unknown') for info in dataset_info.values()]
                }

            # Add language distribution if available
            if "language_config" in self.session_data:
                lang_dist = self.session_data["language_config"].get("language_distribution", {})
                if lang_dist:
                    self.session_data["dataset_info"]["language_distribution"] = lang_dist

        except Exception as e:
            logger.error(f"Failed to analyze datasets: {e}")
            self.session_data["errors"].append(f"Dataset analysis error: {str(e)}")

    def _analyze_single_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset file for statistics."""
        info = {
            "file": str(file_path.name),
            "sample_count": 0,
            "label_distribution": {},
            "language_distribution": {},
            "text_length_stats": {},
            "num_labels": 0
        }

        try:
            if file_path.suffix == '.jsonl':
                # Read JSONL file
                samples = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        samples.append(json.loads(line))

                info["sample_count"] = len(samples)

                # Analyze labels
                labels = []
                languages = []
                text_lengths = []

                for sample in samples:
                    if 'label' in sample:
                        labels.append(sample['label'])
                    if 'lang' in sample or 'language' in sample:
                        languages.append(sample.get('lang', sample.get('language')))
                    if 'text' in sample:
                        text_lengths.append(len(sample['text']))

                # Calculate distributions
                if labels:
                    label_counts = Counter(labels)
                    info["label_distribution"] = dict(label_counts)
                    info["num_labels"] = len(label_counts)

                    # Calculate imbalance metrics
                    counts = list(label_counts.values())
                    if len(counts) > 1:
                        info["imbalance_ratio"] = max(counts) / min(counts)
                        info["gini_coefficient"] = self._calculate_gini(counts)

                if languages:
                    info["language_distribution"] = dict(Counter(languages))

                if text_lengths:
                    info["text_length_stats"] = {
                        "mean": np.mean(text_lengths),
                        "median": np.median(text_lengths),
                        "std": np.std(text_lengths),
                        "min": min(text_lengths),
                        "max": max(text_lengths)
                    }

            elif file_path.suffix == '.csv':
                # Read CSV file
                df = pd.read_csv(file_path)
                info["sample_count"] = len(df)

                # Analyze labels
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts().to_dict()
                    info["label_distribution"] = label_counts
                    info["num_labels"] = len(label_counts)

                    # Calculate imbalance
                    counts = list(label_counts.values())
                    if len(counts) > 1:
                        info["imbalance_ratio"] = max(counts) / min(counts)
                        info["gini_coefficient"] = self._calculate_gini(counts)

                # Language distribution
                for lang_col in ['lang', 'language']:
                    if lang_col in df.columns:
                        info["language_distribution"] = df[lang_col].value_counts().to_dict()
                        break

                # Text statistics
                if 'text' in df.columns:
                    text_lengths = df['text'].str.len()
                    info["text_length_stats"] = {
                        "mean": float(text_lengths.mean()),
                        "median": float(text_lengths.median()),
                        "std": float(text_lengths.std()),
                        "min": int(text_lengths.min()),
                        "max": int(text_lengths.max())
                    }

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            info["error"] = str(e)

        return info

    def _calculate_gini(self, counts: List[int]) -> float:
        """Calculate Gini coefficient for imbalance measurement."""
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n)

    def _calculate_time_statistics(self) -> None:
        """Calculate time-related statistics for the training session."""
        try:
            time_info = self.session_data.get("time_info", {})

            # Get start time from metadata
            if "created_at" in self.session_data.get("session_info", {}):
                start_str = self.session_data["session_info"]["created_at"]
                time_info["start_time"] = start_str
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            else:
                # Fallback to session_id timestamp
                try:
                    start_time = datetime.strptime(self.session_id.split('_')[0], "%Y%m%d")
                    time_info["start_time"] = start_time.isoformat()
                except:
                    start_time = None

            # Calculate end time and duration
            if start_time:
                # Check for completion status
                if "execution_status" in self.session_data:
                    exec_status = self.session_data["execution_status"]
                    if exec_status.get("completed_at"):
                        end_str = exec_status["completed_at"]
                        time_info["end_time"] = end_str
                        end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    else:
                        # Use current time as approximation
                        end_time = datetime.now()
                        time_info["end_time"] = end_time.isoformat()
                        time_info["status"] = "in_progress"

                    # Calculate duration
                    duration = end_time - start_time
                    time_info["total_duration_seconds"] = duration.total_seconds()
                    time_info["total_duration_formatted"] = str(duration)

                    # Calculate per-epoch time if available
                    if "metrics" in self.session_data and "total_epochs" in self.session_data["metrics"]:
                        total_epochs = self.session_data["metrics"]["total_epochs"]
                        if total_epochs > 0:
                            time_per_epoch = duration.total_seconds() / total_epochs
                            time_info["avg_time_per_epoch_seconds"] = time_per_epoch
                            time_info["avg_time_per_epoch_formatted"] = str(timedelta(seconds=time_per_epoch))

            # Add training time from execution status
            if "execution_status" in self.session_data:
                exec_status = self.session_data["execution_status"]
                if "training_time_seconds" in exec_status:
                    time_info["actual_training_time_seconds"] = exec_status["training_time_seconds"]

            self.session_data["time_info"] = time_info

        except Exception as e:
            logger.error(f"Failed to calculate time statistics: {e}")
            self.session_data["errors"].append(f"Time calculation error: {str(e)}")

    def _generate_csv_summary(self) -> Path:
        """Generate comprehensive CSV summary file."""
        try:
            # Prepare data for CSV
            csv_data = []

            # If we have raw metrics, use them as base
            if "_raw_metrics_df" in self.session_data:
                metrics_df = self.session_data["_raw_metrics_df"].copy()

                # Remove unnecessary columns
                columns_to_remove = []
                if 'language' in metrics_df.columns:
                    columns_to_remove.append('language')
                if 'label_key' in metrics_df.columns:
                    columns_to_remove.append('label_key')

                if columns_to_remove:
                    metrics_df = metrics_df.drop(columns=columns_to_remove, errors='ignore')

                # Add session-level information to each row
                metrics_df['session_id'] = self.session_id
                metrics_df['training_mode'] = self.session_data.get("time_info", {}).get("mode", "unknown")

                # Add dataset summary
                if "dataset_info" in self.session_data and "summary" in self.session_data["dataset_info"]:
                    summary = self.session_data["dataset_info"]["summary"]
                    metrics_df['total_samples'] = summary.get("total_samples", 0)
                    metrics_df['total_categories'] = summary.get("total_categories", 0)

                # Add time information
                if "time_info" in self.session_data:
                    time_info = self.session_data["time_info"]
                    metrics_df['session_start_time'] = time_info.get("start_time", "")
                    metrics_df['session_end_time'] = time_info.get("end_time", "")
                    metrics_df['total_duration_seconds'] = time_info.get("total_duration_seconds", 0)

                # Add hyperparameters
                if "training_params" in self.session_data:
                    params = self.session_data["training_params"]
                    metrics_df['learning_rate'] = params.get("learning_rate", "")
                    metrics_df['batch_size'] = params.get("batch_size", "")
                    metrics_df['warmup_ratio'] = params.get("warmup_ratio", "")
                    metrics_df['weight_decay'] = params.get("weight_decay", "")

                # Add reinforced learning info
                if "reinforced_learning" in self.session_data:
                    rl_config = self.session_data["reinforced_learning"]
                    metrics_df['reinforced_learning_enabled'] = rl_config.get("enabled", False)
                    metrics_df['rl_f1_threshold'] = rl_config.get("f1_threshold", "")
                    metrics_df['rl_epochs'] = rl_config.get("reinforced_epochs", "")

                # Prepare label legend
                label_legend = self._prepare_label_legend()

                # Save CSV with label legend in header
                self._save_csv_with_legend(metrics_df, self.csv_summary_path, label_legend)
                self._save_csv_with_legend(metrics_df, self.csv_versioned_path, label_legend)
                logger.info(f"Generated CSV summary with {len(metrics_df)} rows")

            else:
                # Fallback: Create summary from available data
                summary_row = {
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'training_mode': self.session_data.get("time_info", {}).get("mode", "unknown"),
                    'status': self.session_data.get("execution_status", {}).get("status", "unknown")
                }

                # Add metrics
                if "metrics" in self.session_data:
                    metrics = self.session_data["metrics"]
                    summary_row.update({
                        'best_f1_macro': metrics.get("best_f1_macro", ""),
                        'best_accuracy': metrics.get("best_accuracy", ""),
                        'best_model': metrics.get("best_model", ""),
                        'total_epochs': metrics.get("total_epochs", 0),
                        'models_trained': metrics.get("models_trained", 0)
                    })

                # Add dataset info
                if "dataset_info" in self.session_data and "summary" in self.session_data["dataset_info"]:
                    summary = self.session_data["dataset_info"]["summary"]
                    summary_row.update({
                        'total_samples': summary.get("total_samples", 0),
                        'total_categories': summary.get("total_categories", 0),
                        'categories': ', '.join(summary.get("categories", []))
                    })

                # Add time info
                if "time_info" in self.session_data:
                    time_info = self.session_data["time_info"]
                    summary_row.update({
                        'start_time': time_info.get("start_time", ""),
                        'end_time': time_info.get("end_time", ""),
                        'total_duration_seconds': time_info.get("total_duration_seconds", 0),
                        'avg_time_per_epoch': time_info.get("avg_time_per_epoch_seconds", 0)
                    })

                # Add errors if any
                if self.session_data.get("errors"):
                    summary_row['errors'] = '; '.join(self.session_data["errors"])

                # Write single row summary to both paths
                pd.DataFrame([summary_row]).to_csv(self.csv_summary_path, index=False)
                pd.DataFrame([summary_row]).to_csv(self.csv_versioned_path, index=False)
                logger.info("Generated fallback CSV summary")

            return self.csv_summary_path

        except Exception as e:
            logger.error(f"Failed to generate CSV summary: {e}")
            self.session_data["errors"].append(f"CSV generation error: {str(e)}")
            # Create minimal CSV
            self._create_minimal_csv()
            return self.csv_summary_path

    def _generate_jsonl_summary(self) -> Path:
        """Generate comprehensive JSONL summary file."""
        try:
            # Prepare comprehensive data for JSONL
            jsonl_records = []

            # Main session summary record with comprehensive information
            main_record = {
                "record_type": "session_summary",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "session_info": self.session_data.get("session_info", {}),
                "execution_status": self.session_data.get("execution_status", {}),
                "time_statistics": self.session_data.get("time_info", {}),
                "overall_metrics": self.session_data.get("metrics", {}),
                "training_parameters": self.session_data.get("training_params", {}),
                "model_configuration": self.session_data.get("model_info", {}).get("config", {}),
                "reinforced_learning": self.session_data.get("reinforced_learning", {}),
                "language_configuration": self.session_data.get("language_config", {}),
                "split_configuration": self.session_data.get("split_config", {}),
                "dataset_summary": self.session_data.get("dataset_info", {}).get("summary", {}),
                "best_models": self.session_data.get("model_info", {}).get("top_models", []),
                "training_mode": self.session_data.get("time_info", {}).get("mode", "unknown"),
                "hardware_info": {
                    "gpu_available": self.session_data.get("training_params", {}).get("gpu_available", False),
                    "device": self.session_data.get("training_params", {}).get("device", "cpu")
                },
                "reproducibility_info": {
                    "random_seed": self.session_data.get("training_params", {}).get("seed", 42),
                    "deterministic": self.session_data.get("training_params", {}).get("deterministic", True)
                },
                "errors": self.session_data.get("errors", [])
            }
            jsonl_records.append(main_record)

            # Dataset analysis records
            if "dataset_info" in self.session_data and "detailed" in self.session_data["dataset_info"]:
                for category, info in self.session_data["dataset_info"]["detailed"].items():
                    dataset_record = {
                        "record_type": "dataset_analysis",
                        "session_id": self.session_id,
                        "category": category,
                        "statistics": info
                    }
                    jsonl_records.append(dataset_record)

            # Model performance records
            if "model_info" in self.session_data and "top_models" in self.session_data["model_info"]:
                for model in self.session_data["model_info"]["top_models"]:
                    model_record = {
                        "record_type": "model_performance",
                        "session_id": self.session_id,
                        "model": model
                    }
                    jsonl_records.append(model_record)

            # Per-epoch metrics if available
            if "_raw_metrics_df" in self.session_data:
                metrics_df = self.session_data["_raw_metrics_df"]
                for _, row in metrics_df.iterrows():
                    epoch_record = {
                        "record_type": "epoch_metrics",
                        "session_id": self.session_id,
                        "epoch_data": row.to_dict()
                    }
                    jsonl_records.append(epoch_record)

            # Write JSONL file to both paths
            for path in [self.jsonl_summary_path, self.jsonl_versioned_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    for record in jsonl_records:
                        f.write(json.dumps(record, default=str) + '\n')

            logger.info(f"Generated JSONL summary with {len(jsonl_records)} records")
            return self.jsonl_summary_path

        except Exception as e:
            logger.error(f"Failed to generate JSONL summary: {e}")
            self.session_data["errors"].append(f"JSONL generation error: {str(e)}")
            # Create minimal JSONL
            self._create_minimal_jsonl()
            return self.jsonl_summary_path

    def _create_minimal_csv(self) -> None:
        """Create minimal CSV with available data."""
        try:
            minimal_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'partial',
                'errors': '; '.join(self.session_data.get("errors", ["Unknown error"]))
            }
            df = pd.DataFrame([minimal_data])
            df.to_csv(self.csv_summary_path, index=False)
            df.to_csv(self.csv_versioned_path, index=False)
        except:
            # Last resort: create empty CSV in both locations
            for path in [self.csv_summary_path, self.csv_versioned_path]:
                with open(path, 'w') as f:
                    f.write("session_id,status,error\n")
                    f.write(f"{self.session_id},failed,Could not generate summary\n")

    def _create_minimal_jsonl(self) -> None:
        """Create minimal JSONL with available data."""
        try:
            minimal_record = {
                "record_type": "error_summary",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "partial",
                "errors": self.session_data.get("errors", ["Unknown error"])
            }
            for path in [self.jsonl_summary_path, self.jsonl_versioned_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(minimal_record, default=str) + '\n')
        except:
            # Last resort: create error record in both locations
            for path in [self.jsonl_summary_path, self.jsonl_versioned_path]:
                with open(path, 'w') as f:
                    f.write(f'{{"session_id": "{self.session_id}", "status": "failed"}}\n')

    def _generate_partial_summaries(self) -> Tuple[Path, Path]:
        """Generate partial summaries with whatever data is available."""
        logger.warning("Generating partial summaries due to errors")

        # Try to generate CSV with available data
        try:
            self._generate_csv_summary()
        except:
            self._create_minimal_csv()

        # Try to generate JSONL with available data
        try:
            self._generate_jsonl_summary()
        except:
            self._create_minimal_jsonl()

        return self.csv_summary_path, self.jsonl_summary_path

    def add_custom_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add custom metrics to the session data.

        Args:
            metrics: Dictionary of custom metrics to add
        """
        if "custom_metrics" not in self.session_data:
            self.session_data["custom_metrics"] = {}
        self.session_data["custom_metrics"].update(metrics)

    def add_error(self, error_msg: str) -> None:
        """
        Add an error message to the session data.

        Args:
            error_msg: Error message to record
        """
        if "errors" not in self.session_data:
            self.session_data["errors"] = []
        self.session_data["errors"].append(f"{datetime.now().isoformat()}: {error_msg}")

    def _prepare_label_legend(self) -> str:
        """Prepare label legend for CSV headers."""
        legend_lines = []

        # Get unique labels from dataset info
        if "dataset_info" in self.session_data and "summary" in self.session_data["dataset_info"]:
            label_dist = self.session_data["dataset_info"]["summary"].get("overall_label_distribution", {})
            if label_dist:
                legend_lines.append("# LABEL LEGEND:")
                for idx, label in enumerate(sorted(label_dist.keys()), 1):
                    legend_lines.append(f"# {idx}. {label}: {label_dist[label]} samples")
                legend_lines.append("#")

        return "\n".join(legend_lines) if legend_lines else ""

    def _save_csv_with_legend(self, df: pd.DataFrame, path: Path, legend: str) -> None:
        """Save CSV with optional legend header."""
        with open(path, 'w', encoding='utf-8') as f:
            # Write legend as comments if available
            if legend:
                f.write(legend + "\n")

            # Write the dataframe
            df.to_csv(f, index=False)

    def generate_best_models_final_csv(self) -> Path:
        """Generate best_models_final.csv with only the best models and all metrics."""
        best_models_path = self.session_dir / "best_models_final.csv"

        try:
            if "_best_models_df" in self.session_data and not self.session_data["_best_models_df"].empty:
                best_df = self.session_data["_best_models_df"].copy()

                # Remove unnecessary columns if they exist
                columns_to_remove = ['language', 'label_key']
                best_df = best_df.drop(columns=[col for col in columns_to_remove if col in best_df.columns], errors='ignore')

                # Add session info
                best_df['session_id'] = self.session_id
                best_df['training_mode'] = self.session_data.get("time_info", {}).get("mode", "unknown")
                best_df['timestamp'] = datetime.now().isoformat()

                # Prepare label legend
                label_legend = self._prepare_label_legend()

                # Save with legend
                self._save_csv_with_legend(best_df, best_models_path, label_legend)

                logger.info(f"Generated best_models_final.csv with {len(best_df)} best models")
            else:
                # Create empty file with headers only if no best models
                pd.DataFrame(columns=['session_id', 'model_name', 'accuracy', 'f1_score', 'status']).to_csv(best_models_path, index=False)
                logger.warning("No best models found, created empty best_models_final.csv")

            return best_models_path

        except Exception as e:
            logger.error(f"Failed to generate best_models_final.csv: {e}")
            # Create minimal file
            pd.DataFrame([{'session_id': self.session_id, 'status': 'error', 'message': str(e)}]).to_csv(best_models_path, index=False)
            return best_models_path


def generate_training_summaries(session_id: str, session_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Convenience function to generate training summaries.

    Args:
        session_id: Training session identifier
        session_dir: Optional session directory path (defaults to logs/training_arena/{session_id})

    Returns:
        Tuple of (csv_path, jsonl_path) for generated files
    """
    if session_dir is None:
        session_dir = Path("logs/training_arena") / session_id

    generator = TrainingSummaryGenerator(session_id, session_dir)
    return generator.generate_comprehensive_summaries()