#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
ssh_training_worker.py

MAIN OBJECTIVE:
---------------
Remote training worker for distributed SSH training.
Executed on remote machines via SSH to train a single model for one category.

This script is designed to be invoked remotely:
    ssh remote-host "cd /path/to/project && python -m llm_tool.trainers.ssh_training_worker --config /tmp/llm_tool_training/category/task_config.json"

It:
1. Reads a task_config.json with all training parameters
2. Loads training data (CSV or JSONL)
3. Trains a model using the existing MultiLabelTrainer
4. Writes progress.json periodically (atomic writes) for remote monitoring
5. Writes results.json upon completion

Dependencies:
-------------
- llm_tool.trainers.multi_label_trainer
- llm_tool.trainers.data_utils
- torch

Author:
-------
Antoine Lemor
"""

import os
import gc
import ast
import sys
import csv
import json
import time
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_task_config(config_path: str) -> Dict[str, Any]:
    """
    Load a task configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the task_config.json file

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate required fields
    required = ['task_id', 'category_name', 'model_name', 'data_path', 'output_path']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in task config: {field}")

    return config


def write_progress(progress_path: str, update: Dict[str, Any]):
    """
    Write a progress update to a JSON file atomically.

    Uses write-to-temp-then-rename pattern to prevent partial reads
    when the file is polled from the local machine via SSH.

    Parameters
    ----------
    progress_path : str
        Path to the progress.json file
    update : dict
        Progress data to write
    """
    tmp_path = progress_path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(update, f, indent=2)
        os.rename(tmp_path, progress_path)
    except Exception as e:
        logger.warning(f"Failed to write progress: {e}")
        # Try to clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def load_training_data(data_path: str) -> List[Any]:
    """
    Load training data from CSV or JSONL format.

    Automatically detects format based on file extension and content.
    Returns a list of DataSample objects.

    Parameters
    ----------
    data_path : str
        Path to the training data file

    Returns
    -------
    List[DataSample]
        List of training samples
    """
    from llm_tool.trainers.data_utils import DataSample

    samples = []

    # Determine format
    if data_path.endswith('.csv'):
        samples = _load_csv_data(data_path)
    elif data_path.endswith('.jsonl') or data_path.endswith('.json'):
        samples = _load_jsonl_data(data_path)
    else:
        # Try JSONL first, then CSV
        try:
            samples = _load_jsonl_data(data_path)
        except Exception:
            samples = _load_csv_data(data_path)

    return samples


def _load_jsonl_data(data_path: str) -> List[Any]:
    """Load data from JSON Lines format."""
    from llm_tool.trainers.data_utils import DataSample

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            label_data = record.get('label') or record.get('labels', [])

            # Handle string labels that look like lists
            if isinstance(label_data, str) and label_data.startswith('['):
                try:
                    label_data = ast.literal_eval(label_data)
                except Exception:
                    pass

            sample_lang = record.get('language') or record.get('lang', 'EN')
            samples.append(DataSample(
                text=record.get('text', ''),
                label=label_data,
                lang=str(sample_lang).upper() if sample_lang else 'EN',
            ))

    return samples


def _load_csv_data(data_path: str) -> List[Any]:
    """Load data from CSV format."""
    from llm_tool.trainers.data_utils import DataSample

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '')
            label = row.get('label', row.get('labels', ''))

            # Handle multi-label string representation
            if isinstance(label, str) and label.startswith('['):
                try:
                    label = ast.literal_eval(label)
                except Exception:
                    pass

            lang = row.get('language', row.get('lang', 'EN'))
            samples.append(DataSample(
                text=text,
                label=label,
                lang=str(lang).upper() if lang else 'EN',
            ))

    return samples


def run_training(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute model training based on a task configuration.

    Uses the existing MultiLabelTrainer to train a single binary classifier.
    Writes progress updates during training for remote monitoring.

    Parameters
    ----------
    task_config : dict
        Complete task configuration with all training parameters

    Returns
    -------
    Dict[str, Any]
        Training results including metrics and model path
    """
    from llm_tool.trainers.multi_label_trainer import MultiLabelTrainer, TrainingConfig
    from sklearn.model_selection import train_test_split

    start_time = time.time()

    # Extract parameters
    task_id = task_config['task_id']
    category_name = task_config['category_name']
    model_name = task_config['model_name']
    data_path = task_config['data_path']
    output_path = task_config['output_path']
    progress_path = task_config.get('progress_path', '')
    config = task_config.get('config', {})

    epochs = config.get('epochs', 10)
    learning_rate = config.get('learning_rate', 2e-5)
    batch_size = config.get('batch_size', 16)
    multi_label = config.get('multi_label', False)
    multi_label_threshold = config.get('multi_label_threshold', 0.5)
    reinforced_learning = config.get('reinforced_learning', False)
    reinforced_epochs = config.get('reinforced_epochs', None)
    rl_f1_threshold = config.get('rl_f1_threshold', 0.7)
    rl_oversample_factor = config.get('rl_oversample_factor', 2.0)
    rl_class_weight_factor = config.get('rl_class_weight_factor', 2.0)
    training_approach = config.get('training_approach', 'one-vs-all')
    session_id = config.get('session_id', '')

    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model_dir = str(Path(output_path) / "model")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Write initial progress
    if progress_path:
        write_progress(progress_path, {
            "task_id": task_id,
            "category_name": category_name,
            "status": "loading_data",
            "current_epoch": 0,
            "total_epochs": epochs,
            "f1_score": 0.0,
            "accuracy": 0.0,
        })

    # Load and split data
    logger.info(f"Loading training data from {data_path}")
    samples = load_training_data(data_path)

    if not samples:
        return {
            "task_id": task_id,
            "category_name": category_name,
            "status": "error",
            "error": "No training samples found",
            "elapsed_seconds": time.time() - start_time,
        }

    logger.info(f"Loaded {len(samples)} samples for {category_name}")

    # Split into train/val
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, random_state=42
    )
    logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")

    # Determine labels
    all_labels = set()
    for s in samples:
        if isinstance(s.label, list):
            for lbl in s.label:
                if lbl:
                    all_labels.add(str(lbl))
        elif s.label:
            all_labels.add(str(s.label))

    class_names = sorted(all_labels)
    num_labels = len(class_names) if len(class_names) > 2 else 2

    # Create trainer config
    trainer_config = TrainingConfig(
        model_name=model_name,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        multi_label=multi_label,
        multi_label_threshold=multi_label_threshold,
        training_approach=training_approach,
        # Propagate tokenizer-aware filter from the task config so that remote
        # workers honour the "Exclude long texts" strategy chosen in the CLI.
        exclude_long_texts=bool(task_config.get('exclude_long_texts', False)),
        max_length=int(task_config.get('max_length', 512) or 512),
    )

    # Create progress callback
    def progress_callback(**metrics):
        """Write epoch progress for remote monitoring."""
        if not progress_path:
            return
        epoch = metrics.get('epoch', 0)
        write_progress(progress_path, {
            "task_id": task_id,
            "category_name": category_name,
            "status": "training",
            "current_epoch": epoch,
            "total_epochs": epochs,
            "train_loss": metrics.get('train_loss', 0.0),
            "val_loss": metrics.get('val_loss', 0.0),
            "f1_score": metrics.get('f1_macro', 0.0),
            "accuracy": metrics.get('accuracy', 0.0),
            "combined_score": metrics.get('combined_score', 0.0),
            "per_class_f1": metrics.get('per_class_f1'),
            "class_names": metrics.get('class_names'),
        })

    # Write training start progress
    if progress_path:
        write_progress(progress_path, {
            "task_id": task_id,
            "category_name": category_name,
            "status": "training",
            "current_epoch": 0,
            "total_epochs": epochs,
            "f1_score": 0.0,
            "accuracy": 0.0,
        })

    # Create trainer and train
    trainer = MultiLabelTrainer(trainer_config, verbose=False)

    try:
        result = trainer.train_single_model(
            label_name=category_name,
            train_samples=train_samples,
            val_samples=val_samples,
            num_labels=num_labels,
            class_names=class_names,
            session_id=session_id,
            progress_callback=progress_callback,
            suppress_display=True,
            model_output_dir=model_dir,
            training_approach=training_approach,
            reinforced_learning=reinforced_learning,
            reinforced_epochs=reinforced_epochs,
            rl_f1_threshold=rl_f1_threshold,
            rl_oversample_factor=rl_oversample_factor,
            rl_class_weight_factor=rl_class_weight_factor,
        )

        # Extract metrics
        perf = result.performance_metrics if hasattr(result, 'performance_metrics') else {}
        elapsed = time.time() - start_time

        results = {
            "task_id": task_id,
            "category_name": category_name,
            "status": "success",
            "f1_score": perf.get('f1_macro', 0.0),
            "accuracy": perf.get('accuracy', 0.0),
            "best_epoch": perf.get('best_epoch', 0),
            "elapsed_seconds": elapsed,
            "model_path": model_dir,
            "per_class_f1": perf.get('per_class_f1', {}),
            "class_names": class_names,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
        }

        # Write final progress
        if progress_path:
            write_progress(progress_path, {
                "task_id": task_id,
                "category_name": category_name,
                "status": "completed",
                "current_epoch": epochs,
                "total_epochs": epochs,
                "f1_score": results["f1_score"],
                "accuracy": results["accuracy"],
                "elapsed_seconds": elapsed,
            })

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"Training failed for {category_name}: {error_msg}")

        # Write error progress
        if progress_path:
            write_progress(progress_path, {
                "task_id": task_id,
                "category_name": category_name,
                "status": "failed",
                "error": error_msg,
                "elapsed_seconds": elapsed,
            })

        return {
            "task_id": task_id,
            "category_name": category_name,
            "status": "error",
            "error": error_msg,
            "elapsed_seconds": elapsed,
        }

    finally:
        # Clean up memory
        del trainer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    """
    Main entry point for the remote training worker.

    Usage:
        python -m llm_tool.trainers.ssh_training_worker --config /path/to/task_config.json
    """
    parser = argparse.ArgumentParser(
        description="LLM_Tool Remote Training Worker - trains a single model for SSH distributed training"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the task_config.json file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"SSH Training Worker starting with config: {args.config}")

    # Load config
    try:
        task_config = load_task_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load task config: {e}")
        sys.exit(1)

    logger.info(f"Training task: {task_config['category_name']} with model {task_config['model_name']}")

    # Run training
    results = run_training(task_config)

    # Write results to output directory
    output_path = task_config['output_path']
    results_path = os.path.join(output_path, "results.json")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    if results.get("status") == "success":
        logger.info(
            f"Training completed successfully for {task_config['category_name']}: "
            f"F1={results.get('f1_score', 0):.4f}, "
            f"Acc={results.get('accuracy', 0):.4f}, "
            f"Time={results.get('elapsed_seconds', 0):.1f}s"
        )
        sys.exit(0)
    else:
        logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
