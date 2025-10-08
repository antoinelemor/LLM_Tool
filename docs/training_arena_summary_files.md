# Training Arena Summary Files Documentation

## Overview

The Training Arena generates comprehensive summary files at the end of each training cycle to ensure full reproducibility and facilitate analysis of training results. These files capture every important detail about the training process, from dataset characteristics to model performance metrics.

## Generated Files

After each complete training cycle, the system generates the following summary files in the session directory (`logs/training_arena/{session_id}/`):

### 1. Standard Summary Files (Always Generated)
- **`training_summary_final.csv`** - Comprehensive CSV with all training metrics
- **`training_summary_final.jsonl`** - Detailed JSONL with complete session data

### 2. Versioned Summary Files (For Tracking)
- **`{session_id}_training_summary_final.csv`** - Versioned CSV for historical tracking
- **`{session_id}_training_summary_final.jsonl`** - Versioned JSONL for historical tracking

## File Contents

### CSV Summary (`training_summary_final.csv`)

The CSV file contains a tabular view of all training metrics with the following columns:

#### Model Information
- `model_name` - Name of the trained model
- `model_identifier` - Internal model identifier
- `model_type` - Type of model architecture

#### Training Metrics
- `epoch` - Epoch number
- `train_loss` - Training loss
- `val_loss` - Validation loss
- `accuracy` - Overall accuracy
- `macro_f1` - Macro-averaged F1 score

#### Per-Class Metrics
- `precision_[class]` - Precision for each class
- `recall_[class]` - Recall for each class
- `f1_[class]` - F1 score for each class
- `support_[class]` - Number of samples for each class

#### Language-Specific Metrics
- `[LANG]_accuracy` - Accuracy per language (e.g., EN_accuracy, FR_accuracy)
- `[LANG]_precision_[class]` - Language-specific precision
- `[LANG]_recall_[class]` - Language-specific recall
- `[LANG]_f1_[class]` - Language-specific F1 score

#### Session Information
- `session_id` - Unique training session identifier
- `training_mode` - Training mode used (benchmark/quick/custom)
- `timestamp` - When the training occurred
- `session_start_time` - Training start timestamp
- `session_end_time` - Training end timestamp
- `total_duration_seconds` - Total training time

#### Hyperparameters
- `learning_rate` - Learning rate used
- `batch_size` - Batch size
- `warmup_ratio` - Warmup ratio
- `weight_decay` - Weight decay
- `reinforced_learning_enabled` - Whether reinforced learning was used
- `rl_f1_threshold` - F1 threshold for reinforced learning
- `rl_epochs` - Number of reinforced learning epochs

### JSONL Summary (`training_summary_final.jsonl`)

The JSONL file contains multiple record types for comprehensive analysis:

#### 1. Session Summary Record
```json
{
  "record_type": "session_summary",
  "session_id": "training_session_20251008_125605",
  "timestamp": "2025-10-08T15:55:22",
  "session_info": {
    "tool_version": "LLMTool v1.0",
    "workflow": "Training Arena - Quick"
  },
  "execution_status": {
    "status": "completed",
    "models_trained": ["bert-base-uncased", "camembert-base"],
    "best_model": "bert-base-uncased",
    "best_f1": 0.85
  },
  "overall_metrics": {
    "total_epochs": 10,
    "models_trained": 2,
    "averages": {
      "avg_train_loss": 0.45,
      "avg_val_loss": 0.52
    }
  },
  "dataset_summary": {
    "total_samples": 5000,
    "total_categories": 3,
    "overall_imbalance_ratio": 1.5,
    "overall_label_distribution": {
      "positive": 2000,
      "neutral": 1800,
      "negative": 1200
    }
  },
  "best_models": [
    {
      "model_name": "bert-base-uncased",
      "combined_score": 0.85,
      "macro_f1": 0.85,
      "accuracy": 0.88
    }
  ],
  "hardware_info": {
    "gpu_available": true,
    "device": "cuda"
  },
  "reproducibility_info": {
    "random_seed": 42,
    "deterministic": true
  }
}
```

#### 2. Dataset Analysis Records
```json
{
  "record_type": "dataset_analysis",
  "session_id": "training_session_20251008_125605",
  "category": "sentiment",
  "statistics": {
    "sample_count": 5000,
    "num_labels": 3,
    "label_distribution": {
      "positive": 2000,
      "neutral": 1800,
      "negative": 1200
    },
    "language_distribution": {
      "en": 3000,
      "fr": 2000
    },
    "text_length_stats": {
      "mean": 150.5,
      "median": 125,
      "std": 45.2,
      "min": 10,
      "max": 512
    },
    "imbalance_ratio": 1.67
  }
}
```

#### 3. Model Performance Records
```json
{
  "record_type": "model_performance",
  "session_id": "training_session_20251008_125605",
  "model": {
    "model_name": "bert-base-uncased",
    "combined_score": 0.85,
    "macro_f1": 0.85,
    "accuracy": 0.88
  }
}
```

#### 4. Epoch Metrics Records
```json
{
  "record_type": "epoch_metrics",
  "session_id": "training_session_20251008_125605",
  "epoch_data": {
    "model_name": "bert-base-uncased",
    "epoch": 1,
    "train_loss": 0.65,
    "val_loss": 0.70,
    "accuracy": 0.75,
    "macro_f1": 0.72
  }
}
```

## Usage Examples

### Loading CSV Summary in Python
```python
import pandas as pd

# Load the CSV summary
df = pd.read_csv('logs/training_arena/session_id/training_summary_final.csv')

# Get best performing model
best_model = df.loc[df['macro_f1'].idxmax()]
print(f"Best model: {best_model['model_name']} with F1: {best_model['macro_f1']}")

# Analyze performance by language
en_metrics = df[['model_name', 'EN_accuracy', 'EN_macro_f1']].dropna()
fr_metrics = df[['model_name', 'FR_accuracy', 'FR_macro_f1']].dropna()
```

### Loading JSONL Summary in Python
```python
import json

# Load JSONL records
records = []
with open('logs/training_arena/session_id/training_summary_final.jsonl', 'r') as f:
    for line in f:
        records.append(json.loads(line))

# Get session summary
session_summary = next(r for r in records if r['record_type'] == 'session_summary')
print(f"Training completed: {session_summary['execution_status']['status']}")
print(f"Best F1: {session_summary['execution_status']['best_f1']}")

# Get all epoch metrics
epoch_metrics = [r for r in records if r['record_type'] == 'epoch_metrics']
```

## Reproducibility

The summary files contain all information necessary to reproduce the training:

1. **Dataset Information**
   - File paths and names
   - Sample counts and distributions
   - Train/validation/test splits
   - Language distributions

2. **Model Configuration**
   - Selected models
   - Hyperparameters
   - Training mode
   - Reinforced learning settings

3. **Environment Information**
   - Random seeds
   - Hardware used
   - Software versions

4. **Complete Metrics**
   - Per-epoch performance
   - Per-class metrics
   - Per-language metrics
   - Training and validation losses

## Best Practices

1. **Always Save Summaries**: The system automatically generates these files, but ensure the training completes properly.

2. **Version Control**: The versioned files (with session_id prefix) help track multiple experiments.

3. **Backup Important Results**: Copy summary files to a backup location for important experiments.

4. **Use for Comparison**: Load multiple summary files to compare different training runs.

5. **Share for Collaboration**: These files contain everything needed for collaborators to understand and reproduce your results.

## Troubleshooting

### Missing Summary Files
If summary files are not generated:
1. Check if training completed successfully
2. Verify the session directory exists
3. Check logs for any errors during summary generation

### Incomplete Data
If summary files have missing data:
1. Some metrics may not be available for all training modes
2. Dataset analysis requires training data files to be present
3. Check training logs for any warnings

### Large File Sizes
For very long training sessions:
1. CSV files may become large with many epochs
2. JSONL files include per-epoch records which can accumulate
3. Consider archiving older summary files

## Related Documentation

- [Training Arena Guide](training_arena_guide.md)
- [Metadata Persistence](training_arena_metadata_persistence.md)
- [Benchmark Mode Documentation](benchmark_mode.md)