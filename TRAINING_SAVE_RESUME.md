# Training Session Save/Resume System (Mode 3)

## 🎯 Overview

LLMTool's **Training Studio (Mode 3)** now includes a sophisticated **save and resume system**, identical to the annotation modes (Modes 1 & 2). This feature ensures **scientific reproducibility** and allows you to **resume interrupted training sessions**.

---

## ✨ Key Features

- **📋 Automatic Metadata Saving**: All training parameters (dataset, model, hyperparameters) saved to JSON
- **🔄 Resume/Relaunch**: Continue interrupted training or relaunch with same parameters
- **📊 Session History**: Browse and select from 20 most recent training sessions
- **✅ Status Tracking**: Track training status (pending, running, completed, failed)
- **🎯 Reproducibility**: Document exact configurations for scientific papers
- **📁 Organized Storage**: All sessions stored in `logs/training_sessions/`

---

## 🚀 How to Use

### 1. Starting Training Studio

When you enter **Mode 3 (Training Studio)**, you'll see a new menu:

```
🎯 Training Session Options

┌────────┬──────────────────────────────────────────────────────────────┐
│ Option │ Description                                                  │
├────────┼──────────────────────────────────────────────────────────────┤
│ 1      │ 🔄 Resume/Relaunch Training                                  │
│        │    Load saved parameters from previous training sessions     │
├────────┼──────────────────────────────────────────────────────────────┤
│ 2      │ 🆕 New Training Session                                      │
│        │    Start fresh with dataset selection and configuration      │
├────────┼──────────────────────────────────────────────────────────────┤
│ 3      │ ← Back to Main Menu                                          │
└────────┴──────────────────────────────────────────────────────────────┘

Select an option [1/2/3]:
```

**Options:**
- **1 - Resume/Relaunch Training**: Browse and load previous training sessions
- **2 - New Training Session**: Start a fresh training with full configuration
- **3 - Back**: Return to main menu

---

### 2. Saving Training Parameters (New Sessions)

Before training starts, you'll be asked to save metadata:

```
📋 Reproducibility & Metadata
  1. Resume Capability
     • Save parameters to resume later if interrupted
     • Access via 'Resume/Relaunch Training' option

  2. Scientific Reproducibility
     • Document exact training configuration
     • Track model, dataset, and hyperparameters
     • Share configurations with collaborators

  ⚠️  If you choose NO:
     • You CANNOT resume this training later
     • Parameters will not be saved for future reference

Save training parameters to JSON? [Y/n]:
```

**Recommendation**: Always select **Yes** (default) to enable:
- Resume capability if training is interrupted
- Scientific reproducibility for research papers
- Configuration sharing with collaborators
- Training history tracking

---

### 3. Resume/Relaunch Previous Sessions

Select **Option 1** from the main menu to see all previous sessions:

```
📚 Previous Training Sessions (20 most recent)

┌───┬──────────────────┬──────────┬───────────────────────┬────────────────┬──────────┐
│ # │ Date             │ Mode     │ Dataset               │ Model          │ Status   │
├───┼──────────────────┼──────────┼───────────────────────┼────────────────┼──────────┤
│ 1 │ 2025-10-05 23:45 │ quick    │ annotated_data.csv    │ camembert-base │ ✓ compl… │
│ 2 │ 2025-10-05 22:30 │ benchm…  │ training_data.jsonl   │ xlm-roberta-b… │ ✓ compl… │
│ 3 │ 2025-10-05 20:15 │ custom   │ french_texts.csv      │ flaubert-large │ ⏸ pendin…│
│ 4 │ 2025-10-05 18:00 │ distri…  │ multilabel_data.jsonl │ (multiple)     │ ✓ compl… │
└───┴──────────────────┴──────────┴───────────────────────┴────────────────┴──────────┘

Select session to resume/relaunch [1/2/3/4/back]:
```

**Columns explained:**
- **#**: Session number for selection
- **Date**: When the training session was created
- **Mode**: Training mode (quick, benchmark, custom, distributed)
- **Dataset**: Name of the dataset file used
- **Model**: Model name (or "multiple" for distributed training)
- **Status**: Training status with color coding
  - ✓ completed (green) - Training finished successfully
  - ✗ failed (red) - Training failed with errors
  - ⏸ pending (yellow) - Training not yet completed

---

### 4. Viewing Session Details

After selecting a session, you'll see detailed parameters:

```
📋 Selected Session Details

┌─────────────────────────┬──────────────────────────────────────┐
│ Parameter               │ Value                                │
├─────────────────────────┼──────────────────────────────────────┤
│ Timestamp               │ 20251005_234530                      │
│ Workflow                │ Training Studio - Quick              │
│ Dataset                 │ annotated_data.csv                   │
│ Strategy                │ single-label                         │
│ Total Samples           │ 1000                                 │
│ Training Mode           │ quick                                │
│ Model                   │ camembert-base                       │
│ Epochs                  │ 10                                   │
│ Batch Size              │ 16                                   │
│ Status                  │ completed                            │
└─────────────────────────┴──────────────────────────────────────┘

🎯 Action Mode
  • resume   - Continue incomplete training (if interrupted)
  • relaunch - Start fresh with same parameters

Resume or relaunch? [resume/relaunch]:
```

**Choose:**
- **resume**: Continue from where it left off
  - Use when training was interrupted (power outage, system crash, etc.)
  - Picks up from the last saved checkpoint
  - Only available for incomplete trainings

- **relaunch**: Start completely fresh with exact same parameters
  - Use to reproduce results
  - Use to apply same configuration to new similar data
  - Creates a new training session with identical settings

---

## 📂 Metadata File Structure

All training sessions are saved in `logs/training_sessions/training_metadata_TIMESTAMP.json`:

```json
{
  "training_session": {
    "timestamp": "20251005_234530",
    "tool_version": "LLMTool v1.0",
    "workflow": "Training Studio - Quick",
    "session_id": "train_20251005_234530"
  },
  "dataset_config": {
    "primary_file": "/Users/username/data/annotated_data.csv",
    "format": "llm-json",
    "strategy": "single-label",
    "text_column": "text",
    "label_column": "label",
    "total_samples": 1000,
    "num_categories": 5,
    "category_distribution": {
      "positive": 300,
      "negative": 250,
      "neutral": 450
    },
    "training_files": {
      "single_label": "/path/to/training_file.jsonl"
    }
  },
  "language_config": {
    "confirmed_languages": ["FR", "EN"],
    "language_distribution": {
      "FR": 800,
      "EN": 200
    },
    "model_strategy": "multilingual",
    "language_model_mapping": {}
  },
  "text_analysis": {
    "text_length_stats": {
      "token_mean": 350,
      "token_median": 300,
      "token_max": 1024,
      "token_p95": 800
    },
    "requires_long_document_model": false,
    "avg_token_length": 350,
    "max_token_length": 1024
  },
  "model_config": {
    "training_mode": "quick",
    "selected_model": "camembert-base",
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "early_stopping": true,
    "recommended_model": "camembert-base"
  },
  "execution_status": {
    "status": "completed",
    "started_at": "2025-10-05T23:45:30.123456",
    "completed_at": "2025-10-05T23:55:42.789012",
    "models_trained": ["camembert-base"],
    "best_model": "camembert-base",
    "best_f1": 0.89
  },
  "output_paths": {
    "models_dir": "/Users/username/LLM_Tool/models",
    "logs_dir": "/Users/username/LLM_Tool/logs",
    "results_csv": "/path/to/results.csv"
  }
}
```

### Metadata Sections Explained

1. **training_session**: Session identification
   - `timestamp`: Unique session identifier (YYYYMMdd_HHMMSS)
   - `tool_version`: LLMTool version used
   - `workflow`: Training mode description

2. **dataset_config**: Dataset information
   - `primary_file`: Path to the dataset
   - `strategy`: single-label or multi-label
   - `text_column`: Column containing text data
   - `label_column`: Column containing labels
   - `category_distribution`: Distribution of categories/labels

3. **language_config**: Language detection results
   - `confirmed_languages`: List of detected languages
   - `language_distribution`: Number of samples per language
   - `model_strategy`: multilingual or language-specific

4. **text_analysis**: Text statistics
   - `text_length_stats`: Token length statistics
   - `requires_long_document_model`: Boolean flag for long documents

5. **model_config**: Training parameters
   - `training_mode`: quick, benchmark, custom, or distributed
   - `selected_model`: Name of the model used
   - `epochs`, `batch_size`, `learning_rate`: Hyperparameters

6. **execution_status**: Training execution tracking
   - `status`: pending, running, completed, or failed
   - `started_at`, `completed_at`: ISO timestamps
   - `best_model`, `best_f1`: Best performing model and F1 score

---

## 🎓 Use Cases

### 1. Resume Interrupted Training

**Scenario**: Training was interrupted due to power outage or system crash

**Steps:**
1. Launch Training Studio (Mode 3)
2. Select **Option 1 - Resume/Relaunch Training**
3. Select the interrupted session (status: ⏸ pending)
4. Choose **resume**
5. Training continues from last checkpoint

**Benefits:**
- No need to restart from scratch
- Saves time and computational resources
- All previous progress is preserved

---

### 2. Reproduce Experiments

**Scenario**: Need to reproduce results for a research paper

**Steps:**
1. Original training session saved with metadata
2. Launch Training Studio months later
3. Select the original session
4. Choose **relaunch**
5. Exact same configuration is applied

**Benefits:**
- Perfect reproducibility for scientific papers
- Document exact methodology in Methods section
- Verify results independently

---

### 3. Compare Configurations

**Scenario**: Test different hyperparameters on the same dataset

**Steps:**
1. Train with configuration A (e.g., epochs=5)
2. Train with configuration B (e.g., epochs=10)
3. Train with configuration C (e.g., epochs=15)
4. Review all sessions in history
5. Compare best_f1 scores
6. Relaunch best configuration

**Benefits:**
- Systematic hyperparameter tuning
- Track performance across experiments
- Identify optimal configurations

---

### 4. Apply to New Data

**Scenario**: New dataset similar to previous one

**Steps:**
1. Select successful previous session
2. Choose **relaunch**
3. Before training starts, manually edit dataset path
4. Training uses same model/hyperparameters

**Benefits:**
- Consistent approach across similar datasets
- Save time on configuration
- Ensure comparable results

---

### 5. Share with Collaborators

**Scenario**: Share training configuration with team members

**Steps:**
1. Complete training and save metadata
2. Share the JSON file from `logs/training_sessions/`
3. Collaborator places file in their `logs/training_sessions/`
4. Collaborator selects session and relaunches
5. Identical training is executed

**Benefits:**
- Easy knowledge transfer
- Consistent methodology across team
- No ambiguity in configuration

---

## 🔧 Advanced Features

### Automatic Status Updates

The system automatically updates the execution status:

- **Before training**: `status: "pending"`, `started_at: null`
- **During training**: `status: "running"`, `started_at: "ISO timestamp"`
- **After success**: `status: "completed"`, `completed_at: "ISO timestamp"`, `best_f1: 0.89`
- **After failure**: `status: "failed"`, `error_message: "error details"`

### Session Management

**Location**: All metadata files stored in `logs/training_sessions/`

**Filename format**: `training_metadata_YYYYMMdd_HHMMSS.json`

**Maximum displayed**: 20 most recent sessions (sorted by modification time)

**Manual cleanup**: Delete old JSON files to remove from history

---

## 📊 Comparison with Annotation Modes

This system is **identical** to the save/resume functionality in Modes 1 & 2 (Annotation):

| Feature | Mode 1 & 2 (Annotation) | Mode 3 (Training) |
|---------|-------------------------|-------------------|
| **Metadata JSON** | ✅ Yes | ✅ Yes |
| **Resume/Relaunch Menu** | ✅ Yes | ✅ Yes |
| **Session History Table** | ✅ 20 most recent | ✅ 20 most recent |
| **Status Tracking** | ✅ Yes | ✅ Yes |
| **Rich UI Panels** | ✅ Yes | ✅ Yes |
| **Reproducibility** | ✅ Yes | ✅ Yes |

---

## 🎯 Best Practices

### 1. Always Save Metadata
- Select **Yes** when asked to save parameters
- Enables resume capability and reproducibility
- Negligible storage overhead (~5-10 KB per session)

### 2. Use Descriptive Datasets
- Name datasets clearly (e.g., `political_debates_2024.csv`)
- Helps identify sessions in history table
- Easier to find relevant previous configurations

### 3. Document Results
- Add notes to metadata JSON files if needed
- Track which configurations worked best
- Reference session IDs in research notes

### 4. Clean Old Sessions
- Periodically delete very old metadata files
- Keep relevant sessions for reproducibility
- Archive important configurations separately

### 5. Share Configurations
- Share metadata JSON files with collaborators
- Include in research data repositories
- Attach to paper submissions for reproducibility

---

## ❓ FAQ

**Q: What happens if I delete a metadata file?**
A: The session disappears from the resume menu but doesn't affect trained models.

**Q: Can I edit metadata JSON files manually?**
A: Yes, but be careful with format. Use a JSON validator to avoid errors.

**Q: What if the dataset file was moved/deleted?**
A: The system will show an error when trying to resume. Update the path in JSON.

**Q: How much disk space do metadata files use?**
A: Approximately 5-10 KB per session. 100 sessions = ~500 KB total.

**Q: Can I resume a completed training?**
A: No, resume is only for interrupted trainings. Use "relaunch" for completed sessions.

**Q: What's the difference between resume and relaunch?**
A: Resume continues from checkpoint. Relaunch starts fresh with same parameters.

---

## 🐛 Troubleshooting

### Problem: "No training sessions found"

**Solution**:
- Check `logs/training_sessions/` directory exists
- Ensure you've completed at least one training with metadata saved
- Verify JSON files are present in the directory

### Problem: "Failed to reconstruct bundle from metadata"

**Solution**:
- Check that dataset file still exists at original path
- Verify JSON file is not corrupted (use JSON validator)
- Ensure all required fields are present in JSON

### Problem: Session shows "pending" but training is complete

**Solution**:
- Metadata was not updated after training (possibly interrupted)
- Manually edit JSON: change `"status": "pending"` to `"status": "completed"`

---

## 📝 Technical Implementation

### Functions Added

1. **`_save_training_metadata()`**
   - Saves comprehensive metadata before training
   - Returns path to saved JSON file
   - Located in: `llm_tool/cli/advanced_cli.py:9814`

2. **`_update_training_metadata()`**
   - Updates metadata after training completion
   - Adds execution status and results
   - Located in: `llm_tool/cli/advanced_cli.py:9905`

3. **`_reconstruct_bundle_from_metadata()`**
   - Reconstructs TrainingDataBundle from JSON
   - Validates dataset file existence
   - Located in: `llm_tool/cli/advanced_cli.py:9948`

4. **`_resume_training_studio()`**
   - Main resume/relaunch interface
   - Displays session history with Rich tables
   - Handles session selection and action choice
   - Located in: `llm_tool/cli/advanced_cli.py:10018`

### Modified Functions

1. **`training_studio()`**
   - Added session options menu (Resume/New/Back)
   - Routes to resume or new training flow
   - Located in: `llm_tool/cli/advanced_cli.py:5867`

2. **`_training_studio_confirm_and_execute()`**
   - Added metadata save prompt
   - Saves metadata before training
   - Updates metadata after training
   - Handles exceptions and failure status
   - Located in: `llm_tool/cli/advanced_cli.py:6001`

---

## 🚀 Future Enhancements

Potential future improvements to the save/resume system:

- **Session comparison**: Side-by-side comparison of multiple sessions
- **Export configurations**: Export metadata as shareable templates
- **Import templates**: Import configurations from other users
- **Search and filter**: Search sessions by model, dataset, date, etc.
- **Session notes**: Add custom notes to sessions
- **Performance graphs**: Visualize F1 scores across sessions
- **Auto-cleanup**: Automatic removal of old sessions (configurable)

---

**Version**: 1.0.0
**Last Updated**: 2025-10-05
**Author**: Antoine Lemor
**Repository**: [LLMTool](https://github.com/antoine-lemor/LLMTool)
