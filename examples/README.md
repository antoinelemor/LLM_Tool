# LLM Tool - Usage Examples

This directory contains example scripts demonstrating how to use LLM Tool programmatically.

## Available Examples

### 1. `quickstart_annotation.py` - LLM Annotation with Ollama

Demonstrates zero-shot annotation using a local Ollama model for sentiment analysis.

**Prerequisites:**
- Ollama installed: `curl -fsSL https://ollama.ai/install.sh | sh`
- Model pulled: `ollama pull llama3.2`

**Run:**
```bash
python examples/quickstart_annotation.py
```

**What it does:**
- Annotates 5 sample texts for sentiment (positive/negative/neutral)
- Uses Ollama's llama3.2 model locally (no API keys needed)
- Saves results to `annotation_results.json`
- Displays sentiment distribution

**Expected output:**
```
==================================================================
LLM TOOL - Quick Start: Sentiment Analysis with Ollama
==================================================================

Initializing Ollama client...
✓ Ollama connected successfully

Annotating texts...
------------------------------------------------------------------

[1/5] Processing: This product is amazing! Best purchase I've made...
    → Sentiment: POSITIVE (confidence: 0.95)

[2/5] Processing: Terrible experience, would not recommend to anyon...
    → Sentiment: NEGATIVE (confidence: 0.92)

...

Summary: 5/5 texts annotated successfully
✓ Results saved to: annotation_results.json

Sentiment Distribution:
  Positive: ████████████ 2 (40.0%)
  Negative: ████████████ 2 (40.0%)
  Neutral : ██████       1 (20.0%)
```

---

### 2. `quickstart_training.py` - BERT Model Training

Demonstrates training a BERT model for text classification.

**Prerequisites:**
- LLM Tool installed with training dependencies
- (Optional) GPU for faster training

**Run:**
```bash
python examples/quickstart_training.py
```

**What it does:**
- Creates a sample sentiment dataset (600 examples)
- Splits data into train/val/test sets (70/15/15)
- Trains a BERT-base model for 3 epochs
- Evaluates on test set
- Saves trained model to `./models/sentiment_classifier`

**Expected output:**
```
==================================================================
LLM TOOL - Quick Start: BERT Model Training
==================================================================

Step 1: Loading dataset...
------------------------------------------------------------------
✓ Sample dataset created: 600 examples
  Columns: ['text', 'label']
  Label distribution:
    - positive: 200 (33.3%)
    - negative: 200 (33.3%)
    - neutral: 200 (33.3%)

...

Step 5: Training model...
------------------------------------------------------------------
Epoch 1/3: 100%|████████████████| 26/26 [00:45<00:00]
Validation F1: 0.8234

...

✓ Training completed successfully!

Step 6: Training Results
------------------------------------------------------------------
Best validation F1 score: 0.8542
Best validation accuracy: 0.8556
Training time: 142.35 seconds
Best model saved at: ./models/sentiment_classifier/best_model

Step 7: Evaluating on test set...
------------------------------------------------------------------
✓ Test set results:
  F1 Score: 0.8467
  Accuracy: 0.8444
  Precision: 0.8523
  Recall: 0.8412
```

---

### 3. `system_resources_demo.py` - System Resource Monitoring

Demonstrates how to monitor system resources (GPU, CPU, RAM) during operations.

**Run:**
```bash
python examples/system_resources_demo.py
```

**What it does:**
- Detects available hardware (CUDA, MPS, CPU)
- Displays GPU/CPU/RAM information
- Monitors resource usage in real-time

---

## Using Examples as Templates

These examples serve as templates for your own projects. Key patterns:

### Pattern 1: Programmatic Annotation

```python
from llm_tool.annotators.llm_annotator import LLMAnnotator
from llm_tool.annotators.local_models import OllamaClient

# Initialize client
client = OllamaClient(model="llama3.2")

# Define schema
schema = {
    "category": {
        "type": "string",
        "enum": ["tech", "sports", "politics"],
        "description": "Article category"
    }
}

# Create annotator
annotator = LLMAnnotator(llm_client=client, schema=schema)

# Annotate
result = annotator.annotate("Text to annotate...")
print(result)  # {"category": "tech"}
```

### Pattern 2: Programmatic Training

```python
from llm_tool.trainers.model_trainer import ModelTrainer
import pandas as pd

# Load data
df = pd.read_csv("training_data.csv")

# Initialize trainer
trainer = ModelTrainer(
    model_name="bert-base-uncased",
    num_labels=3,
    output_dir="./my_model"
)

# Train
results = trainer.train(
    train_data=df,
    num_epochs=10,
    batch_size=16
)

# Use model
predictions = trainer.predict(["New text to classify"])
```

### Pattern 3: Batch Processing

```python
from llm_tool.annotators.llm_annotator import LLMAnnotator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Load large dataset
df = pd.read_csv("large_dataset.csv")

# Initialize annotator
annotator = LLMAnnotator(...)

# Parallel annotation
def annotate_row(text):
    return annotator.annotate(text)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(annotate_row, df['text']))

df['annotations'] = results
df.to_csv("annotated_dataset.csv", index=False)
```

---

## Tips for Production Use

### 1. Error Handling

Always wrap annotation/training in try-except:

```python
try:
    result = annotator.annotate(text)
except Exception as e:
    print(f"Error: {e}")
    result = None
```

### 2. Progress Tracking

Use tqdm for batch operations:

```python
from tqdm import tqdm

for text in tqdm(texts, desc="Annotating"):
    result = annotator.annotate(text)
    results.append(result)
```

### 3. Incremental Saving

Save results periodically to avoid data loss:

```python
results = []
for i, text in enumerate(texts):
    result = annotator.annotate(text)
    results.append(result)

    # Save every 100 items
    if (i + 1) % 100 == 0:
        pd.DataFrame(results).to_csv(f"checkpoint_{i+1}.csv")
```

### 4. Resource Management

Monitor memory for large datasets:

```python
from llm_tool.utils.system_resources import get_memory_usage

if get_memory_usage() > 0.9:  # >90% RAM used
    # Reduce batch size or process in chunks
    pass
```

---

## Need More Help?

- **Main README**: See [../README.md](../README.md) for CLI usage
- **Documentation**: Check [../docs/](../docs/) for detailed guides
- **Issues**: Report bugs at GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions

---

**Happy coding! 🚀**
