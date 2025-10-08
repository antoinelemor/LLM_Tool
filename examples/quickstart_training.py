#!/usr/bin/env python3
"""
Quick Start Example: Model Training

This example demonstrates how to train a BERT model for text classification
using LLM Tool.

Prerequisites:
- LLM Tool installed (pip install -e .)
- Training data in CSV/JSON/JSONL format with 'text' and 'label' columns
- GPU (optional but recommended for faster training)

Author: Antoine Lemor
"""

from llm_tool.trainers.model_trainer import ModelTrainer
from llm_tool.utils.training_data_utils import prepare_training_data
import pandas as pd
import sys


def create_sample_dataset():
    """Create a sample sentiment dataset for demonstration."""
    data = {
        "text": [
            "This is absolutely fantastic! I love it!",
            "Best product ever, highly recommend!",
            "Amazing quality, exceeded expectations!",
            "Terrible, waste of money.",
            "Very disappointed, poor quality.",
            "Do not buy this, awful experience.",
            "It's okay, nothing special.",
            "Average product, meets basic needs.",
            "Neither good nor bad, just meh.",
            "Outstanding service and quality!",
            "Worst purchase I've ever made.",
            "Decent for the price.",
        ] * 50,  # Replicate for larger dataset
        "label": [
            "positive", "positive", "positive",
            "negative", "negative", "negative",
            "neutral", "neutral", "neutral",
            "positive", "negative", "neutral"
        ] * 50
    }
    return pd.DataFrame(data)


def main():
    print("=" * 70)
    print("LLM TOOL - Quick Start: BERT Model Training")
    print("=" * 70)
    print()

    # Step 1: Create or load dataset
    print("Step 1: Loading dataset...")
    print("-" * 70)

    # Option A: Use sample dataset
    df = create_sample_dataset()
    print(f"✓ Sample dataset created: {len(df)} examples")

    # Option B: Load your own dataset (uncomment to use)
    # df = pd.read_csv("your_dataset.csv")
    # df = pd.read_json("your_dataset.jsonl", lines=True)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"    - {label}: {count} ({count/len(df)*100:.1f}%)")
    print()

    # Step 2: Prepare training data
    print("Step 2: Preparing training data...")
    print("-" * 70)

    train_data, val_data, test_data = prepare_training_data(
        df=df,
        text_column="text",
        label_column="label",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True  # Maintain label distribution across splits
    )

    print(f"✓ Data split completed:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print()

    # Step 3: Initialize trainer
    print("Step 3: Initializing model trainer...")
    print("-" * 70)

    trainer = ModelTrainer(
        model_name="bert-base-uncased",  # You can change this to any supported model
        num_labels=3,  # Number of unique labels
        output_dir="./models/sentiment_classifier",
        logging_dir="./logs/training"
    )

    print(f"✓ Trainer initialized")
    print(f"  Model: bert-base-uncased")
    print(f"  Device: {trainer.device}")
    print(f"  Output directory: ./models/sentiment_classifier")
    print()

    # Step 4: Configure training
    print("Step 4: Configuring training parameters...")
    print("-" * 70)

    training_config = {
        "num_epochs": 3,  # Quick training for demo (use 10+ for production)
        "batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "early_stopping_patience": 2,
        "save_strategy": "best",  # Save only the best model
    }

    print("✓ Training configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print()

    # Step 5: Train model
    print("Step 5: Training model...")
    print("-" * 70)
    print("This may take several minutes depending on your hardware.")
    print("GPU is recommended for faster training.")
    print()

    try:
        results = trainer.train(
            train_data=train_data,
            val_data=val_data,
            **training_config
        )

        print()
        print("-" * 70)
        print("✓ Training completed successfully!")
        print()

        # Step 6: Display results
        print("Step 6: Training Results")
        print("-" * 70)
        print(f"Best validation F1 score: {results['best_f1']:.4f}")
        print(f"Best validation accuracy: {results['best_accuracy']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Best model saved at: {results['model_path']}")
        print()

        # Step 7: Evaluate on test set
        print("Step 7: Evaluating on test set...")
        print("-" * 70)

        test_results = trainer.evaluate(test_data)

        print("✓ Test set results:")
        print(f"  F1 Score: {test_results['f1_score']:.4f}")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall: {test_results['recall']:.4f}")
        print()

        # Display confusion matrix
        if 'confusion_matrix' in test_results:
            print("Confusion Matrix:")
            print(test_results['confusion_matrix'])
            print()

        print("=" * 70)
        print("Training complete! 🎉")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Find your trained model at: ./models/sentiment_classifier")
        print("  2. Use it for inference with BERT Annotation Studio")
        print("  3. Or load it programmatically:")
        print("     from llm_tool.trainers.model_trainer import ModelTrainer")
        print("     trainer = ModelTrainer.load('./models/sentiment_classifier')")
        print("     predictions = trainer.predict(['Your text here'])")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
