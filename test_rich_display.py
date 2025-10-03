"""Test script for Rich Live Display"""
import time
from rich.live import Live
from llm_tool.trainers.bert_base import TrainingDisplay

# Create display
display = TrainingDisplay(
    model_name="bert-base-multilingual",
    label_key="themes",
    label_value="transportation",
    language="MULTI",
    n_epochs=5,
    is_reinforced=False
)

# Simulate training
with Live(display.create_panel(), refresh_per_second=4) as live:
    for epoch in range(1, 6):
        display.current_epoch = epoch
        display.current_phase = "Training"
        display.train_total = 18

        # Simulate training progress
        for step in range(18):
            display.train_progress = step + 1
            display.train_loss = 0.5 - (epoch * 0.05) - (step * 0.01)
            live.update(display.create_panel())
            time.sleep(0.1)

        display.current_phase = "Validation"
        display.train_time = 3.5
        display.val_total = 5

        # Simulate validation progress
        for step in range(5):
            display.val_progress = step + 1
            display.val_loss = 0.3 - (epoch * 0.03)
            live.update(display.create_panel())
            time.sleep(0.05)

        # Update metrics
        display.accuracy = 0.85 + (epoch * 0.02)
        display.precision = [0.88, 0.82]
        display.recall = [0.90, 0.80]
        display.f1_scores = [0.89, 0.81]
        display.f1_macro = 0.85 + (epoch * 0.01)
        display.support = [65, 15]
        display.val_time = 0.5
        display.epoch_time = 4.0

        # Per-language metrics
        display.language_metrics = {
            'EN': {
                'samples': 45,
                'accuracy': 0.83 + (epoch * 0.02),
                'f1_class_0': 0.87,
                'f1_class_1': 0.79,
                'f1_macro': 0.83 + (epoch * 0.01)
            },
            'FR': {
                'samples': 35,
                'accuracy': 0.87 + (epoch * 0.02),
                'f1_class_0': 0.91,
                'f1_class_1': 0.83,
                'f1_macro': 0.87 + (epoch * 0.01)
            }
        }

        # Best model tracking
        if display.f1_macro > display.best_f1:
            display.improvement = display.f1_macro - display.best_f1
            display.best_f1 = display.f1_macro
            display.best_epoch = epoch

        display.total_time = time.time() - display.start_time
        live.update(display.create_panel())
        time.sleep(1)

print("\n✅ Training completed!")
