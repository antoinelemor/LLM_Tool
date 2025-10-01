"""
Smart Reinforced Training Parameters
Adapts reinforced training parameters based on model characteristics and failure mode
"""

def get_reinforced_params(model_name: str, best_f1_1: float, original_lr: float = 5e-5):
    """
    Get intelligent reinforced training parameters based on model and performance

    Args:
        model_name: Name of the model
        best_f1_1: Best F1 score for class 1 after normal training
        original_lr: Original learning rate used in normal training

    Returns:
        dict: Reinforced training parameters
    """

    # More balanced parameters to avoid overcorrection
    if best_f1_1 == 0.0:
        # Model ignores class 1 - needs careful rebalancing, not overcorrection
        params = {
            'learning_rate': original_lr * 1.5,  # Moderate increase to avoid instability
            'class_1_weight': 2.5,  # Balanced weight to avoid flipping all predictions
            'batch_size_multiplier': 0.5,  # Smaller batches for more updates
            'n_epochs': 15,  # More epochs needed for recovery
            'warmup_ratio': 0.2,  # Gradual warmup to avoid instability
            'gradient_accumulation': 2,  # Some accumulation for stability
            'label_smoothing': 0.05,  # Light smoothing to prevent overconfidence
            'dropout_increase': 0.05,  # Light dropout to prevent overfitting
        }
    elif best_f1_1 < 0.3:
        # Model has some signal but very poor - needs moderate intervention
        params = {
            'learning_rate': original_lr * 1.25,
            'class_1_weight': 2.0,
            'batch_size_multiplier': 0.5,
            'n_epochs': 10,
            'warmup_ratio': 0.1,
            'gradient_accumulation': 2,
            'label_smoothing': 0.03,
            'dropout_increase': 0.03,
        }
    else:
        # Model is below threshold but not catastrophic - light intervention
        params = {
            'learning_rate': original_lr * 1.2,
            'class_1_weight': 1.5,
            'batch_size_multiplier': 0.75,
            'n_epochs': 8,
            'warmup_ratio': 0.1,
            'gradient_accumulation': 1,
            'label_smoothing': 0.0,
            'dropout_increase': 0.0,
        }

    # Model-specific adjustments
    model_lower = model_name.lower()

    # XLM-RoBERTa: Known to struggle with imbalanced data
    if 'xlm' in model_lower and 'roberta' in model_lower:
        params['learning_rate'] *= 1.1  # Slight increase, avoid instability
        params['class_1_weight'] *= 1.2  # Moderate increase to avoid overcorrection
        params['n_epochs'] = min(20, int(params['n_epochs'] * 1.3))  # More epochs
        params['use_focal_loss'] = True  # Use focal loss for hard examples
        params['focal_gamma'] = 2.0

    # ALBERT: Parameter sharing needs more iterations
    elif 'albert' in model_lower:
        params['n_epochs'] = min(20, int(params['n_epochs'] * 2))  # Double epochs
        params['learning_rate'] *= 0.7  # More conservative LR
        params['gradient_accumulation'] *= 2  # More accumulation for stability

    # Large models: Risk of overfitting
    elif 'large' in model_lower or 'xlarge' in model_lower:
        params['dropout_increase'] += 0.05  # More dropout
        params['learning_rate'] *= 0.8  # Slightly lower LR
        params['weight_decay'] = 0.1  # Add weight decay

    # Longformer/BigBird: Designed for long sequences, struggle with short
    elif 'longformer' in model_lower or 'bigbird' in model_lower:
        params['batch_size_multiplier'] = 0.125  # Very small batches
        params['learning_rate'] *= 2.0  # Much higher LR
        params['position_weight_decay'] = True  # Special handling for position embeddings

    # DeBERTa: Usually performs well, so if it fails, data might be the issue
    elif 'deberta' in model_lower:
        params['augmentation'] = True  # Enable data augmentation
        params['augmentation_prob'] = 0.2
        params['mixup_alpha'] = 0.2  # Use mixup for better generalization

    return params


def get_early_stopping_params(model_name: str):
    """
    Get early stopping parameters for reinforced training

    Args:
        model_name: Name of the model

    Returns:
        dict: Early stopping parameters
    """
    return {
        'patience': 3,  # Stop if no improvement for 3 epochs
        'min_delta': 0.01,  # Minimum improvement to reset patience
        'monitor': 'f1_1',  # Monitor F1 for class 1
        'mode': 'max',  # Higher is better
        'restore_best': True,  # Restore best model at end
    }


def should_use_advanced_techniques(best_f1_1: float):
    """
    Determine if advanced techniques should be used

    Args:
        best_f1_1: Best F1 score for class 1

    Returns:
        dict: Advanced techniques to use
    """
    techniques = {
        'use_mixup': False,
        'use_focal_loss': False,
        'use_label_smoothing': False,
        'use_gradient_clipping': True,  # Always use
        'use_cosine_schedule': False,
        'use_adversarial_training': False,
    }

    if best_f1_1 == 0.0:
        # Complete failure - use everything
        techniques.update({
            'use_focal_loss': True,
            'use_label_smoothing': True,
            'use_cosine_schedule': True,
        })
    elif best_f1_1 < 0.2:
        # Very poor - use some advanced techniques
        techniques.update({
            'use_focal_loss': True,
            'use_cosine_schedule': True,
        })
    elif best_f1_1 < 0.4:
        # Poor - use light techniques
        techniques.update({
            'use_label_smoothing': True,
        })

    return techniques