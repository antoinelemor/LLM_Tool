#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
reinforced_params.py

MAIN OBJECTIVE:
---------------
Derive reinforced training hyperparameters based on prior model performance
so low-performing classes receive corrective optimisation.

Dependencies:
-------------
- builtins

MAIN FEATURES:
--------------
1) Compute learning rate, class weights, and epoch adjustments per scenario
2) Apply model-specific tweaks for architectures such as XLM-R, ALBERT, or DeBERTa
3) Support multi-class weighting using per-class F1 feedback
4) Expose early stopping recommendations tuned for reinforcement runs
5) Suggest additional techniques (focal loss, mixup, scheduling) when needed

Author:
-------
Antoine Lemor
"""

def get_reinforced_params(model_name: str, best_f1_1: float, original_lr: float = 5e-5,
                          num_classes: int = 2, class_f1_scores: list = None,
                          class_pos_ratios: list = None):
    """
    Get intelligent reinforced training parameters based on model, performance,
    and optionally the actual data distribution.

    Args:
        model_name: Name of the model
        best_f1_1: Best F1 score for class 1 after normal training (for binary) or worst F1 (for multi-class)
        original_lr: Original learning rate used in normal training
        num_classes: Number of classes (2 for binary, >2 for multi-class)
        class_f1_scores: List of F1 scores for each class (for multi-class)
        class_pos_ratios: List of per-class frequency ratios from training data (e.g., [0.85, 0.15] for binary).
                          When provided, class weights are derived from actual data distribution
                          combined with F1 performance, instead of arbitrary F1-only thresholds.

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

    # Override class_1_weight with distribution-aware weight when data ratios are available
    if class_pos_ratios is not None and num_classes == 2:
        pos_ratio = max(class_pos_ratios[1], 1e-6)
        base_weight = min((1.0 - pos_ratio) / pos_ratio, 100.0)
        # Scale by F1 severity: bad F1 on a rare class needs more correction
        if best_f1_1 == 0.0:
            severity = 1.5
        elif best_f1_1 < 0.3:
            severity = 1.3
        elif best_f1_1 < 0.5:
            severity = 1.1
        else:
            severity = 1.0
        params['class_1_weight'] = min(base_weight * severity, 100.0)

    # Calculate class weights for multi-class
    if num_classes > 2 and class_f1_scores is not None:
        if class_pos_ratios is not None:
            # Distribution-aware: combine frequency ratio with F1 severity
            mean_ratio = sum(class_pos_ratios) / len(class_pos_ratios)
            class_weights = []
            for i, f1 in enumerate(class_f1_scores):
                ratio_i = max(class_pos_ratios[i], 1e-6)
                distribution_factor = mean_ratio / ratio_i  # rare classes get higher weight
                if f1 == 0.0:
                    f1_severity = 2.0
                elif f1 < 0.3:
                    f1_severity = 1.5
                elif f1 < 0.5:
                    f1_severity = 1.2
                else:
                    f1_severity = 1.0
                w = min(max(distribution_factor * f1_severity, 0.5), 100.0)
                class_weights.append(w)
        else:
            # Legacy: F1-only weights (no distribution info)
            class_weights = []
            for f1 in class_f1_scores:
                if f1 == 0.0:
                    class_weights.append(params['class_1_weight'])
                elif f1 < 0.3:
                    class_weights.append(params['class_1_weight'] * 0.8)
                elif f1 < 0.5:
                    class_weights.append(params['class_1_weight'] * 0.6)
                else:
                    class_weights.append(1.0)

        params['class_weights'] = class_weights
    else:
        # Binary classification: keep existing class_1_weight
        params['class_weights'] = None

    return params


def get_multi_label_reinforced_params(
    model_name: str,
    label_f1_scores: list,
    label_pos_ratios: list,
    label_names: list = None,
    original_lr: float = 5e-5,
    reinforced_f1_threshold: float = 0.7,
    max_pos_weight: float = 100.0,
):
    """
    Derive per-label reinforced training parameters for multi-label classification.

    Unlike get_reinforced_params() which treats all labels uniformly, this function
    analyses each label independently based on its F1 score AND positive ratio,
    then produces per-label strategies (gamma, pos_weight, freeze mask).

    Parameters
    ----------
    model_name : str
        Model name for architecture-specific adjustments
    label_f1_scores : list of float
        Per-label F1 scores from initial training
    label_pos_ratios : list of float
        Positive ratio per label (e.g., 0.053 for 5.3% positive)
    label_names : list of str, optional
        Label names for logging
    original_lr : float
        Learning rate from initial training
    reinforced_f1_threshold : float
        F1 threshold below which a label is considered underperforming
    max_pos_weight : float
        Cap for pos_weight to prevent training instability (default: 100)

    Returns
    -------
    dict with keys:
        underperforming_labels, per_label_gamma_neg, per_label_pos_weight,
        freeze_mask, n_epochs, learning_rate, should_use_asl, label_tiers
    """
    import math

    num_labels = len(label_f1_scores)
    if label_names is None:
        label_names = [str(i) for i in range(num_labels)]

    underperforming = []
    per_label_gamma_neg = []
    per_label_pos_weight = []
    freeze_mask = []
    label_tiers = {0: [], 1: [], 2: [], 3: []}

    for i in range(num_labels):
        f1 = label_f1_scores[i]
        pos_ratio = max(label_pos_ratios[i], 1e-6)

        # Base gamma from label rarity: rarer labels need stronger negative down-weighting
        gamma = min(max(4.0 + math.log10(1.0 / pos_ratio), 4.0), 8.0)

        # pos_weight = neg/pos, capped
        pw = min((1.0 - pos_ratio) / pos_ratio, max_pos_weight)

        if f1 >= reinforced_f1_threshold:
            # Tier 0: label is performing well — freeze during RL
            label_tiers[0].append(i)
            freeze_mask.append(True)
        elif f1 < 0.2 or (f1 == 0.0 and pos_ratio < 0.01):
            # Tier 3: critical — F1 near zero on a rare label
            label_tiers[3].append(i)
            underperforming.append(i)
            freeze_mask.append(False)
            gamma = min(gamma + 1.0, 8.0)  # boost gamma further
        elif pos_ratio < 0.05:
            # Tier 2: severe — underperforming AND rare
            label_tiers[2].append(i)
            underperforming.append(i)
            freeze_mask.append(False)
            gamma = min(gamma + 0.5, 8.0)
        else:
            # Tier 1: mild — underperforming but not rare
            label_tiers[1].append(i)
            underperforming.append(i)
            freeze_mask.append(False)

        per_label_gamma_neg.append(gamma)
        per_label_pos_weight.append(pw)

    # Epochs determined by worst tier present
    if label_tiers[3]:
        n_epochs = 15
    elif label_tiers[2]:
        n_epochs = 10
    elif label_tiers[1]:
        n_epochs = 8
    else:
        n_epochs = 0

    # Learning rate adjusted by average severity of underperforming labels
    if underperforming:
        avg_f1_under = sum(label_f1_scores[i] for i in underperforming) / len(underperforming)
        if avg_f1_under == 0.0:
            lr_mult = 1.5
        elif avg_f1_under < 0.3:
            lr_mult = 1.3
        else:
            lr_mult = 1.2
    else:
        lr_mult = 1.0
    learning_rate = original_lr * lr_mult

    # Model-specific adjustments
    model_lower = model_name.lower()
    if 'xlm' in model_lower and 'roberta' in model_lower:
        n_epochs = min(20, int(n_epochs * 1.3))
    elif 'albert' in model_lower:
        n_epochs = min(20, n_epochs * 2)
        learning_rate *= 0.7

    # Use ASL when any label has < 10% positive ratio
    should_use_asl = any(r < 0.1 for r in label_pos_ratios)

    return {
        'underperforming_labels': underperforming,
        'per_label_gamma_neg': per_label_gamma_neg,
        'per_label_pos_weight': per_label_pos_weight,
        'freeze_mask': freeze_mask,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'should_use_asl': should_use_asl,
        'label_tiers': label_tiers,
        'label_names': label_names,
    }


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
