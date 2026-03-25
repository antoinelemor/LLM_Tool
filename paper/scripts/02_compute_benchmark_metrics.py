"""
02 — Compute benchmark metrics.

Computes F1, precision, recall (micro, macro, weighted) for each model
on each annotation task, compared against the consensus ground truth.
Also computes per-class metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER, RESULTS_DIR


def load_clean_data():
    """Load the cleaned benchmark dataset."""
    path = DATA_DIR / "benchmark_clean.parquet"
    df = pd.read_parquet(path)
    return df


def compute_multiclass_metrics(y_true, y_pred):
    """Compute metrics for single-label classification."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t and p and str(t) != "nan"]
    if not valid:
        return {}
    yt, yp = zip(*valid)
    return {
        "micro_f1": f1_score(yt, yp, average="micro", zero_division=0),
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "accuracy": accuracy_score(yt, yp),
        "micro_precision": precision_score(yt, yp, average="micro", zero_division=0),
        "micro_recall": recall_score(yt, yp, average="micro", zero_division=0),
        "macro_precision": precision_score(yt, yp, average="macro", zero_division=0),
        "macro_recall": recall_score(yt, yp, average="macro", zero_division=0),
        "n_samples": len(yt),
    }


def _to_list(x):
    """Convert numpy arrays or other iterables to list."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def compute_multilabel_metrics(y_true_lists, y_pred_lists):
    """Compute metrics for multi-label classification."""
    # Convert numpy arrays to lists and filter valid entries
    valid = [(_to_list(t), _to_list(p)) for t, p in zip(y_true_lists, y_pred_lists)
             if hasattr(t, '__iter__') and hasattr(p, '__iter__')]
    if not valid:
        return {}
    yt, yp = zip(*valid)

    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in list(yt) + list(yp):
        all_labels.update(labels)
    mlb.fit([sorted(all_labels)])

    yt_bin = mlb.transform(yt)
    yp_bin = mlb.transform(yp)

    return {
        "micro_f1": f1_score(yt_bin, yp_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(yt_bin, yp_bin, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt_bin, yp_bin, average="weighted", zero_division=0),
        "samples_f1": f1_score(yt_bin, yp_bin, average="samples", zero_division=0),
        "micro_precision": precision_score(yt_bin, yp_bin, average="micro", zero_division=0),
        "micro_recall": recall_score(yt_bin, yp_bin, average="micro", zero_division=0),
        "macro_precision": precision_score(yt_bin, yp_bin, average="macro", zero_division=0),
        "macro_recall": recall_score(yt_bin, yp_bin, average="macro", zero_division=0),
        "n_samples": len(yt),
        "n_labels": len(all_labels),
    }


def compute_per_class_metrics_multiclass(y_true, y_pred):
    """Per-class precision, recall, F1 for multiclass."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t and p and str(t) != "nan"]
    if not valid:
        return {}
    yt, yp = zip(*valid)
    report = classification_report(yt, yp, output_dict=True, zero_division=0)
    per_class = {}
    for cls, metrics in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            continue
        per_class[cls] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1-score"],
            "support": metrics["support"],
        }
    return per_class


def compute_per_class_metrics_multilabel(y_true_lists, y_pred_lists):
    """Per-class precision, recall, F1 for multilabel."""
    valid = [(_to_list(t), _to_list(p)) for t, p in zip(y_true_lists, y_pred_lists)
             if hasattr(t, '__iter__') and hasattr(p, '__iter__')]
    if not valid:
        return {}
    yt, yp = zip(*valid)

    mlb = MultiLabelBinarizer()
    all_labels = set()
    for labels in list(yt) + list(yp):
        all_labels.update(labels)
    sorted_labels = sorted(all_labels)
    mlb.fit([sorted_labels])

    yt_bin = mlb.transform(yt)
    yp_bin = mlb.transform(yp)

    per_class = {}
    for i, cls in enumerate(mlb.classes_):
        if yt_bin[:, i].sum() == 0 and yp_bin[:, i].sum() == 0:
            continue
        per_class[cls] = {
            "precision": precision_score(yt_bin[:, i], yp_bin[:, i], zero_division=0),
            "recall": recall_score(yt_bin[:, i], yp_bin[:, i], zero_division=0),
            "f1": f1_score(yt_bin[:, i], yp_bin[:, i], zero_division=0),
            "support": int(yt_bin[:, i].sum()),
            "predicted": int(yp_bin[:, i].sum()),
        }
    return per_class


def main():
    print("Computing benchmark metrics...")
    df = load_clean_data()

    # ── Overall metrics per model × task ──────────────────────────────────
    overall_records = []

    for model_key in MODEL_ORDER:
        for task_key in TASK_ORDER:
            task_cfg = TASKS[task_key]
            gt_col = f"gt_{task_key}"
            pred_col = f"pred_{model_key}_{task_key}"

            if pred_col not in df.columns:
                continue

            y_true = df[gt_col].tolist()
            y_pred = df[pred_col].tolist()

            if task_cfg["type"] == "multiclass":
                metrics = compute_multiclass_metrics(y_true, y_pred)
            else:
                metrics = compute_multilabel_metrics(y_true, y_pred)

            if metrics:
                record = {
                    "model": model_key,
                    "display_name": MODELS[model_key]["display_name"],
                    "short_name": MODELS[model_key]["short_name"],
                    "task": task_key,
                    "task_display": task_cfg["display_name"],
                    **metrics,
                }
                overall_records.append(record)

    overall_df = pd.DataFrame(overall_records)
    overall_df.to_csv(DATA_DIR / "benchmark_overall_metrics.csv", index=False)
    print(f"  Saved overall metrics: {len(overall_df)} rows")

    # ── Aggregate score per model (mean across tasks) ─────────────────────
    agg_records = []
    for model_key in MODEL_ORDER:
        model_rows = overall_df[overall_df["model"] == model_key]
        if model_rows.empty:
            continue
        agg_records.append({
            "model": model_key,
            "display_name": MODELS[model_key]["display_name"],
            "short_name": MODELS[model_key]["short_name"],
            "mean_micro_f1": model_rows["micro_f1"].mean(),
            "mean_macro_f1": model_rows["macro_f1"].mean(),
            "mean_weighted_f1": model_rows["weighted_f1"].mean(),
            "mean_micro_precision": model_rows["micro_precision"].mean(),
            "mean_micro_recall": model_rows["micro_recall"].mean(),
        })

    agg_df = pd.DataFrame(agg_records).sort_values("mean_micro_f1", ascending=False)
    agg_df.to_csv(DATA_DIR / "benchmark_aggregate_metrics.csv", index=False)
    print(f"  Saved aggregate metrics")

    # ── Per-class metrics ─────────────────────────────────────────────────
    per_class_records = []

    for model_key in MODEL_ORDER:
        for task_key in TASK_ORDER:
            task_cfg = TASKS[task_key]
            gt_col = f"gt_{task_key}"
            pred_col = f"pred_{model_key}_{task_key}"

            if pred_col not in df.columns:
                continue

            y_true = df[gt_col].tolist()
            y_pred = df[pred_col].tolist()

            if task_cfg["type"] == "multiclass":
                pc = compute_per_class_metrics_multiclass(y_true, y_pred)
            else:
                pc = compute_per_class_metrics_multilabel(y_true, y_pred)

            for cls, metrics in pc.items():
                per_class_records.append({
                    "model": model_key,
                    "short_name": MODELS[model_key]["short_name"],
                    "task": task_key,
                    "class": cls,
                    **metrics,
                })

    pc_df = pd.DataFrame(per_class_records)
    pc_df.to_csv(DATA_DIR / "benchmark_per_class_metrics.csv", index=False)
    print(f"  Saved per-class metrics: {len(pc_df)} rows")

    # ── Per-language metrics ──────────────────────────────────────────────
    lang_records = []
    for lang in ["EN", "FR"]:
        mask = df["lang"] == lang
        for model_key in MODEL_ORDER:
            for task_key in TASK_ORDER:
                task_cfg = TASKS[task_key]
                gt_col = f"gt_{task_key}"
                pred_col = f"pred_{model_key}_{task_key}"

                if pred_col not in df.columns:
                    continue

                y_true = df.loc[mask, gt_col].tolist()
                y_pred = df.loc[mask, pred_col].tolist()

                if task_cfg["type"] == "multiclass":
                    metrics = compute_multiclass_metrics(y_true, y_pred)
                else:
                    metrics = compute_multilabel_metrics(y_true, y_pred)

                if metrics:
                    lang_records.append({
                        "language": lang,
                        "model": model_key,
                        "short_name": MODELS[model_key]["short_name"],
                        "task": task_key,
                        **metrics,
                    })

    lang_df = pd.DataFrame(lang_records)
    lang_df.to_csv(DATA_DIR / "benchmark_per_language_metrics.csv", index=False)
    print(f"  Saved per-language metrics: {len(lang_df)} rows")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print("\n── Overall Ranking (Mean Micro F1 across tasks) ──")
    for _, row in agg_df.iterrows():
        print(f"  {row['display_name']:20s}  Micro F1={row['mean_micro_f1']:.4f}  "
              f"Macro F1={row['mean_macro_f1']:.4f}  Weighted F1={row['mean_weighted_f1']:.4f}")

    print("\n── Per-Task Results ──")
    for task_key in TASK_ORDER:
        task_rows = overall_df[overall_df["task"] == task_key].sort_values("micro_f1", ascending=False)
        if task_rows.empty:
            continue
        print(f"\n  {TASKS[task_key]['display_name']}:")
        for _, row in task_rows.iterrows():
            print(f"    {row['short_name']:12s}  Micro F1={row['micro_f1']:.4f}  "
                  f"Macro F1={row['macro_f1']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
