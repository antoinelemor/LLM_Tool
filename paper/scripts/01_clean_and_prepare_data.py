#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
01_clean_and_prepare_data.py

MAIN OBJECTIVE:
---------------
Clean and prepare benchmark data for paper analysis.
Reads the raw benchmark CSV, normalizes column names and label formats,
extracts annotation statistics and training metrics, and outputs clean
datasets ready for analysis.

Dependencies:
-------------
- ast
- json
- sys
- pathlib
- numpy
- pandas

MAIN FEATURES:
--------------
1) Parse and normalize label lists from raw annotations
2) Extract ground-truth consensus labels for all annotation tasks
3) Extract and align model predictions across five LLM sources
4) Compute annotation statistics from annotator logs
5) Extract training metrics from comprehensive summaries

Author:
-------
Antoine Lemor
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BENCHMARK_CSV, DATA_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER,
    TRAINING_ARENA_DIR, ANNOTATOR_DIR,
)


def parse_label_list(raw):
    """Parse a string-encoded list of labels, returning a set of cleaned labels."""
    if pd.isna(raw) or raw == "" or raw == "[]":
        return set()
    if isinstance(raw, list):
        return set(raw)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return set(parsed)
        return {str(parsed)}
    except (ValueError, SyntaxError):
        return {str(raw).strip()}


def strip_prefix(labels: set, prefix: str) -> set:
    """Remove a known prefix from each label in the set."""
    out = set()
    for lbl in labels:
        lbl = str(lbl).strip()
        if lbl.startswith(prefix):
            lbl = lbl[len(prefix):]
        out.add(lbl)
    out.discard("")
    out.discard("null")
    out.discard("null,")
    return out


def clean_benchmark():
    """Load and normalize the benchmark CSV."""
    print("Loading benchmark CSV...")
    df = pd.read_csv(BENCHMARK_CSV)
    print(f"  Raw shape: {df.shape}")

    # ── Core columns ──────────────────────────────────────────────────────
    clean = df[["id", "date", "text", "lang", "source"]].copy()

    # ── Ground truth (consensus) ──────────────────────────────────────────
    for task_key, task_cfg in TASKS.items():
        consensus_col = task_cfg["consensus_col"]
        prefix = task_cfg["consensus_prefix"]

        if task_cfg["type"] == "multiclass":
            # sentiment: simple string
            clean[f"gt_{task_key}"] = (
                df[consensus_col]
                .astype(str)
                .str.replace(prefix, "", regex=False)
                .str.strip()
            )
        else:
            # multilabel: list of strings
            clean[f"gt_{task_key}"] = df[consensus_col].apply(
                lambda x: sorted(strip_prefix(parse_label_list(x), prefix))
            )

    # ── Human annotators ──────────────────────────────────────────────────
    annotators = ["shdin", "Jeremy", "jdrouin", "BenjaminCarignan"]
    for task_key, task_cfg in TASKS.items():
        prefix = task_cfg["consensus_prefix"]
        for ann in annotators:
            src_col = f"{ann}_{task_cfg['consensus_col'].replace('consensus_', '')}"
            if src_col not in df.columns:
                continue
            if task_cfg["type"] == "multiclass":
                clean[f"human_{ann}_{task_key}"] = (
                    df[src_col].astype(str)
                    .str.replace(prefix, "", regex=False)
                    .str.strip()
                )
            else:
                clean[f"human_{ann}_{task_key}"] = df[src_col].apply(
                    lambda x: sorted(strip_prefix(parse_label_list(x), prefix))
                )

    # ── Model predictions ─────────────────────────────────────────────────
    for model_key in MODEL_ORDER:
        model_cfg = MODELS[model_key]
        col_prefix = model_cfg["col_prefix"]

        for task_key, task_cfg in TASKS.items():
            suffix = task_cfg["col_suffix"]
            pred_prefix = task_cfg["pred_prefix"]

            label_col = f"{col_prefix}_normal_training_{suffix}_label"
            prob_col = f"{col_prefix}_normal_training_{suffix}_probability"

            if label_col not in df.columns:
                continue

            # Parse predictions
            if task_cfg["type"] == "multiclass":
                clean[f"pred_{model_key}_{task_key}"] = (
                    df[label_col].astype(str)
                    .str.replace(pred_prefix, "", regex=False)
                    .str.strip()
                )
            else:
                clean[f"pred_{model_key}_{task_key}"] = df[label_col].apply(
                    lambda x: sorted(strip_prefix(parse_label_list(x), pred_prefix))
                )

            # Probability
            if prob_col in df.columns:
                clean[f"prob_{model_key}_{task_key}"] = pd.to_numeric(
                    df[prob_col], errors="coerce"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "benchmark_clean.parquet"
    clean.to_parquet(out_path, index=False)
    print(f"  Saved clean benchmark: {out_path}  ({clean.shape})")

    # Also save a CSV for inspection
    csv_path = DATA_DIR / "benchmark_clean.csv"
    # For CSV, convert lists to JSON strings
    csv_df = clean.copy()
    for col in csv_df.columns:
        if csv_df[col].apply(lambda x: isinstance(x, list)).any():
            csv_df[col] = csv_df[col].apply(json.dumps)
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV copy: {csv_path}")

    return clean


def extract_annotation_data():
    """Extract annotation statistics from annotator logs."""
    print("\nExtracting annotation statistics...")
    records = []

    # Metrics files with timing
    metrics_files = {
        "gpt_oss": ANNOTATOR_DIR / "GPT-OSS_EXTRA_EXTRA_LARGE_20260127_175500" / "annotated_data" / "ollama" / "gpt-oss_120b" / "extra-extra-large" / "annotation_metrics_GPT-OSS_EXTRA_EXTRA_LARGE_20260127_175500_llm_model_20260131_065700.json",
        "mixtral": ANNOTATOR_DIR / "MIXTRAL_EXTRA_EXTRA_LARGE_20260131_225510" / "annotated_data" / "ollama" / "mixtral_8x22b" / "extra-extra-large" / "annotation_metrics_MIXTRAL_EXTRA_EXTRA_LARGE_20260131_225510_llm_model_20260203_093942.json",
    }

    for model_key in MODEL_ORDER:
        cfg = MODELS[model_key]
        record = {
            "model": model_key,
            "display_name": cfg["display_name"],
            "short_name": cfg["short_name"],
            "provider": cfg["provider"],
            "params": cfg["params"],
            "total_annotations": cfg["total_annotations"],
            "mean_inference_time": cfg["mean_inference_time"],
        }

        # Try to load detailed metrics
        if model_key in metrics_files and metrics_files[model_key].exists():
            with open(metrics_files[model_key]) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            timing = metrics.get("timing_metrics", {})
            record["mean_inference_time"] = timing.get("mean_inference_time", record["mean_inference_time"])
            record["success_count"] = metrics.get("success_count", cfg["total_annotations"])
            record["error_count"] = metrics.get("error_count", 0)

            # Label distributions
            value_counts = metrics.get("value_counts_per_key", {})
            for task_key in ["themes_long", "sentiment_long", "political_parties_long", "specific_themes_long"]:
                if task_key in value_counts:
                    record[f"n_unique_{task_key}"] = len(value_counts[task_key])
        else:
            record["success_count"] = cfg["total_annotations"]
            record["error_count"] = 0

        records.append(record)

    ann_df = pd.DataFrame(records)
    out_path = DATA_DIR / "annotation_stats.csv"
    ann_df.to_csv(out_path, index=False)
    print(f"  Saved annotation stats: {out_path}")
    return ann_df


def extract_training_data():
    """Extract training metrics from comprehensive summaries."""
    print("\nExtracting training metrics...")
    records = []

    for model_key in MODEL_ORDER:
        cfg = MODELS[model_key]
        session_dir = TRAINING_ARENA_DIR / cfg["session_id"]

        # Find comprehensive summary
        summaries = list(session_dir.glob("comprehensive_summary_*.json"))
        if not summaries:
            continue

        with open(summaries[0]) as f:
            data = json.load(f)

        total_time = data.get("total_training_time", 0)

        per_model = data.get("per_model", {})
        for task_name, task_metrics in per_model.items():
            # Map task name to our task key
            task_key = None
            for tk, tcfg in TASKS.items():
                if tcfg["col_suffix"] == task_name:
                    task_key = tk
                    break

            records.append({
                "model": model_key,
                "display_name": cfg["display_name"],
                "short_name": cfg["short_name"],
                "task": task_key or task_name,
                "task_name": task_name,
                "f1_macro": task_metrics.get("f1_macro", None),
                "f1_micro": task_metrics.get("f1_micro", None),
                "accuracy": task_metrics.get("accuracy", None),
                "best_epoch": task_metrics.get("best_epoch", None),
                "num_samples_train": task_metrics.get("num_samples_train", None),
                "num_samples_val": task_metrics.get("num_samples_val", None),
                "total_training_time_hours": total_time / 3600,
            })

    train_df = pd.DataFrame(records)
    out_path = DATA_DIR / "training_metrics.csv"
    train_df.to_csv(out_path, index=False)
    print(f"  Saved training metrics: {out_path}")
    return train_df


if __name__ == "__main__":
    clean_benchmark()
    extract_annotation_data()
    extract_training_data()
    print("\nDone.")
