"""
06 — Per-category F1 heatmaps.

Heatmap of F1 scores: models (columns) × classes (rows) for each task.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, FIGURES_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER,
    TASK_COLORS, setup_matplotlib, save_figure,
)

setup_matplotlib()


def plot_heatmap_for_task(task_key, pc_df, min_support=3):
    """Generate a heatmap for one task."""
    task_cfg = TASKS[task_key]
    task_data = pc_df[pc_df["task"] == task_key].copy()

    if task_data.empty:
        return

    # Filter classes with minimum support
    class_support = task_data.groupby("class")["support"].max()
    valid_classes = class_support[class_support >= min_support].index.tolist()
    task_data = task_data[task_data["class"].isin(valid_classes)]

    if task_data.empty:
        return

    # Pivot: classes × models
    pivot = task_data.pivot_table(
        index="class", columns="short_name", values="f1", aggfunc="first"
    )

    # Order models
    model_names = [MODELS[m]["short_name"] for m in MODEL_ORDER if MODELS[m]["short_name"] in pivot.columns]
    pivot = pivot.reindex(columns=model_names)

    # Sort by mean F1 (descending)
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=True)
    pivot = pivot.drop("_mean", axis=1)

    # Clean class names for display, with support (N) from test set
    support_map = task_data.groupby("class")["support"].max().to_dict()
    display_names = [f"{c.replace('_', ' ').title()}  (N={int(support_map.get(c, 0))})"
                     for c in pivot.index]

    fig_height = max(4, len(pivot) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    data = pivot.values
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(display_names, fontsize=9)

    # Cell values
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_title(f"Per-Class F1 Scores — {task_cfg['display_name']}", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")

    fig.tight_layout()
    save_figure(fig, f"heatmap_{task_key}")


def plot_combined_best_classes():
    """Bar chart of best-performing classes across all models (top 15 + bottom 5)."""
    pc_df = pd.read_csv(DATA_DIR / "benchmark_per_class_metrics.csv")

    # Mean F1 per class across all models
    class_means = pc_df.groupby(["task", "class"]).agg(
        mean_f1=("f1", "mean"),
        max_f1=("f1", "max"),
        support=("support", "max"),
    ).reset_index()
    class_means = class_means[class_means["support"] >= 3]

    # Top 15
    top = class_means.nlargest(15, "mean_f1")
    # Bottom 5 (non-zero support)
    bottom = class_means[class_means["mean_f1"] > 0].nsmallest(5, "mean_f1")
    combined = pd.concat([top, bottom]).drop_duplicates()
    combined = combined.sort_values("mean_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [TASK_COLORS.get(t, "#999") for t in combined["task"]]
    display = [f"{r['class'].replace('_', ' ').title()} ({TASKS[r['task']]['display_name'][:6]})"
               for _, r in combined.iterrows()]

    bars = ax.barh(range(len(combined)), combined["mean_f1"], color=colors,
                   edgecolor="white", linewidth=0.5)

    for i, (bar, row) in enumerate(zip(bars, combined.itertuples())):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f} (n={row.support})",
                ha="left", va="center", fontsize=8)

    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(display, fontsize=9)
    ax.set_xlabel("Mean F1 Score (across models)")
    ax.set_title("Per-Class Performance (Top 15 + Bottom 5)", fontweight="bold")
    ax.set_xlim(0, 1.0)

    # Legend for task colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TASK_COLORS[t], label=TASKS[t]["display_name"])
                       for t in TASK_ORDER if t in combined["task"].values]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    save_figure(fig, "per_class_top_bottom")


if __name__ == "__main__":
    print("Generating category heatmaps...")
    pc_df = pd.read_csv(DATA_DIR / "benchmark_per_class_metrics.csv")

    for task_key in TASK_ORDER:
        plot_heatmap_for_task(task_key, pc_df)

    plot_combined_best_classes()
    print("Done.")
