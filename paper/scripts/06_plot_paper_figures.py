#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
06_plot_paper_figures.py

MAIN OBJECTIVE:
---------------
Generate publication-ready composite figures for the paper.
Produces all main manuscript figures with consistent styling suitable
for top-tier academic journals.

Dependencies:
-------------
- sys
- pathlib
- itertools
- matplotlib
- numpy
- pandas
- scipy

MAIN FEATURES:
--------------
1) Per-task performance bars with bootstrap CIs and significance brackets
2) Combined aggregate bars and radar chart
3) Computational efficiency: inference time bars and scaling table
4) Ground-truth quality: agreement, distributions, and category breakdowns
5) Sample size vs F1 relationship scatter and bucket analysis
6) Cross-lingual and cross-domain robustness comparison

Author:
-------
Antoine Lemor
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, FIGURES_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER,
    COLORS, TASK_COLORS, setup_matplotlib, save_figure,
)

setup_matplotlib()


# ══════════════════════════════════════════════════════════════════════
# Figure A: Per-task performance with bootstrap CIs
# ══════════════════════════════════════════════════════════════════════
def _add_sig_bracket(ax, x1, x2, y, p_val, h=0.012):
    """Draw a significance bracket between two bars."""
    if p_val >= 0.05:
        return
    stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="0.3")
    ax.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom",
            fontsize=9, color="0.3", fontweight="bold")


def _pairwise_test(f1_i, f1_j, n_pairs, method="wilcoxon"):
    """Return Bonferroni-corrected p-value for two per-class F1 vectors.
    Uses Wilcoxon when >= 6 paired classes; bootstrap permutation otherwise."""
    from scipy.stats import wilcoxon
    diff = f1_i - f1_j
    if np.all(diff == 0):
        return 1.0
    if method == "wilcoxon" and len(f1_i) >= 6:
        try:
            _, p = wilcoxon(f1_i, f1_j, alternative="two-sided")
        except ValueError:
            return 1.0
    else:
        # Bootstrap permutation test
        rng = np.random.default_rng(42)
        obs = np.abs(np.mean(diff))
        n_perm = 10000
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=len(diff))
            if np.abs(np.mean(diff * signs)) >= obs:
                count += 1
        p = count / n_perm
    return min(p * n_pairs, 1.0)


def plot_per_task_with_ci():
    """Grouped bar chart per task with significance brackets on all tasks."""
    from itertools import combinations

    overall = pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")
    pc = pd.read_csv(DATA_DIR / "benchmark_per_class_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes = axes.flatten()

    for idx, task_key in enumerate(TASK_ORDER):
        ax = axes[idx]
        task_overall = overall[overall["task"] == task_key].set_index("model")
        task_pc = pc[pc["task"] == task_key]

        models_present = [m for m in MODEL_ORDER if m in task_overall.index]
        x = np.arange(len(models_present))
        n_models = len(models_present)

        means = [task_overall.loc[m, "micro_f1"] for m in models_present]

        bars = ax.bar(
            x, means,
            color=[COLORS[m] for m in models_present],
            edgecolor="white", linewidth=0.8, width=0.6,
        )

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Pairwise tests with Bonferroni correction
        n_pairs = n_models * (n_models - 1) // 2
        sig_pairs = []
        for i, j in combinations(range(n_models), 2):
            mi, mj = models_present[i], models_present[j]
            df_i = task_pc[task_pc["model"] == mi][["class", "f1"]].set_index("class")
            df_j = task_pc[task_pc["model"] == mj][["class", "f1"]].set_index("class")
            shared = df_i.index.intersection(df_j.index)
            if len(shared) < 2:
                continue
            vi = df_i.loc[shared, "f1"].values
            vj = df_j.loc[shared, "f1"].values
            p = _pairwise_test(vi, vj, n_pairs,
                               method="wilcoxon" if len(shared) >= 6 else "permutation")
            if p < 0.05:
                sig_pairs.append((i, j, p))

        # Draw up to 3 most significant brackets; if none, annotate "n.s."
        max_val = max(means)
        bracket_y = max_val + 0.03
        drawn = 0
        for i, j, p in sorted(sig_pairs, key=lambda t: t[2]):
            if drawn >= 3:
                break
            _add_sig_bracket(ax, x[i], x[j], bracket_y + drawn * 0.04, p, h=0.01)
            drawn += 1

        if drawn == 0:
            ax.text(0.5, 0.95, "n.s.", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color="0.5",
                    fontstyle="italic")

        top = bracket_y + max(drawn, 1) * 0.04 + 0.05
        ax.set_title(TASKS[task_key]["display_name"], fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODELS[m]["short_name"] for m in models_present], fontsize=9.5)
        ax.set_ylabel("Micro F1")
        ax.set_ylim(0, min(1.18, top))

    fig.suptitle("Model Performance by Annotation Task (pairwise tests, Bonferroni-corrected)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_per_task_ci")


# ══════════════════════════════════════════════════════════════════════
# Figure B: Combined per-task + radar (side by side)
# ══════════════════════════════════════════════════════════════════════
def plot_combined_performance():
    """Two-panel figure: aggregate bars (left) + radar (right)."""
    agg = pd.read_csv(DATA_DIR / "benchmark_aggregate_metrics.csv").set_index("model")
    agg = agg.loc[[m for m in MODEL_ORDER if m in agg.index]].reset_index()

    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1])

    # ── Left panel: aggregate bars ──
    ax1 = fig.add_subplot(gs[0])

    x = np.arange(len(agg))
    width = 0.25
    metrics = ["mean_micro_f1", "mean_macro_f1", "mean_weighted_f1"]
    labels = ["Micro F1", "Macro F1", "Weighted F1"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        bars = ax1.bar(x + i * width, agg[metric], width, label=label,
                       color=[COLORS[m] for m in agg["model"]],
                       alpha=[1.0, 0.65, 0.4][i],
                       edgecolor="white", linewidth=0.6)

    ax1.set_xlabel("")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("(a) Aggregate Performance", fontweight="bold", fontsize=11)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([MODELS[m]["short_name"] for m in agg["model"]], fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, 0.82)

    # ── Right panel: radar ──
    ax2 = fig.add_subplot(gs[1], polar=True)

    categories = ["Micro F1", "Macro F1", "Precision", "Recall"]
    cols = ["mean_micro_f1", "mean_macro_f1", "mean_micro_precision", "mean_micro_recall"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i in range(len(agg)):
        row = agg.iloc[i]
        model_key = row["model"]
        values = [row[c] for c in cols]
        values += values[:1]
        ax2.plot(angles, values, "o-", linewidth=2,
                 label=MODELS[model_key]["short_name"],
                 color=COLORS[model_key], markersize=5)
        ax2.fill(angles, values, alpha=0.06, color=COLORS[model_key])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 0.8)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax2.set_title("(b) Multi-Metric Profiles", fontweight="bold", fontsize=11, pad=20)

    fig.tight_layout()
    save_figure(fig, "fig_combined_performance")


# ══════════════════════════════════════════════════════════════════════
# Figure C: Computational efficiency (clean single figure)
# ══════════════════════════════════════════════════════════════════════
def plot_efficiency_figure():
    """Two-panel efficiency figure: bar chart + scaling projection."""
    ann = pd.read_csv(DATA_DIR / "annotation_stats.csv")
    ann = ann[ann["mean_inference_time"].notna()].copy()
    ann = ann.sort_values("mean_inference_time", ascending=True)

    BERT_TIME = 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: bar chart (log scale) ──
    models_list = ann["model"].tolist()
    all_names = [r.short_name for r in ann.itertuples()] + ["XLM-RoBERTa"]
    all_params = [f"({r.params})" for r in ann.itertuples()] + ["(278M)"]
    all_times = ann["mean_inference_time"].tolist() + [BERT_TIME]
    bar_colors = [COLORS.get(m, "#999") for m in models_list] + ["#2CA02C"]

    x = np.arange(len(all_names))
    bars = ax1.bar(x, all_times, color=bar_colors, edgecolor="white", width=0.55)
    ax1.set_yscale("log")
    ax1.set_ylabel("Seconds per sentence (log scale)")
    ax1.set_title("(a) Inference Time per Sentence", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n}\n{p}" for n, p in zip(all_names, all_params)],
                        fontsize=8, rotation=0)

    for bar, t in zip(bars, all_times):
        label = f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                 label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.axhline(y=BERT_TIME, color="#2CA02C", linestyle="--", alpha=0.4, linewidth=1)
    ax1.set_ylim(top=max(all_times) * 3)

    # ── Right: grouped horizontal bar chart for corpus of 38,451 sentences ──
    # Show total annotation time in days for each model + BERT
    models_list_r = ann["model"].tolist()
    all_names_r = [r.short_name for r in ann.itertuples()] + ["XLM-RoBERTa"]
    all_times_r = ann["mean_inference_time"].tolist() + [BERT_TIME]
    bar_colors_r = [COLORS.get(m, "#999") for m in models_list_r] + ["#2CA02C"]

    corpus_n = 38_451
    total_days = [t * corpus_n / 86400 for t in all_times_r]

    y_r = np.arange(len(all_names_r))
    bars_r = ax2.barh(y_r, total_days, color=bar_colors_r, edgecolor="white",
                      height=0.55)
    ax2.set_xlabel("Total annotation time (days)")
    ax2.set_title("(b) Total Time for 38,451 Sentences", fontweight="bold",
                  fontsize=11)
    ax2.set_yticks(y_r)
    ax2.set_yticklabels(all_names_r, fontsize=9)
    ax2.invert_yaxis()

    # Annotate each bar with readable time
    def _fmt_time(days):
        if days < 1 / 24:
            return f"{days * 24 * 60:.0f} min"
        if days < 1:
            return f"{days * 24:.1f} h"
        return f"{days:.1f} d"

    for bar, d in zip(bars_r, total_days):
        label = _fmt_time(d)
        # Place label outside bar if bar is short relative to axis
        ax2.text(bar.get_width() + max(total_days) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 label, ha="left", va="center", fontsize=9, fontweight="bold")

    # Add speedup annotations vs BERT
    bert_days = total_days[-1]
    for i, d in enumerate(total_days[:-1]):
        speedup = d / bert_days
        ax2.text(bar.get_width() + max(total_days) * 0.02,
                 y_r[i] + 0.25,
                 f"{speedup:,.0f}x slower", ha="left", va="top",
                 fontsize=7, color="#E15759", fontstyle="italic")

    # Reference lines
    for days, label in [(1, "1 day"), (7, "1 week")]:
        ax2.axvline(x=days, color="0.6", linestyle=":", alpha=0.5, linewidth=0.8)
        ax2.text(days, -0.45, label, fontsize=7.5, color="0.5", ha="center")

    ax2.set_xlim(0, max(total_days) * 1.35)

    fig.tight_layout()
    save_figure(fig, "fig_efficiency")


# ══════════════════════════════════════════════════════════════════════
# Figure D: Agreement quality (for Methods)
# ══════════════════════════════════════════════════════════════════════
def plot_agreement_quality():
    """Simplified 2×2 figure: agreement, sentiment, parties+specific, themes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [1, 1.3],
                                          "hspace": 0.38, "wspace": 0.3})

    # ══ (a) Inter-annotator agreement ══
    ax = axes[0, 0]
    metrics = ["Krippendorff's $\\alpha$", "Fleiss' $\\kappa$"]
    values = [0.63, 0.96]
    colors_m = ["#4E79A7", "#59A14F"]
    ci_lo = [0.63 - 0.617, abs(0.96 - 0.962)]
    ci_hi = [0.644 - 0.63, 0.964 - 0.96]

    bars = ax.barh([0, 1], values, color=colors_m, edgecolor="white", height=0.45,
                   xerr=[ci_lo, ci_hi], capsize=4, error_kw={"linewidth": 1.3})
    ax.set_yticks([0, 1])
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_xlabel("Agreement Score")
    ax.set_xlim(0, 1.12)
    ax.set_title("(a) Inter-Annotator Agreement", fontweight="bold", fontsize=10.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.025, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.axvline(x=0.67, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.68, -0.35, "substantial", fontsize=7, color="gray", ha="left")
    ax.axvline(x=0.80, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.81, -0.35, "strong", fontsize=7, color="gray", ha="left")

    # ══ (b) Sentiment distribution (donut) ══
    ax = axes[0, 1]
    labels_sent = ["Neutral\n(863)", "Negative\n(534)", "Positive\n(152)"]
    sizes_sent = [863, 534, 152]
    colors_sent = ["#4E79A7", "#E15759", "#59A14F"]
    wedges, texts, autotexts = ax.pie(
        sizes_sent, labels=labels_sent, colors=colors_sent,
        autopct="%1.0f%%", pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title("(b) Sentiment (N = 1,549)", fontweight="bold", fontsize=10.5)

    # ══ (c) Policy Themes (main distribution) ══
    ax = axes[1, 0]
    theme_names = [
        "Gov./Governance", "Int'l Affairs", "Macroeconomics", "Law & Crime",
        "Health", "Environment", "Defence", "Education", "Housing",
        "Rights/Minorities", "Labor", "Immigration", "Culture/Nationalism",
        "Transportation", "Social Welfare", "Agriculture", "Energy",
        "Domestic Commerce", "Technology", "Public Lands", "Foreign Trade",
    ]
    theme_counts = [226, 180, 156, 137, 106, 76, 66, 59, 56, 47, 43, 39, 36,
                    34, 32, 19, 16, 14, 10, 7, 6]
    y_th = np.arange(len(theme_names))
    norm = plt.Normalize(vmin=0, vmax=max(theme_counts))
    th_colors = [plt.cm.Greys(norm(c) * 0.55 + 0.20) for c in theme_counts]
    ax.barh(y_th, theme_counts, color=th_colors, edgecolor="white",
            height=0.75, linewidth=0.5)
    ax.set_yticks(y_th)
    ax.set_yticklabels(theme_names, fontsize=7.5)
    ax.set_xlabel("Count (test set)")
    ax.set_title("(c) Policy Themes (993 texts annotated)",
                 fontweight="bold", fontsize=10.5)
    ax.invert_yaxis()
    for i, v in enumerate(theme_counts):
        ax.text(v + 3, i, str(v), va="center", fontsize=7, color="0.3")

    # ══ (d) Political Parties + Specific Themes (combined) ══
    ax = axes[1, 1]
    # Parties
    parties = ["LPC", "CPC", "NDP", "BQ", "CAQ", "PLQ", "GPC", "QS", "PQ"]
    pp_counts = [97, 73, 28, 16, 8, 8, 5, 1, 1]
    # Specific themes
    st_labels = ["Welfare St.", "Public Fin.", "Childcare"]
    st_counts = [60, 25, 8]

    all_labels = parties + [""] + st_labels
    all_counts = pp_counts + [0] + st_counts
    norm_pp = plt.Normalize(vmin=0, vmax=max(pp_counts))
    party_colors = [plt.cm.Greys(norm_pp(c) * 0.50 + 0.30) for c in pp_counts]
    all_colors = (party_colors
                  + ["white"]
                  + ["#4E79A7", "#76B7B2", "#B07AA1"])

    y_all = np.arange(len(all_labels))
    ax.barh(y_all, all_counts, color=all_colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_all)
    ax.set_yticklabels(all_labels, fontsize=8.5)
    ax.set_xlabel("Count")
    ax.set_title("(d) Political Parties (189 texts) & Specific Themes (88 texts)",
                 fontweight="bold", fontsize=9.5)
    ax.invert_yaxis()
    for i, v in enumerate(all_counts):
        if v > 0:
            ax.text(v + 1, i, str(v), va="center", fontsize=8)

    # Separator line between the two groups
    sep_y = len(parties) - 0.5
    ax.axhline(y=sep_y + 0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(max(pp_counts) * 0.7, sep_y + 0.55, "Specific Themes",
            fontsize=7.5, color="gray", fontstyle="italic", va="bottom")

    fig.suptitle("Ground Truth: Quality and Category Distributions (N = 1,549)",
                 fontsize=13, fontweight="bold", y=0.99)
    save_figure(fig, "fig_agreement_quality")


# ══════════════════════════════════════════════════════════════════════
# Figure E: Sample size vs F1 (confidence/N relationship)
# ══════════════════════════════════════════════════════════════════════
def plot_sample_size_vs_f1():
    """Scatter: per-class test-set N vs F1, colored by model. Shows that rare
    categories produce lower F1 regardless of model."""
    pc = pd.read_csv(DATA_DIR / "benchmark_per_class_metrics.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: scatter for all tasks ──
    for m in MODEL_ORDER:
        sub = pc[pc["model"] == m]
        ax1.scatter(sub["support"], sub["f1"], s=25, alpha=0.6,
                    color=COLORS[m], label=MODELS[m]["short_name"],
                    edgecolors="white", linewidth=0.3)

    # Trend line (all models pooled)
    from numpy.polynomial import polynomial as P
    mask = pc["support"] > 0
    log_support = np.log1p(pc.loc[mask, "support"].values)
    f1_vals = pc.loc[mask, "f1"].values
    coeffs = P.polyfit(log_support, f1_vals, 1)
    xs = np.linspace(log_support.min(), log_support.max(), 100)
    ax1.plot(np.expm1(xs), P.polyval(xs, coeffs), "--", color="gray", linewidth=1.5,
             label="Log-linear trend", zorder=0)

    ax1.set_xlabel("Test-set support (N per class)")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("(a) Per-Class F1 vs. Test-Set Support", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=8, ncol=2, loc="lower right")
    ax1.set_xscale("symlog", linthresh=5)

    # ── Right: mean F1 by support bucket ──
    pc_copy = pc[pc["support"] > 0].copy()
    bins = [0, 5, 15, 50, 150, 1000]
    labels_bin = ["1-5", "6-15", "16-50", "51-150", "151+"]
    pc_copy["bucket"] = pd.cut(pc_copy["support"], bins=bins, labels=labels_bin)

    bucket_means = pc_copy.groupby("bucket", observed=True)["f1"].agg(["mean", "std", "count"])
    x_b = np.arange(len(bucket_means))
    ax2.bar(x_b, bucket_means["mean"],
            yerr=bucket_means["std"] / np.sqrt(bucket_means["count"]),
            capsize=4, color="#4E79A7", edgecolor="white", width=0.6,
            error_kw={"linewidth": 1.2})
    ax2.set_xticks(x_b)
    ax2.set_xticklabels([f"{l}\n(n={int(c)})" for l, c in
                         zip(labels_bin, bucket_means["count"])], fontsize=9)
    ax2.set_xlabel("Test-set support bucket")
    ax2.set_ylabel("Mean F1")
    ax2.set_title("(b) Mean F1 by Support Bucket", fontweight="bold", fontsize=11)

    for i, v in enumerate(bucket_means["mean"]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Effect of Category Prevalence on Classification Performance",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, "fig_sample_size_vs_f1")


# ══════════════════════════════════════════════════════════════════════
# Figure F: Language and source performance
# ══════════════════════════════════════════════════════════════════════
def _bootstrap_ci(y_true, y_pred, n_boot=2000, ci=0.95):
    """Bootstrap CI for micro-F1."""
    from sklearn.metrics import f1_score as sk_f1
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(sk_f1(y_true.iloc[idx], y_pred.iloc[idx], average="micro"))
        except Exception:
            continue
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi


def plot_language_source():
    """Two-panel figure: F1 by language (EN vs FR) and by source (parl vs media),
    with 95 % CIs."""
    lang = pd.read_csv(DATA_DIR / "benchmark_per_language_metrics.csv")

    # Normalise language column to uppercase (CSV uses EN/FR)
    lang["language"] = lang["language"].str.upper()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: language (CI = SE across 4 tasks) ──
    models_present = [m for m in MODEL_ORDER if m in lang["model"].values]
    x = np.arange(len(models_present))
    width = 0.35

    for i, ln in enumerate(["EN", "FR"]):
        sub = lang[lang["language"] == ln]
        vals, errs = [], []
        for m in models_present:
            task_f1s = sub[sub["model"] == m]["micro_f1"].values
            mean_f1 = task_f1s.mean()
            se = task_f1s.std(ddof=1) / np.sqrt(len(task_f1s)) if len(task_f1s) > 1 else 0
            vals.append(mean_f1)
            errs.append(1.96 * se)  # 95% CI
        label = "English" if ln == "EN" else "French"
        bars = ax1.bar(x + i * width, vals, width,
                       yerr=errs, capsize=3,
                       error_kw={"linewidth": 1.0, "capthick": 1.0},
                       label=label, alpha=0.85,
                       color=["#4E79A7", "#E15759"][i],
                       edgecolor="white")
        for bar, v, e in zip(bars, vals, errs):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + e + 0.008,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([MODELS[m]["short_name"] for m in models_present], fontsize=10)
    ax1.set_ylabel("Micro F1 (mean across tasks)")
    ax1.set_title("(a) Performance by Language", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 0.85)

    # ── Right: source (bootstrap CIs on sentiment) ──
    bench = pd.read_csv(DATA_DIR / "benchmark_clean.csv")
    from sklearn.metrics import f1_score as sk_f1

    source_results = []
    for src in ["parliament", "headlines"]:
        sub = bench[bench["source"] == src]
        for m in models_present:
            pred_col = f"pred_{m}_sentiment"
            true_col = "gt_sentiment"
            if pred_col not in sub.columns:
                continue
            mask = sub[pred_col].notna() & sub[true_col].notna()
            if mask.sum() > 0:
                y_t = sub.loc[mask, true_col].reset_index(drop=True)
                y_p = sub.loc[mask, pred_col].reset_index(drop=True)
                f1 = sk_f1(y_t, y_p, average="micro")
                lo, hi = _bootstrap_ci(y_t, y_p)
                source_results.append({
                    "model": m, "source": src, "micro_f1": f1,
                    "ci_lo": f1 - lo, "ci_hi": hi - f1,
                })

    src_df = pd.DataFrame(source_results)
    for i, src in enumerate(["parliament", "headlines"]):
        sub = src_df[src_df["source"] == src].set_index("model")
        vals = [sub.loc[m, "micro_f1"] if m in sub.index else 0 for m in models_present]
        err_lo = [sub.loc[m, "ci_lo"] if m in sub.index else 0 for m in models_present]
        err_hi = [sub.loc[m, "ci_hi"] if m in sub.index else 0 for m in models_present]
        label = "Parliamentary" if src == "parliament" else "Media Headlines"
        bars = ax2.bar(x + i * width, vals, width,
                       yerr=[err_lo, err_hi], capsize=3,
                       error_kw={"linewidth": 1.0, "capthick": 1.0},
                       label=label, alpha=0.85,
                       color=["#59A14F", "#F28E2B"][i],
                       edgecolor="white")
        for bar, v, eh in zip(bars, vals, err_hi):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + eh + 0.008,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels([MODELS[m]["short_name"] for m in models_present], fontsize=10)
    ax2.set_ylabel("Micro F1 (sentiment)")
    ax2.set_title("(b) Performance by Source", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 0.90)

    fig.suptitle("Cross-Lingual and Cross-Domain Robustness",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_figure(fig, "fig_language_source")


if __name__ == "__main__":
    print("Generating publication figures...")
    plot_per_task_with_ci()
    plot_combined_performance()
    plot_efficiency_figure()
    plot_agreement_quality()
    plot_sample_size_vs_f1()
    plot_language_source()
    print("Done.")
