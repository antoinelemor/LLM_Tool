#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
03_compute_statistical_tests.py

MAIN OBJECTIVE:
---------------
Perform statistical significance tests for model comparison.
Applies non-parametric tests (Friedman, Kruskal-Wallis) with post-hoc
pairwise Wilcoxon signed-rank tests and Bonferroni correction, computes
effect sizes (Kendall's W, eta-squared), and bootstrap confidence intervals.

Dependencies:
-------------
- itertools
- json
- sys
- pathlib
- numpy
- pandas
- scipy

MAIN FEATURES:
--------------
1) Friedman test with Kendall's W effect size
2) Kruskal-Wallis H-test with eta-squared
3) Post-hoc pairwise Wilcoxon signed-rank tests with Bonferroni correction
4) Bootstrap confidence intervals (10,000 resamples, 95% level)
5) Per-task and global statistical comparisons

Author:
-------
Antoine Lemor
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, RESULTS_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER


def load_metrics():
    """Load per-class metrics as the basis for statistical comparison."""
    return pd.read_csv(DATA_DIR / "benchmark_per_class_metrics.csv")


def load_overall():
    """Load overall metrics."""
    return pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")


def friedman_test(scores_per_model):
    """
    Friedman test: non-parametric repeated-measures ANOVA.
    Each 'subject' is a class label; each 'treatment' is a model.
    """
    # scores_per_model: dict of model -> array of per-class F1 scores
    models = list(scores_per_model.keys())
    arrays = [np.array(scores_per_model[m]) for m in models]

    # Align lengths (only use classes present in all models)
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    if min_len < 3:
        return {"statistic": None, "p_value": None, "note": "Too few classes"}

    stat, p = stats.friedmanchisquare(*arrays)
    k = len(models)
    n = min_len
    W = stat / (n * (k - 1))  # Kendall's W

    return {
        "chi_square": round(stat, 2),
        "p_value": p,
        "p_display": f"< 0.0001" if p < 0.0001 else f"{p:.4f}",
        "kendall_w": round(W, 3),
        "n_classes": n,
        "n_models": k,
    }


def kruskal_wallis_test(scores_per_model):
    """Kruskal-Wallis H-test: non-parametric one-way ANOVA."""
    arrays = list(scores_per_model.values())
    if any(len(a) < 2 for a in arrays):
        return {"statistic": None, "p_value": None}

    H, p = stats.kruskal(*arrays)
    N = sum(len(a) for a in arrays)
    k = len(arrays)
    eta_sq = (H - k + 1) / (N - k) if N > k else 0

    return {
        "H_statistic": round(H, 2),
        "p_value": p,
        "p_display": f"< 0.0001" if p < 0.0001 else f"{p:.4f}",
        "eta_squared": round(eta_sq, 3),
    }


def pairwise_wilcoxon(scores_per_model, alpha=0.05):
    """Post-hoc pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    models = list(scores_per_model.keys())
    n_comparisons = len(models) * (len(models) - 1) // 2
    corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    results = []
    for m1, m2 in itertools.combinations(models, 2):
        a1 = np.array(scores_per_model[m1])
        a2 = np.array(scores_per_model[m2])
        min_len = min(len(a1), len(a2))
        a1, a2 = a1[:min_len], a2[:min_len]

        if min_len < 5 or np.all(a1 == a2):
            continue

        try:
            stat, p = stats.wilcoxon(a1, a2, alternative="two-sided")
            # Effect size: r = Z / sqrt(N)
            z = stats.norm.ppf(1 - p / 2) if p > 0 else np.inf
            r = z / np.sqrt(min_len) if min_len > 0 else 0
        except ValueError:
            continue

        results.append({
            "model_1": MODELS[m1]["short_name"],
            "model_2": MODELS[m2]["short_name"],
            "model_1_key": m1,
            "model_2_key": m2,
            "statistic": round(stat, 2),
            "p_value": p,
            "p_corrected": min(p * n_comparisons, 1.0),
            "significant": p * n_comparisons < alpha,
            "effect_size_r": round(abs(r), 3),
            "mean_diff": round(np.mean(a1 - a2), 4),
        })

    return pd.DataFrame(results)


def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for mean F1."""
    scores = np.array(scores)
    if len(scores) < 2:
        return {"mean": float(np.mean(scores)), "ci_lower": None, "ci_upper": None}

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(np.percentile(boot_means, alpha * 100)),
        "ci_upper": float(np.percentile(boot_means, (1 - alpha) * 100)),
        "std": float(np.std(scores)),
    }


def main():
    print("Computing statistical tests...")
    pc_df = load_metrics()
    overall_df = load_overall()

    all_results = {}

    # ── Per-task Friedman & Kruskal-Wallis ─────────────────────────────────
    for task_key in TASK_ORDER:
        task_data = pc_df[pc_df["task"] == task_key]
        if task_data.empty:
            continue

        # Build per-class F1 vectors for each model
        classes = sorted(task_data["class"].unique())
        scores_per_model = {}
        for model_key in MODEL_ORDER:
            model_data = task_data[task_data["model"] == model_key]
            if model_data.empty:
                continue
            f1_by_class = model_data.set_index("class")["f1"].reindex(classes, fill_value=0).values
            scores_per_model[model_key] = f1_by_class

        if len(scores_per_model) < 2:
            continue

        task_name = TASKS[task_key]["display_name"]
        print(f"\n  {task_name}:")

        # Friedman
        fr = friedman_test(scores_per_model)
        print(f"    Friedman: χ²={fr['chi_square']}, p={fr['p_display']}, W={fr['kendall_w']}")

        # Kruskal-Wallis
        kw = kruskal_wallis_test(scores_per_model)
        print(f"    Kruskal-Wallis: H={kw['H_statistic']}, p={kw['p_display']}, η²={kw['eta_squared']}")

        # Pairwise Wilcoxon
        pw = pairwise_wilcoxon(scores_per_model)
        if not pw.empty and "significant" in pw.columns:
            sig = pw[pw["significant"]]
            print(f"    Significant pairwise comparisons: {len(sig)}/{len(pw)}")
        else:
            sig = pd.DataFrame()
            print(f"    Pairwise comparisons: insufficient data")

        all_results[task_key] = {
            "friedman": fr,
            "kruskal_wallis": kw,
            "pairwise": pw.to_dict(orient="records"),
        }

    # ── Global test (across all tasks) ────────────────────────────────────
    print("\n  Global (all tasks combined):")
    global_scores = {}
    for model_key in MODEL_ORDER:
        model_data = pc_df[pc_df["model"] == model_key]
        if not model_data.empty:
            global_scores[model_key] = model_data["f1"].values

    if len(global_scores) >= 2:
        fr_global = friedman_test(global_scores)
        kw_global = kruskal_wallis_test(global_scores)
        pw_global = pairwise_wilcoxon(global_scores)
        sig_global = pw_global[pw_global["significant"]] if not pw_global.empty and "significant" in pw_global.columns else pd.DataFrame()

        print(f"    Friedman: χ²={fr_global['chi_square']}, p={fr_global['p_display']}, W={fr_global['kendall_w']}")
        print(f"    Kruskal-Wallis: H={kw_global['H_statistic']}, p={kw_global['p_display']}, η²={kw_global['eta_squared']}")
        print(f"    Significant pairwise: {len(sig_global)}/{len(pw_global)}")

        all_results["global"] = {
            "friedman": fr_global,
            "kruskal_wallis": kw_global,
            "pairwise": pw_global.to_dict(orient="records"),
        }

    # ── Bootstrap CIs for overall metrics ─────────────────────────────────
    print("\n  Bootstrap confidence intervals (Micro F1):")
    ci_records = []
    for model_key in MODEL_ORDER:
        model_data = pc_df[pc_df["model"] == model_key]
        if model_data.empty:
            continue
        ci = bootstrap_ci(model_data["f1"].values)
        ci_records.append({
            "model": model_key,
            "short_name": MODELS[model_key]["short_name"],
            **ci,
        })
        print(f"    {MODELS[model_key]['short_name']:12s}: {ci['mean']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    ci_df = pd.DataFrame(ci_records)
    ci_df.to_csv(DATA_DIR / "bootstrap_confidence_intervals.csv", index=False)

    # ── Save pairwise results ─────────────────────────────────────────────
    if "global" in all_results:
        pw_df = pd.DataFrame(all_results["global"]["pairwise"])
        pw_df.to_csv(DATA_DIR / "pairwise_wilcoxon_global.csv", index=False)

    # ── Save all results as JSON ──────────────────────────────────────────
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return obj

    out_path = RESULTS_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n  Saved all results: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
