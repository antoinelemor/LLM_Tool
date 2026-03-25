"""
11 — Generate all LaTeX tables for the paper.

Outputs .tex files ready for \\input{} in the manuscript.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, TABLES_DIR, RESULTS_DIR, MODELS, MODEL_ORDER, TASKS, TASK_ORDER,
    HYPERPARAMETERS, TRAINING_MODEL,
)


def fmt(val, digits=4):
    """Format a float value."""
    if pd.isna(val):
        return "—"
    return f"{val:.{digits}f}"


def bold(text):
    return f"\\textbf{{{text}}}"


def generate_main_results_table():
    """Table: Overall model performance (ranked by mean Micro F1)."""
    agg = pd.read_csv(DATA_DIR / "benchmark_aggregate_metrics.csv")
    overall = pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")

    agg = agg.sort_values("mean_micro_f1", ascending=False).reset_index(drop=True)

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Overall model performance on the benchmark test set (N=1,549). Models ranked by mean Micro F1 across all annotation tasks.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{clcccc}",
        r"\toprule",
        r"\textbf{Rank} & \textbf{Model} & \textbf{Micro F1} & \textbf{Macro F1} & \textbf{Weighted F1} & \textbf{Precision} \\",
        r"\midrule",
    ]

    best_micro = agg["mean_micro_f1"].max()

    for i, row in agg.iterrows():
        rank = i + 1
        name = row["display_name"]
        micro = fmt(row["mean_micro_f1"])
        macro = fmt(row["mean_macro_f1"])
        weighted = fmt(row["mean_weighted_f1"])
        prec = fmt(row["mean_micro_precision"])

        if row["mean_micro_f1"] == best_micro:
            micro = bold(micro)

        lines.append(f"{rank} & {name} & {micro} & {macro} & {weighted} & {prec} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "main_results.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_per_task_table():
    """Table: Per-task performance for each model."""
    overall = pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Model performance by annotation task (Micro F1).}",
        r"\label{tab:per_task_results}",
        r"\begin{tabular}{l" + "c" * len(TASK_ORDER) + r"c}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join([bold(TASKS[t]["display_name"]) for t in TASK_ORDER]) + r" & \textbf{Mean} \\",
        r"\midrule",
    ]

    for model_key in MODEL_ORDER:
        name = MODELS[model_key]["short_name"]
        vals = []
        for task_key in TASK_ORDER:
            row = overall[(overall["model"] == model_key) & (overall["task"] == task_key)]
            if not row.empty:
                vals.append(row.iloc[0]["micro_f1"])
            else:
                vals.append(None)

        mean_val = np.nanmean([v for v in vals if v is not None])
        cells = [fmt(v) if v is not None else "—" for v in vals]
        cells.append(fmt(mean_val))

        # Bold best per column
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "per_task_results.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_statistical_tests_table():
    """Table: Statistical test results."""
    results_path = RESULTS_DIR / "statistical_tests.json"
    if not results_path.exists():
        print("  Skipping statistical tests table (no data)")
        return

    with open(results_path) as f:
        results = json.load(f)

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Statistical tests for model comparison.}",
        r"\label{tab:statistical_tests}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Test} & \textbf{Statistic} & \textbf{p-value} & \textbf{Effect Size} \\",
        r"\midrule",
    ]

    # Global tests
    if "global" in results:
        gl = results["global"]
        fr = gl.get("friedman", {})
        kw = gl.get("kruskal_wallis", {})

        if fr.get("chi_square"):
            lines.append(
                f"Friedman test & $\\chi^2 = {fr['chi_square']}$ & ${fr['p_display']}$ "
                f"& Kendall's W = {fr['kendall_w']} \\\\"
            )
        if kw.get("H_statistic"):
            lines.append(
                f"Kruskal-Wallis test & H = {kw['H_statistic']} & ${kw['p_display']}$ "
                f"& $\\eta^2 = {kw['eta_squared']}$ \\\\"
            )

        # Significant pairwise comparisons
        pairwise = gl.get("pairwise", [])
        sig_pairs = [p for p in pairwise if p.get("significant")]
        if sig_pairs:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{4}{l}{\textit{Significant pairwise comparisons (Wilcoxon with Bonferroni correction)}} \\")
            for p in sig_pairs:
                p_disp = f"< 0.001" if p["p_corrected"] < 0.001 else f"{p['p_corrected']:.4f}"
                lines.append(
                    f"{p['model_1']} vs. {p['model_2']} & — & ${p_disp}$ "
                    f"& $r = {p['effect_size_r']}$ \\\\"
                )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "statistical_tests.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_llm_configs_table():
    """Table: LLM configurations used in experiments."""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{LLM configurations used for annotation.}",
        r"\label{tab:llm_configs}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Parameters} & \textbf{Disk Size} & \textbf{Quant.} & \textbf{Context} & \textbf{Provider} & \textbf{Time/sent} \\",
        r"\midrule",
    ]

    for model_key in MODEL_ORDER:
        cfg = MODELS[model_key]
        time_str = f"{cfg['mean_inference_time']:.1f}s" if cfg['mean_inference_time'] else "API"
        lines.append(
            f"{cfg['short_name']} & {cfg['params']} & {cfg['disk_size']} & "
            f"{cfg['quantization']} & {cfg['context_window']} & {cfg['provider']} & {time_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "llm_configs.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_hyperparameters_table():
    """Table: BERT training hyperparameters."""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{BERT fine-tuning hyperparameters.}",
        r"\label{tab:hyperparameters}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
        f"Base model & {TRAINING_MODEL} \\\\",
        f"Learning rate & {HYPERPARAMETERS['learning_rate']} \\\\",
        f"Batch size & {HYPERPARAMETERS['batch_size']} \\\\",
        f"Max epochs & {HYPERPARAMETERS['epochs']} \\\\",
        f"Warmup ratio & {HYPERPARAMETERS['warmup_ratio']} \\\\",
        f"Weight decay & {HYPERPARAMETERS['weight_decay']} \\\\",
        f"Optimizer & {HYPERPARAMETERS['optimizer']} \\\\",
        f"Scheduler & {HYPERPARAMETERS['scheduler']} \\\\",
        f"Max sequence length & {HYPERPARAMETERS['max_seq_length']} \\\\",
        f"Early stopping patience & {HYPERPARAMETERS['early_stopping_patience']} \\\\",
        f"Random seed & {HYPERPARAMETERS['seed']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "hyperparameters.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_training_metrics_table():
    """Table: Training validation metrics per model × task."""
    train = pd.read_csv(DATA_DIR / "training_metrics.csv")

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{BERT training validation performance per LLM source and task.}",
        r"\label{tab:training_metrics}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{LLM Source} & \textbf{Task} & \textbf{F1 Macro} & \textbf{F1 Micro} & \textbf{Accuracy} & \textbf{Best Epoch} \\",
        r"\midrule",
    ]

    prev_model = None
    for _, row in train.iterrows():
        model_name = MODELS.get(row["model"], {}).get("short_name", row["model"])
        task_name = TASKS.get(row["task"], {}).get("display_name", row["task_name"])

        if prev_model and prev_model != row["model"]:
            lines.append(r"\midrule")
        prev_model = row["model"]

        lines.append(
            f"{model_name} & {task_name} & {fmt(row['f1_macro'])} & {fmt(row['f1_micro'])} "
            f"& {fmt(row['accuracy'])} & {int(row['best_epoch']) if pd.notna(row['best_epoch']) else '—'} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "training_validation_metrics.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_corpus_statistics_table():
    """Table: Corpus composition."""
    df = pd.read_parquet(DATA_DIR / "benchmark_clean.parquet")

    n_total = len(df)
    n_en = (df["lang"] == "EN").sum()
    n_fr = (df["lang"] == "FR").sum()
    n_parl = (df["source"] == "parliament").sum()
    n_head = (df["source"] == "headlines").sum()

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Corpus composition.}",
        r"\label{tab:corpus_stats}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{Language}} & \multicolumn{2}{c}{\textbf{Source}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Dataset} & \textbf{N} & \textbf{EN} & \textbf{FR} & \textbf{Parl.} & \textbf{Media} \\",
        r"\midrule",
        f"Test set & {n_total:,} & {n_en:,} ({n_en/n_total*100:.0f}\\%) & {n_fr:,} ({n_fr/n_total*100:.0f}\\%) & {n_parl:,} ({n_parl/n_total*100:.0f}\\%) & {n_head:,} ({n_head/n_total*100:.0f}\\%) \\\\",
        r"Training (per LLM) & 38,451 & 19,209 (50\%) & 19,242 (50\%) & 19,202 (50\%) & 19,249 (50\%) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "corpus_statistics.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_category_performance_table():
    """Table: Best model per annotation dimension."""
    overall = pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Best performing models by annotation dimension.}",
        r"\label{tab:category_performance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Dimension} & \textbf{Best Model} & \textbf{Micro F1} & \textbf{Macro F1} & \textbf{Weighted F1} \\",
        r"\midrule",
    ]

    for task_key in TASK_ORDER:
        task_data = overall[overall["task"] == task_key]
        if task_data.empty:
            continue
        best = task_data.loc[task_data["micro_f1"].idxmax()]
        lines.append(
            f"{TASKS[task_key]['display_name']} & {best['short_name']} "
            f"& {fmt(best['micro_f1'])} & {fmt(best['macro_f1'])} & {fmt(best['weighted_f1'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "category_performance.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_efficiency_table():
    """Table: Inference time comparison."""
    ann = pd.read_csv(DATA_DIR / "annotation_stats.csv")

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Inference time comparison: LLM annotation vs. BERT classification.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & \textbf{Parameters} & \textbf{Time/sent} & \textbf{Total (38k)} & \textbf{Speedup} \\",
        r"\midrule",
    ]

    bert_time = 0.05  # seconds per sample

    for _, row in ann.iterrows():
        params = row["params"] if row["params"] != "NaN" and str(row["params"]) != "nan" else "Proprietary"
        if pd.isna(row["mean_inference_time"]):
            time_str = "API"
            total_str = "—"
            speedup = "—"
        else:
            t = row["mean_inference_time"]
            total_h = t * 38_451 / 3600
            time_str = f"{t:.1f}s"
            total_str = f"{total_h:.1f}h"
            speedup = f"{t / bert_time:,.0f}$\\times$"

        lines.append(
            f"{row['short_name']} & LLM & {params} & {time_str} & {total_str} & {speedup} \\\\"
        )

    # BERT row
    bert_total = bert_time * 38_451 / 60
    lines.append(r"\midrule")
    lines.append(
        f"XLM-RoBERTa & BERT & 278M & {bert_time*1000:.0f}ms & {bert_total:.0f}min & 1$\\times$ \\\\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "efficiency.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


def generate_results_commands():
    """Generate LaTeX \\newcommand macros for inline results."""
    overall = pd.read_csv(DATA_DIR / "benchmark_overall_metrics.csv")
    agg = pd.read_csv(DATA_DIR / "benchmark_aggregate_metrics.csv")

    # Best model
    best = agg.iloc[0]
    best_model = best["display_name"]

    # Load statistical tests
    stat_results = {}
    stat_path = RESULTS_DIR / "statistical_tests.json"
    if stat_path.exists():
        with open(stat_path) as f:
            stat_results = json.load(f)

    cmds = [
        "% LaTeX commands for inline results",
        "% Auto-generated — do not edit manually",
        "",
        f"\\newcommand{{\\bestModelName}}{{{best_model}}}",
        f"\\newcommand{{\\bestMeanMicroF}}{{{fmt(best['mean_micro_f1'])}}}",
        f"\\newcommand{{\\bestMeanMacroF}}{{{fmt(best['mean_macro_f1'])}}}",
        f"\\newcommand{{\\bestMeanWeightedF}}{{{fmt(best['mean_weighted_f1'])}}}",
        "",
        f"\\newcommand{{\\testSetSize}}{{{len(pd.read_parquet(DATA_DIR / 'benchmark_clean.parquet'))}}}",
        f"\\newcommand{{\\trainingSetSize}}{{38,451}}",
        f"\\newcommand{{\\numModels}}{{{len(MODEL_ORDER)}}}",
        f"\\newcommand{{\\numTasks}}{{{len(TASK_ORDER)}}}",
        "",
    ]

    # Statistical test results
    if "global" in stat_results:
        gl = stat_results["global"]
        fr = gl.get("friedman", {})
        kw = gl.get("kruskal_wallis", {})
        if fr.get("chi_square"):
            cmds.append(f"\\newcommand{{\\friedmanChiSquare}}{{{fr['chi_square']}}}")
            cmds.append(f"\\newcommand{{\\friedmanP}}{{{fr['p_display']}}}")
            cmds.append(f"\\newcommand{{\\kendallW}}{{{fr['kendall_w']}}}")
        if kw.get("H_statistic"):
            cmds.append(f"\\newcommand{{\\kruskalH}}{{{kw['H_statistic']}}}")
            cmds.append(f"\\newcommand{{\\kruskalP}}{{{kw['p_display']}}}")
            cmds.append(f"\\newcommand{{\\etaSquared}}{{{kw['eta_squared']}}}")

    # Per-task best
    cmds.append("")
    for task_key in TASK_ORDER:
        task_data = overall[overall["task"] == task_key]
        if task_data.empty:
            continue
        best_row = task_data.loc[task_data["micro_f1"].idxmax()]
        safe_key = task_key.replace("_", "")
        cmds.append(f"\\newcommand{{\\best{safe_key.title()}Model}}{{{best_row['short_name']}}}")
        cmds.append(f"\\newcommand{{\\best{safe_key.title()}MicroF}}{{{fmt(best_row['micro_f1'])}}}")

    out = RESULTS_DIR / "results_commands.tex"
    out.write_text("\n".join(cmds))
    print(f"  Saved: {out.name}")


if __name__ == "__main__":
    print("Generating LaTeX tables...")
    generate_main_results_table()
    generate_per_task_table()
    generate_statistical_tests_table()
    generate_llm_configs_table()
    generate_hyperparameters_table()
    generate_training_metrics_table()
    generate_corpus_statistics_table()
    generate_category_performance_table()
    generate_efficiency_table()
    generate_results_commands()
    print("Done.")
