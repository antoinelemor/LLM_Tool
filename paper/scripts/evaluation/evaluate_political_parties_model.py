#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
evaluate_political_parties_model.py

MAIN OBJECTIVE:
---------------
Evaluate political parties classification model performance with publication-quality
visualizations. Compares model predictions against human consensus benchmark,
excluding 'null' class from all calculations.

Dependencies:
-------------
- json
- warnings
- pathlib
- typing
- collections
- logging
- numpy
- pandas
- matplotlib
- scipy
- sklearn

MAIN FEATURES:
--------------
1) Calculate per-party F1 scores with support analysis
2) Generate Micro/Macro F1 with bootstrap confidence intervals
3) Analyze LLM-to-Human transfer efficiency
4) Compare model performance against human annotators
5) Visualize support vs F1 correlation
6) Create publication-quality evaluation dashboard

Author:
-------
Antoine Lemor
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': 'white',
})

# Color palette
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'success': '#059669',      # Green
    'warning': '#d97706',      # Orange
    'danger': '#dc2626',       # Red
    'info': '#0891b2',         # Cyan
    'dark': '#1f2937',         # Dark gray
    'light': '#f3f4f6',        # Light gray
    'muted': '#6b7280',        # Muted gray
}

# Canadian political party colors
PARTY_COLORS = {
    'LPC': '#D71920',
    'CPC': '#1A4782',
    'NDP': '#F37021',
    'BQ': '#33B2CC',
    'GPC': '#3D9B35',
    'PQ': '#004C9D',
    'CAQ': '#00A7E1',
    'QS': '#FF5605',
    'PLQ': '#ED1B24',
    'PCQ': '#003DA5',
}


class PoliticalPartiesEvaluator:
    """Evaluator for political parties multi-label classification."""

    def __init__(
        self,
        annotations_path: str,
        output_dir: Optional[str] = None,
        consensus_col: str = 'consensus_political_parties',
        prediction_col: str = 'labels_label',
        probability_col: str = 'labels_probability',
        annotator_cols: Optional[List[str]] = None,
        training_metadata_path: Optional[str] = None,
    ):
        self.annotations_path = Path(annotations_path)
        self.output_dir = Path(output_dir) if output_dir else self.annotations_path.parent / 'evaluation_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.consensus_col = consensus_col
        self.prediction_col = prediction_col
        self.probability_col = probability_col

        self.annotator_cols = annotator_cols or [
            'shdin_political_parties',
            'Jeremy_political_parties',
            'jdrouin_political_parties',
            'BenjaminCarignan_political_parties'
        ]

        self.training_metadata_path = training_metadata_path or \
            'models/GPT_5_EXTRA_EXTRA_LARGE_20260124_213418/normal_training/labels/model_epoch_16/training_metadata.json'
        self.training_metadata = self._load_training_metadata()

        self.df = pd.read_csv(self.annotations_path)
        logger.info(f"Loaded {len(self.df)} samples from {self.annotations_path}")

        self.all_parties = self._get_all_parties()
        self.mlb = MultiLabelBinarizer(classes=sorted(self.all_parties))
        self.y_true, self.y_pred, self.raw_probs = self._prepare_data()

        logger.info(f"Evaluating {len(self.all_parties)} political parties (null excluded)")

    def _load_training_metadata(self) -> Dict:
        try:
            base_dir = Path(__file__).parent.parent.parent.parent
            metadata_path = base_dir / self.training_metadata_path
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training metadata: {e}")
        return {}

    def _parse_labels(self, val) -> List[str]:
        if pd.isna(val) or val == '[]' or val == '':
            return []
        try:
            if isinstance(val, str):
                parsed = json.loads(val.replace("'", '"'))
                return parsed if isinstance(parsed, list) else []
            return []
        except:
            return []

    def _normalize(self, label: str) -> str:
        return label.replace('political_parties_long_', '').replace('political_parties_', '')

    def _get_all_parties(self) -> List[str]:
        parties = set()
        for _, row in self.df.iterrows():
            for label in self._parse_labels(row[self.consensus_col]):
                parties.add(self._normalize(label))
            for label in self._parse_labels(row[self.prediction_col]):
                parties.add(self._normalize(label))
        parties.discard('')
        parties.discard('null')
        return sorted(list(parties))

    def _prepare_data(self):
        y_true_labels, y_pred_labels, probs = [], [], []

        for _, row in self.df.iterrows():
            true = [self._normalize(l) for l in self._parse_labels(row[self.consensus_col])]
            true = [l for l in true if l != 'null' and l != '']
            y_true_labels.append(true)

            pred = [self._normalize(l) for l in self._parse_labels(row[self.prediction_col])]
            pred = [l for l in pred if l != 'null' and l != '']
            y_pred_labels.append(pred)

            try:
                p = json.loads(str(row[self.probability_col]).replace("'", '"'))
                probs.append(p[0] if isinstance(p, list) and p else 0.0)
            except:
                probs.append(0.0)

        self.mlb.fit([self.all_parties])
        return self.mlb.transform(y_true_labels), self.mlb.transform(y_pred_labels), np.array(probs)

    def calculate_metrics(self) -> Dict[str, Any]:
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, zero_division=0
        )

        per_class = {}
        for i, party in enumerate(self.mlb.classes_):
            per_class[party] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        metrics = {'per_class': per_class}
        for avg in ['macro', 'micro', 'weighted']:
            p, r, f, _ = precision_recall_fscore_support(
                self.y_true, self.y_pred, average=avg, zero_division=0
            )
            metrics[f'{avg}_precision'] = float(p)
            metrics[f'{avg}_recall'] = float(r)
            metrics[f'{avg}_f1'] = float(f)

        metrics['hamming_loss'] = float(hamming_loss(self.y_true, self.y_pred))
        metrics['subset_accuracy'] = float(accuracy_score(self.y_true, self.y_pred))
        metrics['jaccard_micro'] = float(jaccard_score(self.y_true, self.y_pred, average='micro', zero_division=0))
        metrics['jaccard_macro'] = float(jaccard_score(self.y_true, self.y_pred, average='macro', zero_division=0))

        return metrics

    def calculate_bootstrap_ci(self, n_bootstrap: int = 1000) -> Dict[str, tuple]:
        results = {}
        for metric in ['f1_macro', 'f1_micro', 'precision_macro', 'recall_macro']:
            values = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(self.y_true), len(self.y_true), replace=True)
                y_t, y_p = self.y_true[idx], self.y_pred[idx]
                avg = metric.split('_')[1]
                p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average=avg, zero_division=0)
                val = f if 'f1' in metric else (p if 'precision' in metric else r)
                values.append(val)
            results[metric] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        return results

    def analyze_ambivalence(self) -> Dict[str, Any]:
        ambivalent_mask = np.zeros(len(self.df), dtype=bool)
        ambivalence_by_party = defaultdict(int)

        for idx, row in self.df.iterrows():
            annotations = []
            for col in self.annotator_cols:
                if col in self.df.columns:
                    labels = frozenset(self._normalize(l) for l in self._parse_labels(row[col]) if self._normalize(l) != 'null')
                    annotations.append(labels)

            if len(set(annotations)) > 1:
                ambivalent_mask[idx] = True
                for ann in annotations:
                    for party in ann:
                        ambivalence_by_party[party] += 1

        n_ambivalent = int(ambivalent_mask.sum())
        n_clear = int((~ambivalent_mask).sum())

        perf_clear, perf_amb = {}, {}
        if n_clear > 0:
            p, r, f, _ = precision_recall_fscore_support(
                self.y_true[~ambivalent_mask], self.y_pred[~ambivalent_mask], average='macro', zero_division=0
            )
            perf_clear = {'precision': float(p), 'recall': float(r), 'f1': float(f)}

        if n_ambivalent > 0:
            p, r, f, _ = precision_recall_fscore_support(
                self.y_true[ambivalent_mask], self.y_pred[ambivalent_mask], average='macro', zero_division=0
            )
            perf_amb = {'precision': float(p), 'recall': float(r), 'f1': float(f)}

        return {
            'total': len(self.df),
            'ambivalent': n_ambivalent,
            'clear': n_clear,
            'rate': n_ambivalent / len(self.df),
            'by_party': dict(ambivalence_by_party),
            'perf_clear': perf_clear,
            'perf_ambivalent': perf_amb,
        }

    def calculate_inter_annotator_agreement(self) -> Dict[str, float]:
        annotator_matrices = []
        for col in self.annotator_cols:
            if col in self.df.columns:
                labels_list = []
                for _, row in self.df.iterrows():
                    labels = [self._normalize(l) for l in self._parse_labels(row[col]) if self._normalize(l) != 'null']
                    labels_list.append(labels)
                annotator_matrices.append(self.mlb.transform(labels_list))

        if len(annotator_matrices) < 2:
            return {'fleiss_kappa': float('nan')}

        kappas = []
        for i, party in enumerate(self.mlb.classes_):
            annotations = np.array([m[:, i] for m in annotator_matrices]).T
            kappa = self._fleiss_kappa(annotations)
            kappas.append(kappa)

        return {
            'fleiss_kappa_mean': float(np.nanmean(kappas)),
            'fleiss_kappa_per_class': {party: kappas[i] for i, party in enumerate(self.mlb.classes_)}
        }

    def _fleiss_kappa(self, annotations: np.ndarray) -> float:
        n, k = annotations.shape
        p_j = [np.mean(annotations == cat) for cat in [0, 1]]
        P_e = sum(p ** 2 for p in p_j)
        P_i = []
        for i in range(n):
            counts = np.bincount(annotations[i].astype(int), minlength=2)
            P_i.append((np.sum(counts ** 2) - k) / (k * (k - 1)) if k > 1 else 1)
        P_o = np.mean(P_i)
        return (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

    def analyze_llm_transfer(self) -> Dict[str, Any]:
        results = {
            'training_source': 'GPT-5 annotations',
            'evaluation_target': 'Human consensus',
        }

        if self.training_metadata:
            results['training_macro_f1'] = self.training_metadata.get('macro_f1', None)
            results['training_accuracy'] = self.training_metadata.get('accuracy', None)
            # Estimate micro F1 from accuracy for multi-label (approximation)
            results['training_micro_f1'] = self.training_metadata.get('accuracy', None)

        metrics = self.calculate_metrics()
        results['benchmark_macro_f1'] = metrics['macro_f1']
        results['benchmark_micro_f1'] = metrics['micro_f1']
        results['benchmark_accuracy'] = metrics['subset_accuracy']

        if results.get('training_macro_f1'):
            results['transfer_efficiency_macro'] = results['benchmark_macro_f1'] / results['training_macro_f1']
        if results.get('training_micro_f1'):
            results['transfer_efficiency_micro'] = results['benchmark_micro_f1'] / results['training_micro_f1']

        return results

    def analyze_annotator_vs_model(self) -> Dict[str, Any]:
        results = {'annotators': {}, 'model': {}}

        p, r, f, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='micro', zero_division=0
        )
        results['model'] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}

        for col in self.annotator_cols:
            if col not in self.df.columns:
                continue

            annotator_name = col.replace('_political_parties', '')
            y_annotator = []

            for _, row in self.df.iterrows():
                labels = [self._normalize(l) for l in self._parse_labels(row[col])]
                labels = [l for l in labels if l != 'null' and l != '']
                y_annotator.append(labels)

            y_ann_binary = self.mlb.transform(y_annotator)
            p, r, f, _ = precision_recall_fscore_support(
                self.y_true, y_ann_binary, average='micro', zero_division=0
            )
            results['annotators'][annotator_name] = {
                'precision': float(p), 'recall': float(r), 'f1': float(f)
            }

        if results['annotators']:
            avg_f1 = np.mean([v['f1'] for v in results['annotators'].values()])
            results['annotator_avg_f1'] = float(avg_f1)
            results['model_vs_annotator_gap'] = results['model']['f1'] - avg_f1

        return results

    def plot_evaluation_dashboard(self) -> plt.Figure:
        """Create publication-quality evaluation dashboard."""
        metrics = self.calculate_metrics()
        bootstrap_ci = self.calculate_bootstrap_ci(n_bootstrap=500)
        ambivalence = self.analyze_ambivalence()
        agreement = self.calculate_inter_annotator_agreement()
        llm_transfer = self.analyze_llm_transfer()
        annotator_comparison = self.analyze_annotator_vs_model()

        # Create figure
        fig = plt.figure(figsize=(20, 18))
        fig.patch.set_facecolor('white')

        gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35,
                     height_ratios=[0.12, 1.2, 1, 1, 1])

        # =================================================================
        # ROW 0: Title banner
        # =================================================================
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.set_facecolor(COLORS['dark'])

        title_text = "Political Parties Classification — Multi-Label Model Evaluation"
        ax_title.text(0.5, 0.5, title_text, transform=ax_title.transAxes,
                     fontsize=20, fontweight='bold', color=COLORS['dark'],
                     ha='center', va='center')

        subtitle = f"n = {len(self.df):,} samples  |  {len(self.all_parties)} classes  |  null excluded"
        ax_title.text(0.5, 0.1, subtitle, transform=ax_title.transAxes,
                     fontsize=12, color=COLORS['muted'], ha='center', va='center')

        # =================================================================
        # ROW 1: Per-class F1 scores (main visualization)
        # =================================================================
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.set_facecolor('#fafafa')

        per_class = metrics['per_class']
        parties = sorted(per_class.keys(), key=lambda x: per_class[x]['f1'], reverse=True)
        f1_scores = [per_class[p]['f1'] for p in parties]
        supports = [per_class[p]['support'] for p in parties]
        colors = [PARTY_COLORS.get(p, COLORS['muted']) for p in parties]

        x = np.arange(len(parties))
        bars = ax_main.bar(x, f1_scores, color=colors, edgecolor='white', linewidth=2, alpha=0.9, width=0.7)

        # Add gradient effect with edge highlight
        for bar, color in zip(bars, colors):
            bar.set_edgecolor('white')
            bar.set_linewidth(2)

        # Add Micro and Macro F1 lines
        micro_f1 = metrics['micro_f1']
        macro_f1 = metrics['macro_f1']

        ax_main.axhline(y=micro_f1, color=COLORS['success'], linestyle='-', linewidth=3, zorder=10, alpha=0.9)
        ax_main.axhline(y=macro_f1, color=COLORS['danger'], linestyle='--', linewidth=3, zorder=10, alpha=0.9)

        # Labels for lines with background boxes
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9)
        ax_main.text(len(parties) - 0.3, micro_f1 + 0.025, f'Micro F1 = {micro_f1:.3f}',
                    fontsize=12, fontweight='bold', color=COLORS['success'], ha='right', bbox=bbox_props)
        ax_main.text(len(parties) - 0.3, macro_f1 - 0.055, f'Macro F1 = {macro_f1:.3f}',
                    fontsize=12, fontweight='bold', color=COLORS['danger'], ha='right', bbox=bbox_props)

        # Annotations on bars
        for i, (bar, support, f1) in enumerate(zip(bars, supports, f1_scores)):
            # F1 value on top
            ax_main.annotate(f'{f1:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 8), textcoords="offset points",
                           ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color=COLORS['dark'])
            # Support at bottom
            ax_main.annotate(f'n={support}',
                           xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                           ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

        ax_main.set_xticks(x)
        ax_main.set_xticklabels(parties, fontsize=13, fontweight='bold')
        ax_main.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
        ax_main.set_ylim(0, 1.15)
        ax_main.set_xlim(-0.5, len(parties) - 0.5)
        ax_main.set_title('Per-Party F1-Score Performance', fontsize=16, pad=15, fontweight='bold')
        ax_main.grid(axis='y', alpha=0.3, linestyle='--')

        # =================================================================
        # ROW 2: Metrics table + Precision/Recall + Ambivalence
        # =================================================================

        # Global metrics table
        ax_metrics = fig.add_subplot(gs[2, 0:2])
        ax_metrics.axis('off')

        table_data = [
            ['Macro F1', f"{metrics['macro_f1']:.4f}",
             f"[{bootstrap_ci['f1_macro'][0]:.3f}, {bootstrap_ci['f1_macro'][1]:.3f}]"],
            ['Micro F1', f"{metrics['micro_f1']:.4f}",
             f"[{bootstrap_ci['f1_micro'][0]:.3f}, {bootstrap_ci['f1_micro'][1]:.3f}]"],
            ['Weighted F1', f"{metrics['weighted_f1']:.4f}", '—'],
            ['Precision (Macro)', f"{metrics['macro_precision']:.4f}",
             f"[{bootstrap_ci['precision_macro'][0]:.3f}, {bootstrap_ci['precision_macro'][1]:.3f}]"],
            ['Recall (Macro)', f"{metrics['macro_recall']:.4f}",
             f"[{bootstrap_ci['recall_macro'][0]:.3f}, {bootstrap_ci['recall_macro'][1]:.3f}]"],
            ['Hamming Loss', f"{metrics['hamming_loss']:.4f}", '—'],
            ['Subset Accuracy', f"{metrics['subset_accuracy']:.4f}", '—'],
            ['Jaccard (Macro)', f"{metrics['jaccard_macro']:.4f}", '—'],
        ]

        table = ax_metrics.table(
            cellText=table_data,
            colLabels=['Metric', 'Value', '95% CI'],
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.25, 0.35]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.15, 1.8)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            color = '#f8fafc' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_edgecolor('#e2e8f0')

        ax_metrics.set_title('Global Classification Metrics', fontsize=14, pad=20, fontweight='bold')

        # Precision/Recall horizontal bars
        ax_pr = fig.add_subplot(gs[2, 2])

        y_pos = np.arange(len(parties))
        precisions = [per_class[p]['precision'] for p in parties]
        recalls = [per_class[p]['recall'] for p in parties]

        ax_pr.barh(y_pos + 0.2, precisions, 0.35, label='Precision', color=COLORS['primary'], alpha=0.85)
        ax_pr.barh(y_pos - 0.2, recalls, 0.35, label='Recall', color=COLORS['success'], alpha=0.85)

        ax_pr.set_yticks(y_pos)
        ax_pr.set_yticklabels(parties, fontsize=10)
        ax_pr.set_xlim(0, 1.1)
        ax_pr.set_xlabel('Score', fontsize=11)
        ax_pr.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax_pr.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax_pr.invert_yaxis()
        ax_pr.grid(axis='x', alpha=0.3, linestyle='--')

        # Ambivalence donut chart
        ax_amb = fig.add_subplot(gs[2, 3])

        sizes = [ambivalence['clear'], ambivalence['ambivalent']]
        colors_pie = [COLORS['success'], COLORS['danger']]

        wedges, texts, autotexts = ax_amb.pie(
            sizes, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=3),
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        autotexts[0].set_color('white')
        autotexts[1].set_color('white')

        # Center text
        ax_amb.text(0, 0, f"κ = {agreement.get('fleiss_kappa_mean', 0):.2f}",
                   ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['dark'])

        ax_amb.legend(wedges, [f'Clear ({ambivalence["clear"]})', f'Ambivalent ({ambivalence["ambivalent"]})'],
                     loc='lower center', fontsize=10, framealpha=0.95)
        ax_amb.set_title('Annotator Agreement', fontsize=14, fontweight='bold')

        # =================================================================
        # ROW 3: Support vs F1 Analysis
        # =================================================================

        # Support vs F1 scatter plot with regression
        ax_support = fig.add_subplot(gs[3, 0:2])

        per_class = metrics['per_class']
        support_vals = [per_class[p]['support'] for p in self.all_parties]
        f1_vals = [per_class[p]['f1'] for p in self.all_parties]
        party_colors_scatter = [PARTY_COLORS.get(p, COLORS['muted']) for p in self.all_parties]

        # Scatter plot with party-colored points
        for i, (support, f1, party) in enumerate(zip(support_vals, f1_vals, self.all_parties)):
            ax_support.scatter(support, f1, s=200, c=PARTY_COLORS.get(party, COLORS['muted']),
                              edgecolors='white', linewidths=2, zorder=10, alpha=0.9)
            # Add party label
            ax_support.annotate(party, (support, f1), xytext=(8, 0),
                               textcoords='offset points', fontsize=10, fontweight='bold',
                               color=COLORS['dark'], va='center')

        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(support_vals, f1_vals)
        x_line = np.array([0, max(support_vals) * 1.1])
        y_line = slope * x_line + intercept
        ax_support.plot(x_line, y_line, '--', color=COLORS['danger'], linewidth=2.5, alpha=0.7,
                       label=f'Regression (r={r_value:.2f}, p={p_value:.3f})')

        ax_support.set_xlabel('Support (N samples)', fontsize=12, fontweight='bold')
        ax_support.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax_support.set_title('Support vs F1: Low-Support Classes Have Lower F1', fontsize=14, fontweight='bold')
        ax_support.set_xlim(-10, max(support_vals) * 1.15)
        ax_support.set_ylim(-0.05, 1.1)
        ax_support.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax_support.grid(alpha=0.3, linestyle='--')

        # Add annotation for low-support warning
        low_support_parties = [p for p in self.all_parties if per_class[p]['support'] < 20]
        if low_support_parties:
            warning_text = f"Low support (<20): {', '.join(low_support_parties)}"
            ax_support.text(0.02, 0.98, warning_text, transform=ax_support.transAxes,
                          fontsize=10, color=COLORS['warning'], fontweight='bold',
                          va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fef3c7', edgecolor=COLORS['warning'], linewidth=1))

        # Support distribution histogram
        ax_support_hist = fig.add_subplot(gs[3, 2:4])

        # Sort parties by support for histogram
        sorted_parties = sorted(self.all_parties, key=lambda p: per_class[p]['support'], reverse=True)
        sorted_supports = [per_class[p]['support'] for p in sorted_parties]
        sorted_colors = [PARTY_COLORS.get(p, COLORS['muted']) for p in sorted_parties]

        bars_support = ax_support_hist.bar(range(len(sorted_parties)), sorted_supports,
                                           color=sorted_colors, edgecolor='white', linewidth=2)

        # Add F1 values as text on bars
        for i, (bar, party) in enumerate(zip(bars_support, sorted_parties)):
            f1 = per_class[party]['f1']
            ax_support_hist.annotate(f'F1={f1:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

        ax_support_hist.set_xticks(range(len(sorted_parties)))
        ax_support_hist.set_xticklabels(sorted_parties, fontsize=11, fontweight='bold', rotation=30, ha='right')
        ax_support_hist.set_ylabel('Support (N)', fontsize=12, fontweight='bold')
        ax_support_hist.set_title('Support Distribution per Party', fontsize=14, fontweight='bold')
        ax_support_hist.grid(axis='y', alpha=0.3, linestyle='--')

        # =================================================================
        # ROW 4: Transfer Analysis + Model vs Annotators + Summary
        # =================================================================

        # LLM Transfer Analysis - dual bars for Macro and Micro
        ax_transfer = fig.add_subplot(gs[4, 0:2])

        transfer_x = np.arange(2)
        width = 0.35

        training_vals = [
            llm_transfer.get('training_macro_f1', 0) or 0,
            llm_transfer.get('training_micro_f1', 0) or 0
        ]
        benchmark_vals = [
            llm_transfer.get('benchmark_macro_f1', 0),
            llm_transfer.get('benchmark_micro_f1', 0)
        ]

        bars1 = ax_transfer.bar(transfer_x - width/2, training_vals, width,
                                label='Training (LLM Data)', color=COLORS['secondary'], alpha=0.85)
        bars2 = ax_transfer.bar(transfer_x + width/2, benchmark_vals, width,
                                label='Benchmark (Human)', color=COLORS['primary'], alpha=0.85)

        ax_transfer.set_xticks(transfer_x)
        ax_transfer.set_xticklabels(['Macro F1', 'Micro F1'], fontsize=12, fontweight='bold')
        ax_transfer.set_ylim(0, 1.15)
        ax_transfer.set_ylabel('F1 Score', fontsize=11)
        ax_transfer.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax_transfer.set_title('LLM → Human Transfer Performance', fontsize=14, fontweight='bold')
        ax_transfer.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax_transfer.annotate(f'{bar.get_height():.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                   xytext=(0, 5), textcoords="offset points",
                                   ha='center', fontsize=11, fontweight='bold', color=COLORS['secondary'])
        for bar in bars2:
            ax_transfer.annotate(f'{bar.get_height():.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

        # Transfer efficiency annotation
        eff_macro = llm_transfer.get('transfer_efficiency_macro', 0)
        eff_micro = llm_transfer.get('transfer_efficiency_micro', 0)
        if eff_macro and eff_micro:
            eff_text = f"Transfer: Macro {eff_macro:.0%} | Micro {eff_micro:.0%}"
            eff_color = COLORS['success'] if min(eff_macro, eff_micro) >= 0.9 else COLORS['warning']
            ax_transfer.text(0.5, 1.02, eff_text, transform=ax_transfer.transAxes,
                           ha='center', fontsize=11, fontweight='bold', color=eff_color)

        # Model vs Annotators
        ax_annotators = fig.add_subplot(gs[4, 2])

        comparison_names = ['Model']
        comparison_f1s = [annotator_comparison['model']['f1']]
        comparison_colors = [COLORS['danger']]

        for name, data in sorted(annotator_comparison['annotators'].items()):
            comparison_names.append(name.split('_')[0][:8])  # Truncate names
            comparison_f1s.append(data['f1'])
            comparison_colors.append(COLORS['primary'])

        x_comp = np.arange(len(comparison_names))
        bars_comp = ax_annotators.bar(x_comp, comparison_f1s, color=comparison_colors,
                                      edgecolor='white', linewidth=2, alpha=0.85)

        if annotator_comparison.get('annotator_avg_f1'):
            avg_f1 = annotator_comparison['annotator_avg_f1']
            ax_annotators.axhline(y=avg_f1, color=COLORS['info'], linestyle='--', linewidth=2.5,
                                 label=f'Human Avg: {avg_f1:.3f}')

        for bar in bars_comp:
            ax_annotators.annotate(f'{bar.get_height():.2f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                  xytext=(0, 5), textcoords="offset points",
                                  ha='center', fontsize=10, fontweight='bold')

        ax_annotators.set_xticks(x_comp)
        ax_annotators.set_xticklabels(comparison_names, fontsize=10, rotation=30, ha='right')
        ax_annotators.set_ylim(0, 1.1)
        ax_annotators.set_ylabel('Micro F1', fontsize=11)
        ax_annotators.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_annotators.set_title('Model vs Annotators', fontsize=14, fontweight='bold')
        ax_annotators.grid(axis='y', alpha=0.3, linestyle='--')

        # Summary box
        ax_summary = fig.add_subplot(gs[4, 3])
        ax_summary.axis('off')

        # Key findings
        model_gap = annotator_comparison.get('model_vs_annotator_gap', 0)
        gap_text = f"+{model_gap:.1%}" if model_gap >= 0 else f"{model_gap:.1%}"
        gap_color = COLORS['success'] if model_gap >= 0 else COLORS['warning']

        summary_items = [
            ('Micro F1', f"{metrics['micro_f1']:.3f}", COLORS['success']),
            ('Macro F1', f"{metrics['macro_f1']:.3f}", COLORS['primary']),
            ('Hamming Loss', f"{metrics['hamming_loss']:.4f}", COLORS['info']),
            ("Fleiss' κ", f"{agreement.get('fleiss_kappa_mean', 0):.3f}", COLORS['secondary']),
            ('vs Human Avg', gap_text, gap_color),
        ]

        y_offset = 0.9
        ax_summary.text(0.5, 0.98, 'KEY METRICS', transform=ax_summary.transAxes,
                       fontsize=13, fontweight='bold', ha='center', va='top', color=COLORS['dark'])

        for label, value, color in summary_items:
            ax_summary.text(0.1, y_offset, label, transform=ax_summary.transAxes,
                           fontsize=11, ha='left', va='top', color=COLORS['muted'])
            ax_summary.text(0.9, y_offset, value, transform=ax_summary.transAxes,
                           fontsize=13, fontweight='bold', ha='right', va='top', color=color)
            y_offset -= 0.15

        # Add decorative box
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0.02, 0.15), 0.96, 0.82, transform=ax_summary.transAxes,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=2)
        ax_summary.add_patch(rect)

        # Save
        fig.savefig(self.output_dir / 'political_parties_evaluation_dashboard.png', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=300)
        fig.savefig(self.output_dir / 'political_parties_evaluation_dashboard.pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        logger.info(f"Saved dashboard to {self.output_dir}")

        return fig

    def run_evaluation(self):
        """Run full evaluation and generate outputs."""
        logger.info("=" * 60)
        logger.info("Political Parties Classification Evaluation")
        logger.info("=" * 60)

        metrics = self.calculate_metrics()
        bootstrap_ci = self.calculate_bootstrap_ci()
        ambivalence = self.analyze_ambivalence()
        agreement = self.calculate_inter_annotator_agreement()
        llm_transfer = self.analyze_llm_transfer()
        annotator_comparison = self.analyze_annotator_vs_model()

        self.plot_evaluation_dashboard()

        results = {
            'dataset': {
                'samples': len(self.df),
                'classes': len(self.all_parties),
                'parties': self.all_parties,
            },
            'metrics': metrics,
            'bootstrap_ci': {k: list(v) for k, v in bootstrap_ci.items()},
            'ambivalence': {k: v for k, v in ambivalence.items() if k != 'by_party'},
            'inter_annotator_agreement': {k: v for k, v in agreement.items() if k != 'fleiss_kappa_per_class'},
            'llm_transfer': {k: v for k, v in llm_transfer.items() if k != 'training_per_class_f1'},
            'annotator_comparison': annotator_comparison,
        }

        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "=" * 65)
        print("  EVALUATION RESULTS (null class excluded)")
        print("=" * 65)
        print(f"\n  {'Dataset:':<20} {len(self.df):,} samples, {len(self.all_parties)} classes")
        print(f"\n  {'─' * 61}")
        print(f"  CLASSIFICATION METRICS")
        print(f"  {'─' * 61}")
        print(f"  {'Micro F1:':<20} {metrics['micro_f1']:.4f}  [{bootstrap_ci['f1_micro'][0]:.3f}, {bootstrap_ci['f1_micro'][1]:.3f}]")
        print(f"  {'Macro F1:':<20} {metrics['macro_f1']:.4f}  [{bootstrap_ci['f1_macro'][0]:.3f}, {bootstrap_ci['f1_macro'][1]:.3f}]")
        print(f"  {'Weighted F1:':<20} {metrics['weighted_f1']:.4f}")
        print(f"  {'Hamming Loss:':<20} {metrics['hamming_loss']:.4f}")
        print(f"\n  {'─' * 61}")
        print(f"  LLM → HUMAN TRANSFER")
        print(f"  {'─' * 61}")
        print(f"  {'Training (LLM):':<20} Macro={llm_transfer.get('training_macro_f1', 0):.4f}  Micro={llm_transfer.get('training_micro_f1', 0):.4f}")
        print(f"  {'Benchmark (Human):':<20} Macro={llm_transfer.get('benchmark_macro_f1', 0):.4f}  Micro={llm_transfer.get('benchmark_micro_f1', 0):.4f}")
        if llm_transfer.get('transfer_efficiency_macro'):
            print(f"  {'Transfer Efficiency:':<20} Macro={llm_transfer['transfer_efficiency_macro']:.1%}  Micro={llm_transfer.get('transfer_efficiency_micro', 0):.1%}")
        print(f"\n  {'─' * 61}")
        print(f"  MODEL VS ANNOTATORS")
        print(f"  {'─' * 61}")
        print(f"  {'Model (Micro F1):':<20} {annotator_comparison['model']['f1']:.4f}")
        if annotator_comparison.get('annotator_avg_f1'):
            print(f"  {'Human Avg:':<20} {annotator_comparison['annotator_avg_f1']:.4f}")
            gap = annotator_comparison.get('model_vs_annotator_gap', 0)
            gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
            print(f"  {'Gap:':<20} {gap_str}")
        print(f"\n  {'─' * 61}")
        print(f"  INTER-ANNOTATOR AGREEMENT")
        print(f"  {'─' * 61}")
        print(f"  {'Fleiss κ:':<20} {agreement.get('fleiss_kappa_mean', 0):.4f}")
        print(f"  {'Ambivalence:':<20} {ambivalence['rate']*100:.1f}%")
        print(f"\n  Output: {self.output_dir}")
        print("=" * 65)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Political Parties Classification Model Evaluation')
    parser.add_argument('--annotations', type=str,
                       default='logs/annotation_studio/benchmark_party_20260125_101922/annotations.csv')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent.parent
    annotations_path = base_dir / args.annotations

    if not annotations_path.exists():
        logger.error(f"File not found: {annotations_path}")
        return

    evaluator = PoliticalPartiesEvaluator(
        annotations_path=str(annotations_path),
        output_dir=args.output_dir,
    )
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
