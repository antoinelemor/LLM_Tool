#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
evaluate_themes_model.py

MAIN OBJECTIVE:
---------------
Evaluate multi-label theme classification model performance with publication-quality
visualizations. Compares model predictions against human consensus benchmark,
taking into account annotation ambiguity and inter-annotator disagreement.
Excludes 'null' class from all calculations.

Dependencies:
-------------
- json
- ast
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
1) Calculate per-theme F1 scores with support analysis
2) Generate Micro/Macro F1 with bootstrap confidence intervals
3) Analyze annotation ambiguity stratified by disagreement level
4) Compare model performance against individual human annotators
5) Compute inter-annotator agreement (Fleiss' kappa, Krippendorff's alpha)
6) Calibration analysis (ECE, reliability diagram)
7) LLM-to-Human transfer efficiency analysis
8) Export publication-quality CSV and dashboard PNG/PDF
9) CLI interface for reuse across different models and annotation runs

Author:
-------
Antoine Lemor
"""

import json
import ast
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from itertools import combinations
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
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
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'info': '#0891b2',
    'dark': '#1f2937',
    'light': '#f3f4f6',
    'muted': '#6b7280',
}

# Theme color palette (categorical)
THEME_PALETTE = [
    '#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed',
    '#0891b2', '#be185d', '#4f46e5', '#ea580c', '#0d9488',
    '#9333ea', '#c026d3', '#65a30d', '#0369a1', '#b91c1c',
    '#4338ca', '#15803d', '#a16207', '#7e22ce', '#0e7490',
    '#9f1239', '#1d4ed8', '#047857',
]

# Theme display names mapping
THEME_DISPLAY = {
    'agriculture': 'Agriculture',
    'culture_nationalism': 'Culture & Nationalism',
    'defense': 'Defense',
    'domestic_commerce': 'Domestic Commerce',
    'education': 'Education',
    'energy': 'Energy',
    'environment': 'Environment',
    'foreign_trade': 'Foreign Trade',
    'governments_governance': 'Government & Governance',
    'health': 'Health',
    'housing': 'Housing',
    'immigration': 'Immigration',
    'indigenous_affairs': 'Indigenous Affairs',
    'international_affairs': 'International Affairs',
    'labor': 'Labor',
    'law_and_crime': 'Law & Crime',
    'macroeconomics': 'Macroeconomics',
    'public_lands': 'Public Lands',
    'rights_liberties_minorities_discrimination': 'Rights & Minorities',
    'social_welfare': 'Social Welfare',
    'technology': 'Technology',
    'transportation': 'Transportation',
}

# Agreement level thresholds (Landis & Koch)
AGREEMENT_LEVELS = {
    'Almost Perfect': {'min': 0.81, 'max': 1.01, 'color': '#2ecc71'},
    'Substantial': {'min': 0.61, 'max': 0.81, 'color': '#3498db'},
    'Moderate': {'min': 0.41, 'max': 0.61, 'color': '#f39c12'},
    'Fair': {'min': 0.21, 'max': 0.41, 'color': '#e74c3c'},
    'Slight': {'min': 0.00, 'max': 0.21, 'color': '#95a5a6'},
}


def get_agreement_level(kappa: float) -> str:
    """Return Landis & Koch agreement level label."""
    for level, cfg in AGREEMENT_LEVELS.items():
        if cfg['min'] <= kappa < cfg['max']:
            return level
    return 'Slight'


class ThemesEvaluator:
    """Evaluator for multi-label theme classification."""

    def __init__(
        self,
        annotations_path: str,
        output_dir: Optional[str] = None,
        consensus_col: str = 'consensus_themes',
        prediction_col: Optional[str] = None,
        probability_col: Optional[str] = None,
        annotator_cols: Optional[List[str]] = None,
        training_metadata_path: Optional[str] = None,
        training_arena_dir: Optional[str] = None,
    ):
        self.annotations_path = Path(annotations_path)
        self.output_dir = Path(output_dir) if output_dir else self.annotations_path.parent / 'evaluation_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.consensus_col = consensus_col

        self.df = pd.read_csv(self.annotations_path)
        logger.info(f"Loaded {len(self.df)} samples from {self.annotations_path}")

        # Auto-detect prediction column if not specified
        self.prediction_col = prediction_col or self._detect_column('_labels_label')
        self.probability_col = probability_col or self._detect_column('_labels_probability')

        # Auto-detect annotator columns
        self.annotator_cols = annotator_cols or self._detect_annotator_cols()

        # Load training metadata
        self.training_metadata_path = training_metadata_path
        self.training_metadata = self._load_training_metadata()

        # Load training arena data (per-class sample counts)
        self.training_arena_dir = training_arena_dir
        self.training_sample_counts = self._load_training_sample_counts()

        # Prepare data
        self.all_themes = self._get_all_themes()
        self.mlb = MultiLabelBinarizer(classes=sorted(self.all_themes))
        self.y_true, self.y_pred, self.raw_probs = self._prepare_data()

        logger.info(f"Prediction column: {self.prediction_col}")
        logger.info(f"Probability column: {self.probability_col}")
        logger.info(f"Annotator columns: {self.annotator_cols}")
        logger.info(f"Evaluating {len(self.all_themes)} themes (null excluded)")
        if self.training_sample_counts:
            logger.info(f"Training sample counts loaded for {len(self.training_sample_counts)} themes")

    def _detect_column(self, suffix: str) -> str:
        """Auto-detect column ending with a given suffix."""
        candidates = [c for c in self.df.columns if c.endswith(suffix)]
        if candidates:
            return candidates[0]
        raise ValueError(f"No column ending with '{suffix}' found. Available: {list(self.df.columns)}")

    def _detect_annotator_cols(self) -> List[str]:
        """Auto-detect annotator theme columns."""
        cols = [c for c in self.df.columns
                if c.endswith('_themes')
                and c != self.consensus_col
                and not c.startswith('consensus_')
                and 'specific' not in c.lower()]
        return cols

    def _load_training_metadata(self) -> Dict:
        """Load training metadata JSON if available."""
        if self.training_metadata_path:
            try:
                p = Path(self.training_metadata_path)
                if not p.is_absolute():
                    p = Path(__file__).parent.parent.parent.parent / p
                if p.exists():
                    with open(p, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load training metadata: {e}")
        # Try auto-detect from model directory structure
        try:
            base_dir = Path(__file__).parent.parent.parent.parent
            model_dirs = sorted(base_dir.glob('models/*THEMES*/normal_training/labels/model_epoch_*/training_metadata.json'))
            if model_dirs:
                with open(model_dirs[-1], 'r') as f:
                    logger.info(f"Auto-detected training metadata: {model_dirs[-1]}")
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _load_training_sample_counts(self) -> Dict[str, int]:
        """Load per-theme training sample counts from training arena metadata."""
        base_dir = Path(__file__).parent.parent.parent.parent

        # 1. Explicit path
        if self.training_arena_dir:
            p = Path(self.training_arena_dir)
            if not p.is_absolute():
                p = base_dir / p
            meta = p / 'training_session_metadata' / 'training_metadata.json'
            if meta.exists():
                try:
                    with open(meta, 'r') as f:
                        data = json.load(f)
                    counts = data.get('dataset_config', {}).get('value_counts_by_key', {}).get('themes_long', {})
                    if counts:
                        # Clean keys: skip garbage like "[", normalize "null,"
                        return {k: v for k, v in counts.items()
                                if k not in ('[', 'null,', 'null') and len(k) > 1}
                except Exception as e:
                    logger.warning(f"Could not load training arena metadata: {e}")

        # 2. Auto-detect from logs/training_arena/*THEMES*
        try:
            arena_dirs = sorted(base_dir.glob('logs/training_arena/*THEMES*/training_session_metadata/training_metadata.json'))
            if arena_dirs:
                with open(arena_dirs[-1], 'r') as f:
                    data = json.load(f)
                counts = data.get('dataset_config', {}).get('value_counts_by_key', {}).get('themes_long', {})
                if counts:
                    logger.info(f"Auto-detected training arena: {arena_dirs[-1].parent.parent.name}")
                    return {k: v for k, v in counts.items()
                            if k not in ('[', 'null,', 'null') and len(k) > 1}
        except Exception:
            pass

        return {}

    def _parse_labels(self, val) -> List[str]:
        """Parse label list from CSV cell."""
        if pd.isna(val) or val == '[]' or val == '':
            return []
        try:
            if isinstance(val, str):
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            return []
        except Exception:
            try:
                return json.loads(val.replace("'", '"'))
            except Exception:
                return []

    def _normalize(self, label: str) -> str:
        """Normalize theme label to a clean short name."""
        if not label:
            return ''
        l = label.lower()
        prefixes = [
            'themes_long_',
            'consensus_themes_extra_large_',
            'consensus_themes_large_',
            'consensus_themes_',
            'theme_',
        ]
        for prefix in prefixes:
            if l.startswith(prefix):
                l = l[len(prefix):]
                break
        return l

    def _get_all_themes(self) -> List[str]:
        """Get all unique theme labels (excluding null)."""
        themes = set()
        for _, row in self.df.iterrows():
            for label in self._parse_labels(row[self.consensus_col]):
                themes.add(self._normalize(label))
            for label in self._parse_labels(row[self.prediction_col]):
                themes.add(self._normalize(label))
        themes.discard('')
        themes.discard('null')
        return sorted(list(themes))

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binarize ground truth and predictions."""
        y_true_labels, y_pred_labels, probs = [], [], []

        for _, row in self.df.iterrows():
            true = [self._normalize(l) for l in self._parse_labels(row[self.consensus_col])]
            true = [l for l in true if l != 'null' and l != '']
            y_true_labels.append(true)

            pred = [self._normalize(l) for l in self._parse_labels(row[self.prediction_col])]
            pred = [l for l in pred if l != 'null' and l != '']
            y_pred_labels.append(pred)

            try:
                p = self._parse_labels(str(row.get(self.probability_col, '[]')))
                probs.append(float(p[0]) if p else 0.0)
            except Exception:
                probs.append(0.0)

        self.mlb.fit([self.all_themes])
        return self.mlb.transform(y_true_labels), self.mlb.transform(y_pred_labels), np.array(probs)

    # =====================================================================
    # METRICS
    # =====================================================================

    def calculate_metrics(self) -> Dict[str, Any]:
        """Compute all classification metrics."""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, zero_division=0
        )

        per_class = {}
        for i, theme in enumerate(self.mlb.classes_):
            tp = int(((self.y_true[:, i] == 1) & (self.y_pred[:, i] == 1)).sum())
            fp = int(((self.y_true[:, i] == 0) & (self.y_pred[:, i] == 1)).sum())
            fn = int(((self.y_true[:, i] == 1) & (self.y_pred[:, i] == 0)).sum())
            per_class[theme] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
                'tp': tp, 'fp': fp, 'fn': fn,
                'display_name': THEME_DISPLAY.get(theme, theme.replace('_', ' ').title()),
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

    def calculate_bootstrap_ci(self, n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
        """Bootstrap 95% confidence intervals for key metrics."""
        rng = np.random.RandomState(42)
        results = {}
        for metric in ['f1_macro', 'f1_micro', 'precision_macro', 'recall_macro']:
            values = []
            for _ in range(n_bootstrap):
                idx = rng.choice(len(self.y_true), len(self.y_true), replace=True)
                y_t, y_p = self.y_true[idx], self.y_pred[idx]
                avg = metric.split('_')[1]
                p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average=avg, zero_division=0)
                val = f if 'f1' in metric else (p if 'precision' in metric else r)
                values.append(val)
            results[metric] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        return results

    # =====================================================================
    # AMBIGUITY / INTER-ANNOTATOR AGREEMENT
    # =====================================================================

    def analyze_ambiguity(self) -> Dict[str, Any]:
        """Analyze annotation ambiguity from annotator disagreement."""
        disagreement = np.zeros(len(self.df))
        ambivalent_mask = np.zeros(len(self.df), dtype=bool)

        for idx, (_, row) in enumerate(self.df.iterrows()):
            annotations = []
            for col in self.annotator_cols:
                if col in self.df.columns:
                    labels = frozenset(
                        self._normalize(l) for l in self._parse_labels(row[col])
                        if self._normalize(l) not in ('null', '')
                    )
                    annotations.append(labels)

            if len(annotations) < 2:
                continue

            # Pairwise Jaccard → disagreement index
            non_empty = [a for a in annotations if a]
            if len(non_empty) >= 2:
                jaccards = []
                for a, b in combinations(non_empty, 2):
                    inter = len(a & b)
                    union = len(a | b)
                    jaccards.append(inter / union if union > 0 else 0)
                disagreement[idx] = 1 - np.mean(jaccards)

            if len(set(annotations)) > 1:
                ambivalent_mask[idx] = True

        # Stratified performance
        low_mask = disagreement < 0.33
        med_mask = (disagreement >= 0.33) & (disagreement < 0.66)
        high_mask = disagreement >= 0.66

        strat = {}
        for name, mask in [('low', low_mask), ('medium', med_mask), ('high', high_mask)]:
            n = int(mask.sum())
            perf = {}
            if n > 0:
                p, r, f, _ = precision_recall_fscore_support(
                    self.y_true[mask], self.y_pred[mask], average='macro', zero_division=0
                )
                pi, ri, fi, _ = precision_recall_fscore_support(
                    self.y_true[mask], self.y_pred[mask], average='micro', zero_division=0
                )
                perf = {'precision_macro': float(p), 'recall_macro': float(r), 'f1_macro': float(f),
                        'f1_micro': float(fi)}
            strat[name] = {'n_samples': n, 'metrics': perf}

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
            'rate': n_ambivalent / len(self.df) if len(self.df) > 0 else 0,
            'perf_clear': perf_clear,
            'perf_ambivalent': perf_amb,
            'stratified': strat,
            'disagreement_index': disagreement,
        }

    def calculate_inter_annotator_agreement(self) -> Dict[str, Any]:
        """Compute Fleiss' kappa per theme and overall."""
        annotator_matrices = []
        for col in self.annotator_cols:
            if col in self.df.columns:
                labels_list = []
                for _, row in self.df.iterrows():
                    labels = [self._normalize(l) for l in self._parse_labels(row[col])
                              if self._normalize(l) not in ('null', '')]
                    labels_list.append(labels)
                annotator_matrices.append(self.mlb.transform(labels_list))

        if len(annotator_matrices) < 2:
            return {'fleiss_kappa_mean': float('nan'), 'fleiss_kappa_per_class': {}}

        kappas = {}
        for i, theme in enumerate(self.mlb.classes_):
            annotations = np.array([m[:, i] for m in annotator_matrices]).T
            kappas[theme] = self._fleiss_kappa(annotations)

        return {
            'fleiss_kappa_mean': float(np.nanmean(list(kappas.values()))),
            'fleiss_kappa_per_class': kappas,
        }

    def _fleiss_kappa(self, annotations: np.ndarray) -> float:
        """Compute Fleiss' kappa for a single binary variable."""
        n, k = annotations.shape
        if k < 2:
            return float('nan')
        p_j = [np.mean(annotations == cat) for cat in [0, 1]]
        P_e = sum(p ** 2 for p in p_j)
        P_i = []
        for i in range(n):
            counts = np.bincount(annotations[i].astype(int), minlength=2)
            P_i.append((np.sum(counts ** 2) - k) / (k * (k - 1)) if k > 1 else 1)
        P_o = np.mean(P_i)
        return (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

    # =====================================================================
    # CALIBRATION
    # =====================================================================

    def compute_calibration(self, n_bins: int = 10) -> Dict[str, Any]:
        """Compute Expected Calibration Error and reliability data."""
        probs = self.raw_probs
        # Correctness: at least one predicted theme matches consensus
        y_true_labels = [set(self.mlb.classes_[self.y_true[i] == 1]) for i in range(len(self.y_true))]
        y_pred_labels = [set(self.mlb.classes_[self.y_pred[i] == 1]) for i in range(len(self.y_pred))]
        correct = np.array([bool(t & p) if t else (not p) for t, p in zip(y_true_labels, y_pred_labels)])

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        calibration_data = []
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (probs > lo) & (probs <= hi)
            prop = in_bin.mean()
            if prop > 0:
                avg_conf = probs[in_bin].mean()
                avg_acc = correct[in_bin].mean()
                ece += prop * abs(avg_acc - avg_conf)
                calibration_data.append({
                    'bin_mid': (lo + hi) / 2,
                    'confidence': float(avg_conf),
                    'accuracy': float(avg_acc),
                    'count': int(in_bin.sum()),
                })

        return {'ece': float(ece), 'calibration_data': calibration_data}

    # =====================================================================
    # LLM TRANSFER + ANNOTATOR COMPARISON
    # =====================================================================

    def analyze_llm_transfer(self) -> Dict[str, Any]:
        """Compare training performance (LLM data) vs benchmark (human)."""
        results = {'training_source': 'LLM annotations', 'evaluation_target': 'Human consensus'}
        if self.training_metadata:
            fm = self.training_metadata.get('final_metrics', {}).get('overall', {})
            results['training_macro_f1'] = fm.get('macro_f1') or self.training_metadata.get('macro_f1')
            results['training_accuracy'] = fm.get('accuracy') or self.training_metadata.get('accuracy')
            results['training_micro_f1'] = fm.get('micro_f1') or fm.get('accuracy')

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
        """Compare model vs each annotator (all measured against consensus)."""
        results = {'annotators': {}, 'model': {}}

        p, r, f, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='micro', zero_division=0
        )
        results['model'] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}

        for col in self.annotator_cols:
            if col not in self.df.columns:
                continue
            annotator_name = col.replace('_themes', '')
            y_annotator = []
            for _, row in self.df.iterrows():
                labels = [self._normalize(l) for l in self._parse_labels(row[col])]
                labels = [l for l in labels if l not in ('null', '')]
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

    # =====================================================================
    # CSV EXPORT
    # =====================================================================

    def export_csv(self, metrics: Dict, agreement: Dict, ambiguity: Dict, bootstrap_ci: Dict):
        """Export publication-quality CSV files."""
        # 1. Per-theme metrics
        rows = []
        kappas = agreement.get('fleiss_kappa_per_class', {})
        for theme in sorted(metrics['per_class'].keys()):
            d = metrics['per_class'][theme]
            kappa = kappas.get(theme, float('nan'))
            rows.append({
                'Theme': d['display_name'],
                'Theme_Key': theme,
                'Precision': round(d['precision'], 4),
                'Recall': round(d['recall'], 4),
                'F1': round(d['f1'], 4),
                'Support': d['support'],
                'TP': d['tp'],
                'FP': d['fp'],
                'FN': d['fn'],
                'Fleiss_Kappa': round(kappa, 4) if not np.isnan(kappa) else '',
                'Agreement_Level': get_agreement_level(kappa) if not np.isnan(kappa) else '',
            })
        df_per_theme = pd.DataFrame(rows).sort_values('F1', ascending=False)
        df_per_theme.to_csv(self.output_dir / 'per_theme_metrics.csv', index=False)
        logger.info(f"Saved per_theme_metrics.csv ({len(rows)} themes)")

        # 2. Global summary
        summary_rows = [
            {'Metric': 'Micro F1', 'Value': f"{metrics['micro_f1']:.4f}",
             '95% CI': f"[{bootstrap_ci['f1_micro'][0]:.3f}, {bootstrap_ci['f1_micro'][1]:.3f}]"},
            {'Metric': 'Macro F1', 'Value': f"{metrics['macro_f1']:.4f}",
             '95% CI': f"[{bootstrap_ci['f1_macro'][0]:.3f}, {bootstrap_ci['f1_macro'][1]:.3f}]"},
            {'Metric': 'Weighted F1', 'Value': f"{metrics['weighted_f1']:.4f}", '95% CI': ''},
            {'Metric': 'Precision (Macro)', 'Value': f"{metrics['macro_precision']:.4f}",
             '95% CI': f"[{bootstrap_ci['precision_macro'][0]:.3f}, {bootstrap_ci['precision_macro'][1]:.3f}]"},
            {'Metric': 'Recall (Macro)', 'Value': f"{metrics['macro_recall']:.4f}",
             '95% CI': f"[{bootstrap_ci['recall_macro'][0]:.3f}, {bootstrap_ci['recall_macro'][1]:.3f}]"},
            {'Metric': 'Hamming Loss', 'Value': f"{metrics['hamming_loss']:.4f}", '95% CI': ''},
            {'Metric': 'Subset Accuracy', 'Value': f"{metrics['subset_accuracy']:.4f}", '95% CI': ''},
            {'Metric': 'Jaccard (Micro)', 'Value': f"{metrics['jaccard_micro']:.4f}", '95% CI': ''},
            {'Metric': 'Jaccard (Macro)', 'Value': f"{metrics['jaccard_macro']:.4f}", '95% CI': ''},
            {'Metric': "Fleiss' Kappa (Mean)", 'Value': f"{agreement.get('fleiss_kappa_mean', 0):.4f}", '95% CI': ''},
            {'Metric': 'Ambiguity Rate', 'Value': f"{ambiguity['rate']*100:.1f}%", '95% CI': ''},
        ]
        pd.DataFrame(summary_rows).to_csv(self.output_dir / 'global_metrics_summary.csv', index=False)
        logger.info("Saved global_metrics_summary.csv")

        # 3. Ambiguity-stratified performance
        strat_rows = []
        for level in ['low', 'medium', 'high']:
            s = ambiguity['stratified'][level]
            m = s.get('metrics', {})
            strat_rows.append({
                'Disagreement_Level': level.title(),
                'N_Samples': s['n_samples'],
                'F1_Macro': round(m.get('f1_macro', 0), 4),
                'F1_Micro': round(m.get('f1_micro', 0), 4),
                'Precision_Macro': round(m.get('precision_macro', 0), 4),
                'Recall_Macro': round(m.get('recall_macro', 0), 4),
            })
        pd.DataFrame(strat_rows).to_csv(self.output_dir / 'ambiguity_stratified_performance.csv', index=False)
        logger.info("Saved ambiguity_stratified_performance.csv")

        return df_per_theme

    # =====================================================================
    # VISUALIZATION
    # =====================================================================

    def plot_evaluation_dashboard(self) -> plt.Figure:
        """Create publication-quality multi-panel evaluation dashboard."""
        metrics = self.calculate_metrics()
        bootstrap_ci = self.calculate_bootstrap_ci(n_bootstrap=500)
        ambiguity = self.analyze_ambiguity()
        agreement = self.calculate_inter_annotator_agreement()
        llm_transfer = self.analyze_llm_transfer()
        annotator_comparison = self.analyze_annotator_vs_model()
        calibration = self.compute_calibration()

        # Export CSV
        self.export_csv(metrics, agreement, ambiguity, bootstrap_ci)

        per_class = metrics['per_class']
        themes_sorted = sorted(per_class.keys(), key=lambda x: per_class[x]['f1'], reverse=True)
        kappas = agreement.get('fleiss_kappa_per_class', {})
        micro_f1 = metrics['micro_f1']
        macro_f1 = metrics['macro_f1']

        # ── Figure layout ────────────────────────────────────────────────
        fig = plt.figure(figsize=(28, 36))
        fig.patch.set_facecolor('white')

        gs = GridSpec(7, 6, figure=fig, hspace=0.55, wspace=0.45,
                      height_ratios=[0.15, 1.2, 0.95, 1.0, 0.85, 0.85, 0.85])

        # =================================================================
        # ROW 0: Title banner
        # =================================================================
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.text(0.5, 0.72,
                      "Multi-Label Theme Classification \u2014 SOTA Model Evaluation",
                      transform=ax_title.transAxes, fontsize=22, fontweight='bold',
                      color=COLORS['dark'], ha='center', va='center')
        n_train = sum(self.training_sample_counts.values()) if self.training_sample_counts else 0
        sub_parts = [f"n = {len(self.df):,} benchmark samples",
                     f"{len(self.all_themes)} themes",
                     f"{len(self.annotator_cols)} annotators"]
        if n_train:
            sub_parts.insert(1, f"{n_train:,} training samples")
        ax_title.text(0.5, 0.18, "  |  ".join(sub_parts),
                      transform=ax_title.transAxes, fontsize=12,
                      color=COLORS['muted'], ha='center', va='center')

        # =================================================================
        # ROW 1: Per-theme F1 bar chart (full width)
        # =================================================================
        ax_main = fig.add_subplot(gs[1, :])

        f1_scores = [per_class[t]['f1'] for t in themes_sorted]
        supports = [per_class[t]['support'] for t in themes_sorted]
        display_names = [per_class[t]['display_name'] for t in themes_sorted]

        bar_colors = []
        for t in themes_sorted:
            k = kappas.get(t, 0.5)
            bar_colors.append(AGREEMENT_LEVELS.get(get_agreement_level(k), {}).get('color', COLORS['muted']))

        x = np.arange(len(themes_sorted))
        bars = ax_main.bar(x, f1_scores, color=bar_colors, edgecolor='white',
                           linewidth=1.5, alpha=0.9, width=0.7)

        ax_main.axhline(y=micro_f1, color=COLORS['success'], linestyle='-',
                        linewidth=2.5, zorder=10, alpha=0.85)
        ax_main.axhline(y=macro_f1, color=COLORS['danger'], linestyle='--',
                        linewidth=2.5, zorder=10, alpha=0.85)

        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='none', alpha=0.92)
        ax_main.text(len(themes_sorted) - 0.3, micro_f1 + 0.03,
                     f'Micro F1 = {micro_f1:.3f}', fontsize=11, fontweight='bold',
                     color=COLORS['success'], ha='right', bbox=bbox_props)
        ax_main.text(len(themes_sorted) - 0.3, macro_f1 - 0.06,
                     f'Macro F1 = {macro_f1:.3f}', fontsize=11, fontweight='bold',
                     color=COLORS['danger'], ha='right', bbox=bbox_props)

        for i, (bar, support, f1) in enumerate(zip(bars, supports, f1_scores)):
            ax_main.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                         f'.{int(f1*100):02d}', ha='center', va='bottom',
                         fontsize=9, fontweight='bold', color=COLORS['dark'])
            ax_main.text(bar.get_x() + bar.get_width() / 2, 0.025,
                         f'n={support}', ha='center', va='bottom',
                         fontsize=7.5, color='white', fontweight='bold')

        ax_main.set_xticks(x)
        ax_main.set_xticklabels(display_names, fontsize=9.5, fontweight='bold',
                                rotation=35, ha='right')
        ax_main.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax_main.set_ylim(0, 1.08)
        ax_main.set_xlim(-0.6, len(themes_sorted) - 0.4)
        ax_main.set_title('Per-Theme F1-Score (colored by annotator agreement level)',
                          fontsize=15, pad=12, fontweight='bold')
        ax_main.grid(axis='y', alpha=0.3, linestyle='--')

        legend_patches = [mpatches.Patch(color=cfg['color'], label=level)
                          for level, cfg in AGREEMENT_LEVELS.items()]
        ax_main.legend(handles=legend_patches, loc='upper right', fontsize=9,
                       title='Agreement Level', title_fontsize=9, framealpha=0.95,
                       ncol=1)

        # =================================================================
        # ROW 2: Global metrics table (left) + Ambiguity donut (right)
        # =================================================================
        ax_metrics = fig.add_subplot(gs[2, 0:4])
        ax_metrics.axis('off')

        table_data = [
            ['Macro F1', f"{metrics['macro_f1']:.4f}",
             f"[{bootstrap_ci['f1_macro'][0]:.3f}, {bootstrap_ci['f1_macro'][1]:.3f}]"],
            ['Micro F1', f"{metrics['micro_f1']:.4f}",
             f"[{bootstrap_ci['f1_micro'][0]:.3f}, {bootstrap_ci['f1_micro'][1]:.3f}]"],
            ['Weighted F1', f"{metrics['weighted_f1']:.4f}", '\u2014'],
            ['Precision (Macro)', f"{metrics['macro_precision']:.4f}",
             f"[{bootstrap_ci['precision_macro'][0]:.3f}, {bootstrap_ci['precision_macro'][1]:.3f}]"],
            ['Recall (Macro)', f"{metrics['macro_recall']:.4f}",
             f"[{bootstrap_ci['recall_macro'][0]:.3f}, {bootstrap_ci['recall_macro'][1]:.3f}]"],
            ['Hamming Loss', f"{metrics['hamming_loss']:.4f}", '\u2014'],
            ['Subset Accuracy', f"{metrics['subset_accuracy']:.4f}", '\u2014'],
            ['Jaccard (Macro)', f"{metrics['jaccard_macro']:.4f}", '\u2014'],
        ]
        table = ax_metrics.table(
            cellText=table_data, colLabels=['Metric', 'Value', '95% CI'],
            loc='center', cellLoc='center', colWidths=[0.35, 0.2, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 1.7)
        for i in range(3):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(table_data) + 1):
            bg = '#f8fafc' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(bg)
                table[(i, j)].set_edgecolor('#e2e8f0')
        ax_metrics.set_title('Global Classification Metrics',
                             fontsize=14, pad=18, fontweight='bold')

        ax_amb = fig.add_subplot(gs[2, 4:6])
        sizes = [ambiguity['clear'], ambiguity['ambivalent']]
        colors_pie = [COLORS['success'], COLORS['danger']]
        wedges, texts, autotexts = ax_amb.pie(
            sizes, colors=colors_pie, autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3),
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color('white')
        ax_amb.text(0, 0, f"\u03ba={agreement.get('fleiss_kappa_mean', 0):.2f}",
                    ha='center', va='center', fontsize=15, fontweight='bold',
                    color=COLORS['dark'])
        ax_amb.legend(wedges,
                      [f'Clear ({ambiguity["clear"]})',
                       f'Ambivalent ({ambiguity["ambivalent"]})'],
                      loc='lower center', fontsize=10, framealpha=0.95,
                      bbox_to_anchor=(0.5, -0.08))
        ax_amb.set_title('Annotator Agreement', fontsize=14, fontweight='bold')

        # =================================================================
        # ROW 3: Training Samples vs F1 (left 4 cols) + Precision/Recall (right 2 cols)
        # =================================================================
        ax_scatter = fig.add_subplot(gs[3, 0:4])

        # Use training sample counts if available, otherwise benchmark support
        use_training = bool(self.training_sample_counts)
        scatter_x = []
        scatter_f1 = []
        scatter_themes = []
        for t in self.all_themes:
            f1_val = per_class[t]['f1']
            if use_training:
                x_val = self.training_sample_counts.get(t, 0)
            else:
                x_val = per_class[t]['support']
            if x_val > 0:
                scatter_x.append(x_val)
                scatter_f1.append(f1_val)
                scatter_themes.append(t)

        # Use adjustText-style manual offset to avoid overlaps
        from matplotlib.transforms import offset_copy
        for i, theme in enumerate(scatter_themes):
            cidx = i % len(THEME_PALETTE)
            ax_scatter.scatter(scatter_x[i], scatter_f1[i], s=160,
                               c=THEME_PALETTE[cidx], edgecolors='white',
                               linewidths=1.5, zorder=10, alpha=0.9)

        # Label placement: use alternating offsets to reduce overlap
        sorted_by_x = sorted(range(len(scatter_themes)),
                              key=lambda j: scatter_x[j])
        prev_y = -1
        for rank, j in enumerate(sorted_by_x):
            short = THEME_DISPLAY.get(scatter_themes[j], scatter_themes[j])
            if len(short) > 16:
                short = short[:14] + '..'
            dy = 4 if (scatter_f1[j] - prev_y) > 0.03 or rank == 0 else 12
            if rank % 2 == 0:
                offset = (8, dy)
            else:
                offset = (8, -dy - 8)
            ax_scatter.annotate(
                short, (scatter_x[j], scatter_f1[j]),
                xytext=offset, textcoords='offset points',
                fontsize=8, color=COLORS['dark'], va='center',
                arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.5)
            )
            prev_y = scatter_f1[j]

        if len(scatter_x) > 2:
            log_x = np.log1p(scatter_x)
            slope, intercept, r_value, p_value, _ = stats.linregress(log_x, scatter_f1)
            x_fit = np.linspace(min(scatter_x), max(scatter_x), 200)
            y_fit = slope * np.log1p(x_fit) + intercept
            ax_scatter.plot(x_fit, y_fit, '--', color=COLORS['danger'],
                            linewidth=2.5, alpha=0.6,
                            label=f'Log regression (r={r_value:.2f}, p={p_value:.3f})')

        x_label = 'Training Samples (LLM data)' if use_training else 'Benchmark Support'
        ax_scatter.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel('Benchmark F1-Score', fontsize=12, fontweight='bold')
        title_suffix = ' (from training data)' if use_training else ''
        ax_scatter.set_title(f'Label Frequency{title_suffix} vs Benchmark F1',
                             fontsize=14, fontweight='bold')
        max_x = max(scatter_x) if scatter_x else 100
        ax_scatter.set_xlim(-max_x * 0.03, max_x * 1.12)
        ax_scatter.set_ylim(-0.02, 1.05)
        ax_scatter.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_scatter.grid(alpha=0.3, linestyle='--')

        # Precision / Recall horizontal bars
        ax_pr = fig.add_subplot(gs[3, 4:6])
        top_n = min(12, len(themes_sorted))
        top_themes = themes_sorted[:top_n]
        top_display = [per_class[t]['display_name'] for t in top_themes]
        # Truncate long names
        top_display = [n if len(n) <= 18 else n[:16] + '..' for n in top_display]
        y_pos = np.arange(top_n)
        precisions = [per_class[t]['precision'] for t in top_themes]
        recalls = [per_class[t]['recall'] for t in top_themes]

        ax_pr.barh(y_pos + 0.18, precisions, 0.32, label='Precision',
                   color=COLORS['primary'], alpha=0.85)
        ax_pr.barh(y_pos - 0.18, recalls, 0.32, label='Recall',
                   color=COLORS['success'], alpha=0.85)
        ax_pr.set_yticks(y_pos)
        ax_pr.set_yticklabels(top_display, fontsize=8.5)
        ax_pr.set_xlim(0, 1.08)
        ax_pr.set_xlabel('Score', fontsize=10)
        ax_pr.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_pr.set_title('Precision & Recall (top themes)', fontsize=13, fontweight='bold')
        ax_pr.invert_yaxis()
        ax_pr.grid(axis='x', alpha=0.3, linestyle='--')

        # =================================================================
        # ROW 4: Disagreement-stratified (left 3) + Model vs Annotators (right 3)
        # =================================================================
        ax_strat = fig.add_subplot(gs[4, 0:3])
        strat = ambiguity['stratified']
        strat_names = ['Low (<0.33)', 'Medium (0.33\u20130.66)', 'High (\u22650.66)']
        strat_f1_macro = [strat[l]['metrics'].get('f1_macro', 0) for l in ['low', 'medium', 'high']]
        strat_f1_micro = [strat[l]['metrics'].get('f1_micro', 0) for l in ['low', 'medium', 'high']]
        strat_counts = [strat[l]['n_samples'] for l in ['low', 'medium', 'high']]
        strat_colors = ['#2ecc71', '#f39c12', '#e74c3c']

        x_s = np.arange(3)
        w = 0.32
        bars_macro = ax_strat.bar(x_s - w / 2, strat_f1_macro, w, label='Macro F1',
                                  color=strat_colors, edgecolor='white', linewidth=2, alpha=0.85)
        bars_micro = ax_strat.bar(x_s + w / 2, strat_f1_micro, w, label='Micro F1',
                                  color=[c + '80' for c in strat_colors],
                                  edgecolor='white', linewidth=2, alpha=0.65)
        for bar, count in zip(bars_macro, strat_counts):
            ax_strat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                          f'n={count}', ha='center', fontsize=10, fontweight='bold')
        ax_strat.set_xticks(x_s)
        ax_strat.set_xticklabels(strat_names, fontsize=10)
        ax_strat.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax_strat.set_ylim(0, 1.1)
        ax_strat.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax_strat.set_title('Performance by Disagreement Level',
                           fontsize=13, fontweight='bold')
        ax_strat.grid(axis='y', alpha=0.3, linestyle='--')

        ax_annotators = fig.add_subplot(gs[4, 3:6])
        comparison_names = ['Model']
        comparison_f1s = [annotator_comparison['model']['f1']]
        comparison_colors = [COLORS['danger']]
        for name, data in sorted(annotator_comparison['annotators'].items()):
            comparison_names.append(name[:12])
            comparison_f1s.append(data['f1'])
            comparison_colors.append(COLORS['primary'])
        x_comp = np.arange(len(comparison_names))
        bars_comp = ax_annotators.bar(x_comp, comparison_f1s, color=comparison_colors,
                                       edgecolor='white', linewidth=2, alpha=0.85,
                                       width=0.6)
        if annotator_comparison.get('annotator_avg_f1'):
            avg_f1 = annotator_comparison['annotator_avg_f1']
            ax_annotators.axhline(y=avg_f1, color=COLORS['info'], linestyle='--',
                                  linewidth=2.5, label=f'Human Avg: {avg_f1:.3f}')
        for bar in bars_comp:
            ax_annotators.text(bar.get_x() + bar.get_width() / 2,
                               bar.get_height() + 0.015,
                               f'{bar.get_height():.3f}', ha='center',
                               fontsize=10, fontweight='bold')
        ax_annotators.set_xticks(x_comp)
        ax_annotators.set_xticklabels(comparison_names, fontsize=10, rotation=20, ha='right')
        ax_annotators.set_ylim(0, 1.08)
        ax_annotators.set_ylabel('Micro F1', fontsize=11)
        ax_annotators.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax_annotators.set_title('Model vs Annotators (Micro F1 vs Consensus)',
                                fontsize=13, fontweight='bold')
        ax_annotators.grid(axis='y', alpha=0.3, linestyle='--')

        # =================================================================
        # ROW 5: LLM Transfer (left 3) + Calibration (right 3)
        # =================================================================
        ax_transfer = fig.add_subplot(gs[5, 0:3])
        transfer_x = np.arange(2)
        width = 0.32
        training_vals = [
            llm_transfer.get('training_macro_f1', 0) or 0,
            llm_transfer.get('training_micro_f1', 0) or 0
        ]
        benchmark_vals = [
            llm_transfer.get('benchmark_macro_f1', 0),
            llm_transfer.get('benchmark_micro_f1', 0)
        ]
        has_training_data = any(v > 0 for v in training_vals)

        if has_training_data:
            bars1 = ax_transfer.bar(transfer_x - width / 2, training_vals, width,
                                    label='Training (LLM Data)', color=COLORS['secondary'],
                                    alpha=0.85)
            bars2 = ax_transfer.bar(transfer_x + width / 2, benchmark_vals, width,
                                    label='Benchmark (Human)', color=COLORS['primary'],
                                    alpha=0.85)
            for bar in list(bars1) + list(bars2):
                if bar.get_height() > 0:
                    clr = COLORS['secondary'] if bar in bars1 else COLORS['primary']
                    ax_transfer.text(bar.get_x() + bar.get_width() / 2,
                                     bar.get_height() + 0.015,
                                     f'{bar.get_height():.3f}', ha='center',
                                     fontsize=10, fontweight='bold', color=clr)
            eff_macro = llm_transfer.get('transfer_efficiency_macro', 0)
            eff_micro = llm_transfer.get('transfer_efficiency_micro', 0)
            if eff_macro and eff_micro:
                eff_text = f"Transfer: Macro {eff_macro:.0%} | Micro {eff_micro:.0%}"
                eff_color = COLORS['success'] if min(eff_macro, eff_micro) >= 0.9 else COLORS['warning']
                ax_transfer.text(0.5, 1.03, eff_text, transform=ax_transfer.transAxes,
                                 ha='center', fontsize=11, fontweight='bold', color=eff_color)
        else:
            # No training data available - show benchmark only
            bars2 = ax_transfer.bar(transfer_x, benchmark_vals, width * 1.5,
                                    label='Benchmark (Human)', color=COLORS['primary'],
                                    alpha=0.85)
            for bar in bars2:
                ax_transfer.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.015,
                                 f'{bar.get_height():.3f}', ha='center',
                                 fontsize=10, fontweight='bold', color=COLORS['primary'])
            ax_transfer.text(0.5, 0.5, 'Training metadata\nnot available',
                             transform=ax_transfer.transAxes, ha='center', va='center',
                             fontsize=11, color=COLORS['muted'], style='italic',
                             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        ax_transfer.set_xticks(transfer_x)
        ax_transfer.set_xticklabels(['Macro F1', 'Micro F1'], fontsize=11, fontweight='bold')
        ax_transfer.set_ylim(0, 1.1)
        ax_transfer.set_ylabel('F1 Score', fontsize=11)
        ax_transfer.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax_transfer.set_title('LLM \u2192 Human Transfer Performance',
                              fontsize=13, fontweight='bold')
        ax_transfer.grid(axis='y', alpha=0.3, linestyle='--')

        ax_cal = fig.add_subplot(gs[5, 3:6])
        cal_data = calibration['calibration_data']
        if cal_data:
            # Use bin_mid for evenly spaced bars
            bin_mids = [d['bin_mid'] for d in cal_data]
            accuracies = [d['accuracy'] for d in cal_data]
            counts = [d['count'] for d in cal_data]

            # Calculate bar width based on number of bins
            bar_width = 0.8 / len(cal_data) if len(cal_data) > 1 else 0.08

            bars_cal = ax_cal.bar(bin_mids, accuracies, width=bar_width, alpha=0.8,
                                  color=COLORS['primary'], edgecolor='white', linewidth=1.5,
                                  label='Observed Accuracy')

            # Add count labels on bars
            for bar, count in zip(bars_cal, counts):
                if bar.get_height() > 0.05:
                    ax_cal.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.08,
                                f'n={count}', ha='center', va='top', fontsize=7,
                                color='white', fontweight='bold')

            ax_cal.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
            ax_cal.set_xlabel('Mean Predicted Probability', fontsize=10)
            ax_cal.set_ylabel('Fraction Correct', fontsize=10)
            ax_cal.set_xlim(0, 1.05)
            ax_cal.set_ylim(0, 1.05)
            ax_cal.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ece = calibration['ece']
            ax_cal.text(0.95, 0.05, f'ECE = {ece:.4f}', fontsize=12,
                        fontweight='bold', transform=ax_cal.transAxes,
                        ha='right', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax_cal.set_title('Calibration Reliability', fontsize=13, fontweight='bold')
        ax_cal.grid(alpha=0.3, linestyle='--')

        # =================================================================
        # ROW 6: Key metrics card (left 2) + Agreement vs F1 (right 4)
        # =================================================================
        ax_summary = fig.add_subplot(gs[6, 0:2])
        ax_summary.axis('off')

        model_gap = annotator_comparison.get('model_vs_annotator_gap', 0)
        gap_text = f"+{model_gap:.1%}" if model_gap >= 0 else f"{model_gap:.1%}"
        gap_color = COLORS['success'] if model_gap >= 0 else COLORS['warning']

        summary_items = [
            ('Micro F1', f"{metrics['micro_f1']:.3f}", COLORS['success']),
            ('Macro F1', f"{metrics['macro_f1']:.3f}", COLORS['primary']),
            ('Hamming Loss', f"{metrics['hamming_loss']:.4f}", COLORS['info']),
            ("Fleiss' \u03ba", f"{agreement.get('fleiss_kappa_mean', 0):.3f}", COLORS['secondary']),
            ('vs Human Avg', gap_text, gap_color),
            ('Ambiguity', f"{ambiguity['rate']*100:.1f}%", COLORS['warning']),
        ]
        rect = FancyBboxPatch((0.03, 0.02), 0.94, 0.92, transform=ax_summary.transAxes,
                               boxstyle="round,pad=0.02,rounding_size=0.02",
                               facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=2)
        ax_summary.add_patch(rect)
        ax_summary.text(0.5, 0.95, 'KEY METRICS', transform=ax_summary.transAxes,
                        fontsize=13, fontweight='bold', ha='center', va='top',
                        color=COLORS['dark'])
        y_off = 0.84
        for label, value, color in summary_items:
            ax_summary.text(0.1, y_off, label, transform=ax_summary.transAxes,
                            fontsize=11, ha='left', va='top', color=COLORS['muted'])
            ax_summary.text(0.9, y_off, value, transform=ax_summary.transAxes,
                            fontsize=13, fontweight='bold', ha='right', va='top', color=color)
            y_off -= 0.13

        ax_kappa = fig.add_subplot(gs[6, 2:6])
        kappa_themes = sorted(kappas.keys(), key=lambda t: kappas.get(t, 0), reverse=True)
        kappa_vals = [kappas.get(t, 0) for t in kappa_themes]
        kappa_display = [THEME_DISPLAY.get(t, t) for t in kappa_themes]
        kappa_display = [n if len(n) <= 16 else n[:14] + '..' for n in kappa_display]
        f1_ordered = [per_class[t]['f1'] for t in kappa_themes]

        x_k = np.arange(len(kappa_themes))
        w_k = 0.38
        kappa_colors = [AGREEMENT_LEVELS[get_agreement_level(k)]['color'] for k in kappa_vals]
        ax_kappa.bar(x_k - w_k / 2, kappa_vals, w_k,
                     color=kappa_colors,
                     edgecolor='white', linewidth=1.5, alpha=0.9)
        ax_kappa.bar(x_k + w_k / 2, f1_ordered, w_k, label='Model F1',
                     color=COLORS['primary'], edgecolor='white', linewidth=1.5, alpha=0.7)
        ax_kappa.set_xticks(x_k)
        ax_kappa.set_xticklabels(kappa_display, fontsize=8.5, rotation=38, ha='right')
        ax_kappa.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax_kappa.set_ylim(0, 1.08)

        # Create legend with agreement level colors + Model F1
        legend_handles = [mpatches.Patch(color=cfg['color'], label=f"\u03ba {level}")
                          for level, cfg in AGREEMENT_LEVELS.items()]
        legend_handles.append(mpatches.Patch(color=COLORS['primary'], alpha=0.7, label='Model F1'))
        ax_kappa.legend(handles=legend_handles, loc='upper right', fontsize=8,
                        framealpha=0.95, ncol=2)
        ax_kappa.set_title("Per-Theme: Agreement (\u03ba) vs Model F1",
                           fontsize=13, fontweight='bold')
        ax_kappa.grid(axis='y', alpha=0.3, linestyle='--')

        # ── Save ─────────────────────────────────────────────────────────
        fig.savefig(self.output_dir / 'themes_evaluation_dashboard.png',
                    bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
        fig.savefig(self.output_dir / 'themes_evaluation_dashboard.pdf',
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved dashboard to {self.output_dir}")
        plt.close(fig)

        return fig

    # =====================================================================
    # MAIN RUN
    # =====================================================================

    def run_evaluation(self):
        """Run full evaluation pipeline."""
        logger.info("=" * 65)
        logger.info("Multi-Label Theme Classification Evaluation")
        logger.info("=" * 65)

        metrics = self.calculate_metrics()
        bootstrap_ci = self.calculate_bootstrap_ci()
        ambiguity = self.analyze_ambiguity()
        agreement = self.calculate_inter_annotator_agreement()
        llm_transfer = self.analyze_llm_transfer()
        annotator_comparison = self.analyze_annotator_vs_model()
        calibration = self.compute_calibration()

        # Export CSV
        self.export_csv(metrics, agreement, ambiguity, bootstrap_ci)

        # Dashboard
        self.plot_evaluation_dashboard()

        # JSON results
        results = {
            'dataset': {
                'samples': len(self.df),
                'classes': len(self.all_themes),
                'themes': self.all_themes,
                'prediction_col': self.prediction_col,
            },
            'metrics': {k: v for k, v in metrics.items() if k != 'per_class'},
            'per_class': {t: {k: v for k, v in d.items()} for t, d in metrics['per_class'].items()},
            'bootstrap_ci': {k: list(v) for k, v in bootstrap_ci.items()},
            'ambiguity': {k: v for k, v in ambiguity.items() if k != 'disagreement_index'},
            'inter_annotator_agreement': {
                'fleiss_kappa_mean': agreement.get('fleiss_kappa_mean'),
                'fleiss_kappa_per_class': {k: round(v, 4) for k, v in agreement.get('fleiss_kappa_per_class', {}).items()},
            },
            'llm_transfer': {k: v for k, v in llm_transfer.items()},
            'annotator_comparison': annotator_comparison,
            'calibration': {'ece': calibration['ece']},
        }

        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Console report
        self._print_report(metrics, bootstrap_ci, ambiguity, agreement,
                           llm_transfer, annotator_comparison, calibration)

        return results

    def _print_report(self, metrics, bootstrap_ci, ambiguity, agreement,
                      llm_transfer, annotator_comparison, calibration):
        """Print detailed console report."""
        print("\n" + "=" * 70)
        print("  MULTI-LABEL THEME CLASSIFICATION \u2014 EVALUATION RESULTS")
        print("  (null class excluded)")
        print("=" * 70)
        print(f"\n  {'Dataset:':<22} {len(self.df):,} samples, {len(self.all_themes)} themes")
        print(f"  {'Prediction col:':<22} {self.prediction_col}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  CLASSIFICATION METRICS")
        print(f"  {'\u2500' * 66}")
        print(f"  {'Micro F1:':<22} {metrics['micro_f1']:.4f}  [{bootstrap_ci['f1_micro'][0]:.3f}, {bootstrap_ci['f1_micro'][1]:.3f}]")
        print(f"  {'Macro F1:':<22} {metrics['macro_f1']:.4f}  [{bootstrap_ci['f1_macro'][0]:.3f}, {bootstrap_ci['f1_macro'][1]:.3f}]")
        print(f"  {'Weighted F1:':<22} {metrics['weighted_f1']:.4f}")
        print(f"  {'Precision (Macro):':<22} {metrics['macro_precision']:.4f}  [{bootstrap_ci['precision_macro'][0]:.3f}, {bootstrap_ci['precision_macro'][1]:.3f}]")
        print(f"  {'Recall (Macro):':<22} {metrics['macro_recall']:.4f}  [{bootstrap_ci['recall_macro'][0]:.3f}, {bootstrap_ci['recall_macro'][1]:.3f}]")
        print(f"  {'Hamming Loss:':<22} {metrics['hamming_loss']:.4f}")
        print(f"  {'Subset Accuracy:':<22} {metrics['subset_accuracy']:.4f}")
        print(f"  {'Jaccard (Micro):':<22} {metrics['jaccard_micro']:.4f}")
        print(f"  {'Jaccard (Macro):':<22} {metrics['jaccard_macro']:.4f}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  PER-THEME PERFORMANCE")
        print(f"  {'\u2500' * 66}")
        print(f"  {'Theme':<30} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Supp':>7} {'Kappa':>8}")
        print(f"  {'-' * 69}")
        kappas = agreement.get('fleiss_kappa_per_class', {})
        per_class = metrics['per_class']
        for t in sorted(per_class.keys(), key=lambda x: per_class[x]['f1'], reverse=True):
            d = per_class[t]
            k_str = f"{kappas.get(t, 0):.3f}" if t in kappas else "  N/A"
            name = d['display_name'][:28]
            print(f"  {name:<30} {d['f1']:>8.3f} {d['precision']:>8.3f} {d['recall']:>8.3f} {d['support']:>7} {k_str:>8}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  AMBIGUITY ANALYSIS")
        print(f"  {'\u2500' * 66}")
        print(f"  {'Ambivalence rate:':<22} {ambiguity['rate']*100:.1f}%  ({ambiguity['ambivalent']}/{ambiguity['total']})")
        for level in ['low', 'medium', 'high']:
            s = ambiguity['stratified'][level]
            m = s.get('metrics', {})
            f1_m = m.get('f1_macro', 0)
            f1_i = m.get('f1_micro', 0)
            print(f"  {level.title() + ' disagreement:':<22} F1 Macro={f1_m:.4f}  Micro={f1_i:.4f}  (n={s['n_samples']})")

        print(f"\n  {'\u2500' * 66}")
        print(f"  INTER-ANNOTATOR AGREEMENT")
        print(f"  {'\u2500' * 66}")
        print(f"  {'Fleiss\u2019 \u03ba (mean):':<22} {agreement.get('fleiss_kappa_mean', 0):.4f}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  LLM \u2192 HUMAN TRANSFER")
        print(f"  {'\u2500' * 66}")
        t_macro = llm_transfer.get('training_macro_f1', 0) or 0
        t_micro = llm_transfer.get('training_micro_f1', 0) or 0
        print(f"  {'Training (LLM):':<22} Macro={t_macro:.4f}  Micro={t_micro:.4f}")
        print(f"  {'Benchmark (Human):':<22} Macro={llm_transfer.get('benchmark_macro_f1', 0):.4f}  Micro={llm_transfer.get('benchmark_micro_f1', 0):.4f}")
        if llm_transfer.get('transfer_efficiency_macro'):
            print(f"  {'Transfer Efficiency:':<22} Macro={llm_transfer['transfer_efficiency_macro']:.1%}  Micro={llm_transfer.get('transfer_efficiency_micro', 0):.1%}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  MODEL VS ANNOTATORS (Micro F1 vs consensus)")
        print(f"  {'\u2500' * 66}")
        print(f"  {'Model:':<22} {annotator_comparison['model']['f1']:.4f}")
        for name, data in sorted(annotator_comparison['annotators'].items()):
            print(f"  {name + ':':<22} {data['f1']:.4f}")
        if annotator_comparison.get('annotator_avg_f1'):
            avg_f1 = annotator_comparison['annotator_avg_f1']
            gap = annotator_comparison.get('model_vs_annotator_gap', 0)
            gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
            print(f"  {'Human Avg:':<22} {avg_f1:.4f}")
            print(f"  {'Gap:':<22} {gap_str}")

        print(f"\n  {'\u2500' * 66}")
        print(f"  CALIBRATION")
        print(f"  {'\u2500' * 66}")
        print(f"  {'ECE:':<22} {calibration['ece']:.4f}")

        print(f"\n  Output: {self.output_dir}")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Label Theme Classification \u2014 SOTA Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate the GPT-OSS themes model
  python evaluate_themes_model.py \\
      --annotations logs/annotation_studio/gpt_oss_themes_20260131_183144/annotations.csv

  # Custom prediction column and output directory
  python evaluate_themes_model.py \\
      --annotations path/to/annotations.csv \\
      --prediction-col my_model_labels_label \\
      --output-dir paper/results/my_evaluation

  # With training metadata for transfer efficiency
  python evaluate_themes_model.py \\
      --annotations path/to/annotations.csv \\
      --training-metadata models/MY_MODEL/normal_training/labels/model_epoch_14/training_metadata.json

  # With training arena for per-class sample count analysis
  python evaluate_themes_model.py \\
      --annotations logs/annotation_studio/gpt_oss_themes_20260131_183144/annotations.csv \\
      --training-arena-dir logs/training_arena/GPT_OSS_THEMES_EXTRA_EXTRA_LARGE_20260131_084105
        """
    )

    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to annotations CSV (relative to project root or absolute)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: <annotations_dir>/evaluation_results)')
    parser.add_argument('--prediction-col', type=str, default=None,
                        help='Column name for model predictions (auto-detected if not specified)')
    parser.add_argument('--probability-col', type=str, default=None,
                        help='Column name for prediction probabilities (auto-detected if not specified)')
    parser.add_argument('--consensus-col', type=str, default='consensus_themes',
                        help='Column name for consensus annotations (default: consensus_themes)')
    parser.add_argument('--training-metadata', type=str, default=None,
                        help='Path to training_metadata.json for transfer efficiency analysis')
    parser.add_argument('--training-arena-dir', type=str, default=None,
                        help='Path to training arena dir for per-class sample counts (auto-detected if not specified)')

    args = parser.parse_args()

    # Resolve path (relative to project root or absolute)
    annotations_path = Path(args.annotations)
    if not annotations_path.is_absolute():
        base_dir = Path(__file__).parent.parent.parent.parent
        annotations_path = base_dir / args.annotations

    if not annotations_path.exists():
        logger.error(f"File not found: {annotations_path}")
        return 1

    evaluator = ThemesEvaluator(
        annotations_path=str(annotations_path),
        output_dir=args.output_dir,
        consensus_col=args.consensus_col,
        prediction_col=args.prediction_col,
        probability_col=args.probability_col,
        training_metadata_path=args.training_metadata,
        training_arena_dir=args.training_arena_dir,
    )
    evaluator.run_evaluation()
    return 0


if __name__ == '__main__':
    exit(main())
