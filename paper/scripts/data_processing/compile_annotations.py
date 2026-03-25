#!/usr/bin/env python3
"""
PROJECT:
--------
LLMTool

TITLE:
------
compile_annotations.py

MAIN OBJECTIVE:
---------------
Compile all annotation metrics and data from multiple annotation sessions.
Parses label columns in format "{model_name} IS" / "{model_name} IS NOT" to
extract binary predictions and maps them back to the original model structure.

Dependencies:
-------------
- pandas
- pathlib
- json
- dataclasses
- logging

MAIN FEATURES:
--------------
1) Parse annotation files from Label Studio format
2) Extract binary predictions per model/theme combination
3) Map predictions to original model structure
4) Generate consolidated CSV with all pipelines
5) Produce summary statistics per model

OUTPUT FILES:
-------------
- final_test_small_annotated.csv: Consolidated annotations with all pipelines
- models_summary.csv: Summary of all models evaluated

Author:
-------
Antoine Lemor
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

ALL_THEMES = [
    'theme_agriculture', 'theme_culture_nationalism', 'theme_defense',
    'theme_domestic_commerce', 'theme_education', 'theme_energy',
    'theme_environment', 'theme_foreign_trade', 'theme_governments_governance',
    'theme_health', 'theme_housing', 'theme_immigration', 'theme_indigenous_affairs',
    'theme_international_affairs', 'theme_labor', 'theme_law_and_crime',
    'theme_macroeconomics', 'theme_public_lands',
    'theme_rights_liberties_minorities_discrimination', 'theme_social_welfare',
    'theme_technology', 'theme_transportation'
]

ALL_PARTIES = [
    'political_parties_BQ', 'political_parties_CAQ', 'political_parties_CPC',
    'political_parties_GPC', 'political_parties_LPC', 'political_parties_NDP',
    'political_parties_PCQ', 'political_parties_PLQ', 'political_parties_PQ',
    'political_parties_QS'
]

ALL_SPECIFIC_THEMES = [
    'specific_themes_early_learning_childcare',
    'specific_themes_public_finance',
    'specific_themes_welfare_state'
]

ALL_SENTIMENTS = ['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']

LLMS = ['manual', 'gemma', 'gpt_4_1', 'gpt_5', 'gpt_oss', 'llama', 'nemotron']

THEME_NAMES = [
    'agriculture', 'culture_nationalism', 'defense', 'domestic_commerce',
    'education', 'energy', 'environment', 'foreign_trade', 'governments_governance',
    'health', 'housing', 'immigration', 'indigenous_affairs', 'international_affairs',
    'labor', 'law_and_crime', 'macroeconomics', 'public_lands',
    'rights_liberties_minorities_discrimination', 'social_welfare',
    'technology', 'transportation'
]

PARTY_CODES = ['BQ', 'CAQ', 'CPC', 'GPC', 'LPC', 'NDP', 'PCQ', 'PLQ', 'PQ', 'QS']


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelMapping:
    """Mapping from column prefix to model details."""
    prefix: str
    model_id: str
    llm: str
    category: str
    subcategory: str
    label_column: str
    probability_column: str


@dataclass
class SessionData:
    """Data from an annotation session."""
    name: str
    session_type: str
    df: pd.DataFrame
    mappings: List[ModelMapping]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_label_value(label_str: str) -> Tuple[str, bool]:
    """Parse a label string in format "{model_name} IS" or "{model_name} IS NOT"."""
    if pd.isna(label_str) or not label_str:
        return ('', False)

    label_str = str(label_str).strip()

    if label_str.endswith(' IS NOT'):
        return (label_str[:-7].strip(), False)
    elif label_str.endswith(' IS'):
        return (label_str[:-3].strip(), True)
    else:
        return (label_str, True)


def extract_category_from_model_id(model_id: str) -> Tuple[str, str, str]:
    """
    Extract LLM, category, and subcategory from model ID.

    Formats:
    - 'gemma_themes_large_agriculture' -> ('gemma', 'themes', 'agriculture')
    - 'gemma_political_parties_large_BQ' -> ('gemma', 'political_parties', 'BQ')
    - 'gemma_sentiment_large' -> ('gemma', 'sentiment', '')
    - 'gemma_specific_themes_large' -> ('gemma', 'specific_themes', '')
    - 'manual_themes_theme_transportation' -> ('manual', 'themes', 'transportation')
    """
    # Remove /transformer/model suffix
    if '/' in model_id:
        model_id = model_id.split('/')[0]

    # Identify LLM
    llm = None
    for l in LLMS:
        if model_id.startswith(l + '_'):
            llm = l
            break
    if llm is None:
        llm = model_id.split('_')[0]

    remaining = model_id[len(llm)+1:] if llm else model_id

    # Check for political_parties pattern: political_parties_{size}_{party}
    if remaining.startswith('political_parties_'):
        rest = remaining[len('political_parties_'):]
        # Format: large_BQ or small_CPC
        for party in PARTY_CODES:
            if rest.endswith('_' + party):
                return (llm, 'political_parties', party)
        # Fallback
        parts = rest.split('_')
        if len(parts) >= 2:
            return (llm, 'political_parties', parts[-1])
        return (llm, 'political_parties', rest)

    # Check for themes pattern: themes_{size}_{theme} or themes_theme_{theme}
    if remaining.startswith('themes_'):
        rest = remaining[len('themes_'):]
        # Check for manual format: theme_{theme_name}
        if rest.startswith('theme_'):
            theme_name = rest[len('theme_'):]
            return (llm, 'themes', theme_name)
        # Check for LLM format: {size}_{theme_name}
        for theme in THEME_NAMES:
            if rest.endswith('_' + theme):
                return (llm, 'themes', theme)
            # Handle themes with underscores
            if theme in rest:
                return (llm, 'themes', theme)
        # Fallback: remove size prefix
        for size in ['large_', 'small_', 'manual_']:
            if rest.startswith(size):
                return (llm, 'themes', rest[len(size):])
        return (llm, 'themes', rest)

    # Check for specific_themes pattern
    if remaining.startswith('specific_themes'):
        return (llm, 'specific_themes', '')

    # Check for sentiment pattern
    if remaining.startswith('sentiment'):
        return (llm, 'sentiment', '')

    return (llm, 'unknown', remaining)


def load_session_metadata(session_path: Path) -> List[Dict]:
    """Load model mappings from session_metadata.json."""
    metadata_path = session_path / 'session_metadata.json'
    if not metadata_path.exists():
        return []
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata.get('models', [])


def load_output_columns(session_path: Path) -> List[Dict]:
    """Load model mappings from 05_output_columns.json."""
    output_cols_path = session_path / 'steps' / '05_output_columns.json'
    if not output_cols_path.exists():
        return []
    with open(output_cols_path, 'r') as f:
        data = json.load(f)
    return data.get('data', {}).get('plan', [])


def build_model_mappings(session_path: Path) -> List[ModelMapping]:
    """Build model mappings from session metadata."""
    mappings = []
    models = load_output_columns(session_path)
    if not models:
        models = load_session_metadata(session_path)

    for model_data in models:
        model_id = model_data.get('id', model_data.get('relative_name', ''))
        columns = model_data.get('columns', {})
        prefix = model_data.get('prefix', '')

        if not prefix:
            label_col = columns.get('label', '')
            if label_col:
                prefix = label_col.replace('_label', '')

        llm, category, subcategory = extract_category_from_model_id(model_id)

        mappings.append(ModelMapping(
            prefix=prefix,
            model_id=model_id,
            llm=llm,
            category=category,
            subcategory=subcategory,
            label_column=columns.get('label', f'{prefix}_label'),
            probability_column=columns.get('probability', f'{prefix}_probability')
        ))

    return mappings


def load_session(session_path: Path, session_name: str, session_type: str) -> Optional[SessionData]:
    """Load a complete annotation session."""
    annotations_path = session_path / 'annotations.csv'
    if not annotations_path.exists():
        logger.warning(f"No annotations.csv found for session {session_name}")
        return None

    try:
        df = pd.read_csv(annotations_path)
        logger.info(f"Loaded {len(df)} rows from {session_name}")
    except Exception as e:
        logger.error(f"Error loading annotations for {session_name}: {e}")
        return None

    mappings = build_model_mappings(session_path)
    logger.info(f"Found {len(mappings)} model mappings for {session_name}")

    return SessionData(name=session_name, session_type=session_type, df=df, mappings=mappings)


# =============================================================================
# ANNOTATION EXTRACTION
# =============================================================================

def extract_predictions_for_row(
    row: pd.Series,
    mappings: List[ModelMapping],
    session_type: str
) -> Dict[str, Dict]:
    """Extract predictions for a single row from all models."""
    llm_predictions = defaultdict(lambda: {
        'themes': [],
        'political_parties': [],
        'specific_themes': [],
        'sentiment': None
    })

    for mapping in mappings:
        label_col = mapping.label_column
        if label_col not in row.index:
            continue

        label_value = row[label_col]
        if pd.isna(label_value):
            continue

        model_name, is_present = parse_label_value(label_value)
        if not model_name:
            continue

        pipeline_name = f"{mapping.llm}_{session_type}"

        if mapping.category == 'themes':
            if is_present:
                theme_label = f"theme_{mapping.subcategory}"
                if theme_label not in llm_predictions[pipeline_name]['themes']:
                    llm_predictions[pipeline_name]['themes'].append(theme_label)

        elif mapping.category == 'political_parties':
            if is_present:
                party_label = f"political_parties_{mapping.subcategory}"
                if party_label not in llm_predictions[pipeline_name]['political_parties']:
                    llm_predictions[pipeline_name]['political_parties'].append(party_label)

        elif mapping.category == 'specific_themes':
            # Handle multi-class specific themes
            # Labels format: {llm}_specific_themes_{size}_{theme} or manual_specific_themes_specific_themes_{theme}
            label_str = str(label_value)
            for theme_suffix in ['welfare_state', 'early_learning_childcare', 'public_finance']:
                if label_str.endswith('_' + theme_suffix):
                    full_label = f'specific_themes_{theme_suffix}'
                    if full_label not in llm_predictions[pipeline_name]['specific_themes']:
                        llm_predictions[pipeline_name]['specific_themes'].append(full_label)
                    break

        elif mapping.category == 'sentiment':
            # Labels can be: sentiment_neutral OR gemma_sentiment_large_neutral
            label_str = str(label_value).lower()
            if '_positive' in label_str or label_str == 'positive':
                llm_predictions[pipeline_name]['sentiment'] = 'sentiment_positive'
            elif '_negative' in label_str or label_str == 'negative':
                llm_predictions[pipeline_name]['sentiment'] = 'sentiment_negative'
            elif '_neutral' in label_str or label_str == 'neutral':
                llm_predictions[pipeline_name]['sentiment'] = 'sentiment_neutral'

    return dict(llm_predictions)


def parse_benchmark_annotations(annotations_str: str) -> Dict:
    """Parse the benchmark annotation JSON string."""
    if pd.isna(annotations_str) or not annotations_str:
        return {'manual_themes': [], 'manual_sentiment': None,
                'manual_specific_themes': [], 'manual_political_parties': []}
    try:
        return json.loads(annotations_str)
    except:
        return {'manual_themes': [], 'manual_sentiment': None,
                'manual_specific_themes': [], 'manual_political_parties': []}


# =============================================================================
# MAIN COMPILATION
# =============================================================================

def compile_all_annotations(benchmark_df: pd.DataFrame, sessions: List[SessionData]) -> pd.DataFrame:
    """Compile all annotations from all sessions into the benchmark DataFrame."""
    result_df = benchmark_df.copy()

    session_id_maps = {}
    for session in sessions:
        session_id_maps[session.name] = {row['id']: idx for idx, row in session.df.iterrows()}

    all_annotations_list = []

    for _, row in result_df.iterrows():
        row_id = row['id']
        benchmark = parse_benchmark_annotations(row.get('annotations', '{}'))
        all_pipelines = {}

        for session in sessions:
            if row_id not in session_id_maps[session.name]:
                continue
            session_row_idx = session_id_maps[session.name][row_id]
            session_row = session.df.iloc[session_row_idx]
            predictions = extract_predictions_for_row(session_row, session.mappings, session.session_type)
            all_pipelines.update(predictions)

        consolidated = {'benchmark': benchmark, 'pipelines': all_pipelines}
        all_annotations_list.append(json.dumps(consolidated, ensure_ascii=False))

    result_df['all_annotations'] = all_annotations_list
    return result_df


def generate_models_summary(sessions: List[SessionData]) -> pd.DataFrame:
    """Generate summary of all models."""
    rows = []
    for session in sessions:
        for mapping in session.mappings:
            rows.append({
                'session_name': session.name,
                'session_type': session.session_type,
                'model_id': mapping.model_id,
                'llm': mapping.llm,
                'category': mapping.category,
                'subcategory': mapping.subcategory,
                'prefix': mapping.prefix,
                'label_column': mapping.label_column,
                'pipeline_name': f"{mapping.llm}_{session.session_type}"
            })
    return pd.DataFrame(rows)


def main():
    """Main function."""
    # Paths relative to paper/scripts/
    base_path = Path(__file__).parent.parent.parent  # LLM_Tool root
    logs_path = base_path / 'logs' / 'annotation_studio'
    data_path = base_path / 'data' / 'sets'

    # Find annotation sessions
    session_dirs = {}
    for session_dir in sorted(logs_path.iterdir()):
        if not session_dir.is_dir():
            continue
        dir_name = session_dir.name
        for session_type in ['manual', 'small', 'large']:
            if dir_name.startswith(f"{session_type}_"):
                session_dirs[session_type] = session_dir

    logger.info("Found annotation sessions:")
    for session_type, path in session_dirs.items():
        logger.info(f"  {session_type}: {path.name}")

    # Load sessions
    sessions = []
    for session_type, path in session_dirs.items():
        session = load_session(path, path.name, session_type)
        if session:
            sessions.append(session)

    if not sessions:
        logger.error("No sessions loaded!")
        return

    # Load benchmark
    benchmark_path = data_path / 'final_test_small.csv'
    logger.info(f"Loading benchmark from {benchmark_path}")
    benchmark_df = pd.read_csv(benchmark_path)
    logger.info(f"Loaded {len(benchmark_df)} benchmark rows")

    # Compile annotations
    logger.info("Compiling all annotations...")
    result_df = compile_all_annotations(benchmark_df, sessions)

    # Save consolidated file
    output_path = data_path / 'final_test_small_annotated.csv'
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved consolidated annotations to {output_path}")

    # Generate and save models summary
    summary_df = generate_models_summary(sessions)
    summary_path = data_path / 'models_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved models summary to {summary_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPILATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total rows: {len(result_df)}")
    logger.info(f"Total model mappings: {len(summary_df)}")

    pipelines = set()
    for _, row in result_df.iterrows():
        ann = json.loads(row['all_annotations'])
        pipelines.update(ann.get('pipelines', {}).keys())

    logger.info(f"\nUnique pipelines: {len(pipelines)}")
    for p in sorted(pipelines):
        logger.info(f"  - {p}")

    # Check for unknown categories
    unknown_count = len(summary_df[summary_df['category'] == 'unknown'])
    if unknown_count > 0:
        logger.warning(f"\n{unknown_count} models with unknown category!")
        logger.warning(summary_df[summary_df['category'] == 'unknown'][['model_id', 'category']].head(10).to_string())

    return result_df, summary_df


if __name__ == '__main__':
    result_df, summary_df = main()
