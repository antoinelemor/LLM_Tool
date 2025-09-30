#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
annotation_validator.py

MAIN OBJECTIVE:
---------------
This script provides validation and quality control for LLM annotations,
including export to Doccano format for human review and correction.

Dependencies:
-------------
- pandas
- numpy
- json
- typing

MAIN FEATURES:
--------------
1) Annotation validation and quality checks
2) Export to Doccano JSONL format
3) Inter-annotator agreement calculation
4) Confidence score analysis
5) Label distribution analysis
6) Error detection and reporting
7) Sample selection for human review
8) Batch validation with progress tracking

Author:
-------
Antoine Lemor
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from collections import Counter, defaultdict


@dataclass
class ValidationConfig:
    """Configuration for annotation validation"""
    sample_size: int = 100
    stratified_sampling: bool = True
    confidence_threshold: float = 0.8
    export_format: str = "jsonl"  # jsonl, csv, both
    export_to_doccano: bool = True
    check_label_consistency: bool = True
    check_schema_compliance: bool = True
    calculate_agreement: bool = True
    min_samples_per_label: int = 5
    max_samples_per_label: int = 50
    include_metadata: bool = True
    random_seed: int = 42


@dataclass
class DoccanoAnnotation:
    """Doccano format annotation"""
    text: str
    label: str
    meta: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_jsonl(self) -> str:
        """Convert to Doccano JSONL format"""
        data = {
            "text": self.text,
            "label": self.label
        }
        if self.meta:
            data["meta"] = self.meta
        if self.id is not None:
            data["id"] = self.id
        return json.dumps(data, ensure_ascii=False)


@dataclass
class ValidationResult:
    """Results from validation process"""
    total_annotations: int
    validated_samples: int
    label_distribution: Dict[str, int]
    confidence_stats: Dict[str, float]
    quality_score: float
    issues_found: List[Dict[str, Any]]
    export_path: Optional[str] = None
    doccano_export_path: Optional[str] = None
    agreement_score: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AnnotationValidator:
    """Validates and exports LLM annotations for quality control"""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the validator"""
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def load_annotations(self, file_path: str) -> pd.DataFrame:
        """Load annotations from file"""
        self.logger.info(f"Loading annotations from {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")

        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.logger.info(f"Loaded {len(df)} annotations")
        return df

    def validate_schema(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate annotation schema and structure"""
        issues = []

        # Check required columns
        required_columns = ['text']
        for col in required_columns:
            if col not in df.columns:
                issues.append({
                    'type': 'missing_column',
                    'column': col,
                    'severity': 'error'
                })

        # Check for label columns (could be 'label', 'labels', 'annotation', etc.)
        label_columns = [col for col in df.columns if 'label' in col.lower() or 'annotation' in col.lower()]
        if not label_columns:
            issues.append({
                'type': 'no_label_column',
                'severity': 'error',
                'message': 'No label or annotation column found'
            })

        # Check for empty values
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append({
                    'type': 'null_values',
                    'column': col,
                    'count': int(null_count),
                    'percentage': float(null_count / len(df) * 100),
                    'severity': 'warning'
                })

        # Check for duplicate texts
        if 'text' in df.columns:
            duplicates = df[df.duplicated(subset=['text'], keep=False)]
            if len(duplicates) > 0:
                issues.append({
                    'type': 'duplicate_texts',
                    'count': len(duplicates),
                    'severity': 'warning',
                    'examples': duplicates['text'].head(5).tolist()
                })

        return issues

    def analyze_label_distribution(self, df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
        """Analyze the distribution of labels"""
        label_counts = df[label_column].value_counts()

        # Calculate statistics
        stats = {
            'unique_labels': len(label_counts),
            'label_counts': label_counts.to_dict(),
            'most_common': label_counts.index[0] if len(label_counts) > 0 else None,
            'least_common': label_counts.index[-1] if len(label_counts) > 0 else None,
            'imbalance_ratio': float(label_counts.max() / label_counts.min()) if len(label_counts) > 0 and label_counts.min() > 0 else None
        }

        # Check for severe imbalance
        if stats['imbalance_ratio'] and stats['imbalance_ratio'] > 10:
            self.logger.warning(f"Severe label imbalance detected: {stats['imbalance_ratio']:.2f}x")

        return stats

    def analyze_confidence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze confidence scores if available"""
        confidence_stats = {}

        # Look for confidence column
        confidence_columns = [col for col in df.columns if 'confidence' in col.lower() or 'score' in col.lower()]

        if confidence_columns:
            conf_col = confidence_columns[0]
            confidence_stats = {
                'mean': float(df[conf_col].mean()),
                'median': float(df[conf_col].median()),
                'std': float(df[conf_col].std()),
                'min': float(df[conf_col].min()),
                'max': float(df[conf_col].max()),
                'low_confidence_count': int((df[conf_col] < self.config.confidence_threshold).sum()),
                'low_confidence_percentage': float((df[conf_col] < self.config.confidence_threshold).mean() * 100)
            }

            self.logger.info(f"Average confidence: {confidence_stats['mean']:.3f}")

        return confidence_stats

    def select_validation_samples(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """Select samples for validation using stratified sampling"""
        if self.config.stratified_sampling and label_column in df.columns:
            # Stratified sampling
            samples = []
            label_counts = df[label_column].value_counts()

            for label, count in label_counts.items():
                label_df = df[df[label_column] == label]

                # Determine number of samples for this label
                n_samples = min(
                    max(self.config.min_samples_per_label, int(self.config.sample_size * count / len(df))),
                    min(self.config.max_samples_per_label, len(label_df))
                )

                # Sample from this label
                label_samples = label_df.sample(n=n_samples, random_state=self.config.random_seed)
                samples.append(label_samples)

            validation_df = pd.concat(samples, ignore_index=True)

        else:
            # Random sampling
            n_samples = min(self.config.sample_size, len(df))
            validation_df = df.sample(n=n_samples, random_state=self.config.random_seed)

        self.logger.info(f"Selected {len(validation_df)} samples for validation")
        return validation_df

    def export_to_doccano(self, df: pd.DataFrame, output_path: str,
                         text_column: str = "text", label_column: str = "label") -> str:
        """Export annotations to Doccano format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting {len(df)} annotations to Doccano format")

        # Create Doccano annotations
        annotations = []
        for idx, row in df.iterrows():
            # Prepare metadata
            meta = {}
            if self.config.include_metadata:
                # Add all non-text, non-label columns as metadata
                for col in df.columns:
                    if col not in [text_column, label_column]:
                        value = row[col]
                        # Convert numpy types to Python types
                        if pd.notna(value):
                            if isinstance(value, (np.integer, np.int64)):
                                value = int(value)
                            elif isinstance(value, (np.floating, np.float64)):
                                value = float(value)
                            meta[col] = value

                # Add validation metadata
                meta['validation_timestamp'] = datetime.now().isoformat()
                meta['original_index'] = int(idx)

            annotation = DoccanoAnnotation(
                text=str(row[text_column]),
                label=str(row[label_column]) if pd.notna(row[label_column]) else "",
                meta=meta,
                id=int(idx)
            )
            annotations.append(annotation)

        # Write to JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                f.write(annotation.to_jsonl() + '\n')

        self.logger.info(f"Exported to Doccano format: {output_path}")
        return str(output_path)

    def calculate_agreement(self, df: pd.DataFrame, label1_column: str,
                           label2_column: str) -> float:
        """Calculate inter-annotator agreement if multiple annotations exist"""
        if label1_column in df.columns and label2_column in df.columns:
            # Remove rows where either annotation is missing
            valid_df = df.dropna(subset=[label1_column, label2_column])

            if len(valid_df) > 0:
                # Calculate Cohen's Kappa
                kappa = cohen_kappa_score(
                    valid_df[label1_column],
                    valid_df[label2_column]
                )
                self.logger.info(f"Inter-annotator agreement (Cohen's Kappa): {kappa:.3f}")
                return float(kappa)

        return None

    def calculate_quality_score(self, df: pd.DataFrame, issues: List[Dict],
                               confidence_stats: Dict[str, float]) -> float:
        """Calculate overall quality score for annotations"""
        score = 100.0

        # Deduct points for issues
        for issue in issues:
            if issue['severity'] == 'error':
                score -= 20
            elif issue['severity'] == 'warning':
                score -= 5

        # Factor in confidence if available
        if confidence_stats and 'mean' in confidence_stats:
            confidence_factor = confidence_stats['mean']
            score = score * confidence_factor

        # Factor in completeness
        if 'text' in df.columns:
            completeness = 1.0 - (df['text'].isna().sum() / len(df))
            score = score * completeness

        # Ensure score is between 0 and 100
        score = max(0, min(100, score))

        return score

    def validate(self, input_file: str, output_dir: Optional[str] = None) -> ValidationResult:
        """Main validation pipeline"""
        self.logger.info("Starting annotation validation")

        # Load annotations
        df = self.load_annotations(input_file)

        # Determine label column
        label_columns = [col for col in df.columns if 'label' in col.lower() or 'annotation' in col.lower()]
        label_column = label_columns[0] if label_columns else 'label'

        # Validate schema
        issues = []
        if self.config.check_schema_compliance:
            issues = self.validate_schema(df)

        # Analyze label distribution
        label_stats = {}
        if label_column in df.columns:
            label_stats = self.analyze_label_distribution(df, label_column)

        # Analyze confidence
        confidence_stats = self.analyze_confidence(df)

        # Calculate quality score
        quality_score = self.calculate_quality_score(df, issues, confidence_stats)

        # Select validation samples
        validation_df = self.select_validation_samples(df, label_column)

        # Export to various formats
        output_dir = Path(output_dir) if output_dir else Path("./validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        export_paths = {}

        # Export to Doccano
        if self.config.export_to_doccano:
            doccano_path = output_dir / f"doccano_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            export_paths['doccano'] = self.export_to_doccano(
                validation_df, doccano_path,
                text_column='text', label_column=label_column
            )

        # Export to CSV
        if self.config.export_format in ['csv', 'both']:
            csv_path = output_dir / f"validation_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            validation_df.to_csv(csv_path, index=False)
            export_paths['csv'] = str(csv_path)

        # Export to JSONL
        if self.config.export_format in ['jsonl', 'both']:
            jsonl_path = output_dir / f"validation_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            validation_df.to_json(jsonl_path, orient='records', lines=True)
            export_paths['jsonl'] = str(jsonl_path)

        # Create validation result
        result = ValidationResult(
            total_annotations=len(df),
            validated_samples=len(validation_df),
            label_distribution=label_stats.get('label_counts', {}),
            confidence_stats=confidence_stats,
            quality_score=quality_score,
            issues_found=issues,
            export_path=export_paths.get('csv') or export_paths.get('jsonl'),
            doccano_export_path=export_paths.get('doccano')
        )

        # Save validation report
        report_path = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            report_dict = asdict(result)
            # Convert numpy arrays to lists for JSON serialization
            if result.confusion_matrix is not None:
                report_dict['confusion_matrix'] = result.confusion_matrix.tolist()
            json.dump(report_dict, f, indent=2, default=str)

        self.logger.info(f"Validation complete. Quality score: {quality_score:.1f}/100")
        self.logger.info(f"Validation report saved to: {report_path}")

        return result

    async def validate_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for validation (for pipeline integration)"""
        # Update config
        input_file = config.get('input_data') or config.get('input_file')
        if not input_file:
            raise ValueError("No input file specified")

        # Update validation config from dict
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Run validation
        result = self.validate(
            input_file=input_file,
            output_dir=config.get('output_dir', './validation')
        )

        # Convert to dict for pipeline
        return {
            'samples_validated': result.validated_samples,
            'quality_score': result.quality_score,
            'issues_found': len(result.issues_found),
            'export_path': result.export_path,
            'doccano_export_path': result.doccano_export_path
        }


def main():
    """Example usage"""
    # Initialize validator
    validator = AnnotationValidator(
        config=ValidationConfig(
            sample_size=100,
            stratified_sampling=True,
            export_to_doccano=True,
            export_format='both'
        )
    )

    # Validate annotations
    result = validator.validate(
        input_file="data/annotations.csv",
        output_dir="./validation"
    )

    print(f"Quality Score: {result.quality_score:.1f}/100")
    print(f"Validated Samples: {result.validated_samples}")
    print(f"Doccano Export: {result.doccano_export_path}")


if __name__ == "__main__":
    main()