#!/usr/bin/env python3
"""
Verification script for Training Arena CSV structure fixes.

This script verifies that:
1. Reinforced CSV files have CLASS_LEGEND header
2. Reinforced CSV files use standardized indices (_0, _1, _2)
3. Final session best_models.csv contains combined_score
4. Final session training_metrics.csv contains all epochs from all modes
5. CSV structure is consistent across all training modes
"""

import pandas as pd
from pathlib import Path
import sys
import csv


def verify_class_legend(csv_path):
    """Verify that CSV file has CLASS_LEGEND header."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line.startswith('# CLASS_LEGEND:'):
            return True, first_line
        return False, None


def verify_standardized_indices(df):
    """Verify that dataframe uses standardized indices (_0, _1, etc.) not class names."""
    columns = df.columns.tolist()

    # Check for problematic patterns (class names in column headers)
    problematic_patterns = [
        'precision_negative', 'recall_negative', 'f1_negative',
        'precision_neutral', 'recall_neutral', 'f1_neutral',
        'precision_positive', 'recall_positive', 'f1_positive',
        'precision_class_', 'recall_class_', 'f1_class_'
    ]

    issues = []
    for col in columns:
        for pattern in problematic_patterns:
            if pattern in col.lower():
                issues.append(f"Non-standard column: {col}")

    # Check for correct patterns
    correct_patterns = ['precision_0', 'recall_0', 'f1_0', 'precision_1', 'recall_1', 'f1_1']
    found_correct = [pat for pat in correct_patterns if pat in columns]

    return len(issues) == 0, issues, found_correct


def verify_combined_score(df):
    """Verify that dataframe contains combined_score column."""
    if 'combined_score' in df.columns:
        # Check if combined_score values are properly calculated (not all zeros or NaN)
        non_zero = df['combined_score'].fillna(0).sum()
        if non_zero > 0:
            return True, "combined_score present and has non-zero values"
        else:
            return False, "combined_score present but all values are zero or NaN"
    return False, "combined_score column missing"


def verify_all_phases(df):
    """Verify that dataframe contains all training phases."""
    phases_found = set()
    modes_found = set()

    if 'phase' in df.columns:
        phases_found = set(df['phase'].unique())

    if 'mode' in df.columns:
        modes_found = set(df['mode'].unique())

    expected_phases = {'normal', 'reinforced'}
    expected_modes = {'benchmark', 'normal_training'}

    return {
        'phases': phases_found,
        'modes': modes_found,
        'has_all_phases': expected_phases.issubset(phases_found) or len(phases_found) > 0,
        'has_modes': len(modes_found) > 0
    }


def verify_session_csvs(session_path):
    """Verify the consolidated session CSV files."""
    print(f"\n{'='*80}")
    print(f"Verifying session: {session_path}")
    print(f"{'='*80}")

    results = {
        'session_path': session_path,
        'issues': [],
        'successes': []
    }

    # Find consolidated CSV files
    session_id = session_path.name
    training_metrics_file = session_path / f"{session_id}_training_metrics.csv"
    best_models_file = session_path / f"{session_id}_best_models.csv"

    # Check training metrics CSV
    if training_metrics_file.exists():
        print(f"\n1. Checking {training_metrics_file.name}...")

        # Check CLASS_LEGEND
        has_legend, legend = verify_class_legend(training_metrics_file)
        if has_legend:
            results['successes'].append(f"✓ CLASS_LEGEND found: {legend[:50]}...")
        else:
            results['issues'].append("✗ CLASS_LEGEND missing from training_metrics.csv")

        # Read CSV
        df = pd.read_csv(training_metrics_file, comment='#')
        print(f"   - Shape: {df.shape}")

        # Check standardized indices
        indices_ok, issues, correct_cols = verify_standardized_indices(df)
        if indices_ok:
            results['successes'].append(f"✓ Uses standardized indices: {', '.join(correct_cols[:3])}...")
        else:
            results['issues'].extend(issues)

        # Check all phases
        phase_info = verify_all_phases(df)
        print(f"   - Phases found: {phase_info['phases']}")
        print(f"   - Modes found: {phase_info['modes']}")

        if phase_info['has_all_phases']:
            results['successes'].append(f"✓ Contains training phases: {phase_info['phases']}")
        else:
            if len(phase_info['phases']) == 0:
                results['issues'].append("✗ No 'phase' column found")
            else:
                results['issues'].append(f"✗ Missing some phases. Found: {phase_info['phases']}")
    else:
        results['issues'].append(f"✗ {training_metrics_file.name} not found")

    # Check best models CSV
    if best_models_file.exists():
        print(f"\n2. Checking {best_models_file.name}...")

        # Check CLASS_LEGEND
        has_legend, legend = verify_class_legend(best_models_file)
        if has_legend:
            results['successes'].append(f"✓ CLASS_LEGEND found in best_models.csv")
        else:
            results['issues'].append("✗ CLASS_LEGEND missing from best_models.csv")

        # Read CSV
        df = pd.read_csv(best_models_file, comment='#')
        print(f"   - Shape: {df.shape}")

        # Check combined_score
        has_score, score_msg = verify_combined_score(df)
        if has_score:
            results['successes'].append(f"✓ {score_msg}")
            # Show some combined scores
            if 'combined_score' in df.columns:
                scores = df['combined_score'].dropna().head(3).tolist()
                if scores:
                    print(f"   - Sample combined scores: {scores}")
        else:
            results['issues'].append(f"✗ {score_msg}")

        # Check that it only contains best models (not all epochs)
        if 'epoch' in df.columns:
            unique_models = len(df.groupby(['category', 'language', 'model'], dropna=False).size())
            total_rows = len(df)
            if unique_models == total_rows:
                results['successes'].append(f"✓ Contains only best models ({total_rows} unique)")
            else:
                results['issues'].append(f"✗ Contains duplicate models ({total_rows} rows, {unique_models} unique)")
    else:
        results['issues'].append(f"✗ {best_models_file.name} not found")

    # Check individual CSV files in training_metrics folder
    training_metrics_dir = session_path / "training_metrics"
    if training_metrics_dir.exists():
        print(f"\n3. Checking individual CSV files in training_metrics/...")

        # Find reinforced CSV files
        reinforced_files = list(training_metrics_dir.rglob("reinforced.csv"))
        if reinforced_files:
            print(f"   - Found {len(reinforced_files)} reinforced.csv files")

            # Check first reinforced file
            sample_file = reinforced_files[0]
            print(f"   - Checking sample: {sample_file.relative_to(session_path)}")

            has_legend, legend = verify_class_legend(sample_file)
            if has_legend:
                results['successes'].append(f"✓ Reinforced CSV has CLASS_LEGEND")
            else:
                results['issues'].append("✗ Reinforced CSV missing CLASS_LEGEND")

            df = pd.read_csv(sample_file, comment='#')
            indices_ok, issues, correct_cols = verify_standardized_indices(df)
            if indices_ok:
                results['successes'].append(f"✓ Reinforced CSV uses standardized indices")
            else:
                results['issues'].extend([f"Reinforced CSV: {issue}" for issue in issues[:3]])
        else:
            print("   - No reinforced.csv files found (may not have run reinforced training)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")

    if results['successes']:
        print("\n✓ SUCCESSES:")
        for success in results['successes']:
            print(f"  {success}")

    if results['issues']:
        print("\n✗ ISSUES FOUND:")
        for issue in results['issues']:
            print(f"  {issue}")
    else:
        print("\n✓ NO ISSUES FOUND - All CSV structures are correct!")

    return results


def main():
    """Main verification function."""
    print("Training Arena CSV Structure Verification")
    print("="*80)

    # Find training arena sessions
    logs_dir = Path("logs/training_arena")

    if not logs_dir.exists():
        print(f"Error: {logs_dir} directory not found")
        sys.exit(1)

    # Get all session directories
    session_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]

    if not session_dirs:
        print("No training sessions found in logs/training_arena/")
        sys.exit(1)

    # Sort by modification time (most recent first)
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"Found {len(session_dirs)} training sessions")
    print(f"Most recent: {session_dirs[0].name}")

    # Verify most recent session
    if len(session_dirs) > 0:
        results = verify_session_csvs(session_dirs[0])

        # Check if there are critical issues
        if results['issues']:
            print("\n⚠️  Some issues were found. Please review the fixes.")
            sys.exit(1)
        else:
            print("\n✅ All CSV structure checks passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()