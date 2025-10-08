#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_filter_logger.py

MAIN OBJECTIVE:
---------------
Centralized logging system for all data filtering operations throughout the repository.
Provides consistent traceability and debugging capabilities by logging all filtered data
with detailed reasons and context.

Dependencies:
-------------
- logging
- pathlib
- datetime
- typing
- json
- pandas

MAIN FEATURES:
--------------
1) Centralized logging of all filtered data across the entire codebase
2) Detailed tracking with reasons, locations, and timestamps
3) Support for individual items, batches, and DataFrame filtering
4) Automatic session statistics and summaries
5) JSONL format for easy parsing and analysis
6) Sample data preservation for debugging
7) Global singleton pattern for consistent logging
8) Human-readable summary reports

Author:
-------
Antoine Lemor
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import pandas as pd


class DataFilterLogger:
    """Centralized logger for tracking filtered data throughout the application"""

    def __init__(self, log_dir: Optional[Path] = None, logger_name: str = "data_filter", session_id: Optional[str] = None):
        """
        Initialize the data filter logger

        Args:
            log_dir: Directory to store filter logs (default: logs/filtered_data/)
            logger_name: Name for the logger instance
            session_id: Optional Training Arena session ID for contextualized logging
        """
        self.logger = logging.getLogger(logger_name)

        # Setup log directory based on context
        if log_dir is None:
            if session_id:
                # Training Arena context: logs go to session directory
                log_dir = Path.cwd() / "logs" / "training_arena" / session_id / "filtered_data"
            else:
                # General context: logs go to application directory
                log_dir = Path.cwd() / "logs" / "application" / "filtered_data"
        self.log_dir = Path(log_dir)
        # Don't create directory yet - will be created on first write

        # Prepare timestamped log file path for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_file = self.log_dir / f"filter_log_{timestamp}.jsonl"

        # Statistics
        self.session_stats = {
            'total_filtered': 0,
            'filters_by_reason': {},
            'filters_by_location': {}
        }

    def log_filtered_item(
        self,
        item: Any,
        reason: str,
        location: str,
        item_index: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        log_content: bool = True
    ):
        """
        Log a single filtered item

        Args:
            item: The filtered item (text, row, dict, etc.)
            reason: Why it was filtered (e.g., "empty_text", "invalid_format", "nan_value")
            location: Where filtering occurred (e.g., "model_trainer.train_single_model", "csv_loader")
            item_index: Optional index/position of the item
            additional_info: Any additional context information
            log_content: Whether to include the actual content in logs (default: True)
        """
        # Update statistics
        self.session_stats['total_filtered'] += 1
        self.session_stats['filters_by_reason'][reason] = \
            self.session_stats['filters_by_reason'].get(reason, 0) + 1
        self.session_stats['filters_by_location'][location] = \
            self.session_stats['filters_by_location'].get(location, 0) + 1

        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'location': location,
            'item_index': item_index,
            'item_type': type(item).__name__,
        }

        # Add item content if requested
        if log_content:
            if isinstance(item, str):
                log_entry['content'] = item[:500]  # Truncate long strings
                log_entry['content_length'] = len(item)
            elif isinstance(item, dict):
                log_entry['content'] = {k: str(v)[:100] for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                log_entry['content'] = [str(x)[:100] for x in item[:5]]  # First 5 items
            elif pd.notna(item):
                log_entry['content'] = str(item)[:500]
            else:
                log_entry['content'] = 'NaN/None'

        # Add additional info
        if additional_info:
            log_entry['additional_info'] = additional_info

        # Ensure log directory exists before writing
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write to JSONL file
        with open(self.current_session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        # Also log to standard logger
        self.logger.warning(
            f"[{location}] Filtered item {item_index if item_index is not None else ''}: "
            f"reason={reason}, type={type(item).__name__}"
        )

    def log_filtered_batch(
        self,
        items: List[Any],
        reason: str,
        location: str,
        indices: Optional[List[int]] = None,
        log_sample_size: int = 5
    ):
        """
        Log a batch of filtered items (more efficient for large batches)

        Args:
            items: List of filtered items
            reason: Common reason for filtering
            location: Where filtering occurred
            indices: Optional list of indices corresponding to items
            log_sample_size: Number of sample items to log in detail
        """
        count = len(items)

        if count == 0:
            return

        # Update statistics
        self.session_stats['total_filtered'] += count
        self.session_stats['filters_by_reason'][reason] = \
            self.session_stats['filters_by_reason'].get(reason, 0) + count
        self.session_stats['filters_by_location'][location] = \
            self.session_stats['filters_by_location'].get(location, 0) + count

        # Create summary entry
        summary_entry = {
            'timestamp': datetime.now().isoformat(),
            'batch': True,
            'count': count,
            'reason': reason,
            'location': location,
            'indices': indices[:log_sample_size] if indices else None,
        }

        # Add sample items
        summary_entry['sample_items'] = []
        for i, item in enumerate(items[:log_sample_size]):
            sample = {
                'index': indices[i] if indices and i < len(indices) else i,
                'type': type(item).__name__,
            }

            if isinstance(item, str):
                sample['content'] = item[:200]
                sample['length'] = len(item)
            elif isinstance(item, dict):
                sample['keys'] = list(item.keys())
            elif pd.notna(item):
                sample['content'] = str(item)[:200]
            else:
                sample['content'] = 'NaN/None'

            summary_entry['sample_items'].append(sample)

        # Ensure log directory exists before writing
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write to JSONL file
        with open(self.current_session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')

        # Log summary
        self.logger.warning(
            f"[{location}] Filtered {count} items: reason={reason}"
        )

    def log_dataframe_filtering(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        reason: str,
        location: str,
        text_column: Optional[str] = None,
        log_filtered_samples: int = 3
    ):
        """
        Log filtering operation on a pandas DataFrame

        Args:
            df_before: DataFrame before filtering
            df_after: DataFrame after filtering
            reason: Reason for filtering
            location: Where filtering occurred
            text_column: Column name containing text data (to show samples)
            log_filtered_samples: Number of filtered samples to log
        """
        filtered_count = len(df_before) - len(df_after)

        if filtered_count == 0:
            return

        # Update statistics
        self.session_stats['total_filtered'] += filtered_count
        self.session_stats['filters_by_reason'][reason] = \
            self.session_stats['filters_by_reason'].get(reason, 0) + filtered_count
        self.session_stats['filters_by_location'][location] = \
            self.session_stats['filters_by_location'].get(location, 0) + filtered_count

        # Find filtered rows (rows in df_before but not in df_after)
        filtered_indices = df_before.index.difference(df_after.index)

        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'dataframe_filtering': True,
            'reason': reason,
            'location': location,
            'filtered_count': filtered_count,
            'original_count': len(df_before),
            'remaining_count': len(df_after),
            'filtered_indices': filtered_indices.tolist()[:10],  # First 10 indices
        }

        # Add sample filtered rows
        if text_column and text_column in df_before.columns:
            sample_rows = []
            for idx in filtered_indices[:log_filtered_samples]:
                if idx in df_before.index:
                    row_data = {
                        'index': int(idx),
                        'text': str(df_before.loc[idx, text_column])[:200]
                    }
                    # Add other relevant columns
                    for col in df_before.columns:
                        if col != text_column and col not in ['Unnamed: 0']:
                            row_data[col] = str(df_before.loc[idx, col])[:100]
                    sample_rows.append(row_data)
            log_entry['sample_filtered_rows'] = sample_rows

        # Ensure log directory exists before writing
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write to JSONL file
        with open(self.current_session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        # Log summary
        self.logger.warning(
            f"[{location}] DataFrame filtering: {filtered_count}/{len(df_before)} rows removed "
            f"(reason: {reason})"
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session"""
        return {
            **self.session_stats,
            'log_file': str(self.current_session_file),
            'timestamp': datetime.now().isoformat()
        }

    def print_session_summary(self):
        """Print a human-readable summary of filtering in this session"""
        print("\n" + "="*60)
        print("📋 DATA FILTERING SUMMARY")
        print("="*60)
        print(f"Total items filtered: {self.session_stats['total_filtered']}")

        if self.session_stats['filters_by_reason']:
            print("\n🔍 Filtering by reason:")
            for reason, count in sorted(
                self.session_stats['filters_by_reason'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  • {reason}: {count} items")

        if self.session_stats['filters_by_location']:
            print("\n📍 Filtering by location:")
            for location, count in sorted(
                self.session_stats['filters_by_location'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  • {location}: {count} items")

        print(f"\n📄 Detailed log: {self.current_session_file}")
        print("="*60 + "\n")


# Global singleton instance
_global_filter_logger: Optional[DataFilterLogger] = None


def get_filter_logger(log_dir: Optional[Path] = None, session_id: Optional[str] = None) -> DataFilterLogger:
    """
    Get or create the global filter logger instance

    Args:
        log_dir: Optional log directory (only used on first call)
        session_id: Optional Training Arena session ID for contextualized logging

    Returns:
        DataFilterLogger instance
    """
    global _global_filter_logger

    if _global_filter_logger is None:
        _global_filter_logger = DataFilterLogger(log_dir=log_dir, session_id=session_id)

    return _global_filter_logger


def reset_filter_logger():
    """Reset the global filter logger (useful for testing)"""
    global _global_filter_logger
    _global_filter_logger = None
