#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
file_handlers.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive file handling for various data formats
including CSV, Excel, Parquet, and RData/RDS files with support for
incremental saving and Unicode handling.

Dependencies:
-------------
- sys
- pandas
- pathlib
- logging
- typing
- json

MAIN FEATURES:
--------------
1) Read/write CSV files with incremental append
2) Excel file support (.xlsx, .xls)
3) Parquet file support
4) RData/RDS file support via pyreadr
5) Unicode sanitization for safe file writing
6) Batch processing and incremental saving

Author:
-------
Antoine Lemor
"""

import logging
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings

import pandas as pd
import numpy as np

# Try to import pyreadr for RData support
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("pyreadr not installed. RData/RDS support disabled.")

# Constants
CSV_APPEND = True
OTHER_FORMAT_SAVE_EVERY = 50


class FileHandler:
    """Base file handler for various data formats"""

    def __init__(self, file_path: Union[str, Path], format: Optional[str] = None):
        """
        Initialize file handler.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the file
        format : str, optional
            File format. If None, detected from extension
        """
        self.file_path = Path(file_path)
        self.logger = logging.getLogger(__name__)
        
        # Detect format from extension if not provided
        if format:
            self.format = format.lower()
        else:
            self.format = self._detect_format()
        
        # Validate format
        self._validate_format()
        
        # Initialize format-specific settings
        self.append_mode = self.format == 'csv' and CSV_APPEND
        self.save_frequency = OTHER_FORMAT_SAVE_EVERY if self.format != 'csv' else 1
        self.pending_saves = 0

    def _detect_format(self) -> str:
        """Detect file format from extension"""
        suffix = self.file_path.suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.xlsm': 'excel',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.rdata': 'rdata',
            '.rda': 'rdata',
            '.rds': 'rds'
        }
        
        format = format_map.get(suffix)
        if not format:
            raise ValueError(f"Unknown file format: {suffix}")
        
        return format

    def _validate_format(self):
        """Validate that required libraries are available for format"""
        if self.format in ['rdata', 'rds'] and not HAS_PYREADR:
            raise ImportError(
                f"pyreadr is required for {self.format.upper()} files. "
                "Install with: pip install pyreadr"
            )

    def read(self, **kwargs) -> pd.DataFrame:
        """
        Read data from file.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to the read function
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        try:
            if self.format == 'csv':
                return self._read_csv(**kwargs)
            elif self.format == 'excel':
                return self._read_excel(**kwargs)
            elif self.format == 'parquet':
                return self._read_parquet(**kwargs)
            elif self.format == 'rdata':
                return self._read_rdata(**kwargs)
            elif self.format == 'rds':
                return self._read_rds(**kwargs)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
                
        except Exception as e:
            self.logger.error(f"Error reading {self.format} file: {e}")
            raise

    def write(self, data: pd.DataFrame, **kwargs):
        """
        Write data to file.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to write
        **kwargs
            Additional arguments passed to the write function
        """
        # Create parent directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sanitize data before writing
        data = self._sanitize_dataframe(data)
        
        try:
            if self.format == 'csv':
                self._write_csv(data, **kwargs)
            elif self.format == 'excel':
                self._write_excel(data, **kwargs)
            elif self.format == 'parquet':
                self._write_parquet(data, **kwargs)
            elif self.format == 'rdata':
                self._write_rdata(data, **kwargs)
            elif self.format == 'rds':
                self._write_rds(data, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
                
            self.logger.debug(f"Wrote {len(data)} rows to {self.file_path}")
            
        except Exception as e:
            self.logger.error(f"Error writing {self.format} file: {e}")
            raise

    def append_row(
        self,
        row_data: Union[pd.DataFrame, pd.Series, Dict],
        identifier_column: Optional[str] = None,
        identifier_value: Optional[Any] = None
    ):
        """
        Append a single row to file (optimized for CSV).
        
        Parameters
        ----------
        row_data : DataFrame, Series, or dict
            Row data to append
        identifier_column : str, optional
            Column name for identifier
        identifier_value : Any, optional
            Value to match for the row
        """
        # Convert to DataFrame if needed
        if isinstance(row_data, dict):
            row_df = pd.DataFrame([row_data])
        elif isinstance(row_data, pd.Series):
            row_df = row_data.to_frame().T
        elif isinstance(row_data, pd.DataFrame):
            if identifier_column and identifier_value is not None:
                row_df = row_data[row_data[identifier_column] == identifier_value].copy()
            else:
                row_df = row_data.copy()
        else:
            raise ValueError(f"Unsupported row data type: {type(row_data)}")
        
        if self.format == 'csv' and self.append_mode:
            # Direct append for CSV
            self._append_csv(row_df)
        else:
            # For other formats, accumulate and save periodically
            self.pending_saves += 1
            if self.pending_saves >= self.save_frequency:
                # Need to read, append, and write back
                if self.file_path.exists():
                    existing_data = self.read()
                    combined_data = pd.concat([existing_data, row_df], ignore_index=True)
                    self.write(combined_data)
                else:
                    self.write(row_df)
                self.pending_saves = 0

    def _read_csv(self, **kwargs) -> pd.DataFrame:
        """Read CSV file"""
        default_kwargs = {
            'encoding': 'utf-8',
            'on_bad_lines': 'warn'
        }
        default_kwargs.update(kwargs)
        return pd.read_csv(self.file_path, **default_kwargs)

    def _write_csv(self, data: pd.DataFrame, **kwargs):
        """Write CSV file"""
        default_kwargs = {
            'index': False,
            'encoding': 'utf-8',
            'errors': 'replace'
        }
        default_kwargs.update(kwargs)
        data.to_csv(self.file_path, **default_kwargs)

    def _append_csv(self, data: pd.DataFrame):
        """Append to CSV file"""
        header_needed = not self.file_path.exists()
        
        # Sanitize before appending
        data = self._sanitize_dataframe(data)
        
        try:
            data.to_csv(
                self.file_path,
                mode='a',
                index=False,
                header=header_needed,
                encoding='utf-8',
                errors='replace'
            )
        except UnicodeEncodeError as e:
            self.logger.error(f"Unicode error appending to CSV: {e}")
            # Try with ASCII-only as fallback
            data = self._force_ascii(data)
            data.to_csv(
                self.file_path,
                mode='a',
                index=False,
                header=header_needed
            )

    def _read_excel(self, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        return pd.read_excel(self.file_path, **kwargs)

    def _write_excel(self, data: pd.DataFrame, **kwargs):
        """Write Excel file"""
        default_kwargs = {
            'index': False,
            'engine': 'openpyxl'
        }
        default_kwargs.update(kwargs)
        
        # Handle large Excel files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data.to_excel(self.file_path, **default_kwargs)

    def _read_parquet(self, **kwargs) -> pd.DataFrame:
        """Read Parquet file"""
        return pd.read_parquet(self.file_path, **kwargs)

    def _write_parquet(self, data: pd.DataFrame, **kwargs):
        """Write Parquet file"""
        default_kwargs = {
            'index': False,
            'engine': 'pyarrow',
            'compression': 'snappy'
        }
        default_kwargs.update(kwargs)
        data.to_parquet(self.file_path, **default_kwargs)

    def _read_rdata(self, **kwargs) -> pd.DataFrame:
        """Read RData file"""
        if not HAS_PYREADR:
            raise ImportError("pyreadr required for RData files")
        
        result = pyreadr.read_r(str(self.file_path))
        # Return the first dataframe (most common case)
        if result:
            return list(result.values())[0]
        else:
            raise ValueError("No data found in RData file")

    def _write_rdata(self, data: pd.DataFrame, **kwargs):
        """Write RData file"""
        if not HAS_PYREADR:
            raise ImportError("pyreadr required for RData files")
        
        # Default object name if not provided
        object_name = kwargs.get('object_name', 'data')
        pyreadr.write_rdata(
            {object_name: data},
            str(self.file_path)
        )

    def _read_rds(self, **kwargs) -> pd.DataFrame:
        """Read RDS file"""
        if not HAS_PYREADR:
            raise ImportError("pyreadr required for RDS files")
        
        return pyreadr.read_r(str(self.file_path))

    def _write_rds(self, data: pd.DataFrame, **kwargs):
        """Write RDS file"""
        if not HAS_PYREADR:
            raise ImportError("pyreadr required for RDS files")
        
        pyreadr.write_rds(data, str(self.file_path))

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame for safe file writing"""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':  # String columns
                df_copy[col] = df_copy[col].apply(
                    lambda x: self._sanitize_string(x) if isinstance(x, str) else x
                )
        
        return df_copy

    def _sanitize_string(self, text: str) -> str:
        """Remove or replace invalid characters from string"""
        if not text:
            return text
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Replace invalid UTF-8 sequences
        text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')
        
        # Remove or replace problematic Unicode characters
        text = re.sub(r'[\uDC00-\uDFFF]', '', text)  # Surrogate pairs
        text = re.sub(r'[\uFFFE\uFFFF]', '', text)  # Invalid characters
        
        return text

    def _force_ascii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force DataFrame to ASCII-only characters"""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].apply(
                    lambda x: ''.join(
                        char for char in str(x) if ord(char) < 128
                    ) if isinstance(x, str) else x
                )
        
        return df_copy


class IncrementalFileWriter:
    """
    Handler for incremental file writing with batching.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        batch_size: int = 50
    ):
        """
        Initialize incremental writer.
        
        Parameters
        ----------
        file_path : str or Path
            Output file path
        format : str, optional
            File format
        batch_size : int
            Batch size for non-CSV formats
        """
        self.handler = FileHandler(file_path, format)
        self.batch_size = batch_size
        self.buffer = []
        self.logger = logging.getLogger(__name__)

    def write_row(self, row_data: Union[Dict, pd.Series]):
        """Write single row, batching as needed"""
        if self.handler.format == 'csv':
            # Direct append for CSV
            self.handler.append_row(row_data)
        else:
            # Buffer for other formats
            self.buffer.append(row_data)
            
            if len(self.buffer) >= self.batch_size:
                self.flush()

    def flush(self):
        """Flush buffered data to file"""
        if not self.buffer:
            return
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.buffer)
        
        # Write or append
        if self.handler.file_path.exists():
            existing = self.handler.read()
            combined = pd.concat([existing, df], ignore_index=True)
            self.handler.write(combined)
        else:
            self.handler.write(df)
        
        self.buffer = []
        self.logger.debug(f"Flushed {len(df)} rows to {self.handler.file_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


def create_file_handler(
    file_path: Union[str, Path],
    format: Optional[str] = None
) -> FileHandler:
    """
    Factory function to create appropriate file handler.
    
    Parameters
    ----------
    file_path : str or Path
        File path
    format : str, optional
        File format (auto-detected if None)
    
    Returns
    -------
    FileHandler
        Appropriate file handler instance
    """
    return FileHandler(file_path, format)


def strip_log_columns(
    df: pd.DataFrame,
    annotation_column: str,
    suffixes: List[str] = ['raw_per_prompt', 'cleaned_per_prompt', 'status_per_prompt']
) -> pd.DataFrame:
    """
    Remove log columns from DataFrame for cleaner output.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    annotation_column : str
        Base annotation column name
    suffixes : list
        Suffixes to remove
    
    Returns
    -------
    pd.DataFrame
        DataFrame without log columns
    """
    columns_to_drop = [
        f"{annotation_column}_{suffix}"
        for suffix in suffixes
        if f"{annotation_column}_{suffix}" in df.columns
    ]
    
    if columns_to_drop:
        return df.drop(columns=columns_to_drop)
    return df


def write_log_csv(
    log_path: Union[str, Path],
    log_entry: Dict[str, Any]
):
    """
    Write log entry to CSV file.
    
    Parameters
    ----------
    log_path : str or Path
        Path to log file
    log_entry : dict
        Log entry to write
    """
    log_path = Path(log_path)
    
    # Check if file exists to determine if header is needed
    write_header = not log_path.exists()
    
    # Sanitize log entry
    for key in log_entry:
        if isinstance(log_entry[key], str):
            log_entry[key] = log_entry[key].replace('\n', ' ').replace('\r', '')
    
    # Write to CSV
    df = pd.DataFrame([log_entry])
    df.to_csv(
        log_path,
        mode='a',
        index=False,
        header=write_header,
        encoding='utf-8',
        errors='replace'
    )
