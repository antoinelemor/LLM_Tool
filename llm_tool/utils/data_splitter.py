#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
data_splitter.py

MAIN OBJECTIVE:
---------------
This script provides data splitting and sampling utilities for preparing
datasets for annotation and training, including stratified sampling and
train/test/validation splits.

Dependencies:
-------------
- sys
- pandas
- numpy
- sklearn
- logging
- typing

MAIN FEATURES:
--------------
1) Train/test/validation split
2) Stratified sampling
3) Random sampling with seed
4) K-fold cross-validation splits
5) Balanced sampling for imbalanced datasets

Author:
-------
Antoine Lemor
"""

import logging
from typing import Tuple, Optional, List, Union
import math

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    KFold,
    StratifiedKFold
)
from sklearn.utils import resample


class DataSplitter:
    """Utility class for data splitting and sampling"""

    def __init__(self, random_state: int = 42):
        """
        Initialize data splitter.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def train_test_val_split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, test, and validation sets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        test_size : float
            Proportion for test set
        val_size : float
            Proportion for validation set (from train)
        stratify_column : str, optional
            Column for stratified split
        
        Returns
        -------
        tuple
            train_df, val_df, test_df
        """
        # First split: train+val vs test
        stratify = data[stratify_column] if stratify_column else None
        
        train_val_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_val = train_val_df[stratify_column] if stratify_column else None
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_val
        )
        
        self.logger.info(
            f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df

    def stratified_sample(
        self,
        data: pd.DataFrame,
        sample_size: Union[int, float],
        stratify_column: str
    ) -> pd.DataFrame:
        """
        Perform stratified sampling.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        sample_size : int or float
            Number of samples or proportion
        stratify_column : str
            Column for stratification
        
        Returns
        -------
        pd.DataFrame
            Sampled data
        """
        if sample_size > 1:
            # Absolute number
            sample_size = min(sample_size, len(data))
            sample_prop = sample_size / len(data)
        else:
            # Proportion
            sample_prop = sample_size
            sample_size = int(len(data) * sample_prop)
        
        # Use stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1-sample_prop,
            random_state=self.random_state
        )
        
        train_idx, _ = next(splitter.split(data, data[stratify_column]))
        sampled = data.iloc[train_idx]
        
        self.logger.info(
            f"Stratified sample: {len(sampled)} rows from {len(data)}"
        )
        
        return sampled

    def random_sample(
        self,
        data: pd.DataFrame,
        sample_size: Union[int, float]
    ) -> pd.DataFrame:
        """
        Perform random sampling.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        sample_size : int or float
            Number of samples or proportion
        
        Returns
        -------
        pd.DataFrame
            Sampled data
        """
        if sample_size > 1:
            # Absolute number
            sample_size = min(int(sample_size), len(data))
        else:
            # Proportion
            sample_size = int(len(data) * sample_size)
        
        sampled = data.sample(n=sample_size, random_state=self.random_state)
        
        self.logger.info(
            f"Random sample: {len(sampled)} rows from {len(data)}"
        )
        
        return sampled

    def balanced_sample(
        self,
        data: pd.DataFrame,
        target_column: str,
        method: str = 'undersample'
    ) -> pd.DataFrame:
        """
        Balance dataset by class.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_column : str
            Target column with classes
        method : str
            'undersample' or 'oversample'
        
        Returns
        -------
        pd.DataFrame
            Balanced data
        """
        # Get class counts
        class_counts = data[target_column].value_counts()
        
        if method == 'undersample':
            # Undersample to minority class
            min_count = class_counts.min()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_data = data[data[target_column] == class_label]
                sampled = resample(
                    class_data,
                    n_samples=min_count,
                    random_state=self.random_state
                )
                balanced_dfs.append(sampled)
            
        elif method == 'oversample':
            # Oversample to majority class
            max_count = class_counts.max()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_data = data[data[target_column] == class_label]
                sampled = resample(
                    class_data,
                    n_samples=max_count,
                    replace=True,
                    random_state=self.random_state
                )
                balanced_dfs.append(sampled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the result
        balanced = balanced.sample(frac=1, random_state=self.random_state)
        
        self.logger.info(
            f"Balanced sample ({method}): {len(balanced)} rows from {len(data)}"
        )
        
        return balanced

    def kfold_splits(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        stratify_column: Optional[str] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create K-fold cross-validation splits.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        n_splits : int
            Number of folds
        stratify_column : str, optional
            Column for stratified k-fold
        
        Returns
        -------
        list
            List of (train, test) DataFrame tuples
        """
        if stratify_column:
            kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            splits_gen = kfold.split(data, data[stratify_column])
        else:
            kfold = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            splits_gen = kfold.split(data)
        
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits_gen, 1):
            train_fold = data.iloc[train_idx]
            test_fold = data.iloc[test_idx]
            splits.append((train_fold, test_fold))
            
            self.logger.info(
                f"Fold {fold_idx}: Train={len(train_fold)}, Test={len(test_fold)}"
            )
        
        return splits

    def calculate_sample_size(
        self,
        population_size: int,
        confidence_level: float = 0.95,
        margin_error: float = 0.05,
        proportion: float = 0.5
    ) -> int:
        """
        Calculate required sample size for given confidence interval.
        
        Parameters
        ----------
        population_size : int
            Total population size
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        margin_error : float
            Margin of error (e.g., 0.05 for 5%)
        proportion : float
            Expected proportion (0.5 for maximum variability)
        
        Returns
        -------
        int
            Required sample size
        """
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        
        z = z_scores.get(confidence_level, 1.96)
        
        # Sample size formula
        numerator = (z**2) * proportion * (1 - proportion)
        denominator = margin_error**2
        
        # Finite population correction
        sample_size = (numerator / denominator) / (
            1 + ((numerator / denominator - 1) / population_size)
        )
        
        sample_size = min(math.ceil(sample_size), population_size)
        
        self.logger.info(
            f"Calculated sample size: {sample_size} "
            f"(population={population_size}, CI={confidence_level*100}%, ME={margin_error*100}%)"
        )
        
        return sample_size

    def split_by_date(
        self,
        data: pd.DataFrame,
        date_column: str,
        train_end_date: str,
        val_end_date: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split data by date for time series.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with date column
        date_column : str
            Name of date column
        train_end_date : str
            End date for training set
        val_end_date : str, optional
            End date for validation set
        
        Returns
        -------
        tuple
            (train, test) or (train, val, test) DataFrames
        """
        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        data = data.sort_values(date_column)
        
        # Split by dates
        train = data[data[date_column] <= train_end_date]
        
        if val_end_date:
            val = data[
                (data[date_column] > train_end_date) &
                (data[date_column] <= val_end_date)
            ]
            test = data[data[date_column] > val_end_date]
            
            self.logger.info(
                f"Date split: Train={len(train)}, Val={len(val)}, Test={len(test)}"
            )
            
            return train, val, test
        else:
            test = data[data[date_column] > train_end_date]
            
            self.logger.info(
                f"Date split: Train={len(train)}, Test={len(test)}"
            )
            
            return train, test

    def create_holdout_set(
        self,
        data: pd.DataFrame,
        holdout_size: float = 0.1,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a holdout set for final evaluation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        holdout_size : float
            Proportion for holdout set
        stratify_column : str, optional
            Column for stratification
        
        Returns
        -------
        tuple
            (main_data, holdout_data)
        """
        stratify = data[stratify_column] if stratify_column else None
        
        main_data, holdout_data = train_test_split(
            data,
            test_size=holdout_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        self.logger.info(
            f"Holdout split: Main={len(main_data)}, Holdout={len(holdout_data)}"
        )
        
        return main_data, holdout_data
