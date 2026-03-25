#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
bert_abc.py

MAIN OBJECTIVE:
---------------
Define the abstract base contract that all Bert-based trainers must follow,
ensuring consistent encode, training, and prediction interfaces.

Dependencies:
-------------
- abc
- typing
- torch.types
- torch.utils.data

MAIN FEATURES:
--------------
1) Declare constructor requirements shared by every Bert implementation
2) Specify the encode API for turning raw text into tensors
3) Enforce training and prediction method signatures across subclasses
4) Provide hooks for on-disk prediction using saved checkpoints
5) Standardise time formatting helpers used by downstream dashboards

Author:
-------
Antoine Lemor
"""

from abc import ABC, abstractmethod
from typing import List, Any

from torch.types import Device
from torch.utils.data import DataLoader


class BertABC(ABC):
    """
    Abstract class defining that Bert implementations can inherit from.
    """

    @abstractmethod
    def __init__(
            self,
            model_name: str,
            tokenizer: Any,
            model_sequence_classifier: Any,
            device: Device | None = None,
    ):
        pass

    @abstractmethod
    def encode(
            self,
            sequences: List[str],
            labels: List[str | int] | None = None,
            batch_size: int = 32,
            progress_bar: bool = True,
            add_special_tokens: bool = True
    ):
        pass

    @abstractmethod
    def run_training(
            self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            n_epochs: int = 3,
            lr: float = 5e-5,
            random_state: int = 42,
            save_model_as: str | None = None
    ):
        pass

    @abstractmethod
    def predict(
            self,
            dataloader: DataLoader,
            model: Any,
            proba: bool = True,
            progress_bar: bool = True
    ):
        pass

    @abstractmethod
    def predict_with_model(
            self,
            dataloader: DataLoader,
            model_path: str,
            proba: bool = True,
            progress_bar: bool = True
    ):
        pass

    @abstractmethod
    def format_time(
            self,
            elapsed: float | int
    ):
        pass
