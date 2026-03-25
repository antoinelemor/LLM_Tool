#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
prompt_manager.py

MAIN OBJECTIVE:
---------------
This script manages prompt loading, processing, and multi-prompt workflows
for LLM annotation, including prefix management and JSON key extraction.

Dependencies:
-------------
- sys
- os
- pathlib
- typing
- logging
- json

MAIN FEATURES:
--------------
1) Load prompts from files or directories
2) Extract expected JSON keys from prompts
3) Manage multi-prompt workflows
4) Handle prompt prefixes for key renaming
5) Validate prompt structure

Author:
-------
Antoine Lemor
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import re

from .json_cleaner import extract_expected_keys


class PromptManager:
    """Manage prompts for LLM annotation"""

    def __init__(self):
        """Initialize the prompt manager"""
        self.logger = logging.getLogger(__name__)

    def load_prompt(self, prompt_path: str) -> Tuple[str, List[str]]:
        """
        Load a prompt file containing a self-contained prompt.

        We preserve the **entirety** of the text (no splitting before/after
        "Expected JSON Keys" section), so each prompt keeps its own
        definitions/examples.

        The expected JSON keys are detected *read-only* based on the
        complete content; we don't modify the string returned to the model.

        Parameters
        ----------
        prompt_path : str
            Absolute/relative path to the .txt file.

        Returns
        -------
        tuple
            - `full_prompt`  : str  → complete prompt, ready to be concatenated
            - `expected_keys`: list → order of JSON key appearance
        """
        with open(prompt_path, "r", encoding="utf-8") as f:
            full_prompt = f.read().strip()

        # Robust detection of keys, without cutting the prompt
        expected_keys = extract_expected_keys(full_prompt)

        return full_prompt, expected_keys

    def verify_prompt_structure(self, base_prompt: str, expected_keys: List[str]) -> None:
        """
        Verify whether the prompt structure is correct and display
        information about the detected JSON keys.
        """
        print("\n=== Verification of Prompt Structure ===")
        if not base_prompt:
            print("Warning: the main prompt is empty or not detected.")
        else:
            print(f"Length of main prompt: {len(base_prompt)} characters.")

        if expected_keys:
            print(f"Detected JSON keys: {expected_keys}")
            print("Prompt structure: OK")
        else:
            print("No JSON keys have been detected.")
            print("Either the '**Expected JSON Keys**' segment is missing or the block is not recognized.")
        print("==========================================\n")

    def get_prompts_with_prefix(self) -> List[Tuple[str, List[str], str]]:
        """
        Interactively load one or several prompt files **and** ask, for each,
        whether a prefix word should be prepended to every JSON key _after_ the
        model has answered (the prefix is **not** sent to the model).

        Returns
        -------
        list[tuple]  Each tuple contains:
            0. full_prompt   : str   – the complete prompt text
            1. expected_keys : list  – keys detected in the prompt
            2. prefix_word   : str   – '' if no prefix requested
        """
        multiple_prompts = self._prompt_user_yes_no(
            "Do you want to use multiple successive prompts for each text?"
        )

        list_of_prompts: List[Tuple[str, List[str], str]] = []

        if multiple_prompts:
            # Ask if user wants to load from a folder
            load_from_folder = self._prompt_user_yes_no(
                "Do you want to automatically load all .txt prompts from a folder?"
            )

            if load_from_folder:
                list_of_prompts = self._load_prompts_from_folder()
            else:
                list_of_prompts = self._load_prompts_interactively()
        else:
            # Single prompt
            prompt_path = input("Path to the prompt file (.txt): ").strip()
            while not os.path.isfile(prompt_path):
                print("File not found. Please try again.")
                prompt_path = input("Path to the prompt file (.txt): ").strip()

            full_prompt, expected_keys = self.load_prompt(prompt_path)
            self.verify_prompt_structure(full_prompt, expected_keys)

            # Ask for prefix
            use_prefix = self._prompt_user_yes_no(
                "Do you want to add a prefix to the JSON keys returned by the model?"
            )
            prefix_word = ""
            if use_prefix:
                prefix_word = input("Prefix to add (e.g., 'prompt1'): ").strip()
                print(f"Keys will be prefixed with '{prefix_word}_' in the final output.")

            list_of_prompts = [(full_prompt, expected_keys, prefix_word)]

        return list_of_prompts

    def _load_prompts_from_folder(self) -> List[Tuple[str, List[str], str]]:
        """Load all .txt prompts from a folder"""
        folder_path = input("Folder path: ").strip()

        while not os.path.isdir(folder_path):
            print(f"⚠ Directory not found: {folder_path}")
            folder_path = input("Folder path: ").strip()

        # Find all .txt files in the folder
        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

        if not txt_files:
            print(f"❌ No .txt files found in {folder_path}")
            return []

        # Display found files
        print(f"\n📂 Found {len(txt_files)} prompt files:")
        for i, filename in enumerate(txt_files, 1):
            print(f"  {i}. {filename}")

        # Load each file
        list_of_prompts = []
        for i, filename in enumerate(txt_files, 1):
            filepath = os.path.join(folder_path, filename)
            print(f"\n[{i}/{len(txt_files)}] Loading {filename}...")

            try:
                full_prompt, expected_keys = self.load_prompt(filepath)

                if expected_keys:
                    print(f"✓ Detected {len(expected_keys)} JSON keys: {expected_keys[:3]}{'...' if len(expected_keys) > 3 else ''}")

                    # Ask for prefix for this specific prompt
                    print(f"Prefix for '{filename}' keys?")
                    use_prefix = self._prompt_user_yes_no("Add prefix?")
                    prefix_word = ""
                    if use_prefix:
                        # Suggest filename without extension as default prefix
                        default_prefix = Path(filename).stem.lower().replace(' ', '_')
                        prefix_word = input(f"Prefix (default: {default_prefix}): ").strip()
                        if not prefix_word:
                            prefix_word = default_prefix
                        print(f"✓ Keys will be prefixed with '{prefix_word}_'")

                    list_of_prompts.append((full_prompt, expected_keys, prefix_word))
                else:
                    print(f"⚠ No JSON keys detected in {filename}, skipping...")

            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue

        if list_of_prompts:
            print(f"\n✓ Successfully loaded {len(list_of_prompts)} prompts")
        else:
            print("\n❌ No valid prompts could be loaded")

        return list_of_prompts

    def _load_prompts_interactively(self) -> List[Tuple[str, List[str], str]]:
        """Load prompts interactively one by one"""
        num_prompts = int(input("How many prompts do you want to use? ").strip())

        list_of_prompts = []
        for i in range(1, num_prompts + 1):
            print(f"\n=== Prompt {i}/{num_prompts} ===")
            prompt_path = input(f"Path to prompt {i} (.txt): ").strip()

            while not os.path.isfile(prompt_path):
                print("File not found. Please try again.")
                prompt_path = input(f"Path to prompt {i} (.txt): ").strip()

            full_prompt, expected_keys = self.load_prompt(prompt_path)
            self.verify_prompt_structure(full_prompt, expected_keys)

            # Ask for prefix
            use_prefix = self._prompt_user_yes_no(
                f"Add a prefix to keys from prompt {i}?"
            )
            prefix_word = ""
            if use_prefix:
                prefix_word = input(f"Prefix for prompt {i} (e.g., 'p{i}'): ").strip()
                print(f"Keys will be prefixed with '{prefix_word}_'")

            list_of_prompts.append((full_prompt, expected_keys, prefix_word))

        return list_of_prompts

    def build_combined_text(self, row: Dict[str, Any], text_columns: List[str], prefixes: List[str]) -> str:
        """
        Concatenate the requested columns with their respective prefixes.

        Example result with two columns:
        ----------------------------------------------------
        <prefix-1>
        <row[text_columns[0]]>

        <prefix-2>
        <row[text_columns[1]]>

        Parameters
        ----------
        row : dict or pandas.Series
            The current row.
        text_columns : list[str]
            Column names to read, in order.
        prefixes : list[str]
            Prefix messages, in the same order as text_columns.

        Returns
        -------
        str
            A single string ready to be appended to the base prompt.
        """
        segments = []
        for col, pre in zip(text_columns, prefixes):
            # Handle both dict and pandas Series
            if hasattr(row, 'get'):
                raw = str(row.get(col, ""))
            else:
                raw = str(row[col]) if col in row and row[col] is not None else ""
            segments.append(f"{pre}\n{raw}")
        return "\n\n".join(segments).strip()

    def merge_json_objects(self, json_list: List[Dict]) -> Dict:
        """
        Receive a list of JSON objects (dicts) and merge their keys.
        If the same key appears in multiple JSON objects, the last occurrence takes precedence.
        """
        merged = {}
        for d in json_list:
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                merged[k] = v
        return merged

    def apply_prefix(self, data: Dict, prefix: str) -> Dict:
        """
        Prepend *prefix*_ to every top-level key of *data* unless prefix == ''.

        Parameters
        ----------
        data   : dict   – parsed JSON object coming from the model
        prefix : str    – word to prepend, or '' for no change
        """
        # Handle case where data is not a dict
        if not isinstance(data, dict):
            self.logger.warning(f"Expected dict but got {type(data).__name__}. Attempting to convert.")

            # Try to convert list to dict if possible
            if isinstance(data, list):
                if len(data) == 1 and isinstance(data[0], dict):
                    # If it's a list with single dict, extract the dict
                    data = data[0]
                    self.logger.info("Converted single-item list to dict")
                elif all(isinstance(item, dict) for item in data):
                    # If it's a list of dicts, merge them
                    merged = {}
                    for item in data:
                        merged.update(item)
                    data = merged
                    self.logger.info("Merged list of dicts into single dict")
                else:
                    # Can't convert to dict meaningfully
                    self.logger.error(f"Cannot convert list to dict: {data}")
                    return {}
            else:
                # Not a list or dict, return empty
                return {}

        if not prefix:
            return data
        return {f"{prefix}_{k}": v for k, v in data.items()}

    def _prompt_user_yes_no(self, question: str) -> bool:
        """Display a yes/no question to the user"""
        while True:
            choice = input(f"\n❓ {question} (yes/no): ").strip().lower()
            if choice in ['yes', 'y', 'oui', 'o']:
                print("✓ Yes selected")
                return True
            elif choice in ['no', 'n', 'non']:
                print("✗ No selected")
                return False
            else:
                print("⚠ Please answer with 'yes' or 'no'.")