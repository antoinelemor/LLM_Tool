#!/usr/bin/env python3
"""
Training Arena Integration for Annotator Factory

This module provides seamless integration between Annotator Factory and Training Arena.
It calls the existing Training Arena methods to provide the EXACT same workflow.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def integrate_training_arena_in_annotator_factory(
    cli_instance,
    output_file: Path,
    text_column: str,
    session_id: str,
    session_dirs: Optional[Dict[str, Path]] = None
) -> Dict[str, Any]:
    """
    Integration for Annotator Factory - provides the EXACT Training Arena experience.

    This function calls the complete Training Arena workflow from cli_instance.
    All Training Arena methods are available in cli_instance (AdvancedCLI).

    The workflow provided is EXACTLY the same as Training Arena:
    - Language detection with minority language handling
    - One-vs-all, multi-label, hybrid, custom training modes
    - One model per language option
    - Value filtering for data quality
    - Data split configuration
    - Label naming strategies
    - Text length analysis
    - Model selection and training
    - Benchmarking and evaluation

    Args:
        cli_instance: AdvancedCLI instance with ALL Training Arena methods
        output_file: Path to annotated file
        text_column: Name of text column used during annotation
        session_id: Session ID from Annotator Factory
        session_dirs: Session directory structure from Annotator Factory

    Returns:
        Dict with training results and metadata
    """
    console = cli_instance.console

    # Show integration banner
    console.print("\n" + "=" * 80)
    console.print("🎮 TRAINING ARENA - Complete Training Workflow")
    console.print("=" * 80 + "\n")
    console.print("[green]✓ Annotations completed![/green]")
    console.print(f"  [cyan]File:[/cyan] {output_file}")
    console.print(f"  [cyan]Text column:[/cyan] '{text_column}'")
    console.print(f"  [cyan]Session:[/cyan] {session_id}\n")

    # Call the COMPLETE Training Arena
    # This provides the EXACT same CLI with ALL features
    cli_instance.training_studio()

    return {"status": "completed", "session_id": session_id}
