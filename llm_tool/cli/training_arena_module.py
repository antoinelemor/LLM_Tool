"""
Training Arena Module - Modularized training workflow for LLM-JSON format data
This module provides the complete Training Arena workflow that can be integrated
into both the standalone Training Arena and the Annotator Factory.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich import box
import logging

class TrainingArenaWorkflow:
    """
    Encapsulates the complete Training Arena workflow for training models from LLM-JSON data.
    This can be called from multiple entry points (Training Arena, Annotator Factory, etc.)
    """

    def __init__(self, console: Optional[Console] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the Training Arena workflow.

        Parameters
        ----------
        console : Console, optional
            Rich console for output. If None, creates a new one.
        logger : logging.Logger, optional
            Logger instance. If None, creates a default logger.
        """
        self.console = console if console else Console()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.current_session_id = None
        self.current_session_manager = None

    def run_training_workflow(
        self,
        data_file: Optional[Path] = None,
        text_column: Optional[str] = None,
        annotation_column: Optional[str] = None,
        session_id: Optional[str] = None,
        auto_mode: bool = False,
        preloaded_config: Optional[Dict[str, Any]] = None,
        parent_session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run the complete Training Arena workflow for llm-json format data.

        Parameters
        ----------
        data_file : Path, optional
            Path to the data file. If None, will prompt user to select.
        text_column : str, optional
            Name of the text column. If None, will auto-detect or prompt.
        annotation_column : str, optional
            Name of the annotation/label column. If None, will auto-detect or prompt.
        session_id : str, optional
            Session ID for organization. If None, will generate one.
        auto_mode : bool, default False
            If True, uses defaults instead of prompting where possible.
        preloaded_config : dict, optional
            Pre-loaded configuration from saved session.

        Returns
        -------
        dict
            Dictionary containing training results and metadata
        """
        try:
            # Import required components
            from llm_tool.trainers.dataset_builder import TrainingDatasetBuilder, TrainingDataBundle
            from llm_tool.utils.training_data_utils import TrainingDataSessionManager
            from llm_tool.utils.data_detector import DataDetector

            # Step 1: Session Management
            if not session_id:
                if not auto_mode:
                    session_id = self._prompt_for_session_id()
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_id = f"training_session_{timestamp}"

            self.current_session_id = session_id
            self.console.print(f"\n[bold green]✓ Session ID:[/bold green] [cyan]{session_id}[/cyan]")
            self.console.print(f"[dim]This ID will be used consistently across all data, logs, and models[/dim]\n")

            # Initialize session manager
            # If parent_session_dir is provided (from Annotator Factory), use it as base
            if parent_session_dir:
                # Create Training Arena subdirectory within Annotator Factory session
                from pathlib import Path
                training_arena_dir = parent_session_dir / "training_arena"
                training_arena_dir.mkdir(parents=True, exist_ok=True)

                # Create session manager with custom base directory
                self.current_session_manager = TrainingDataSessionManager(
                    session_id=session_id,
                    base_dir=training_arena_dir
                )
            else:
                # Standard Training Arena session management
                self.current_session_manager = TrainingDataSessionManager(session_id=session_id)

            # Initialize dataset builder
            builder = TrainingDatasetBuilder(
                Path("data") / "training_data",
                session_id=session_id
            )

            # Step 2: Dataset Configuration
            if data_file:
                # Use provided file directly
                self.console.print(f"[green]✓ Using provided data file: {data_file.name}[/green]\n")
                bundle = self._configure_llm_json_dataset(
                    builder,
                    data_file,
                    text_column,
                    annotation_column,
                    auto_mode
                )
            else:
                # Run the full dataset wizard
                bundle = self._run_dataset_wizard(builder)

            if bundle is None:
                self.console.print("[yellow]Training cancelled.[/yellow]")
                return {"status": "cancelled"}

            # Step 3: Show Dataset Summary
            self._render_bundle_summary(bundle)

            # Step 4: Configure Learning Parameters and Execute
            self.console.print("\n[bold cyan]Configuring learning parameters...[/bold cyan]\n")

            # Execute training with confirmation
            training_results = self._confirm_and_execute(
                bundle,
                "quick",
                preloaded_config=preloaded_config,
                auto_mode=auto_mode
            )

            return {
                "status": "completed",
                "session_id": session_id,
                "bundle": bundle,
                "results": training_results
            }

        except Exception as e:
            self.logger.exception(f"Training Arena workflow failed: {e}")
            self.console.print(f"[red]Training failed: {e}[/red]")
            return {"status": "failed", "error": str(e)}

    def _prompt_for_session_id(self) -> str:
        """Prompt user for session name and generate full session ID."""
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print("[bold cyan]           📝 Session Name Configuration                       [/bold cyan]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        self.console.print("[bold]Why session names matter:[/bold]")
        self.console.print("  • [green]Organization:[/green] Easily identify experiments (e.g., 'baseline', 'improved_features')")
        self.console.print("  • [green]Traceability:[/green] Track your training runs across data, logs, and models")
        self.console.print("  • [green]Collaboration:[/green] Team members understand what each session represents")
        self.console.print("  • [green]Audit trail:[/green] Timestamp ensures uniqueness\n")

        self.console.print("[dim]Format: {session_name}_{yyyymmdd_hhmmss}[/dim]")
        self.console.print("[dim]Example: sentiment_analysis_20251008_143022[/dim]\n")

        # Ask for user-defined session name
        user_session_name = Prompt.ask(
            "[bold yellow]Enter a descriptive name for this training session[/bold yellow]",
            default="training_session"
        ).strip()

        # Sanitize the user input
        user_session_name = user_session_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        user_session_name = ''.join(c for c in user_session_name if c.isalnum() or c in ['_', '-'])

        # Create full session ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_session_name}_{timestamp}"

    def _configure_llm_json_dataset(
        self,
        builder,
        data_file: Path,
        text_column: Optional[str],
        annotation_column: Optional[str],
        auto_mode: bool = False
    ):
        """Configure dataset for llm-json format with provided or detected columns."""
        from llm_tool.utils.data_detector import DataDetector

        # Analyze file structure
        self.console.print("\n[bold cyan]Analyzing Dataset Structure[/bold cyan]")
        self.console.print("[dim]🔍 Analyzing columns, detecting types, and extracting samples...[/dim]")

        analysis = DataDetector.analyze_file_intelligently(data_file)

        # Load dataframe for column detection
        if data_file.suffix.lower() == '.csv':
            df = pd.read_csv(data_file)
        elif data_file.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_file)
        elif data_file.suffix.lower() == '.json':
            df = pd.read_json(data_file)
        elif data_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # Determine text column
        if not text_column:
            if auto_mode and analysis['text_column_candidates']:
                text_column = analysis['text_column_candidates'][0]['name']
                self.console.print(f"[green]✓ Auto-selected text column: {text_column}[/green]")
            else:
                text_column = self._prompt_for_column(
                    df,
                    "text",
                    analysis['text_column_candidates']
                )

        # Determine annotation column
        if not annotation_column:
            if auto_mode and analysis['annotation_column_candidates']:
                annotation_column = analysis['annotation_column_candidates'][0]['name']
                self.console.print(f"[green]✓ Auto-selected annotation column: {annotation_column}[/green]")
            else:
                annotation_column = self._prompt_for_column(
                    df,
                    "annotation",
                    analysis['annotation_column_candidates']
                )

        # Configure bundle with llm-json strategy
        bundle = builder.prepare_data(
            strategy="llm-json",
            csv_file=data_file,
            text_column=text_column,
            annotation_column=annotation_column
        )

        return bundle

    def _prompt_for_column(
        self,
        df: pd.DataFrame,
        column_type: str,
        candidates: List[Dict]
    ) -> str:
        """Prompt user to select a column with intelligent suggestions."""
        self.console.print(f"\n[bold]Select {column_type} column:[/bold]")

        # Display all columns
        columns_table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=box.ROUNDED)
        columns_table.add_column("#", style="dim", width=3)
        columns_table.add_column("Column Name", style="cyan bold", width=30)
        columns_table.add_column("Sample Values", style="white", width=50)

        for idx, col in enumerate(df.columns, 1):
            samples = df[col].dropna().head(2).tolist()
            if samples:
                sample_str = ", ".join([str(s)[:40] + "..." if len(str(s)) > 40 else str(s) for s in samples])
            else:
                sample_str = "[empty]"

            # Mark recommended columns
            is_candidate = any(c['name'] == col for c in candidates)
            col_display = f"{col} {'✓ (recommended)' if is_candidate else ''}"

            columns_table.add_row(str(idx), col_display, sample_str)

        self.console.print(columns_table)

        # Get default from candidates
        default_col = candidates[0]['name'] if candidates else df.columns[0]
        default_idx = list(df.columns).index(default_col) + 1

        choice = Prompt.ask(
            f"Select {column_type} column number",
            default=str(default_idx)
        )

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(df.columns):
                return df.columns[idx]

        # If not a number, treat as column name
        if choice in df.columns:
            return choice

        # Fallback to default
        return default_col

    def _run_dataset_wizard(self, builder):
        """Run the complete dataset wizard for Training Arena."""
        # This calls the original dataset wizard from the CLI
        # We'll need to extract this logic or import it
        from llm_tool.cli.advanced_cli import AdvancedCLI

        # Create a temporary CLI instance to access the wizard
        # This is a bit hacky but maintains compatibility
        temp_cli = AdvancedCLI()
        temp_cli.console = self.console
        temp_cli.logger = self.logger
        temp_cli.detected_datasets = []  # Will be populated if needed

        return temp_cli._training_studio_dataset_wizard(builder)

    def _render_bundle_summary(self, bundle):
        """Display a summary of the prepared training bundle."""
        self.console.print("\n[bold cyan]Dataset Summary[/bold cyan]\n")

        summary_table = Table(show_header=True, header_style="bold magenta", border_style="green", box=box.ROUNDED)
        summary_table.add_column("Property", style="cyan bold", width=25)
        summary_table.add_column("Value", style="white", width=60)

        summary_table.add_row("📊 Dataset", str(bundle.primary_file.name) if bundle.primary_file else "—")
        summary_table.add_row("📝 Format", bundle.strategy)
        summary_table.add_row("📖 Text Column", bundle.text_column)
        summary_table.add_row("🏷️ Label Column", bundle.label_column)
        summary_table.add_row("📈 Total Samples", f"{len(bundle.df):,}" if hasattr(bundle, 'df') else "—")

        if bundle.metadata.get('confirmed_languages'):
            langs = ', '.join([l.upper() for l in bundle.metadata['confirmed_languages']])
            summary_table.add_row("🌍 Languages", langs)

        self.console.print(summary_table)

    def _confirm_and_execute(
        self,
        bundle,
        mode: str,
        preloaded_config: Optional[Dict] = None,
        auto_mode: bool = False
    ):
        """Configure parameters and execute training."""
        # This is a simplified version - in production you'd import the full logic
        from llm_tool.cli.advanced_cli import AdvancedCLI

        # Create temporary CLI instance for compatibility
        temp_cli = AdvancedCLI()
        temp_cli.console = self.console
        temp_cli.logger = self.logger
        temp_cli.current_session_id = self.current_session_id
        temp_cli.current_session_manager = self.current_session_manager

        # Call the actual training execution
        return temp_cli._training_studio_confirm_and_execute(
            bundle,
            mode,
            preloaded_config=preloaded_config
        )


def integrate_training_arena_in_annotator_factory(
    cli_instance,
    output_file: Path,
    text_column: str,
    session_id: str,
    session_dirs: Optional[Dict[str, Path]] = None
) -> Dict[str, Any]:
    """
    Integration point for Annotator Factory to use Training Arena workflow.

    This function is called from the Annotator Factory after annotations are complete
    and the user chooses to train a model.

    Parameters
    ----------
    cli_instance : AdvancedCLI
        The CLI instance with console and logger
    output_file : Path
        Path to the annotated CSV file from the annotation step
    text_column : str
        Name of the text column used in annotation
    session_id : str
        Session ID from the annotation phase

    Returns
    -------
    dict
        Training results
    """
    console = cli_instance.console
    logger = cli_instance.logger

    console.print("\n[bold cyan]🎓 Post-Annotation Training with Training Arena[/bold cyan]")
    console.print("[dim]Using the complete Training Arena workflow for LLM-JSON format data[/dim]\n")

    # Ask if user wants to train
    train_model = Confirm.ask(
        "[bold]Would you like to train a classifier model from these annotations?[/bold]",
        default=True
    )

    if not train_model:
        console.print("[yellow]Skipping training. Annotations are ready for manual use or export.[/yellow]")
        return {"status": "skipped"}

    # Load the annotated data to check format
    df = pd.read_csv(output_file)

    # Find annotation columns (the LLM output columns)
    annotation_cols = []
    for col in df.columns:
        if col.startswith('annotation_') or col.endswith('_annotation') or 'llm' in col.lower():
            annotation_cols.append(col)
        # Also check for JSON-like content
        elif col not in [text_column, 'id', 'text_id', 'n']:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
            if isinstance(sample, str) and (sample.startswith('{') or sample.startswith('[')):
                annotation_cols.append(col)

    if not annotation_cols:
        console.print("[red]No annotation columns found in LLM-JSON format. Cannot proceed with Training Arena.[/red]")
        return {"status": "failed", "error": "No LLM-JSON annotations found"}

    # Let user select which annotation column to use if multiple
    if len(annotation_cols) == 1:
        annotation_column = annotation_cols[0]
        console.print(f"[green]✓ Found annotation column: {annotation_column}[/green]")
    else:
        console.print(f"\n[yellow]Multiple annotation columns found:[/yellow]")
        for i, col in enumerate(annotation_cols, 1):
            console.print(f"  {i}. {col}")

        choice = IntPrompt.ask(
            "Select annotation column to use for training",
            default=1,
            choices=[str(i) for i in range(1, len(annotation_cols) + 1)]
        )
        annotation_column = annotation_cols[int(choice) - 1]

    # Initialize Training Arena workflow
    training_workflow = TrainingArenaWorkflow(console=console, logger=logger)

    # Run the training workflow with the annotated data
    console.print(f"\n[bold cyan]Starting Training Arena workflow...[/bold cyan]")
    console.print(f"[dim]Session: {session_id}[/dim]\n")

    # Get parent session directory if available
    parent_session_dir = None
    if session_dirs and 'session_root' in session_dirs:
        parent_session_dir = session_dirs['session_root']

    results = training_workflow.run_training_workflow(
        data_file=output_file,
        text_column=text_column,
        annotation_column=annotation_column,
        session_id=session_id,
        auto_mode=False,  # Let user configure everything
        parent_session_dir=parent_session_dir  # Pass parent directory for organized structure
    )

    return results