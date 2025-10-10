"""
Shared annotation workflow utilities bridging Annotator (Mode 1) and
Annotator Factory (Mode 2).

The goal is to centralise the interactive annotation workflow so the CLI
modes delegate to a single module while preserving their specific behaviour.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from ..utils.language_detector import LanguageDetector
from .data_detector import DataDetector


class AnnotationMode(Enum):
    """Workflow contexts supported by the annotation wizard."""

    ANNOTATOR = "annotator"
    FACTORY = "annotator_factory"


def create_session_directories(mode: AnnotationMode, session_id: str) -> Dict[str, Path]:
    """Create organised directory structure for an annotation session."""

    if mode == AnnotationMode.FACTORY:
        base_dir = Path("logs") / "annotator_factory" / session_id
        dirs: Dict[str, Path] = {
            "base": base_dir,
            "session_root": base_dir,
            "annotated_data": base_dir / "annotated_data",
            "metadata": base_dir / "metadata",
            "validation_exports": base_dir / "validation_exports",
            "doccano": base_dir / "validation_exports" / "doccano",
            "labelstudio": base_dir / "validation_exports" / "labelstudio",
            "training_metrics": base_dir / "training_metrics",
            "training_data": base_dir / "training_data",
        }
    else:
        base_dir = Path("logs") / "annotator" / session_id
        dirs = {
            "base": base_dir,
            "session_root": base_dir,
            "annotated_data": base_dir / "annotated_data",
            "metadata": base_dir / "metadata",
            "validation_exports": base_dir / "validation_exports",
            "doccano": base_dir / "validation_exports" / "doccano",
            "labelstudio": base_dir / "validation_exports" / "labelstudio",
        }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def run_annotator_workflow(cli, session_id: str = None, session_dirs: Optional[Dict[str, Path]] = None):
    """Smart guided annotation wizard with all options

    Parameters
    ----------
    session_id : str, optional
        Session identifier for organizing outputs. If None, a timestamp-based ID is generated.
    """
    import pandas as pd
    from datetime import datetime

    # Generate session_id if not provided (for backward compatibility)
    if session_id is None:
        session_id = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create session directories
    if session_dirs is None:
        session_dirs = cli._create_annotator_session_directories(session_id)

    cli.console.print("\n[bold cyan]🎯 Smart Annotate - Guided Wizard[/bold cyan]\n")

    # Step 1: Data Selection
    cli.console.print("[bold]Step 1/7: Data Selection[/bold]")

    if not cli.detected_datasets:
        cli.console.print("[yellow]No datasets auto-detected.[/yellow]")
        data_path = Path(cli._prompt_file_path("Dataset path"))
    else:
        cli.console.print(f"\n[bold cyan]📊 Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

        # Create table for datasets
        datasets_table = Table(border_style="cyan", show_header=True)
        datasets_table.add_column("#", style="bold yellow", width=4)
        datasets_table.add_column("Filename", style="white")
        datasets_table.add_column("Format", style="green", width=10)
        datasets_table.add_column("Size", style="magenta", width=10)
        datasets_table.add_column("Rows", style="cyan", width=10)
        datasets_table.add_column("Columns", style="blue", width=10)

        for i, ds in enumerate(cli.detected_datasets[:20], 1):
            # Format size
            if ds.size_mb < 0.1:
                size_str = f"{ds.size_mb * 1024:.1f} KB"
            else:
                size_str = f"{ds.size_mb:.1f} MB"

            # Format rows and columns
            rows_str = f"{ds.rows:,}" if ds.rows else "?"
            cols_str = str(len(ds.columns)) if ds.columns else "?"

            datasets_table.add_row(
                str(i),
                ds.path.name,
                ds.format.upper(),
                size_str,
                rows_str,
                cols_str
            )

        cli.console.print(datasets_table)
        cli.console.print()

        use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
        if use_detected:
            choice = cli._int_prompt_with_validation("Select dataset", 1, 1, len(cli.detected_datasets))
            data_path = cli.detected_datasets[choice - 1].path
        else:
            data_path = Path(cli._prompt_file_path("Dataset path"))

    # Detect format
    data_format = data_path.suffix[1:].lower()
    if data_format == 'xlsx':
        data_format = 'excel'

    cli.console.print(f"[green]✓ Selected: {data_path.name} ({data_format})[/green]")

    # Step 2: Text column selection with intelligent detection
    cli.console.print("\n[bold]Step 2/7: Text Column Selection[/bold]")

    # Detect text columns
    column_info = cli._detect_text_columns(data_path)

    if column_info['text_candidates']:
        cli.console.print("\n[dim]Detected text columns (sorted by confidence):[/dim]")

        # Create table for candidates
        col_table = Table(border_style="blue")
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white")
        col_table.add_column("Confidence", style="yellow")
        col_table.add_column("Avg Length", style="green")
        col_table.add_column("Sample", style="dim")

        for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
            # Color code confidence
            conf_color = {
                "high": "[green]High[/green]",
                "medium": "[yellow]Medium[/yellow]",
                "low": "[orange1]Low[/orange1]",
                "very_low": "[red]Very Low[/red]"
            }
            conf_display = conf_color.get(candidate['confidence'], candidate['confidence'])

            col_table.add_row(
                str(i),
                candidate['name'],
                conf_display,
                f"{candidate['avg_length']:.0f} chars",
                candidate['sample'][:50] + "..." if len(candidate['sample']) > 50 else candidate['sample']
            )

        cli.console.print(col_table)

        # Show all columns option
        cli.console.print(f"\n[dim]All columns ({len(column_info['all_columns'])}): {', '.join(column_info['all_columns'])}[/dim]")

        # Ask user to select
        default_col = column_info['text_candidates'][0]['name'] if column_info['text_candidates'] else "text"
        text_column = Prompt.ask(
            "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)",
            default=default_col
        )
    else:
        # No candidates detected, show all columns
        if column_info['all_columns']:
            cli.console.print(f"\n[yellow]Could not auto-detect text columns.[/yellow]")
            cli.console.print(f"[dim]Available columns: {', '.join(column_info['all_columns'])}[/dim]")
        text_column = Prompt.ask("Text column name", default="text")

    # Step 2b: ID Column Selection (MODERNIZED)
    # Load dataframe to detect ID candidates
    if data_format == 'csv':
        df_for_id = pd.read_csv(data_path, nrows=1000)
    elif data_format == 'json':
        df_for_id = pd.read_json(data_path, lines=False, nrows=1000)
    elif data_format == 'jsonl':
        df_for_id = pd.read_json(data_path, lines=True, nrows=1000)
    elif data_format == 'excel':
        df_for_id = pd.read_excel(data_path, nrows=1000)
    else:
        df_for_id = pd.read_csv(data_path, nrows=1000)  # Fallback

    # Use new unified ID selection function
    identifier_column = DataDetector.display_and_select_id_column(
        cli.console,
        df_for_id,
        text_column=text_column,
        step_label="Step 2b/7: Identifier Column Selection"
    )

    # Step 3: Model Selection
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 3:[/bold cyan] [bold white]Model Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose from local (Ollama) or cloud (OpenAI/Anthropic) models for annotation.[/dim]\n")

    selected_llm = cli._select_llm_interactive()
    provider = selected_llm.provider
    model_name = selected_llm.name

    # Get API key if needed
    api_key = None
    if selected_llm.requires_api_key:
        api_key = cli._get_or_prompt_api_key(provider, model_name)

    # Step 4: Prompt Configuration
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 4:[/bold cyan] [bold white]Prompt Configuration[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Select from existing prompts or create new annotation instructions.[/dim]")

    # Auto-detect prompts
    detected_prompts = cli._detect_prompts_in_folder()

    if detected_prompts:
        cli.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
        for i, p in enumerate(detected_prompts, 1):
            # Display ALL keys, not truncated
            keys_str = ', '.join(p['keys'])
            cli.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
            cli.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

        # Explain the options clearly
        cli.console.print("\n[bold]Prompt Selection Options:[/bold]")
        cli.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
        cli.console.print("           → Each text will be annotated with all prompts")
        cli.console.print("           → Useful when you want complete annotations from all perspectives")
        cli.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
        cli.console.print("           → Only selected prompts will be used")
        cli.console.print("           → Useful when testing or when you need only certain annotations")
        cli.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
        cli.console.print("           → Interactive guided prompt creation")
        cli.console.print("           → Optional AI assistance for definitions")
        cli.console.print("           → [bold green]Recommended for new research projects![/bold green]")
        cli.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
        cli.console.print("           → Use a prompt from another location")
        cli.console.print("           → Useful for testing new prompts or one-off annotations")

        prompt_choice = Prompt.ask(
            "\n[bold yellow]Prompt selection[/bold yellow]",
            choices=["all", "select", "wizard", "custom"],
            default="all"
        )

        selected_prompts = []
        if prompt_choice == "all":
            selected_prompts = detected_prompts
            cli.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "select":
            indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
            if indices.strip():  # Only process if not empty
                for idx_str in indices.split(','):
                    idx_str = idx_str.strip()
                    if idx_str:  # Skip empty strings
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(detected_prompts):
                                selected_prompts.append(detected_prompts[idx])
                        except ValueError:
                            cli.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
            if not selected_prompts:
                cli.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                selected_prompts = detected_prompts
            else:
                cli.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            # Custom path
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]
    else:
        cli.console.print("[yellow]No prompts found in prompts/ folder[/yellow]")

        # Offer wizard or custom path
        cli.console.print("\n[bold]Prompt Options:[/bold]")
        cli.console.print("  [cyan]wizard[/cyan] - 🧙‍♂️ Create prompt using Social Science Wizard (Recommended)")
        cli.console.print("  [cyan]custom[/cyan] - Provide path to existing prompt file")

        choice = Prompt.ask(
            "\n[bold yellow]Select option[/bold yellow]",
            choices=["wizard", "custom"],
            default="wizard"
        )

        if choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]


    # Language detection moved to training phase
    # Language columns will be detected and handled automatically after annotation
    lang_column = None
    available_columns = column_info.get('all_columns', []) if column_info else []
    if available_columns:
        # Silently detect potential language columns for metadata
        potential_lang_cols = [col for col in available_columns
                              if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

        # If language column exists, note it for later use but don't ask user
        if potential_lang_cols:
            lang_column = potential_lang_cols[0]  # Use first one if found
    # Multi-prompt prefix configuration (if needed)
    prompt_configs = []
    if len(selected_prompts) > 1:
        cli.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
        cli.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

        for i, prompt in enumerate(selected_prompts, 1):
            cli.console.print(f"\n[cyan]Prompt {i}: {prompt['name']}[/cyan]")
            cli.console.print(f"  Keys: {', '.join(prompt['keys'])}")

            add_prefix = Confirm.ask(f"Add prefix to keys for this prompt?", default=True)
            prefix = ""
            if add_prefix:
                default_prefix = prompt['name'].lower().replace(' ', '_')
                prefix = Prompt.ask("Prefix", default=default_prefix)
                cli.console.print(f"  [green]Keys will become: {', '.join([f'{prefix}_{k}' for k in prompt['keys'][:3]])}[/green]")

            prompt_configs.append({
                'prompt': prompt,
                'prefix': prefix
            })
    else:
        # Single prompt - no prefix needed
        prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings for optimal performance.[/dim]")

    # ============================================================
    # DATASET SCOPE
    # ============================================================
    cli.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
    cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

    # Get total rows if possible
    total_rows = None
    if column_info.get('df') is not None:
        # We have a sample, extrapolate
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    if total_rows:
        cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

    # Option 1: Annotate all or limited
    cli.console.print("[yellow]Option 1:[/yellow] Annotate ALL rows vs LIMIT to specific number")
    cli.console.print("  • [cyan]all[/cyan]   - Annotate the entire dataset")
    cli.console.print("           [dim]Use this for production annotations[/dim]")
    cli.console.print("  • [cyan]limit[/cyan] - Specify exact number of rows to annotate")
    cli.console.print("           [dim]Use this for testing or partial annotation[/dim]")

    scope_choice = Prompt.ask(
        "\nAnnotate entire dataset or limit rows?",
        choices=["all", "limit"],
        default="all"
    )

    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None

    if scope_choice == "limit":
        # Option 2: FIRST ask about representative sample calculation (before asking for number)
        if total_rows and total_rows > 1000:
            cli.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
            cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
            cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

            calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

            if calculate_sample:
                # Formula: n = (Z² × p × (1-p)) / E²
                # For 95% CI: Z=1.96, p=0.5 (max variance), E=0.05 (5% margin)
                import math
                z = 1.96
                p = 0.5
                e = 0.05
                n_infinite = (z**2 * p * (1-p)) / (e**2)
                n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                recommended_sample = int(math.ceil(n_adjusted))

                cli.console.print(f"\n[green]📈 Recommended sample size: {recommended_sample} rows[/green]")
                cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

        # THEN ask for specific number (with recommendation as default if calculated)
        default_limit = recommended_sample if recommended_sample else 100
        annotation_limit = cli._int_prompt_with_validation(
            f"How many rows to annotate?",
            default=default_limit,
            min_value=1,
            max_value=total_rows if total_rows else 1000000
        )

        # Check if user chose the recommended sample
        if recommended_sample and annotation_limit == recommended_sample:
            use_sample = True

        # Option 3: Random sampling
        cli.console.print("\n[yellow]Option 3:[/yellow] Sampling Strategy")
        cli.console.print("  Choose how to select the rows to annotate")
        cli.console.print("  • [cyan]head[/cyan]   - Take first N rows (faster, sequential)")
        cli.console.print("           [dim]Good for testing, preserves order[/dim]")
        cli.console.print("  • [cyan]random[/cyan] - Random sample of N rows (representative)")
        cli.console.print("           [dim]Better for statistical validity, unbiased[/dim]")

        sample_strategy = Prompt.ask(
            "\nSampling strategy",
            choices=["head", "random"],
            default="random" if use_sample else "head"
        )

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    cli.console.print("\n[bold cyan]⚙️  Parallel Processing[/bold cyan]")
    cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

    cli.console.print("[yellow]Parallel Workers:[/yellow]")
    cli.console.print("  Number of simultaneous annotation processes")
    cli.console.print("\n  [red]⚠️  IMPORTANT:[/red]")
    cli.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
    cli.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
    cli.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
    cli.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
    cli.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
    cli.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
    cli.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
    cli.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

    num_processes = cli._int_prompt_with_validation("Parallel workers", 1, 1, 16)

    # ============================================================
    # INCREMENTAL SAVE
    # ============================================================
    cli.console.print("\n[bold cyan]💾 Incremental Save[/bold cyan]")
    cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

    cli.console.print("[yellow]Enable incremental save?[/yellow]")
    cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
    cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
    cli.console.print("  • [red]No[/red]  - Save only at the end")
    cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

    save_incrementally = Confirm.ask("\n💿 Enable incremental save?", default=True)

    # Only ask for batch size if incremental save is enabled
    if save_incrementally:
        cli.console.print("\n[yellow]Batch Size:[/yellow]")
        cli.console.print("  Number of rows processed between each save")
        cli.console.print("  • [cyan]Smaller (1-10)[/cyan]   - Very frequent saves, maximum safety")
        cli.console.print("           [dim]Use for: Unstable systems, expensive APIs, testing[/dim]")
        cli.console.print("  • [cyan]Medium (10-50)[/cyan]   - Balanced safety and performance")
        cli.console.print("           [dim]Use for: Most production cases[/dim]")
        cli.console.print("  • [cyan]Larger (50-200)[/cyan]  - Less frequent saves, better performance")
        cli.console.print("           [dim]Use for: Stable systems, large datasets, local models[/dim]")

        batch_size = cli._int_prompt_with_validation("Batch size", 1, 1, 1000)
    else:
        batch_size = None  # Not used when incremental save is disabled

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    cli.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    supports_params = not is_o_series

    if not supports_params:
        cli.console.print(f"[yellow]⚠️  Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
        cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
        configure_params = False
    else:
        cli.console.print("[yellow]Configure model parameters?[/yellow]")
        cli.console.print("  Adjust how the model generates responses")
        cli.console.print("  [dim]• Default values work well for most cases[/dim]")
        cli.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
        configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

    # Default values
    temperature = 0.7
    max_tokens = 1000
    top_p = 1.0
    top_k = 40

    if configure_params:
        cli.console.print("\n[bold]Parameter Explanations:[/bold]\n")

        # Temperature
        cli.console.print("[cyan]🌡️  Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]📏 Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]🎯 Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]🔢 Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    # Step 1.6: Execute
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.6:[/bold cyan] [bold white]Review & Execute[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Review your configuration and start the annotation process.[/dim]")

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True)
    summary_table.add_column("Category", style="bold cyan", width=20)
    summary_table.add_column("Setting", style="yellow", width=25)
    summary_table.add_column("Value", style="white")

    # Data section
    summary_table.add_row("📁 Data", "Dataset", str(data_path.name))
    summary_table.add_row("", "Format", data_format.upper())
    summary_table.add_row("", "Text Column", text_column)
    if total_rows:
        summary_table.add_row("", "Total Rows", f"{total_rows:,}")
    if annotation_limit:
        summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
    else:
        summary_table.add_row("", "Rows to Annotate", "ALL")

    # Model section
    summary_table.add_row("🤖 Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("📝 Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    summary_table.add_row("⚙️  Processing", "Parallel Workers", str(num_processes))
    summary_table.add_row("", "Batch Size", str(batch_size))
    summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
    cli.console.print("[green]✓ Session parameters are automatically saved for:[/green]\n")

    cli.console.print("  [green]1. Resume Capability[/green]")
    cli.console.print("     • Continue this annotation if it stops or crashes")
    cli.console.print("     • Annotate additional rows later with same settings")
    cli.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

    cli.console.print("  [green]2. Scientific Reproducibility[/green]")
    cli.console.print("     • Document exact parameters for research papers")
    cli.console.print("     • Reproduce identical annotations in the future")
    cli.console.print("     • Track model version, prompts, and all settings\n")

    # Metadata is ALWAYS saved automatically for reproducibility
    save_metadata = True

    # ============================================================
    # VALIDATION TOOL EXPORT OPTION
    # ============================================================
    cli.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
    cli.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

    cli.console.print("[yellow]Available validation tools:[/yellow]")
    cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
    cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
    cli.console.print("  • Both are open-source and free\n")

    cli.console.print("[green]Why validate with external tools?[/green]")
    cli.console.print("  • Review and correct LLM annotations")
    cli.console.print("  • Calculate inter-annotator agreement")
    cli.console.print("  • Export validated data for metrics calculation\n")

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None

    # Step 1: Ask if user wants to export
    export_confirm = Confirm.ask(
        "[bold yellow]Export to validation tool?[/bold yellow]",
        default=False
    )

    if export_confirm:
        # Step 2: Ask which tool to export to
        tool_choice = Prompt.ask(
            "[bold yellow]Which validation tool?[/bold yellow]",
            choices=["doccano", "labelstudio"],
            default="doccano"
        )

        # Set the appropriate export flag
        if tool_choice == "doccano":
            export_to_doccano = True
        else:  # labelstudio
            export_to_labelstudio = True

        # Step 2b: If Label Studio, ask export method
        labelstudio_direct_export = False
        labelstudio_api_url = None
        labelstudio_api_key = None

        if export_to_labelstudio:
            cli.console.print("\n[yellow]Label Studio export method:[/yellow]")
            cli.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
            if cli.HAS_REQUESTS:
                cli.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                export_choices = ["jsonl", "direct"]
            else:
                cli.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                export_choices = ["jsonl"]

            export_method = Prompt.ask(
                "[bold yellow]Export method[/bold yellow]",
                choices=export_choices,
                default="jsonl"
            )

            if export_method == "direct":
                labelstudio_direct_export = True

                cli.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                labelstudio_api_url = Prompt.ask(
                    "Label Studio URL",
                    default="http://localhost:8080"
                )

                labelstudio_api_key = Prompt.ask(
                    "API Key (from Label Studio Account & Settings)"
                )

        # Step 3: Ask about LLM predictions inclusion
        cli.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
        cli.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
        cli.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
        cli.console.print("  • [cyan]both[/cyan] - Create two files: one with and one without predictions\n")

        prediction_mode = Prompt.ask(
            "[bold yellow]Prediction mode[/bold yellow]",
            choices=["with", "without", "both"],
            default="with"
        )

        # Step 4: Ask how many sentences to export
        cli.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
        cli.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
        cli.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
        cli.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

        sample_choice = Prompt.ask(
            "[bold yellow]Export sample[/bold yellow]",
            choices=["all", "representative", "number"],
            default="all"
        )

        if sample_choice == "all":
            export_sample_size = "all"
        elif sample_choice == "representative":
            export_sample_size = "representative"
        else:  # number
            export_sample_size = cli._int_prompt_with_validation(
                "Number of sentences to export",
                100,
                1,
                999999
            )

    # ============================================================
    # EXECUTE ANNOTATION
    # ============================================================

    # CRITICAL: Use new organized structure with dataset-specific subfolder
    # Structure: logs/annotator/{session_id}/annotated_data/{dataset_name}/
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create dataset-specific subdirectory (like {category} in Training Arena)
    dataset_name = data_path.stem
    dataset_subdir = session_dirs['annotated_data'] / dataset_name
    dataset_subdir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
    default_output_path = dataset_subdir / output_filename

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")
    cli.console.print()

    # Prepare prompts payload for pipeline
    prompts_payload = []
    for pc in prompt_configs:
        prompts_payload.append({
            'prompt': pc['prompt']['content'],
            'expected_keys': pc['prompt']['keys'],
            'prefix': pc['prefix']
        })

    # Determine annotation mode
    annotation_mode = 'api' if provider in {'openai', 'anthropic', 'google'} else 'local'

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': text_column,
        'text_columns': [text_column],
        'annotation_column': 'annotation',
        'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key if api_key else None,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,  # Use Rich progress instead
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': save_incrementally,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
        'lang_column': lang_column,  # From Step 4b: Language column for training metadata
    }

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        import json

        # Build comprehensive metadata
        metadata = {
            'annotation_session': {
                'timestamp': timestamp,
                'tool_version': 'LLMTool v1.0',
                'workflow': 'The Annotator - Smart Annotate'
            },
            'data_source': {
                'file_path': str(data_path),
                'file_name': data_path.name,
                'data_format': data_format,
                'text_column': text_column,
                'total_rows': annotation_limit if annotation_limit else 'all',
                'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
                'sample_seed': 42 if sample_strategy == 'random' else None
            },
            'model_configuration': {
                'provider': provider,
                'model_name': model_name,
                'annotation_mode': annotation_mode,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'top_k': top_k if provider in ['ollama', 'google'] else None
            },
            'prompts': [
                {
                    'name': pc['prompt']['name'],
                    'file_path': str(pc['prompt']['path']) if 'path' in pc['prompt'] else None,
                    'expected_keys': pc['prompt']['keys'],
                    'prefix': pc['prefix'],
                    'prompt_content': pc['prompt']['content']
                }
                for pc in prompt_configs
            ],
            'processing_configuration': {
                'parallel_workers': num_processes,
                'batch_size': batch_size,
                'incremental_save': save_incrementally,
                'identifier_column': 'annotation_id'
            },
            'output': {
                'output_path': str(default_output_path),
                'output_format': data_format
            },
            'export_preferences': {
                'export_to_doccano': export_to_doccano,
                'export_to_labelstudio': export_to_labelstudio,
                'export_sample_size': export_sample_size,
                'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
            },
            'training_workflow': {
                'enabled': False,  # Will be updated after training workflow
                'training_params_file': None,  # Will be added after training
                'note': 'Training parameters will be saved separately after annotation completes'
            }
        }

        # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
        # Use dataset-specific subdirectory for metadata too
        metadata_subdir = session_dirs['metadata'] / dataset_name
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        metadata_path = metadata_subdir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path}\n")

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id  # Pass session_id for organized logging
        )

        # Use RichProgressManager for elegant display
        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,  # Show JSON sample for every annotation
            compact_mode=False   # Full preview panels
        ) as progress_manager:
            # Wrap pipeline for enhanced JSON tracking
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            # Run pipeline
            state = enhanced_pipeline.run_pipeline(pipeline_config)

            # Check for errors
            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        # Display success message
        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
            cli.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

            try:
                import pandas as pd
                from llm_tool.utils.language_detector import LanguageDetector

                # Load annotated file
                df_annotated = pd.read_csv(output_file)

                # CRITICAL: Only detect languages for ANNOTATED rows
                # The output file may contain ALL original rows, but we only want to detect
                # languages for rows that were actually annotated
                original_row_count = len(df_annotated)

                # Try to identify annotated rows by checking for annotation columns
                # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                if annotation_cols:
                    # Filter to only rows that have annotations (non-null AND non-empty in annotation column)
                    annotation_col = annotation_cols[0]
                    df_annotated = df_annotated[(df_annotated[annotation_col].notna()) & (df_annotated[annotation_col] != '')].copy()
                    cli.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                else:
                    cli.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                elif text_column in df_annotated.columns:
                    # Get ALL texts (including NaN) to maintain index alignment
                    all_texts = df_annotated[text_column].tolist()

                    # Count non-empty texts for display
                    non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                    if non_empty_texts > 0:
                        detector = LanguageDetector()
                        detected_languages = []

                        # Progress indicator
                        from tqdm import tqdm
                        cli.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not cli.HAS_RICH):
                            # Handle NaN and empty texts
                            if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                detected_languages.append('unknown')
                            else:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected and detected.get('language'):
                                        detected_languages.append(detected['language'])
                                    else:
                                        detected_languages.append('unknown')
                                except Exception as e:
                                    cli.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages.append('unknown')

                        # Add language column to the filtered dataframe
                        df_annotated['lang'] = detected_languages

                        # Reload the FULL original file and update only the annotated rows
                        df_full = pd.read_csv(output_file)

                        # Initialize lang column if it doesn't exist
                        if 'lang' not in df_full.columns:
                            df_full['lang'] = 'unknown'

                        # Update language for annotated rows only
                        # Match by index of df_annotated within df_full
                        df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                        # Save updated full file with language column
                        df_full.to_csv(output_file, index=False)

                        # Show distribution
                        lang_counts = {}
                        for lang in detected_languages:
                            if lang != 'unknown':
                                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                        if lang_counts:
                            total = sum(lang_counts.values())
                            cli.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                            lang_table.add_column("Language", style="cyan", width=12)
                            lang_table.add_column("Count", style="yellow", justify="right", width=12)
                            lang_table.add_column("Percentage", style="green", justify="right", width=12)

                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100) if total > 0 else 0
                                lang_table.add_row(
                                    lang.upper(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )

                            cli.console.print(lang_table)
                            cli.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                        else:
                            cli.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")


        # Export to Doccano JSONL if requested
        if export_to_doccano:
            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs,
                data_path=data_path,
                timestamp=timestamp,
                sample_size=export_sample_size,
                session_dirs=session_dirs
            )

        # Export to Label Studio if requested
        if export_to_labelstudio:
            if labelstudio_direct_export:
                # Direct export to Label Studio via API
                cli._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    api_url=labelstudio_api_url,
                    api_key=labelstudio_api_key
                )
            else:
                # Export to JSONL file
                cli._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    session_dirs=session_dirs
                )

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Annotation execution failed")
        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

def run_factory_workflow(cli, session_id: str = None, session_dirs: Optional[Dict[str, Path]] = None):
    """Execute complete annotation → training workflow

    Parameters
    ----------
    session_id : str, optional
        Session identifier for organizing outputs. If None, a timestamp-based ID is generated.
    """
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    # Generate session_id if not provided (for backward compatibility)
    if session_id is None:
        session_id = f"factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create session directories
    if session_dirs is None:
        session_dirs = cli._create_annotator_factory_session_directories(session_id)

    # Display Annotator Factory STEP 1 banner
    from llm_tool.cli.banners import BANNERS, STEP_NUMBERS, STEP_LABEL
    from rich.align import Align

    cli.console.print()

    # Display "STEP" label in ASCII art
    for line in STEP_LABEL.split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    # Display "1/3" in ASCII art
    for line in STEP_NUMBERS['1/3'].split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    cli.console.print()

    # Display main banner (centered)
    for line in BANNERS['llm_annotator']['ascii'].split('\n'):
        cli.console.print(Align.center(f"[bold {BANNERS['llm_annotator']['color']}]{line}[/bold {BANNERS['llm_annotator']['color']}]"))

    # Display tagline (centered)
    cli.console.print(Align.center(f"[{BANNERS['llm_annotator']['color']}]{BANNERS['llm_annotator']['tagline']}[/{BANNERS['llm_annotator']['color']}]"))
    cli.console.print()

    # Step 1.1: Data Source Selection
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.1:[/bold cyan] [bold white]Data Source Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose between file-based datasets or SQL database sources.[/dim]\n")

    # Ask user to choose between files and SQL database
    cli.console.print("[yellow]Available data sources:[/yellow]")
    cli.console.print("  1. 📁 Files (CSV/Excel/JSON/etc.) - Auto-detected or manual")
    cli.console.print("  2. 🗄️  SQL Database (PostgreSQL/MySQL/SQLite/SQL Server)\n")

    data_source_choice = Prompt.ask(
        "Data source",
        choices=["1", "2"],
        default="1"
    )

    use_sql_database = (data_source_choice == "2")

    if use_sql_database:
        # SQL DATABASE WORKFLOW
        cli.console.print("\n[bold cyan]🗄️  SQL Database (Training Sample)[/bold cyan]\n")
        cli.console.print("[yellow]Note: For training, you'll select a representative sample from your database[/yellow]\n")

        # Database type selection
        db_choices = ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server"]
        db_table = Table(title="Database Types", border_style="cyan")
        db_table.add_column("#", style="cyan", width=6)
        db_table.add_column("Database Type", style="white")
        for i, choice in enumerate(db_choices, 1):
            db_table.add_row(str(i), choice)
        cli.console.print(db_table)

        db_choice = cli._int_prompt_with_validation("Select database type", 1, 1, len(db_choices))
        db_type_name = db_choices[db_choice - 1]

        # Connection details
        if db_type_name == "SQLite":
            db_file = Prompt.ask("SQLite database file path")
            connection_string = f"sqlite:///{db_file}"
        else:
            host = Prompt.ask("Database host", default="localhost")
            default_ports = {"PostgreSQL": "5432", "MySQL": "3306", "Microsoft SQL Server": "1433"}
            port = Prompt.ask("Port", default=default_ports.get(db_type_name, "5432"))
            username = Prompt.ask("Username", default="postgres" if db_type_name == "PostgreSQL" else "root")
            password = Prompt.ask("Password", password=True)
            database = Prompt.ask("Database name")

            if db_type_name == "PostgreSQL":
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            elif db_type_name == "MySQL":
                connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            else:
                connection_string = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

        # Test connection
        cli.console.print("\nTesting connection...")
        try:
            from sqlalchemy import create_engine, inspect, text
            import pandas as pd
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            cli.console.print("[green]✓ Connected successfully![/green]\n")
        except Exception as e:
            cli.console.print(f"[red]✗ Connection failed: {str(e)}[/red]")
            input("\nPress Enter to continue...")
            return

        # Table selection
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            cli.console.print("[red]No tables found[/red]")
            input("\nPress Enter to continue...")
            return

        # Get row counts
        table_info = []
        for table in tables:
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    table_info.append((table, result.scalar()))
            except:
                table_info.append((table, None))

        tables_table = Table(title="Available Tables", border_style="cyan")
        tables_table.add_column("#", style="cyan", width=3)
        tables_table.add_column("Table Name", style="white")
        tables_table.add_column("Rows", style="green", justify="right")
        for i, (table, rows) in enumerate(table_info, 1):
            tables_table.add_row(str(i), table, f"{rows:,}" if rows else "?")
        cli.console.print(tables_table)

        table_choice = cli._int_prompt_with_validation("Select table", 1, 1, len(table_info))
        selected_table, total_rows = table_info[table_choice - 1]
        cli.console.print(f"\n[green]✓ Selected: {selected_table} ({total_rows:,} rows)[/green]\n")

        # Load ALL data to temporary CSV (will use SAME workflow as files)
        from datetime import datetime
        import pandas as pd

        df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)

        # Save to CSV in data/annotations
        annotations_dir = cli.settings.paths.data_dir / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_path = annotations_dir / f"quickstart_sql_{selected_table}_{timestamp}.csv"
        df.to_csv(data_path, index=False)
        data_format = 'csv'

        cli.console.print(f"[green]✓ Loaded {len(df):,} rows from {selected_table}[/green]")
        cli.console.print(f"[dim]Saved to: {data_path}[/dim]")

    else:
        # FILE-BASED WORKFLOW (original code)
        if not cli.detected_datasets:
            cli.console.print("[yellow]No datasets auto-detected.[/yellow]")
            data_path = Path(cli._prompt_file_path("Dataset path"))
        else:
            cli.console.print(f"\n[bold cyan]📊 Found {len(cli.detected_datasets)} dataset(s):[/bold cyan]\n")

            # Create table for datasets
            datasets_table = Table(border_style="cyan", show_header=True)
            datasets_table.add_column("#", style="bold yellow", width=4)
            datasets_table.add_column("Filename", style="white")
            datasets_table.add_column("Format", style="green", width=10)
            datasets_table.add_column("Size", style="magenta", width=10)
            datasets_table.add_column("Rows", style="cyan", width=10)
            datasets_table.add_column("Columns", style="blue", width=10)

            for i, ds in enumerate(cli.detected_datasets[:20], 1):
                # Format size
                if ds.size_mb < 0.1:
                    size_str = f"{ds.size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{ds.size_mb:.1f} MB"

                # Format rows and columns
                rows_str = f"{ds.rows:,}" if ds.rows else "?"
                cols_str = str(len(ds.columns)) if ds.columns else "?"

                datasets_table.add_row(
                    str(i),
                    ds.path.name,
                    ds.format.upper(),
                    size_str,
                    rows_str,
                    cols_str
                )

            cli.console.print(datasets_table)
            cli.console.print()

            use_detected = Confirm.ask("[bold yellow]Use detected dataset?[/bold yellow]", default=True)
            if use_detected:
                choice = cli._int_prompt_with_validation("Select dataset", 1, 1, len(cli.detected_datasets))
                data_path = cli.detected_datasets[choice - 1].path
            else:
                data_path = Path(cli._prompt_file_path("Dataset path"))

        # Detect format
        data_format = data_path.suffix[1:].lower()
        if data_format == 'xlsx':
            data_format = 'excel'

        cli.console.print(f"[green]✓ Selected: {data_path.name} ({data_format})[/green]")

    # Step 1.2: Text Column Selection (MODERNIZED - Same format as quick start)
    cli.console.print("\n[bold]Step 1.2/4: Text Column Selection[/bold]\n")

    # Detect text columns using the advanced detection system
    column_info = cli._detect_text_columns(data_path)
    import pandas as pd
    df_sample = pd.read_csv(data_path, nrows=100) if data_path.suffix == '.csv' else pd.read_excel(data_path, nrows=100)

    if column_info['text_candidates']:
        cli.console.print("[dim]Detected text columns (sorted by confidence):[/dim]")

        # Create table for text candidates ONLY
        col_table = Table(border_style="blue")
        col_table.add_column("#", style="cyan", width=3)
        col_table.add_column("Column", style="white")
        col_table.add_column("Confidence", style="yellow")
        col_table.add_column("Avg Length", style="green")
        col_table.add_column("Sample", style="dim")

        for i, candidate in enumerate(column_info['text_candidates'][:10], 1):
            # Color code confidence
            conf_color = {
                "high": "[green]High[/green]",
                "medium": "[yellow]Medium[/yellow]",
                "low": "[orange1]Low[/orange1]",
                "very_low": "[red]Very Low[/red]"
            }
            conf_display = conf_color.get(candidate['confidence'], candidate['confidence'])

            col_table.add_row(
                str(i),
                candidate['name'],
                conf_display,
                f"{candidate['avg_length']:.0f} chars",
                candidate['sample'][:50] + "..." if len(candidate['sample']) > 50 else candidate['sample']
            )

        cli.console.print(col_table)

        # Show all columns list
        cli.console.print(f"\n[dim]All columns ({len(column_info['all_columns'])}): {', '.join(column_info['all_columns'])}[/dim]")

        # Ask user to select
        default_col = column_info['text_candidates'][0]['name'] if column_info['text_candidates'] else "text"
        text_column = Prompt.ask(
            "\n[bold yellow]Enter column name[/bold yellow] (or choose from above)",
            default=default_col
        )
    else:
        # No candidates detected, show all columns
        if column_info['all_columns']:
            cli.console.print(f"\n[yellow]Could not auto-detect text columns.[/yellow]")
            cli.console.print(f"[dim]Available columns: {', '.join(column_info['all_columns'])}[/dim]")
        text_column = Prompt.ask("Text column name", default="text")

    # Step 1.2b: ID Column Selection (MODERNIZED with new system)
    identifier_column = DataDetector.display_and_select_id_column(
        cli.console,
        df_sample,
        text_column=text_column,
        step_label="Step 1.2b/4: Identifier Column Selection"
    )

    # Store column info for later use
    column_info['df'] = df_sample

    # Step 1.3: Model Selection
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.3:[/bold cyan] [bold white]Model Selection[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Choose from local (Ollama) or cloud (OpenAI/Anthropic) models for annotation.[/dim]\n")

    selected_llm = cli._select_llm_interactive()
    provider = selected_llm.provider
    model_name = selected_llm.name

    # Get API key if needed
    api_key = None
    if selected_llm.requires_api_key:
        api_key = cli._get_or_prompt_api_key(provider, model_name)

    # Step 1.4: Prompt Configuration
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.4:[/bold cyan] [bold white]Prompt Configuration[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Select from existing prompts or create new annotation instructions.[/dim]")

    # Auto-detect prompts
    detected_prompts = cli._detect_prompts_in_folder()

    if detected_prompts:
        cli.console.print(f"\n[green]✓ Found {len(detected_prompts)} prompts in prompts/ folder:[/green]")
        for i, p in enumerate(detected_prompts, 1):
            # Display ALL keys, not truncated
            keys_str = ', '.join(p['keys'])
            cli.console.print(f"  {i}. [cyan]{p['name']}[/cyan]")
            cli.console.print(f"     Keys ({len(p['keys'])}): {keys_str}")

        # Explain the options clearly
        cli.console.print("\n[bold]Prompt Selection Options:[/bold]")
        cli.console.print("  [cyan]all[/cyan]     - Use ALL detected prompts (multi-prompt mode)")
        cli.console.print("           → Each text will be annotated with all prompts")
        cli.console.print("           → Useful when you want complete annotations from all perspectives")
        cli.console.print("\n  [cyan]select[/cyan]  - Choose SPECIFIC prompts by number (e.g., 1,3,5)")
        cli.console.print("           → Only selected prompts will be used")
        cli.console.print("           → Useful when testing or when you need only certain annotations")
        cli.console.print("\n  [cyan]wizard[/cyan]  - 🧙‍♂️ Create NEW prompt using Social Science Wizard")
        cli.console.print("           → Interactive guided prompt creation")
        cli.console.print("           → Optional AI assistance for definitions")
        cli.console.print("           → [bold green]Recommended for new research projects![/bold green]")
        cli.console.print("\n  [cyan]custom[/cyan]  - Provide path to a prompt file NOT in prompts/ folder")
        cli.console.print("           → Use a prompt from another location")
        cli.console.print("           → Useful for testing new prompts or one-off annotations")

        prompt_choice = Prompt.ask(
            "\n[bold yellow]Prompt selection[/bold yellow]",
            choices=["all", "select", "wizard", "custom"],
            default="all"
        )

        selected_prompts = []
        if prompt_choice == "all":
            selected_prompts = detected_prompts
            cli.console.print(f"[green]✓ Using all {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "select":
            indices = Prompt.ask("Enter prompt numbers (comma-separated, e.g., 1,3,5)")
            if indices.strip():  # Only process if not empty
                for idx_str in indices.split(','):
                    idx_str = idx_str.strip()
                    if idx_str:  # Skip empty strings
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(detected_prompts):
                                selected_prompts.append(detected_prompts[idx])
                        except ValueError:
                            cli.console.print(f"[yellow]⚠️  Skipping invalid number: '{idx_str}'[/yellow]")
            if not selected_prompts:
                cli.console.print("[yellow]No valid prompts selected. Using all prompts.[/yellow]")
                selected_prompts = detected_prompts
            else:
                cli.console.print(f"[green]✓ Selected {len(selected_prompts)} prompts[/green]")
        elif prompt_choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            # Custom path
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]
    else:
        cli.console.print("[yellow]No prompts found in prompts/ folder[/yellow]")

        # Offer wizard or custom path
        cli.console.print("\n[bold]Prompt Options:[/bold]")
        cli.console.print("  [cyan]wizard[/cyan] - 🧙‍♂️ Create prompt using Social Science Wizard (Recommended)")
        cli.console.print("  [cyan]custom[/cyan] - Provide path to existing prompt file")

        choice = Prompt.ask(
            "\n[bold yellow]Select option[/bold yellow]",
            choices=["wizard", "custom"],
            default="wizard"
        )

        if choice == "wizard":
            # Launch Social Science Wizard
            wizard_prompt = cli._run_social_science_wizard()
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(wizard_prompt)
            selected_prompts = [{
                'path': None,  # Wizard-generated, not from file
                'name': 'wizard_generated',
                'keys': keys,
                'content': wizard_prompt
            }]
            cli.console.print(f"[green]✓ Using wizard-generated prompt with {len(keys)} keys[/green]")
        else:
            custom_path = Path(cli._prompt_file_path("Prompt file path (.txt)"))
            content = custom_path.read_text(encoding='utf-8')
            from ..annotators.json_cleaner import extract_expected_keys
            keys = extract_expected_keys(content)
            selected_prompts = [{
                'path': custom_path,
                'name': custom_path.stem,
                'keys': keys,
                'content': content
            }]


    # Language detection moved to training phase
    # Language columns will be detected and handled automatically after annotation
    lang_column = None
    available_columns = column_info.get('all_columns', []) if column_info else []
    if available_columns:
        # Silently detect potential language columns for metadata
        potential_lang_cols = [col for col in available_columns
                              if col.lower() in ['lang', 'language', 'langue', 'lng', 'iso_lang']]

        # If language column exists, note it for later use but don't ask user
        if potential_lang_cols:
            lang_column = potential_lang_cols[0]  # Use first one if found
    # Multi-prompt prefix configuration (if needed)
    prompt_configs = []
    if len(selected_prompts) > 1:
        cli.console.print("\n[bold]Multi-Prompt Mode:[/bold] Configure key prefixes")
        cli.console.print("[dim]Prefixes help identify which prompt generated which keys[/dim]\n")

        for i, prompt in enumerate(selected_prompts, 1):
            cli.console.print(f"\n[cyan]Prompt {i}: {prompt['name']}[/cyan]")
            cli.console.print(f"  Keys: {', '.join(prompt['keys'])}")

            add_prefix = Confirm.ask(f"Add prefix to keys for this prompt?", default=True)
            prefix = ""
            if add_prefix:
                default_prefix = prompt['name'].lower().replace(' ', '_')
                prefix = Prompt.ask("Prefix", default=default_prefix)
                cli.console.print(f"  [green]Keys will become: {', '.join([f'{prefix}_{k}' for k in prompt['keys'][:3]])}[/green]")

            prompt_configs.append({
                'prompt': prompt,
                'prefix': prefix
            })
    else:
        # Single prompt - no prefix needed
        prompt_configs = [{'prompt': selected_prompts[0], 'prefix': ''}]

    # Step 1.5: Advanced Options
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.5:[/bold cyan] [bold white]Advanced Options[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Configure processing settings for optimal performance.[/dim]")

    # ============================================================
    # DATASET SCOPE
    # ============================================================
    cli.console.print("\n[bold cyan]📊 Dataset Scope[/bold cyan]")
    cli.console.print("[dim]Determine how many rows to annotate from your dataset[/dim]\n")

    # Get total rows if possible
    total_rows = None
    if column_info.get('df') is not None:
        # We have a sample, extrapolate
        total_rows = len(pd.read_csv(data_path)) if data_format == 'csv' else None

    if total_rows:
        cli.console.print(f"[green]✓ Dataset contains {total_rows:,} rows[/green]\n")

    # Option 1: Annotate all or limited
    cli.console.print("[yellow]Option 1:[/yellow] Annotate ALL rows vs LIMIT to specific number")
    cli.console.print("  • [cyan]all[/cyan]   - Annotate the entire dataset")
    cli.console.print("           [dim]Use this for production annotations[/dim]")
    cli.console.print("  • [cyan]limit[/cyan] - Specify exact number of rows to annotate")
    cli.console.print("           [dim]Use this for testing or partial annotation[/dim]")

    scope_choice = Prompt.ask(
        "\nAnnotate entire dataset or limit rows?",
        choices=["all", "limit"],
        default="all"
    )

    annotation_limit = None
    use_sample = False
    sample_strategy = "head"
    recommended_sample = None

    if scope_choice == "limit":
        # Option 2: FIRST ask about representative sample calculation (before asking for number)
        if total_rows and total_rows > 1000:
            cli.console.print("\n[yellow]Option 2:[/yellow] Representative Sample Calculation")
            cli.console.print("  Calculate statistically representative sample size (95% confidence interval)")
            cli.console.print("  [dim]This helps determine the minimum sample needed for statistical validity[/dim]")

            calculate_sample = Confirm.ask("Calculate representative sample size?", default=True)

            if calculate_sample:
                # Formula: n = (Z² × p × (1-p)) / E²
                # For 95% CI: Z=1.96, p=0.5 (max variance), E=0.05 (5% margin)
                import math
                z = 1.96
                p = 0.5
                e = 0.05
                n_infinite = (z**2 * p * (1-p)) / (e**2)
                n_adjusted = n_infinite / (1 + ((n_infinite - 1) / total_rows))
                recommended_sample = int(math.ceil(n_adjusted))

                cli.console.print(f"\n[green]📈 Recommended sample size: {recommended_sample} rows[/green]")
                cli.console.print(f"[dim]   (95% confidence level, 5% margin of error)[/dim]")
                cli.console.print(f"[dim]   Population: {total_rows:,} rows[/dim]\n")

        # THEN ask for specific number (with recommendation as default if calculated)
        default_limit = recommended_sample if recommended_sample else 100
        annotation_limit = cli._int_prompt_with_validation(
            f"How many rows to annotate?",
            default=default_limit,
            min_value=1,
            max_value=total_rows if total_rows else 1000000
        )

        # Check if user chose the recommended sample
        if recommended_sample and annotation_limit == recommended_sample:
            use_sample = True

        # Option 3: Random sampling
        cli.console.print("\n[yellow]Option 3:[/yellow] Sampling Strategy")
        cli.console.print("  Choose how to select the rows to annotate")
        cli.console.print("  • [cyan]head[/cyan]   - Take first N rows (faster, sequential)")
        cli.console.print("           [dim]Good for testing, preserves order[/dim]")
        cli.console.print("  • [cyan]random[/cyan] - Random sample of N rows (representative)")
        cli.console.print("           [dim]Better for statistical validity, unbiased[/dim]")

        sample_strategy = Prompt.ask(
            "\nSampling strategy",
            choices=["head", "random"],
            default="random" if use_sample else "head"
        )

    # ============================================================
    # PARALLEL PROCESSING
    # ============================================================
    cli.console.print("\n[bold cyan]⚙️  Parallel Processing[/bold cyan]")
    cli.console.print("[dim]Configure how many processes run simultaneously[/dim]\n")

    cli.console.print("[yellow]Parallel Workers:[/yellow]")
    cli.console.print("  Number of simultaneous annotation processes")
    cli.console.print("\n  [red]⚠️  IMPORTANT:[/red]")
    cli.console.print("  [dim]Most local machines can only handle 1 worker for LLM inference[/dim]")
    cli.console.print("  [dim]Parallel processing is mainly useful for API models[/dim]")
    cli.console.print("\n  • [cyan]1 worker[/cyan]  - Sequential processing")
    cli.console.print("           [dim]Recommended for: Local models (Ollama), first time users, debugging[/dim]")
    cli.console.print("  • [cyan]2-4 workers[/cyan] - Moderate parallelism")
    cli.console.print("           [dim]Recommended for: API models (OpenAI, Claude) - avoid rate limits[/dim]")
    cli.console.print("  • [cyan]4-8 workers[/cyan] - High parallelism")
    cli.console.print("           [dim]Recommended for: API models only - requires high rate limits[/dim]")

    num_processes = cli._int_prompt_with_validation("Parallel workers", 1, 1, 16)

    # ============================================================
    # INCREMENTAL SAVE
    # ============================================================
    cli.console.print("\n[bold cyan]💾 Incremental Save[/bold cyan]")
    cli.console.print("[dim]Configure how often results are saved during annotation[/dim]\n")

    cli.console.print("[yellow]Enable incremental save?[/yellow]")
    cli.console.print("  • [green]Yes[/green] - Save progress regularly during annotation (recommended)")
    cli.console.print("           [dim]Protects against crashes, allows resuming, safer for long runs[/dim]")
    cli.console.print("  • [red]No[/red]  - Save only at the end")
    cli.console.print("           [dim]Faster but risky - you lose everything if process crashes[/dim]")

    save_incrementally = Confirm.ask("\n💿 Enable incremental save?", default=True)

    # Only ask for batch size if incremental save is enabled
    if save_incrementally:
        cli.console.print("\n[yellow]Batch Size:[/yellow]")
        cli.console.print("  Number of rows processed between each save")
        cli.console.print("  • [cyan]Smaller (1-10)[/cyan]   - Very frequent saves, maximum safety")
        cli.console.print("           [dim]Use for: Unstable systems, expensive APIs, testing[/dim]")
        cli.console.print("  • [cyan]Medium (10-50)[/cyan]   - Balanced safety and performance")
        cli.console.print("           [dim]Use for: Most production cases[/dim]")
        cli.console.print("  • [cyan]Larger (50-200)[/cyan]  - Less frequent saves, better performance")
        cli.console.print("           [dim]Use for: Stable systems, large datasets, local models[/dim]")

        batch_size = cli._int_prompt_with_validation("Batch size", 1, 1, 1000)
    else:
        batch_size = None  # Not used when incremental save is disabled

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    cli.console.print("\n[bold cyan]🎛️  Model Parameters[/bold cyan]")
    cli.console.print("[dim]Configure advanced model generation parameters[/dim]\n")

    # Check if model supports parameter tuning
    model_name_lower = model_name.lower()
    is_o_series = any(x in model_name_lower for x in ['o1', 'o3', 'o4'])
    supports_params = not is_o_series

    if not supports_params:
        cli.console.print(f"[yellow]⚠️  Model '{model_name}' uses fixed parameters (reasoning model)[/yellow]")
        cli.console.print("[dim]   Temperature and top_p are automatically set to 1.0[/dim]")
        configure_params = False
    else:
        cli.console.print("[yellow]Configure model parameters?[/yellow]")
        cli.console.print("  Adjust how the model generates responses")
        cli.console.print("  [dim]• Default values work well for most cases[/dim]")
        cli.console.print("  [dim]• Advanced users can fine-tune for specific needs[/dim]")
        configure_params = Confirm.ask("\nConfigure model parameters?", default=False)

    # Default values
    temperature = 0.7
    max_tokens = 1000
    top_p = 1.0
    top_k = 40

    if configure_params:
        cli.console.print("\n[bold]Parameter Explanations:[/bold]\n")

        # Temperature
        cli.console.print("[cyan]🌡️  Temperature (0.0 - 2.0):[/cyan]")
        cli.console.print("  Controls randomness in responses")
        cli.console.print("  • [green]Low (0.0-0.3)[/green]  - Deterministic, focused, consistent")
        cli.console.print("           [dim]Use for: Structured tasks, factual extraction, classification[/dim]")
        cli.console.print("  • [yellow]Medium (0.4-0.9)[/yellow] - Balanced creativity and consistency")
        cli.console.print("           [dim]Use for: General annotation, most use cases[/dim]")
        cli.console.print("  • [red]High (1.0-2.0)[/red]  - Creative, varied, unpredictable")
        cli.console.print("           [dim]Use for: Brainstorming, diverse perspectives[/dim]")
        temperature = FloatPrompt.ask("Temperature", default=0.7)

        # Max tokens
        cli.console.print("\n[cyan]📏 Max Tokens:[/cyan]")
        cli.console.print("  Maximum length of the response")
        cli.console.print("  • [green]Short (100-500)[/green]   - Brief responses, simple annotations")
        cli.console.print("  • [yellow]Medium (500-2000)[/yellow]  - Standard responses, detailed annotations")
        cli.console.print("  • [red]Long (2000+)[/red]     - Extensive responses, complex reasoning")
        cli.console.print("  [dim]Note: More tokens = higher API costs[/dim]")
        max_tokens = cli._int_prompt_with_validation("Max tokens", 1000, 50, 8000)

        # Top_p (nucleus sampling)
        cli.console.print("\n[cyan]🎯 Top P (0.0 - 1.0):[/cyan]")
        cli.console.print("  Nucleus sampling - alternative to temperature")
        cli.console.print("  • [green]Low (0.1-0.5)[/green]  - Focused on most likely tokens")
        cli.console.print("           [dim]More deterministic, safer outputs[/dim]")
        cli.console.print("  • [yellow]High (0.9-1.0)[/yellow] - Consider broader token range")
        cli.console.print("           [dim]More creative, diverse outputs[/dim]")
        cli.console.print("  [dim]Tip: Use either temperature OR top_p, not both aggressively[/dim]")
        top_p = FloatPrompt.ask("Top P", default=1.0)

        # Top_k (only for some models)
        if provider in ['ollama', 'google']:
            cli.console.print("\n[cyan]🔢 Top K:[/cyan]")
            cli.console.print("  Limits vocabulary to K most likely next tokens")
            cli.console.print("  • [green]Small (1-10)[/green]   - Very focused, repetitive")
            cli.console.print("  • [yellow]Medium (20-50)[/yellow]  - Balanced diversity")
            cli.console.print("  • [red]Large (50+)[/red]    - Maximum diversity")
            top_k = cli._int_prompt_with_validation("Top K", 40, 1, 100)

    # Step 1.6: Execute
    cli.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[bold cyan]  STEP 1.6:[/bold cyan] [bold white]Review & Execute[/bold white]")
    cli.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    cli.console.print("[dim]Review your configuration and start the annotation process.[/dim]")

    # Display comprehensive summary
    summary_table = Table(title="Configuration Summary", border_style="cyan", show_header=True)
    summary_table.add_column("Category", style="bold cyan", width=20)
    summary_table.add_column("Setting", style="yellow", width=25)
    summary_table.add_column("Value", style="white")

    # Data section
    summary_table.add_row("📁 Data", "Dataset", str(data_path.name))
    summary_table.add_row("", "Format", data_format.upper())
    summary_table.add_row("", "Text Column", text_column)
    if total_rows:
        summary_table.add_row("", "Total Rows", f"{total_rows:,}")
    if annotation_limit:
        summary_table.add_row("", "Rows to Annotate", f"{annotation_limit:,} ({sample_strategy})")
    else:
        summary_table.add_row("", "Rows to Annotate", "ALL")

    # Model section
    summary_table.add_row("🤖 Model", "Provider/Model", f"{provider}/{model_name}")
    summary_table.add_row("", "Temperature", f"{temperature}")
    summary_table.add_row("", "Max Tokens", f"{max_tokens}")
    if configure_params:
        summary_table.add_row("", "Top P", f"{top_p}")
        if provider in ['ollama', 'google']:
            summary_table.add_row("", "Top K", f"{top_k}")

    # Prompts section
    summary_table.add_row("📝 Prompts", "Count", f"{len(prompt_configs)}")
    for i, pc in enumerate(prompt_configs, 1):
        prefix_info = f" (prefix: {pc['prefix']}_)" if pc['prefix'] else " (no prefix)"
        summary_table.add_row("", f"  Prompt {i}", f"{pc['prompt']['name']}{prefix_info}")

    # Processing section
    summary_table.add_row("⚙️  Processing", "Parallel Workers", str(num_processes))
    summary_table.add_row("", "Batch Size", str(batch_size))
    summary_table.add_row("", "Incremental Save", "Yes" if save_incrementally else "No")

    cli.console.print("\n")
    cli.console.print(summary_table)

    if not Confirm.ask("\n[bold yellow]Start annotation?[/bold yellow]", default=True):
        return

    # ============================================================
    # REPRODUCIBILITY METADATA
    # ============================================================
    cli.console.print("\n[bold cyan]📋 Reproducibility & Metadata[/bold cyan]")
    cli.console.print("[green]✓ Session parameters are automatically saved for:[/green]\n")

    cli.console.print("  [green]1. Resume Capability[/green]")
    cli.console.print("     • Continue this annotation if it stops or crashes")
    cli.console.print("     • Annotate additional rows later with same settings")
    cli.console.print("     • Access via 'Resume/Relaunch Annotation' workflow\n")

    cli.console.print("  [green]2. Scientific Reproducibility[/green]")
    cli.console.print("     • Document exact parameters for research papers")
    cli.console.print("     • Reproduce identical annotations in the future")
    cli.console.print("     • Track model version, prompts, and all settings\n")

    # Metadata is ALWAYS saved automatically for reproducibility
    save_metadata = True

    # ============================================================
    # VALIDATION TOOL EXPORT OPTION
    # ============================================================
    cli.console.print("\n[bold cyan]📤 Validation Tool Export[/bold cyan]")
    cli.console.print("[dim]Export annotations to JSONL format for human validation[/dim]\n")

    cli.console.print("[yellow]Available validation tools:[/yellow]")
    cli.console.print("  • [cyan]Doccano[/cyan] - Simple, lightweight NLP annotation tool")
    cli.console.print("  • [cyan]Label Studio[/cyan] - Advanced, feature-rich annotation platform")
    cli.console.print("  • Both are open-source and free\n")

    cli.console.print("[green]Why validate with external tools?[/green]")
    cli.console.print("  • Review and correct LLM annotations")
    cli.console.print("  • Calculate inter-annotator agreement")
    cli.console.print("  • Export validated data for metrics calculation\n")

    # Initialize export flags
    export_to_doccano = False
    export_to_labelstudio = False
    export_sample_size = None

    # Step 1: Ask if user wants to export
    export_confirm = Confirm.ask(
        "[bold yellow]Export to validation tool?[/bold yellow]",
        default=False
    )

    if export_confirm:
        # Step 2: Ask which tool to export to
        tool_choice = Prompt.ask(
            "[bold yellow]Which validation tool?[/bold yellow]",
            choices=["doccano", "labelstudio"],
            default="doccano"
        )

        # Set the appropriate export flag
        if tool_choice == "doccano":
            export_to_doccano = True
        else:  # labelstudio
            export_to_labelstudio = True

        # Step 2b: If Label Studio, ask export method
        labelstudio_direct_export = False
        labelstudio_api_url = None
        labelstudio_api_key = None

        if export_to_labelstudio:
            cli.console.print("\n[yellow]Label Studio export method:[/yellow]")
            cli.console.print("  • [cyan]jsonl[/cyan] - Export to JSONL file (manual import)")
            if cli.HAS_REQUESTS:
                cli.console.print("  • [cyan]direct[/cyan] - Direct export to Label Studio via API\n")
                export_choices = ["jsonl", "direct"]
            else:
                cli.console.print("  • [dim]direct[/dim] - Direct export via API [dim](requires 'requests' library)[/dim]\n")
                export_choices = ["jsonl"]

            export_method = Prompt.ask(
                "[bold yellow]Export method[/bold yellow]",
                choices=export_choices,
                default="jsonl"
            )

            if export_method == "direct":
                labelstudio_direct_export = True

                cli.console.print("\n[cyan]Label Studio API Configuration:[/cyan]")
                labelstudio_api_url = Prompt.ask(
                    "Label Studio URL",
                    default="http://localhost:8080"
                )

                labelstudio_api_key = Prompt.ask(
                    "API Key (from Label Studio Account & Settings)"
                )

        # Step 3: Ask about LLM predictions inclusion
        cli.console.print("\n[yellow]Include LLM predictions in export?[/yellow]")
        cli.console.print("  • [cyan]with[/cyan] - Include LLM annotations as predictions (for review/correction)")
        cli.console.print("  • [cyan]without[/cyan] - Export only data without predictions (for manual annotation)")
        cli.console.print("  • [cyan]both[/cyan] - Create two files: one with and one without predictions\n")

        prediction_mode = Prompt.ask(
            "[bold yellow]Prediction mode[/bold yellow]",
            choices=["with", "without", "both"],
            default="with"
        )

        # Step 4: Ask how many sentences to export
        cli.console.print("\n[yellow]How many annotated sentences to export?[/yellow]")
        cli.console.print("  • [cyan]all[/cyan] - Export all annotated sentences")
        cli.console.print("  • [cyan]representative[/cyan] - Representative sample (stratified by labels)")
        cli.console.print("  • [cyan]number[/cyan] - Specify exact number\n")

        sample_choice = Prompt.ask(
            "[bold yellow]Export sample[/bold yellow]",
            choices=["all", "representative", "number"],
            default="all"
        )

        if sample_choice == "all":
            export_sample_size = "all"
        elif sample_choice == "representative":
            export_sample_size = "representative"
        else:  # number
            export_sample_size = cli._int_prompt_with_validation(
                "Number of sentences to export",
                100,
                1,
                999999
            )

    # ============================================================
    # EXECUTE ANNOTATION
    # ============================================================

    # CRITICAL: Use new organized structure with dataset-specific subfolder
    # Structure: logs/annotator/{session_id}/annotated_data/{dataset_name}/
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create dataset-specific subdirectory (like {category} in Training Arena)
    dataset_name = data_path.stem
    dataset_subdir = session_dirs['annotated_data'] / dataset_name
    dataset_subdir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
    default_output_path = dataset_subdir / output_filename

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")
    cli.console.print()

    # Prepare prompts payload for pipeline
    prompts_payload = []
    for pc in prompt_configs:
        prompts_payload.append({
            'prompt': pc['prompt']['content'],
            'expected_keys': pc['prompt']['keys'],
            'prefix': pc['prefix']
        })

    # Determine annotation mode
    annotation_mode = 'api' if provider in {'openai', 'anthropic', 'google'} else 'local'

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': text_column,
        'text_columns': [text_column],
        'annotation_column': 'annotation',
        'identifier_column': identifier_column,  # From Step 2b: User-selected ID strategy
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key if api_key else None,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,  # Use Rich progress instead
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': save_incrementally,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
        'lang_column': lang_column,  # From Step 4b: Language column for training metadata
    }

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # ============================================================
    # SAVE REPRODUCIBILITY METADATA
    # ============================================================
    if save_metadata:
        import json

        # Build comprehensive metadata
        metadata = {
            'annotation_session': {
                'timestamp': timestamp,
                'tool_version': 'LLMTool v1.0',
                'workflow': 'The Annotator - Smart Annotate'
            },
            'data_source': {
                'file_path': str(data_path),
                'file_name': data_path.name,
                'data_format': data_format,
                'text_column': text_column,
                'total_rows': annotation_limit if annotation_limit else 'all',
                'sampling_strategy': sample_strategy if annotation_limit else 'none (all rows)',
                'sample_seed': 42 if sample_strategy == 'random' else None
            },
            'model_configuration': {
                'provider': provider,
                'model_name': model_name,
                'annotation_mode': annotation_mode,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'top_k': top_k if provider in ['ollama', 'google'] else None
            },
            'prompts': [
                {
                    'name': pc['prompt']['name'],
                    'file_path': str(pc['prompt']['path']) if 'path' in pc['prompt'] else None,
                    'expected_keys': pc['prompt']['keys'],
                    'prefix': pc['prefix'],
                    'prompt_content': pc['prompt']['content']
                }
                for pc in prompt_configs
            ],
            'processing_configuration': {
                'parallel_workers': num_processes,
                'batch_size': batch_size,
                'incremental_save': save_incrementally,
                'identifier_column': 'annotation_id'
            },
            'output': {
                'output_path': str(default_output_path),
                'output_format': data_format
            },
            'export_preferences': {
                'export_to_doccano': export_to_doccano,
                'export_to_labelstudio': export_to_labelstudio,
                'export_sample_size': export_sample_size,
                'prediction_mode': prediction_mode if (export_to_doccano or export_to_labelstudio) else 'with',
                'labelstudio_direct_export': labelstudio_direct_export if export_to_labelstudio else False,
                'labelstudio_api_url': labelstudio_api_url if export_to_labelstudio else None,
                'labelstudio_api_key': labelstudio_api_key if export_to_labelstudio else None
            },
            'training_workflow': {
                'enabled': False,  # Will be updated after training workflow
                'training_params_file': None,  # Will be added after training
                'note': 'Training parameters will be saved separately after annotation completes'
            }
        }

        # Save metadata JSON (PRE-ANNOTATION SAVE POINT 1)
        # Use dataset-specific subdirectory for metadata too
        metadata_subdir = session_dirs['metadata'] / dataset_name
        metadata_subdir.mkdir(parents=True, exist_ok=True)

        metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        metadata_path = metadata_subdir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[bold green]✅ Metadata saved for reproducibility[/bold green]")
        cli.console.print(f"[bold cyan]📋 Metadata File:[/bold cyan]")
        cli.console.print(f"   {metadata_path}\n")

    # Execute pipeline with Rich progress
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

        # Create pipeline controller with session_id for organized logging
        from ..pipelines.pipeline_controller import PipelineController
        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=session_id  # Pass session_id for organized logging
        )

        # Use RichProgressManager for elegant display
        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,  # Show JSON sample for every annotation
            compact_mode=False   # Full preview panels
        ) as progress_manager:
            # Wrap pipeline for enhanced JSON tracking
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            # Run pipeline
            state = enhanced_pipeline.run_pipeline(pipeline_config)

            # Check for errors
            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                cli.console.print("[dim]Press Enter to return to menu...[/dim]")
                input()
                return

        # Get results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        # Display success message
        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        # Display statistics if available
        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

        # ============================================================
        # AUTOMATIC LANGUAGE DETECTION (if no language column provided)
        # ============================================================
        if not lang_column:
            cli.console.print("\n[bold cyan]🌍 Language Detection for Training[/bold cyan]")
            cli.console.print("[yellow]No language column was provided. Detecting languages for training...[/yellow]\n")

            try:
                import pandas as pd
                from llm_tool.utils.language_detector import LanguageDetector

                # Load annotated file
                df_annotated = pd.read_csv(output_file)

                # CRITICAL: Only detect languages for ANNOTATED rows
                # The output file may contain ALL original rows, but we only want to detect
                # languages for rows that were actually annotated
                original_row_count = len(df_annotated)

                # Try to identify annotated rows by checking for annotation columns
                # Common annotation column names: 'label', 'category', 'annotation', 'labels'
                annotation_cols = [col for col in df_annotated.columns if col in ['label', 'labels', 'category', 'annotation', 'predicted_label']]

                if annotation_cols:
                    # Filter to only rows that have annotations (non-null AND non-empty in annotation column)
                    annotation_col = annotation_cols[0]
                    df_annotated = df_annotated[(df_annotated[annotation_col].notna()) & (df_annotated[annotation_col] != '')].copy()
                    cli.console.print(f"[dim]Filtering to {len(df_annotated):,} annotated rows (out of {original_row_count:,} total rows in file)[/dim]")
                else:
                    cli.console.print(f"[yellow]⚠️  Could not identify annotation column. Processing all {original_row_count:,} rows.[/yellow]")

                if len(df_annotated) == 0:
                    cli.console.print("[yellow]⚠️  No annotated rows found. Skipping language detection.[/yellow]")
                elif text_column in df_annotated.columns:
                    # Get ALL texts (including NaN) to maintain index alignment
                    all_texts = df_annotated[text_column].tolist()

                    # Count non-empty texts for display
                    non_empty_texts = sum(1 for text in all_texts if pd.notna(text) and len(str(text).strip()) > 10)

                    if non_empty_texts > 0:
                        detector = LanguageDetector()
                        detected_languages = []

                        # Progress indicator
                        from tqdm import tqdm
                        cli.console.print(f"[dim]Analyzing {non_empty_texts} texts...[/dim]")

                        for text in tqdm(all_texts, desc="Detecting languages", disable=not cli.HAS_RICH):
                            # Handle NaN and empty texts
                            if pd.isna(text) or not text or len(str(text).strip()) <= 10:
                                detected_languages.append('unknown')
                            else:
                                try:
                                    detected = detector.detect(str(text))
                                    if detected and detected.get('language'):
                                        detected_languages.append(detected['language'])
                                    else:
                                        detected_languages.append('unknown')
                                except Exception as e:
                                    cli.logger.debug(f"Language detection failed for text: {e}")
                                    detected_languages.append('unknown')

                        # Add language column to the filtered dataframe
                        df_annotated['lang'] = detected_languages

                        # Reload the FULL original file and update only the annotated rows
                        df_full = pd.read_csv(output_file)

                        # Initialize lang column if it doesn't exist
                        if 'lang' not in df_full.columns:
                            df_full['lang'] = 'unknown'

                        # Update language for annotated rows only
                        # Match by index of df_annotated within df_full
                        df_full.loc[df_annotated.index, 'lang'] = df_annotated['lang'].values

                        # Save updated full file with language column
                        df_full.to_csv(output_file, index=False)

                        # Show distribution
                        lang_counts = {}
                        for lang in detected_languages:
                            if lang != 'unknown':
                                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                        if lang_counts:
                            total = sum(lang_counts.values())
                            cli.console.print(f"\n[bold]🌍 Languages Detected ({total:,} texts):[/bold]")

                            lang_table = Table(border_style="cyan", show_header=True, header_style="bold")
                            lang_table.add_column("Language", style="cyan", width=12)
                            lang_table.add_column("Count", style="yellow", justify="right", width=12)
                            lang_table.add_column("Percentage", style="green", justify="right", width=12)

                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / total * 100) if total > 0 else 0
                                lang_table.add_row(
                                    lang.upper(),
                                    f"{count:,}",
                                    f"{percentage:.1f}%"
                                )

                            cli.console.print(lang_table)
                            cli.console.print(f"\n[green]✓ Language column 'lang' added to {output_file}[/green]")
                        else:
                            cli.console.print("[yellow]⚠️  No languages detected successfully[/yellow]")

            except Exception as e:
                cli.console.print(f"[yellow]⚠️  Language detection failed: {e}[/yellow]")
                cli.logger.exception("Language detection failed")

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        cli._post_annotation_training_workflow(
            output_file=output_file,
            text_column=text_column,
            prompt_configs=prompt_configs,
            session_id=session_id,
            session_dirs=session_dirs  # Pass session directories for organized logging
        )

        # Export to Doccano JSONL if requested
        if export_to_doccano:
            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=text_column,
                prompt_configs=prompt_configs,
                data_path=data_path,
                timestamp=timestamp,
                sample_size=export_sample_size,
                session_dirs=session_dirs
            )

        # Export to Label Studio if requested
        if export_to_labelstudio:
            if labelstudio_direct_export:
                # Direct export to Label Studio via API
                cli._export_to_labelstudio_direct(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    api_url=labelstudio_api_url,
                    api_key=labelstudio_api_key
                )
            else:
                # Export to JSONL file
                cli._export_to_labelstudio_jsonl(
                    output_file=output_file,
                    text_column=text_column,
                    prompt_configs=prompt_configs,
                    data_path=data_path,
                    timestamp=timestamp,
                    sample_size=export_sample_size,
                    prediction_mode=prediction_mode,
                    session_dirs=session_dirs
                )

        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

    except Exception as exc:
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Annotation execution failed")
        cli.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        input()

def execute_from_metadata(cli, metadata: dict, action_mode: str, metadata_file: Path, session_dirs: Optional[Dict[str, Path]] = None):
    """Execute annotation based on loaded metadata"""
    import json
    from datetime import datetime

    # Extract all parameters from metadata
    data_source = metadata.get('data_source', {})
    model_config = metadata.get('model_configuration', {})
    prompts = metadata.get('prompts', [])
    proc_config = metadata.get('processing_configuration', {})
    output_config = metadata.get('output', {})
    export_prefs = metadata.get('export_preferences', {})

    # Get export preferences
    export_to_doccano = export_prefs.get('export_to_doccano', False)
    export_to_labelstudio = export_prefs.get('export_to_labelstudio', False)
    export_sample_size = export_prefs.get('export_sample_size', 'all')

    if export_to_doccano or export_to_labelstudio:
        export_tools = []
        if export_to_doccano:
            export_tools.append("Doccano")
        if export_to_labelstudio:
            export_tools.append("Label Studio")
        cli.console.print(f"\n[cyan]ℹ️  Export enabled for: {', '.join(export_tools)} (from saved preferences)[/cyan]")
        if export_sample_size != 'all':
            cli.console.print(f"[cyan]   Sample size: {export_sample_size}[/cyan]")

    # Prepare paths
    data_path = Path(data_source.get('file_path', ''))
    data_format = data_source.get('data_format', 'csv')

    # Check if resuming
    if action_mode == 'resume':
        # Try to find the output file
        original_output = Path(output_config.get('output_path', ''))

        if not original_output.exists():
            cli.console.print(f"\n[yellow]⚠️  Output file not found: {original_output}[/yellow]")
            cli.console.print("[yellow]Switching to relaunch mode (fresh annotation)[/yellow]")
            action_mode = 'relaunch'
        else:
            cli.console.print(f"\n[green]✓ Found output file: {original_output.name}[/green]")

            # Count already annotated rows
            import pandas as pd
            try:
                if data_format == 'csv':
                    df_output = pd.read_csv(original_output)
                elif data_format in ['excel', 'xlsx']:
                    df_output = pd.read_excel(original_output)
                elif data_format == 'parquet':
                    df_output = pd.read_parquet(original_output)

                # Count rows with valid annotations (non-empty, non-null strings)
                if 'annotation' in df_output.columns:
                    # Count only rows where annotation exists and is not empty/whitespace
                    annotated_mask = (
                        df_output['annotation'].notna() &
                        (df_output['annotation'].astype(str).str.strip() != '') &
                        (df_output['annotation'].astype(str) != 'nan')
                    )
                    annotated_count = annotated_mask.sum()
                else:
                    annotated_count = 0

                cli.console.print(f"[cyan]  Rows already annotated: {annotated_count:,}[/cyan]")

                # Get total available rows from source file
                if data_path.exists():
                    if data_format == 'csv':
                        total_available = len(pd.read_csv(data_path))
                    elif data_format in ['excel', 'xlsx']:
                        total_available = len(pd.read_excel(data_path))
                    elif data_format == 'parquet':
                        total_available = len(pd.read_parquet(data_path))
                    else:
                        total_available = len(df_output)
                else:
                    total_available = len(df_output)

                # Calculate remaining based on original target
                original_target = data_source.get('total_rows', 'all')

                if original_target == 'all':
                    total_target = total_available
                else:
                    total_target = original_target

                remaining_from_target = total_target - annotated_count
                remaining_from_source = total_available - annotated_count

                cli.console.print(f"[cyan]  Original target: {total_target:,} rows[/cyan]")
                cli.console.print(f"[cyan]  Remaining from target: {remaining_from_target:,}[/cyan]")
                cli.console.print(f"[cyan]  Total available in source: {total_available:,} rows[/cyan]")
                cli.console.print(f"[cyan]  Maximum you can annotate: {remaining_from_source:,}[/cyan]\n")

                if remaining_from_source <= 0:
                    cli.console.print("\n[yellow]All available rows are already annotated![/yellow]")
                    continue_anyway = Confirm.ask("Continue with relaunch mode?", default=False)
                    if not continue_anyway:
                        return
                    action_mode = 'relaunch'
                else:
                    cli.console.print("[yellow]You can annotate:[/yellow]")
                    cli.console.print(f"  • Up to [cyan]{remaining_from_target:,}[/cyan] more rows to complete original target")
                    cli.console.print(f"  • Or up to [cyan]{remaining_from_source:,}[/cyan] total to use all available data\n")

                    resume_count = cli._int_prompt_with_validation(
                        f"How many more rows to annotate? (max: {remaining_from_source:,})",
                        min(100, remaining_from_target) if remaining_from_target > 0 else 100,
                        1,
                        remaining_from_source
                    )

                    # Update metadata for resume
                    metadata['data_source']['total_rows'] = resume_count
                    metadata['resume_mode'] = True
                    metadata['resume_from_file'] = str(original_output)
                    metadata['already_annotated'] = int(annotated_count)

            except Exception as e:
                cli.console.print(f"\n[red]Error reading output file: {e}[/red]")
                cli.console.print("[yellow]Switching to relaunch mode[/yellow]")
                action_mode = 'relaunch'

    # Prepare output path
    annotations_dir = cli.settings.paths.data_dir / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_config.get('model_name', 'unknown').replace(':', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if action_mode == 'resume':
        output_filename = original_output.name  # Keep same filename
        default_output_path = original_output
    else:
        output_filename = f"{data_path.stem}_{safe_model_name}_annotations_{timestamp}.{data_format}"
        default_output_path = annotations_dir / output_filename

    cli.console.print(f"\n[bold cyan]📁 Output Location:[/bold cyan]")
    cli.console.print(f"   {default_output_path}")

    # Prepare prompts payload
    prompts_payload = []
    for p in prompts:
        prompts_payload.append({
            'prompt': p.get('prompt_content', p.get('prompt', '')),
            'expected_keys': p.get('expected_keys', []),
            'prefix': p.get('prefix', '')
        })

    # Get parameters
    provider = model_config.get('provider', 'ollama')
    model_name = model_config.get('model_name', 'llama2')
    annotation_mode = model_config.get('annotation_mode', 'local')
    temperature = model_config.get('temperature', 0.7)
    max_tokens = model_config.get('max_tokens', 1000)
    top_p = model_config.get('top_p', 1.0)
    top_k = model_config.get('top_k', 40)

    num_processes = proc_config.get('parallel_workers', 1)
    batch_size = proc_config.get('batch_size', 1)

    total_rows = data_source.get('total_rows')
    annotation_limit = None if total_rows == 'all' else total_rows
    sample_strategy = data_source.get('sampling_strategy', 'head')

    # IMPORTANT: In resume mode, always use 'head' strategy to continue sequentially
    # This ensures we pick up exactly where we left off, not random new rows
    if action_mode == 'resume':
        sample_strategy = 'head'
        cli.console.print(f"\n[cyan]ℹ️  Resume mode: Using sequential (head) strategy to continue where you left off[/cyan]")

    # Get API key if needed
    api_key = None
    if provider in ['openai', 'anthropic', 'google']:
        api_key = cli._get_api_key(provider)
        if not api_key:
            cli.console.print(f"[red]API key required for {provider}[/red]")
            return

    # Build pipeline config
    pipeline_config = {
        'mode': 'file',
        'data_source': data_format,
        'data_format': data_format,
        'file_path': str(data_path),
        'text_column': data_source.get('text_column', 'text'),
        'text_columns': [data_source.get('text_column', 'text')],
        'annotation_column': 'annotation',
        'identifier_column': 'annotation_id',
        'run_annotation': True,
        'annotation_mode': annotation_mode,
        'annotation_provider': provider,
        'annotation_model': model_name,
        'api_key': api_key,
        'prompts': prompts_payload,
        'annotation_sample_size': annotation_limit,
        'annotation_sampling_strategy': sample_strategy if annotation_limit else 'head',
        'annotation_sample_seed': 42,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k if provider in ['ollama', 'google'] else None,
        'max_workers': num_processes,
        'num_processes': num_processes,
        'use_parallel': num_processes > 1,
        'warmup': False,
        'disable_tqdm': True,
        'output_format': data_format,
        'output_path': str(default_output_path),
        'save_incrementally': True,
        'batch_size': batch_size,
        'run_validation': False,
        'run_training': False,
    }

    # Add resume information if resuming
    if action_mode == 'resume' and metadata.get('resume_mode'):
        pipeline_config['resume_mode'] = True
        pipeline_config['resume_from_file'] = metadata.get('resume_from_file')
        pipeline_config['skip_annotated'] = True

        # Load already annotated IDs to skip them
        try:
            import pandas as pd
            resume_file = Path(metadata.get('resume_from_file'))
            if resume_file.exists():
                if data_format == 'csv':
                    df_resume = pd.read_csv(resume_file)
                elif data_format in ['excel', 'xlsx']:
                    df_resume = pd.read_excel(resume_file)
                elif data_format == 'parquet':
                    df_resume = pd.read_parquet(resume_file)

                # Get IDs of rows that have valid annotations
                if 'annotation' in df_resume.columns and 'annotation_id' in df_resume.columns:
                    annotated_mask = (
                        df_resume['annotation'].notna() &
                        (df_resume['annotation'].astype(str).str.strip() != '') &
                        (df_resume['annotation'].astype(str) != 'nan')
                    )
                    already_annotated_ids = df_resume.loc[annotated_mask, 'annotation_id'].tolist()
                    pipeline_config['skip_annotation_ids'] = already_annotated_ids

                    cli.console.print(f"[cyan]  Will skip {len(already_annotated_ids)} already annotated row(s)[/cyan]")
        except Exception as e:
            cli.logger.warning(f"Could not load annotated IDs from resume file: {e}")
            cli.console.print(f"[yellow]⚠️  Warning: Could not load annotated IDs - may re-annotate some rows[/yellow]")

    # Add model-specific options
    if provider == 'ollama':
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_p': top_p,
            'top_k': top_k
        }
        pipeline_config['options'] = options

    # Save new metadata for this execution
    if action_mode == 'relaunch':
        new_metadata = metadata.copy()
        new_metadata['annotation_session']['timestamp'] = timestamp
        new_metadata['annotation_session']['relaunch_from'] = str(metadata_file.name)
        new_metadata['output']['output_path'] = str(default_output_path)

        new_metadata_filename = f"{data_path.stem}_{safe_model_name}_metadata_{timestamp}.json"
        new_metadata_path = session_dirs['metadata'] / new_metadata_filename

        with open(new_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)

        cli.console.print(f"\n[green]✅ New session metadata saved[/green]")
        cli.console.print(f"[cyan]📋 Metadata File:[/cyan]")
        cli.console.print(f"   {new_metadata_path}\n")

    # Execute pipeline
    try:
        cli.console.print("\n[bold green]🚀 Starting annotation...[/bold green]\n")

        from ..pipelines.pipeline_controller import PipelineController

        # Extract session_id from metadata or create new one
        resume_session_id = metadata.get("session_id") if metadata else None

        pipeline_with_progress = PipelineController(
            settings=cli.settings,
            session_id=resume_session_id  # Pass session_id for organized logging
        )

        from ..utils.rich_progress_manager import RichProgressManager
        from ..pipelines.enhanced_pipeline_wrapper import EnhancedPipelineWrapper

        with RichProgressManager(
            show_json_every=1,
            compact_mode=False
        ) as progress_manager:
            enhanced_pipeline = EnhancedPipelineWrapper(
                pipeline_with_progress,
                progress_manager
            )

            state = enhanced_pipeline.run_pipeline(pipeline_config)

            if state.errors:
                error_msg = state.errors[0]['error'] if state.errors else "Annotation failed"
                cli.console.print(f"\n[bold red]❌ Error:[/bold red] {error_msg}")
                return

        # Display results
        annotation_results = state.annotation_results or {}
        output_file = annotation_results.get('output_file', str(default_output_path))

        cli.console.print("\n[bold green]✅ Annotation completed successfully![/bold green]")
        cli.console.print(f"\n[bold cyan]📄 Output File:[/bold cyan]")
        cli.console.print(f"   {output_file}")

        total_annotated = annotation_results.get('total_annotated', 0)
        if total_annotated:
            cli.console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            cli.console.print(f"   Rows annotated: {total_annotated:,}")

            success_count = annotation_results.get('success_count', 0)
            if success_count:
                success_rate = (success_count / total_annotated * 100)
                cli.console.print(f"   Success rate: {success_rate:.1f}%")

        # ============================================================
        # INTELLIGENT TRAINING WORKFLOW (Post-Annotation)
        # ============================================================
        # Build prompt_configs for training workflow
        prompt_configs_for_training = []
        for p in prompts:
            prompt_configs_for_training.append({
                'prompt': {
                    'keys': p.get('expected_keys', []),
                    'content': p.get('prompt_content', p.get('prompt', '')),
                    'name': p.get('name', 'prompt')
                },
                'prefix': p.get('prefix', '')
            })

        cli._post_annotation_training_workflow(
            output_file=output_file,
            text_column=data_source.get('text_column', 'text'),
            prompt_configs=prompt_configs_for_training,
            session_id=session_id,
            session_dirs=None  # No session_dirs in resume context
        )

        # Export to Doccano JSONL if enabled in preferences
        if export_to_doccano:
            # Build prompt_configs for export
            prompt_configs_for_export = []
            for p in prompts:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            cli._export_to_doccano_jsonl(
                output_file=output_file,
                text_column=data_source.get('text_column', 'text'),
                prompt_configs=prompt_configs_for_export,
                data_path=data_path,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                sample_size=export_sample_size
            )

        # Export to Label Studio JSONL if enabled in preferences
        if export_to_labelstudio:
            # Build prompt_configs for export
            prompt_configs_for_export = []
            for p in prompts:
                prompt_configs_for_export.append({
                    'prompt': {
                        'keys': p.get('expected_keys', []),
                        'content': p.get('prompt_content', p.get('prompt', '')),
                        'name': p.get('name', 'prompt')
                    },
                    'prefix': p.get('prefix', '')
                })

            cli._export_to_labelstudio_jsonl(
                output_file=output_file,
                text_column=data_source.get('text_column', 'text'),
                prompt_configs=prompt_configs_for_export,
                data_path=data_path,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                sample_size=export_sample_size
            )

    except Exception as exc:
        cli.console.print(f"\n[bold red]❌ Annotation failed:[/bold red] {exc}")
        cli.logger.exception("Resume/Relaunch annotation failed")