#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
prompt_wizard.py

MAIN OBJECTIVE:
---------------
This module provides an interactive wizard for creating structured annotation prompts
for social science research. It guides users through defining their research objectives,
data characteristics, annotation categories, and automatically generates optimized prompts.

Dependencies:
-------------
- rich
- typing
- dataclasses
- json

MAIN FEATURES:
--------------
1) Interactive step-by-step prompt creation
2) LLM-assisted definition generation
3) Support for named entity extraction and categorical annotation
4) Hierarchical category structures (general categories + specific values)
5) Prompt validation and editing
6) Example generation

Author:
-------
Antoine Lemor
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


@dataclass
class AnnotationCategory:
    """Represents an annotation category (key in the JSON output)"""
    name: str  # JSON key name
    description: str  # Full description of the category
    category_type: str  # "entity", "categorical", "sentiment"
    values: List[str] = field(default_factory=list)  # Possible values
    value_definitions: Dict[str, str] = field(default_factory=dict)  # Value -> definition
    allows_multiple: bool = False  # Can have multiple values
    allows_null: bool = True  # Can be null
    parent_category: Optional[str] = None  # For hierarchical structures


@dataclass
class PromptSpecification:
    """Complete specification for generating an annotation prompt"""
    project_description: str
    data_description: str
    annotation_type: str  # "entity_extraction" or "categorical_annotation"
    categories: List[AnnotationCategory] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    domain: str = "social_sciences"


class SocialSciencePromptWizard:
    """Interactive wizard for creating social science annotation prompts"""

    def __init__(self, llm_client=None):
        """Initialize the wizard

        Args:
            llm_client: Optional LLM client for assisted definition generation
        """
        self.console = console if HAS_RICH else None
        self.llm_client = llm_client
        self.spec = None

    def run(self) -> Tuple[str, List[str]]:
        """Run the interactive wizard and return (prompt, expected_keys)

        Returns:
            Tuple of (prompt_text, expected_json_keys)
        """
        self._display_welcome()

        # Step 1: Project description
        project_desc = self._get_project_description()

        # Step 2: Data description
        data_desc = self._get_data_description()

        # Step 3: Annotation type
        annotation_type = self._get_annotation_type()

        # Step 4: Define categories
        categories = self._define_categories(annotation_type)

        # Step 5: Generate examples (optional)
        examples = self._generate_examples(categories)

        # Step 6: Build specification
        self.spec = PromptSpecification(
            project_description=project_desc,
            data_description=data_desc,
            annotation_type=annotation_type,
            categories=categories,
            examples=examples
        )

        # Step 7: Generate prompt
        prompt_text = self._build_prompt()

        # Step 8: Review and edit
        final_prompt = self._review_and_edit_prompt(prompt_text)

        # Extract expected keys
        expected_keys = [cat.name for cat in self.spec.categories]

        return final_prompt, expected_keys

    def _display_welcome(self):
        """Display welcome message"""
        if not self.console:
            print("\n=== Social Science Prompt Wizard ===\n")
            return

        welcome_text = """
[bold cyan]Welcome to the Social Science Prompt Wizard! 🧙‍♂️[/bold cyan]

This interactive wizard will guide you through creating a sophisticated annotation prompt
for your social science research project. You will:

• Define your research objectives
• Describe your data
• Choose annotation strategies (entity extraction or categorical annotation)
• Define categories and their values with AI assistance
• Review and refine the generated prompt

[dim]The wizard supports both named entity extraction and categorical classification.[/dim]
        """

        panel = Panel(
            welcome_text,
            title="[bold]🎓 Social Science Prompt Wizard[/bold]",
            border_style="cyan",
            box=box.DOUBLE
        )
        self.console.print(panel)
        self.console.print()

    def _get_project_description(self) -> str:
        """Get project description from user"""
        if not self.console:
            return input("\nDescribe your research project: ")

        self.console.print(Panel(
            "[bold]Step 1: Project Description[/bold]\n\n"
            "Please provide a brief description of your research project.\n"
            "[dim]This helps contextualize the annotation task.[/dim]",
            border_style="green"
        ))

        # Show example
        example = (
            "[dim italic]Example: Analyzing political discourse in Canadian parliamentary debates "
            "to identify policy themes, party positions, and sentiment towards various "
            "socio-economic issues from 2015-2024.[/dim italic]"
        )
        self.console.print(example)
        self.console.print()

        description = Prompt.ask(
            "[cyan]📋 Project description[/cyan]",
            default=""
        )

        while not description.strip():
            self.console.print("[yellow]⚠️  Description cannot be empty[/yellow]")
            description = Prompt.ask("[cyan]📋 Project description[/cyan]")

        self.console.print(f"\n[green]✓[/green] Project: {description[:100]}...\n")
        return description

    def _get_data_description(self) -> str:
        """Get data description from user"""
        if not self.console:
            return input("\nDescribe your data: ")

        self.console.print(Panel(
            "[bold]Step 2: Data Description[/bold]\n\n"
            "Describe the nature and characteristics of the data you will annotate.\n"
            "[dim]Be specific about the source, format, and content.[/dim]",
            border_style="green"
        ))

        # Show example
        example = (
            "[dim italic]Example: One-sentence excerpts from Hansard transcripts and Canadian news "
            "articles (La Presse, Globe and Mail, CBC) discussing federal and provincial policies. "
            "Texts are in English and French, typically 15-50 words, focusing on policy statements "
            "and political commentary.[/dim italic]"
        )
        self.console.print(example)
        self.console.print()

        description = Prompt.ask(
            "[cyan]📊 Data description[/cyan]",
            default=""
        )

        while not description.strip():
            self.console.print("[yellow]⚠️  Description cannot be empty[/yellow]")
            description = Prompt.ask("[cyan]📊 Data description[/cyan]")

        self.console.print(f"\n[green]✓[/green] Data: {description[:100]}...\n")
        return description

    def _get_annotation_type(self) -> str:
        """Ask user to choose annotation type"""
        if not self.console:
            print("\nAnnotation types:")
            print("1. Named Entity Extraction (extract specific entities/concepts)")
            print("2. Categorical Annotation (classify into predefined categories)")
            choice = input("Choose (1 or 2): ")
            return "entity_extraction" if choice == "1" else "categorical_annotation"

        self.console.print(Panel(
            "[bold]Step 3: Annotation Type[/bold]\n\n"
            "Choose your annotation strategy:",
            border_style="green"
        ))

        # Show options with examples
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Type", style="cyan", width=30)
        table.add_column("Description", style="white", width=50)
        table.add_column("Example", style="dim", width=50)

        table.add_row(
            "1️⃣  Named Entity Extraction",
            "Extract and classify specific entities or concepts from text",
            "Extract: persons → {role: politician, name: Justin Trudeau}"
        )
        table.add_row(
            "2️⃣  Categorical Annotation",
            "Classify entire text into predefined categories",
            "Classify: theme → environment, sentiment → positive"
        )

        self.console.print(table)
        self.console.print()

        choice = Prompt.ask(
            "[cyan]Select annotation type[/cyan]",
            choices=["1", "2"],
            default="2"
        )

        annotation_type = "entity_extraction" if choice == "1" else "categorical_annotation"

        type_name = "Named Entity Extraction" if choice == "1" else "Categorical Annotation"
        self.console.print(f"\n[green]✓[/green] Selected: {type_name}\n")

        return annotation_type

    def _define_categories(self, annotation_type: str) -> List[AnnotationCategory]:
        """Guide user through defining annotation categories"""
        categories = []

        if annotation_type == "entity_extraction":
            categories = self._define_entity_categories()
        else:
            categories = self._define_categorical_categories()

        return categories

    def _define_entity_categories(self) -> List[AnnotationCategory]:
        """Define categories for entity extraction"""
        if not self.console:
            return []

        self.console.print(Panel(
            "[bold]Step 4: Define Entity Categories[/bold]\n\n"
            "You will now define the types of entities to extract.\n"
            "[dim]Example: persons, organizations, policy_topics, etc.[/dim]",
            border_style="green"
        ))

        categories = []
        category_count = 0

        while True:
            category_count += 1
            self.console.print(f"\n[bold cyan]Entity Category #{category_count}[/bold cyan]")

            # Get category name (JSON key)
            cat_name = Prompt.ask(
                "[cyan]Category name (JSON key, e.g., 'persons', 'organizations')[/cyan]"
            )

            if not cat_name.strip():
                break

            # Sanitize name
            cat_name = re.sub(r'[^a-z0-9_]', '_', cat_name.lower())

            # Get category description
            cat_desc = Prompt.ask(
                "[cyan]Brief description of this entity category[/cyan]",
                default=f"Entities of type {cat_name}"
            )

            # Ask if this category has sub-types
            has_subtypes = Confirm.ask(
                f"[cyan]Does '{cat_name}' have sub-categories/types?[/cyan]\n"
                "[dim](e.g., persons → politician, scientist, activist)[/dim]"
            )

            values = []
            value_defs = {}

            if has_subtypes:
                parent_name = Prompt.ask(
                    "[cyan]What is the parent category name?[/cyan]",
                    default=cat_name + "_type"
                )

                # Define subtypes
                values, value_defs = self._define_category_values(
                    cat_name,
                    f"types of {cat_name}"
                )

                # Create parent category
                parent_cat = AnnotationCategory(
                    name=parent_name,
                    description=f"Type/role of {cat_name}",
                    category_type="entity",
                    values=values,
                    value_definitions=value_defs,
                    allows_multiple=False,
                    allows_null=True
                )
                categories.append(parent_cat)

            # Create main entity category
            entity_cat = AnnotationCategory(
                name=cat_name,
                description=cat_desc,
                category_type="entity",
                values=[],  # Entities are free-form
                value_definitions={},
                allows_multiple=True,
                allows_null=True,
                parent_category=parent_name if has_subtypes else None
            )
            categories.append(entity_cat)

            # Ask if user wants to add more categories
            if not Confirm.ask("[cyan]Add another entity category?[/cyan]", default=False):
                break

        return categories

    def _define_categorical_categories(self) -> List[AnnotationCategory]:
        """Define categories for categorical annotation"""
        if not self.console:
            return []

        self.console.print(Panel(
            "[bold]Step 4: Define Annotation Categories[/bold]\n\n"
            "You will now define the classification categories.\n"
            "[dim]Example: theme → environment/health/economy, sentiment → positive/negative[/dim]",
            border_style="green"
        ))

        categories = []
        category_count = 0

        while True:
            category_count += 1
            self.console.print(f"\n[bold cyan]Category #{category_count}[/bold cyan]")

            # Get category name (JSON key)
            cat_name = Prompt.ask(
                "[cyan]Category name (JSON key, e.g., 'theme', 'sentiment', 'party')[/cyan]"
            )

            if not cat_name.strip():
                break

            # Sanitize name
            cat_name = re.sub(r'[^a-z0-9_]', '_', cat_name.lower())

            # Get general description
            cat_desc = Prompt.ask(
                "[cyan]General description of this category[/cyan]",
                default=f"Classification by {cat_name}"
            )

            # Define possible values
            values, value_defs = self._define_category_values(cat_name, cat_desc)

            # Ask about multiple values
            allows_multiple = Confirm.ask(
                f"[cyan]Can a text have multiple '{cat_name}' values simultaneously?[/cyan]",
                default=False
            )

            # Create category
            category = AnnotationCategory(
                name=cat_name,
                description=cat_desc,
                category_type="categorical",
                values=values,
                value_definitions=value_defs,
                allows_multiple=allows_multiple,
                allows_null=True
            )
            categories.append(category)

            # Ask if user wants to add more categories
            if not Confirm.ask("[cyan]Add another category?[/cyan]", default=True):
                break

        return categories

    def _define_category_values(self, cat_name: str, cat_desc: str) -> Tuple[List[str], Dict[str, str]]:
        """Define values and their definitions for a category"""
        if not self.console:
            return [], {}

        self.console.print(f"\n[bold yellow]Define values for '{cat_name}'[/bold yellow]")
        self.console.print(f"[dim]{cat_desc}[/dim]\n")

        values = []
        value_defs = {}
        value_count = 0

        # Ask if user wants LLM assistance
        use_llm = False
        if self.llm_client:
            use_llm = Confirm.ask(
                "[cyan]🤖 Do you want AI assistance to generate value definitions?[/cyan]",
                default=True
            )

        while True:
            value_count += 1

            # Get value name
            value_name = Prompt.ask(
                f"[cyan]Value #{value_count} (e.g., 'environment', 'positive')[/cyan]\n"
                "[dim]Press Enter without input to finish[/dim]"
            )

            if not value_name.strip():
                break

            # Sanitize value
            value_name = re.sub(r'[^a-z0-9_]', '_', value_name.lower())
            values.append(value_name)

            # Get definition
            if use_llm and self.llm_client:
                # Generate definition with LLM
                definition = self._generate_value_definition_with_llm(
                    cat_name, cat_desc, value_name
                )

                self.console.print(Panel(
                    f"[bold]AI-Generated Definition:[/bold]\n\n{definition}",
                    border_style="yellow"
                ))

                # Ask if user wants to edit
                if Confirm.ask("[cyan]Accept this definition?[/cyan]", default=True):
                    value_defs[value_name] = definition
                else:
                    manual_def = Prompt.ask(
                        "[cyan]Enter your definition[/cyan]",
                        default=definition
                    )
                    value_defs[value_name] = manual_def
            else:
                # Manual definition
                definition = Prompt.ask(
                    f"[cyan]Definition for '{value_name}'[/cyan]",
                    default=f"Text relates to {value_name}"
                )
                value_defs[value_name] = definition

            self.console.print(f"[green]✓[/green] Added: {value_name}\n")

        # Always add 'null' option
        if 'null' not in values:
            values.append('null')
            value_defs['null'] = f"Does not explicitly relate to any {cat_name} category"

        # Show summary
        self._show_category_summary(cat_name, values, value_defs)

        return values, value_defs

    def _generate_value_definition_with_llm(self, cat_name: str, cat_desc: str, value_name: str) -> str:
        """Use LLM to generate a definition for a category value"""
        prompt = f"""You are an expert in social science research methodology and text annotation.

Category: {cat_name}
Category Description: {cat_desc}
Value to Define: {value_name}

Generate a precise, clear definition for the value "{value_name}" in the context of the category "{cat_name}".

The definition should:
- Be 1-2 sentences maximum
- Be specific and actionable for annotators
- Include concrete indicators when possible
- Be written for annotation guidelines

Definition:"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )

            if response:
                # Clean up response
                definition = response.strip()
                # Remove any "Definition:" prefix if present
                definition = re.sub(r'^Definition:\s*', '', definition, flags=re.IGNORECASE)
                return definition
            else:
                return f"Text relates to {value_name} in the context of {cat_name}"

        except Exception as e:
            self.console.print(f"[yellow]⚠️  LLM generation failed: {e}[/yellow]")
            return f"Text relates to {value_name} in the context of {cat_name}"

    def _show_category_summary(self, cat_name: str, values: List[str], value_defs: Dict[str, str]):
        """Display a summary table of category values"""
        if not self.console:
            return

        table = Table(title=f"Summary: {cat_name}", box=box.ROUNDED)
        table.add_column("Value", style="cyan", width=20)
        table.add_column("Definition", style="white", width=60)

        for value in values:
            definition = value_defs.get(value, "")
            # Truncate long definitions
            if len(definition) > 100:
                definition = definition[:97] + "..."
            table.add_row(value, definition)

        self.console.print(table)
        self.console.print()

    def _generate_examples(self, categories: List[AnnotationCategory]) -> List[Dict[str, Any]]:
        """Allow user to add example annotations"""
        if not self.console:
            return []

        self.console.print(Panel(
            "[bold]Step 5: Example Annotations (Optional)[/bold]\n\n"
            "You can add example annotations to help guide the LLM.\n"
            "[dim]Examples improve annotation quality and consistency.[/dim]",
            border_style="green"
        ))

        add_examples = Confirm.ask(
            "[cyan]Would you like to add example annotations?[/cyan]",
            default=True
        )

        if not add_examples:
            return []

        examples = []
        example_count = 0

        while True:
            example_count += 1
            self.console.print(f"\n[bold cyan]Example #{example_count}[/bold cyan]")

            # Get example text
            text = Prompt.ask(
                "[cyan]Example text to annotate[/cyan]\n"
                "[dim]Press Enter without input to finish[/dim]"
            )

            if not text.strip():
                break

            # Get annotations for each category
            annotations = {}
            for cat in categories:
                self.console.print(f"\n[yellow]Annotation for '{cat.name}':[/yellow]")
                self.console.print(f"[dim]Values: {', '.join(cat.values)}[/dim]")

                if cat.allows_multiple:
                    value = Prompt.ask(
                        f"[cyan]Value(s) for '{cat.name}' (comma-separated for multiple)[/cyan]"
                    )
                    if ',' in value:
                        annotations[cat.name] = [v.strip() for v in value.split(',')]
                    else:
                        annotations[cat.name] = value.strip()
                else:
                    value = Prompt.ask(f"[cyan]Value for '{cat.name}'[/cyan]")
                    annotations[cat.name] = value.strip()

            examples.append({
                "text": text,
                "annotations": annotations
            })

            self.console.print(f"[green]✓[/green] Example added\n")

            if not Confirm.ask("[cyan]Add another example?[/cyan]", default=False):
                break

        return examples

    def _build_prompt(self) -> str:
        """Build the complete annotation prompt from specification"""
        prompt_parts = []

        # 1. Introduction and role
        intro = self._build_introduction()
        prompt_parts.append(intro)

        # 2. Task description
        task_desc = self._build_task_description()
        prompt_parts.append(task_desc)

        # 3. Category definitions
        category_defs = self._build_category_definitions()
        prompt_parts.append(category_defs)

        # 4. Instructions
        instructions = self._build_instructions()
        prompt_parts.append(instructions)

        # 5. Examples
        if self.spec.examples:
            examples = self._build_examples()
            prompt_parts.append(examples)

        # 6. Expected JSON keys
        expected_keys = self._build_expected_keys()
        prompt_parts.append(expected_keys)

        return "\n\n".join(prompt_parts)

    def _build_introduction(self) -> str:
        """Build prompt introduction"""
        return (
            f"You are a text annotator specializing in {self.spec.domain}. "
            f"{self.spec.project_description}\n\n"
            f"Analyze the following data: {self.spec.data_description}\n\n"
            "You must structure the output in JSON format. "
            "Write exclusively in JSON without any explanatory text."
        )

    def _build_task_description(self) -> str:
        """Build task description"""
        if self.spec.annotation_type == "entity_extraction":
            return (
                "**Task:** Extract and classify the specified entities from each text. "
                "Identify all relevant entities and their types according to the categories below."
            )
        else:
            return (
                "**Task:** Classify each text according to the categories defined below. "
                "The categories must be clear, and appropriate values must be used."
            )

    def _build_category_definitions(self) -> str:
        """Build category definitions section"""
        lines = ["**Expected keys:**"]

        for cat in self.spec.categories:
            # Build value descriptions
            value_parts = []
            for value in cat.values:
                definition = cat.value_definitions.get(value, f"relates to {value}")
                value_parts.append(f'"{value}" if {definition}')

            values_str = ", ".join(value_parts)

            # Category line
            multiple_note = " (can be multiple values)" if cat.allows_multiple else ""
            cat_line = f'- "{cat.name}"{multiple_note}: {values_str}.'

            lines.append(cat_line)

        return "\n".join(lines)

    def _build_instructions(self) -> str:
        """Build instructions section"""
        instructions = [
            "**Instructions**",
            "- Strictly follow the structure of the keys defined above.",
            "- Ensure that all keys are present in the JSON, using `null` when necessary.",
            "- Do not include keys that are not defined in the expected keys above.",
            "- Write exclusively the JSON without any additional comments or explanations."
        ]

        # Add category-specific instructions
        for cat in self.spec.categories:
            if cat.allows_multiple:
                instructions.append(
                    f"- Indicate multiple '{cat.name}' values as an array if multiple values are present."
                )
            else:
                instructions.append(
                    f"- Indicate only one '{cat.name}' value for each text."
                )

        return "\n".join(instructions)

    def _build_examples(self) -> str:
        """Build examples section"""
        lines = []

        for i, example in enumerate(self.spec.examples, 1):
            lines.append(f"**Example of an annotation for the text:**\n")
            lines.append(example["text"])
            lines.append("\n**Example of JSON:**\n")

            json_str = json.dumps(example["annotations"], indent=2, ensure_ascii=False)
            lines.append(json_str)
            lines.append("")

        lines.append(
            "Follow this structure for each text analyzed. "
            "No other comments or additional details beyond the requested JSON structure "
            "and the specified categories should be added."
        )

        return "\n".join(lines)

    def _build_expected_keys(self) -> str:
        """Build expected JSON keys template"""
        lines = ["**Expected JSON Keys**"]

        template = {}
        for cat in self.spec.categories:
            if cat.allows_multiple:
                template[cat.name] = [""]
            else:
                template[cat.name] = ""

        json_str = json.dumps(template, indent=2, ensure_ascii=False)
        lines.append(json_str)

        return "\n".join(lines)

    def _review_and_edit_prompt(self, prompt_text: str) -> str:
        """Display prompt and allow user to review/edit"""
        if not self.console:
            return prompt_text

        self.console.print("\n" + "="*80 + "\n")
        self.console.print(Panel(
            "[bold green]Step 6: Review Generated Prompt[/bold green]",
            border_style="green"
        ))

        # Display prompt with syntax highlighting
        syntax = Syntax(prompt_text, "markdown", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Generated Prompt", border_style="cyan"))

        # Options
        self.console.print("\n[bold]What would you like to do?[/bold]")
        self.console.print("1. ✅ Accept and use this prompt")
        self.console.print("2. ✏️  Edit the prompt manually")
        self.console.print("3. 🔄 Regenerate with modifications")
        self.console.print("4. 💾 Save prompt to file")

        choice = Prompt.ask(
            "\n[cyan]Your choice[/cyan]",
            choices=["1", "2", "3", "4"],
            default="1"
        )

        if choice == "1":
            self.console.print("\n[green]✓ Prompt accepted![/green]\n")
            return prompt_text

        elif choice == "2":
            return self._manual_edit_prompt(prompt_text)

        elif choice == "3":
            self.console.print("[yellow]Regeneration not yet implemented[/yellow]")
            return prompt_text

        elif choice == "4":
            self._save_prompt_to_file(prompt_text)
            return prompt_text

        return prompt_text

    def _manual_edit_prompt(self, prompt_text: str) -> str:
        """Allow manual editing of prompt using external editor"""
        import tempfile
        import subprocess
        import os

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt_text)
            temp_path = f.name

        try:
            # Open in editor
            editor = os.environ.get('EDITOR', 'nano')
            self.console.print(f"\n[cyan]Opening editor: {editor}[/cyan]")
            self.console.print("[dim]Edit the prompt, save, and close the editor to continue.[/dim]\n")

            subprocess.call([editor, temp_path])

            # Read edited content
            with open(temp_path, 'r') as f:
                edited_prompt = f.read()

            self.console.print("\n[green]✓ Prompt edited successfully![/green]\n")
            return edited_prompt

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _save_prompt_to_file(self, prompt_text: str):
        """Save prompt to file"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"prompt_wizard_{timestamp}.txt"

        filename = Prompt.ask(
            "[cyan]Enter filename to save[/cyan]",
            default=default_filename
        )

        # Ensure prompts directory exists
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)

        filepath = prompts_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        self.console.print(f"\n[green]✓ Prompt saved to: {filepath}[/green]\n")


def create_llm_client_for_wizard(provider: str, model: str, api_key: Optional[str] = None):
    """Create an LLM client for the wizard's AI assistance

    Args:
        provider: "ollama", "openai", "anthropic"
        model: Model name
        api_key: API key if needed

    Returns:
        LLM client instance
    """
    from ..annotators.api_clients import create_api_client
    from ..annotators.local_models import OllamaClient

    if provider == "ollama":
        return OllamaClient(model)
    else:
        return create_api_client(provider=provider, api_key=api_key, model=model)
