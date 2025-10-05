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


# UI translations for wizard interface
UI_TRANSLATIONS = {
    "en": {
        "category_values_title": "💡 How to Structure Your Categories",
        "category_values_heading": "📖 Understanding the Category → Values System",
        "category_values_intro": "This system works in TWO LEVELS:",
        "level1_title": "LEVEL 1: CATEGORY (the general question)",
        "level1_point1": "• This is the GLOBAL DIMENSION you want to analyze",
        "level1_point2": "• Becomes the JSON KEY in your final annotation",
        "level1_point3": "• Think: \"What QUESTION am I asking about this text?\"",
        "level1_examples_header": "Category examples (questions):",
        "level1_ex1": "• theme → \"What is this text about?\"",
        "level1_ex2": "• sentiment → \"What is the emotional tone?\"",
        "level1_ex3": "• political_party → \"Which party is mentioned?\"",
        "level2_title": "LEVEL 2: VALUES (the possible answers)",
        "level2_point1": "• These are the SPECIFIC ANSWERS possible to your question",
        "level2_point2": "• These will be the CONCRETE RESULTS of annotation",
        "level2_point3": "• Think: \"What are the POSSIBLE ANSWERS?\"",
        "level2_examples_header": "Value examples (answers) for 'theme':",
        "level2_ex1": "• environment (if text discusses ecology)",
        "level2_ex2": "• health (if text discusses healthcare)",
        "level2_ex3": "• economy (if text discusses economics)",
        "level2_ex4": "• justice (if text discusses judicial system)",
        "complete_example_title": "✨ Complete Example:",
        "complete_ex1_q": "Question → Category: theme",
        "complete_ex1_a": "Answers → Values: environment, health, justice, economy",
        "complete_ex2_q": "Question → Category: sentiment",
        "complete_ex2_a": "Answers → Values: positive, negative, neutral",
        "final_annotation_header": "📝 Your final annotation will look like:",
        "final_annotation_comment1": "← the chosen VALUE",
        "final_annotation_comment2": "← the chosen VALUE",
        "keep_category": "Keep this category '{}'?",
        "category_added": "Category '{}' added",
        "category_rejected": "Category '{}' rejected",
        "add_another_category": "Add another category?",
        "delete_values_prompt": "Do you want to delete values from '{}'?",
        "current_values": "Current values for '{}':",
        "value_to_delete": "Number of value to delete (or 'cancel')",
        "value_deleted": "Value '{}' deleted",
        "keep_entity": "Keep this entity '{}'?",
        "entity_added": "Entity '{}' added",
        "entity_rejected": "Entity '{}' rejected",
        "add_another_entity": "Add another entity?"
    },
    "fr": {
        "category_values_title": "💡 Comment Structurer Vos Catégories",
        "category_values_heading": "📖 Comprendre le Système Catégorie → Valeurs",
        "category_values_intro": "Ce système fonctionne en DEUX NIVEAUX :",
        "level1_title": "NIVEAU 1 : CATÉGORIE (la question générale)",
        "level1_point1": "• C'est la DIMENSION GLOBALE que vous voulez analyser",
        "level1_point2": "• Devient la CLÉ JSON dans votre annotation finale",
        "level1_point3": "• Pensez : \"Quelle QUESTION je pose sur ce texte ?\"",
        "level1_examples_header": "Exemples de catégories (questions) :",
        "level1_ex1": "• thème → \"De quoi parle ce texte ?\"",
        "level1_ex2": "• sentiment → \"Quel est le ton émotionnel ?\"",
        "level1_ex3": "• parti_politique → \"Quel parti est mentionné ?\"",
        "level2_title": "NIVEAU 2 : VALEURS (les réponses possibles)",
        "level2_point1": "• Ce sont les RÉPONSES SPÉCIFIQUES possibles à votre question",
        "level2_point2": "• Ce seront les RÉSULTATS CONCRETS de l'annotation",
        "level2_point3": "• Pensez : \"Quelles sont les RÉPONSES possibles ?\"",
        "level2_examples_header": "Exemples de valeurs (réponses) pour 'thème' :",
        "level2_ex1": "• environnement (si le texte parle d'écologie)",
        "level2_ex2": "• santé (si le texte parle de soins de santé)",
        "level2_ex3": "• économie (si le texte parle d'économie)",
        "level2_ex4": "• justice (si le texte parle de système judiciaire)",
        "complete_example_title": "✨ Exemple Complet :",
        "complete_ex1_q": "Question → Catégorie : thème",
        "complete_ex1_a": "Réponses → Valeurs : environnement, santé, justice, économie",
        "complete_ex2_q": "Question → Catégorie : sentiment",
        "complete_ex2_a": "Réponses → Valeurs : positif, négatif, neutre",
        "final_annotation_header": "📝 Votre annotation finale ressemblera à :",
        "final_annotation_comment1": "← la VALEUR choisie",
        "final_annotation_comment2": "← la VALEUR choisie",
        "keep_category": "Garder cette catégorie '{}' ?",
        "category_added": "Catégorie '{}' ajoutée",
        "category_rejected": "Catégorie '{}' rejetée",
        "add_another_category": "Ajouter une autre catégorie ?",
        "delete_values_prompt": "Voulez-vous supprimer des valeurs de '{}' ?",
        "current_values": "Valeurs actuelles pour '{}' :",
        "value_to_delete": "Numéro de la valeur à supprimer (ou 'annuler')",
        "value_deleted": "Valeur '{}' supprimée",
        "keep_entity": "Garder cette entité '{}' ?",
        "entity_added": "Entité '{}' ajoutée",
        "entity_rejected": "Entité '{}' rejetée",
        "add_another_entity": "Ajouter une autre entité ?"
    },
    "es": {
        "category_values_title": "💡 Cómo Estructurar Sus Categorías",
        "category_values_heading": "📖 Comprender el Sistema Categoría → Valores",
        "category_values_intro": "Este sistema funciona en DOS NIVELES:",
        "level1_title": "NIVEL 1: CATEGORÍA (la pregunta general)",
        "level1_point1": "• Esta es la DIMENSIÓN GLOBAL que desea analizar",
        "level1_point2": "• Se convierte en la CLAVE JSON en su anotación final",
        "level1_point3": "• Piense: \"¿Qué PREGUNTA estoy haciendo sobre este texto?\"",
        "level1_examples_header": "Ejemplos de categorías (preguntas):",
        "level1_ex1": "• tema → \"¿De qué trata este texto?\"",
        "level1_ex2": "• sentimiento → \"¿Cuál es el tono emocional?\"",
        "level1_ex3": "• partido_político → \"¿Qué partido se menciona?\"",
        "level2_title": "NIVEL 2: VALORES (las respuestas posibles)",
        "level2_point1": "• Estas son las RESPUESTAS ESPECÍFICAS posibles a su pregunta",
        "level2_point2": "• Estos serán los RESULTADOS CONCRETOS de la anotación",
        "level2_point3": "• Piense: \"¿Cuáles son las RESPUESTAS POSIBLES?\"",
        "level2_examples_header": "Ejemplos de valores (respuestas) para 'tema':",
        "level2_ex1": "• medio_ambiente (si el texto habla de ecología)",
        "level2_ex2": "• salud (si el texto habla de atención médica)",
        "level2_ex3": "• economía (si el texto habla de economía)",
        "level2_ex4": "• justicia (si el texto habla del sistema judicial)",
        "complete_example_title": "✨ Ejemplo Completo:",
        "complete_ex1_q": "Pregunta → Categoría: tema",
        "complete_ex1_a": "Respuestas → Valores: medio_ambiente, salud, justicia, economía",
        "complete_ex2_q": "Pregunta → Categoría: sentimiento",
        "complete_ex2_a": "Respuestas → Valores: positivo, negativo, neutral",
        "final_annotation_header": "📝 Su anotación final se verá así:",
        "final_annotation_comment1": "← el VALOR elegido",
        "final_annotation_comment2": "← el VALOR elegido",
        "keep_category": "¿Mantener esta categoría '{}'?",
        "category_added": "Categoría '{}' añadida",
        "category_rejected": "Categoría '{}' rechazada",
        "add_another_category": "¿Agregar otra categoría?",
        "delete_values_prompt": "¿Desea eliminar valores de '{}'?",
        "current_values": "Valores actuales para '{}':",
        "value_to_delete": "Número del valor a eliminar (o 'cancelar')",
        "value_deleted": "Valor '{}' eliminado",
        "keep_entity": "¿Mantener esta entidad '{}'?",
        "entity_added": "Entidad '{}' añadida",
        "entity_rejected": "Entidad '{}' rechazada",
        "add_another_entity": "¿Agregar otra entidad?"
    }
}

# Multilingual prompt templates
PROMPT_TEMPLATES = {
    "en": {
        "you_are": "You are a text annotator specializing in {domain}.",
        "analyze_data": "Analyze the following data: {data_description}",
        "json_format": "You must structure the output in JSON format. Write exclusively in JSON without any explanatory text.",
        "task_categorical": "**Task:** Classify each text according to the categories defined below. The categories must be clear, and appropriate values must be used.",
        "task_entity": "**Task:** Extract and classify the specified entities from each text. Identify all relevant entities and their types according to the categories below.",
        "expected_keys": "**Expected keys:**",
        "instructions": "**Instructions**",
        "strict_follow": "Strictly follow the structure of the keys defined above.",
        "all_keys_present": "Ensure that all keys are present in the JSON, using `null` when necessary.",
        "no_extra_keys": "Do not include keys that are not defined in the expected keys above.",
        "json_only": "Write exclusively the JSON without any additional comments or explanations.",
        "examples_intro": "**Example of an annotation for the text:**",
        "example_json": "**Example of JSON:**",
        "follow_structure": "Follow this structure for each text analyzed. No other comments or additional details beyond the requested JSON structure and the specified categories should be added.",
        "expected_json_keys": "**Expected JSON Keys**",
        "if_condition": "if",
        "null_explanation": "if the text does not explicitly relate to any",
        "indicate_multiple": "Indicate multiple '{cat_name}' values as an array if multiple values are present.",
        "indicate_one": "Indicate only one '{cat_name}' value for each text."
    },
    "fr": {
        "you_are": "Vous êtes un annotateur de texte spécialisé en {domain}.",
        "analyze_data": "Analysez les données suivantes : {data_description}",
        "json_format": "Vous devez structurer la sortie en format JSON. Écrivez exclusivement en JSON sans aucun texte explicatif.",
        "task_categorical": "**Tâche:** Classifiez chaque texte selon les catégories définies ci-dessous. Les catégories doivent être claires et les valeurs appropriées doivent être utilisées.",
        "task_entity": "**Tâche:** Extrayez et classifiez les entités spécifiées de chaque texte. Identifiez toutes les entités pertinentes et leurs types selon les catégories ci-dessous.",
        "expected_keys": "**Clés attendues:**",
        "instructions": "**Instructions**",
        "strict_follow": "Suivez strictement la structure des clés définies ci-dessus.",
        "all_keys_present": "Assurez-vous que toutes les clés sont présentes dans le JSON, en utilisant `null` si nécessaire.",
        "no_extra_keys": "N'incluez pas de clés qui ne sont pas définies dans les clés attendues ci-dessus.",
        "json_only": "Écrivez exclusivement le JSON sans aucun commentaire ou explication supplémentaire.",
        "examples_intro": "**Exemple d'annotation pour le texte :**",
        "example_json": "**Exemple de JSON :**",
        "follow_structure": "Suivez cette structure pour chaque texte analysé. Aucun autre commentaire ou détail supplémentaire au-delà de la structure JSON demandée et des catégories spécifiées ne doit être ajouté.",
        "expected_json_keys": "**Clés JSON Attendues**",
        "if_condition": "si",
        "null_explanation": "si le texte ne se rapporte explicitement à aucun",
        "indicate_multiple": "Indiquez plusieurs valeurs de '{cat_name}' sous forme de tableau si plusieurs valeurs sont présentes.",
        "indicate_one": "Indiquez une seule valeur de '{cat_name}' pour chaque texte."
    },
    "es": {
        "you_are": "Usted es un anotador de texto especializado en {domain}.",
        "analyze_data": "Analice los siguientes datos: {data_description}",
        "json_format": "Debe estructurar la salida en formato JSON. Escriba exclusivamente en JSON sin ningún texto explicativo.",
        "task_categorical": "**Tarea:** Clasifique cada texto según las categorías definidas a continuación. Las categorías deben ser claras y se deben usar valores apropiados.",
        "task_entity": "**Tarea:** Extraiga y clasifique las entidades especificadas de cada texto. Identifique todas las entidades relevantes y sus tipos según las categorías a continuación.",
        "expected_keys": "**Claves esperadas:**",
        "instructions": "**Instrucciones**",
        "strict_follow": "Siga estrictamente la estructura de las claves definidas anteriormente.",
        "all_keys_present": "Asegúrese de que todas las claves estén presentes en el JSON, usando `null` cuando sea necesario.",
        "no_extra_keys": "No incluya claves que no estén definidas en las claves esperadas anteriormente.",
        "json_only": "Escriba exclusivamente el JSON sin comentarios o explicaciones adicionales.",
        "examples_intro": "**Ejemplo de anotación para el texto:**",
        "example_json": "**Ejemplo de JSON:**",
        "follow_structure": "Siga esta estructura para cada texto analizado. No se deben agregar otros comentarios o detalles adicionales más allá de la estructura JSON solicitada y las categorías especificadas.",
        "expected_json_keys": "**Claves JSON Esperadas**",
        "if_condition": "si",
        "null_explanation": "si el texto no se relaciona explícitamente con ningún",
        "indicate_multiple": "Indique múltiples valores de '{cat_name}' como un array si hay múltiples valores presentes.",
        "indicate_one": "Indique solo un valor de '{cat_name}' para cada texto."
    },
    "de": {
        "you_are": "Sie sind ein Textannotator, der sich auf {domain} spezialisiert hat.",
        "analyze_data": "Analysieren Sie die folgenden Daten: {data_description}",
        "json_format": "Sie müssen die Ausgabe im JSON-Format strukturieren. Schreiben Sie ausschließlich JSON ohne erklärenden Text.",
        "task_categorical": "**Aufgabe:** Klassifizieren Sie jeden Text gemäß den unten definierten Kategorien. Die Kategorien müssen klar sein und geeignete Werte müssen verwendet werden.",
        "task_entity": "**Aufgabe:** Extrahieren und klassifizieren Sie die angegebenen Entitäten aus jedem Text. Identifizieren Sie alle relevanten Entitäten und ihre Typen gemäß den unten stehenden Kategorien.",
        "expected_keys": "**Erwartete Schlüssel:**",
        "instructions": "**Anweisungen**",
        "strict_follow": "Befolgen Sie strikt die Struktur der oben definierten Schlüssel.",
        "all_keys_present": "Stellen Sie sicher, dass alle Schlüssel im JSON vorhanden sind und verwenden Sie bei Bedarf `null`.",
        "no_extra_keys": "Fügen Sie keine Schlüssel hinzu, die nicht in den oben erwarteten Schlüsseln definiert sind.",
        "json_only": "Schreiben Sie ausschließlich das JSON ohne zusätzliche Kommentare oder Erklärungen.",
        "examples_intro": "**Beispiel einer Annotation für den Text:**",
        "example_json": "**Beispiel-JSON:**",
        "follow_structure": "Befolgen Sie diese Struktur für jeden analysierten Text. Es sollten keine anderen Kommentare oder zusätzlichen Details über die angeforderte JSON-Struktur und die angegebenen Kategorien hinaus hinzugefügt werden.",
        "expected_json_keys": "**Erwartete JSON-Schlüssel**",
        "if_condition": "wenn",
        "null_explanation": "wenn der Text sich nicht ausdrücklich auf",
        "indicate_multiple": "Geben Sie mehrere '{cat_name}'-Werte als Array an, wenn mehrere Werte vorhanden sind.",
        "indicate_one": "Geben Sie nur einen '{cat_name}'-Wert für jeden Text an."
    },
    "it": {
        "you_are": "Lei è un annotatore di testo specializzato in {domain}.",
        "analyze_data": "Analizzi i seguenti dati: {data_description}",
        "json_format": "Deve strutturare l'output in formato JSON. Scriva esclusivamente in JSON senza alcun testo esplicativo.",
        "task_categorical": "**Compito:** Classifichi ogni testo secondo le categorie definite di seguito. Le categorie devono essere chiare e devono essere utilizzati valori appropriati.",
        "task_entity": "**Compito:** Estragga e classifichi le entità specificate da ogni testo. Identifichi tutte le entità rilevanti e i loro tipi secondo le categorie di seguito.",
        "expected_keys": "**Chiavi attese:**",
        "instructions": "**Istruzioni**",
        "strict_follow": "Segua rigorosamente la struttura delle chiavi definite sopra.",
        "all_keys_present": "Si assicuri che tutte le chiavi siano presenti nel JSON, utilizzando `null` quando necessario.",
        "no_extra_keys": "Non includa chiavi che non sono definite nelle chiavi attese sopra.",
        "json_only": "Scriva esclusivamente il JSON senza commenti o spiegazioni aggiuntive.",
        "examples_intro": "**Esempio di annotazione per il testo:**",
        "example_json": "**Esempio di JSON:**",
        "follow_structure": "Segua questa struttura per ogni testo analizzato. Non devono essere aggiunti altri commenti o dettagli aggiuntivi oltre alla struttura JSON richiesta e alle categorie specificate.",
        "expected_json_keys": "**Chiavi JSON Attese**",
        "if_condition": "se",
        "null_explanation": "se il testo non si riferisce esplicitamente a",
        "indicate_multiple": "Indichi più valori di '{cat_name}' come array se sono presenti più valori.",
        "indicate_one": "Indichi solo un valore di '{cat_name}' per ogni testo."
    },
    "pt": {
        "you_are": "Você é um anotador de texto especializado em {domain}.",
        "analyze_data": "Analise os seguintes dados: {data_description}",
        "json_format": "Você deve estruturar a saída em formato JSON. Escreva exclusivamente em JSON sem nenhum texto explicativo.",
        "task_categorical": "**Tarefa:** Classifique cada texto de acordo com as categorias definidas abaixo. As categorias devem ser claras e valores apropriados devem ser usados.",
        "task_entity": "**Tarefa:** Extraia e classifique as entidades especificadas de cada texto. Identifique todas as entidades relevantes e seus tipos de acordo com as categorias abaixo.",
        "expected_keys": "**Chaves esperadas:**",
        "instructions": "**Instruções**",
        "strict_follow": "Siga rigorosamente a estrutura das chaves definidas acima.",
        "all_keys_present": "Certifique-se de que todas as chaves estejam presentes no JSON, usando `null` quando necessário.",
        "no_extra_keys": "Não inclua chaves que não estão definidas nas chaves esperadas acima.",
        "json_only": "Escreva exclusivamente o JSON sem comentários ou explicações adicionais.",
        "examples_intro": "**Exemplo de anotação para o texto:**",
        "example_json": "**Exemplo de JSON:**",
        "follow_structure": "Siga esta estrutura para cada texto analisado. Nenhum outro comentário ou detalhes adicionais além da estrutura JSON solicitada e das categorias especificadas devem ser adicionados.",
        "expected_json_keys": "**Chaves JSON Esperadas**",
        "if_condition": "se",
        "null_explanation": "se o texto não se relaciona explicitamente com",
        "indicate_multiple": "Indique múltiplos valores de '{cat_name}' como um array se vários valores estiverem presentes.",
        "indicate_one": "Indique apenas um valor de '{cat_name}' para cada texto."
    }
}


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
        self.ui_language = "en"  # UI language, set during wizard run

    def _t(self, key: str) -> str:
        """Get translation for UI text based on selected language"""
        lang = self.ui_language
        if lang not in UI_TRANSLATIONS:
            lang = "en"  # Fallback to English
        return UI_TRANSLATIONS[lang].get(key, UI_TRANSLATIONS["en"].get(key, key))

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

        # Step 2b: Prompt language (NEW!)
        language = self._get_prompt_language()
        self.ui_language = language  # Set UI language for all subsequent interactions

        # Step 3: Annotation type
        annotation_type = self._get_annotation_type()

        # Step 4: Define categories
        categories = self._define_categories(annotation_type, language)

        # Build preliminary specification (needed for example generation)
        self.spec = PromptSpecification(
            project_description=project_desc,
            data_description=data_desc,
            annotation_type=annotation_type,
            categories=categories,
            examples=[],  # Will be filled next
            language=language
        )

        # Step 5: Generate examples (optional)
        examples = self._generate_examples(categories)

        # Update spec with examples
        self.spec.examples = examples

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

    def _get_prompt_language(self) -> str:
        """Ask user to choose the language for the prompt"""
        if not self.console:
            return "en"

        self.console.print(Panel(
            "[bold]Step 2b: Prompt Language[/bold]\n\n"
            "Choose the language for your annotation prompt.\n\n"
            "[yellow]💡 Important:[/yellow] LLMs generally perform [bold]better with English prompts[/bold],\n"
            "but you can use any language. The entire prompt will be generated\n"
            "in your chosen language.",
            border_style="green"
        ))

        languages = {
            "en": "🇬🇧 English (Recommended - best LLM performance)",
            "fr": "🇫🇷 Français",
            "es": "🇪🇸 Español",
            "de": "🇩🇪 Deutsch",
            "it": "🇮🇹 Italiano",
            "pt": "🇵🇹 Português"
        }

        self.console.print("\n[bold]Available languages:[/bold]")
        for code, name in languages.items():
            self.console.print(f"  [cyan]{code}[/cyan] - {name}")

        lang_choice = Prompt.ask(
            "\n[cyan]Select prompt language[/cyan]",
            choices=list(languages.keys()),
            default="en"
        )

        display_name = languages.get(lang_choice, "English")
        self.console.print(f"\n[green]✓[/green] Prompt language: {display_name}\n")
        return lang_choice

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

        # Detailed explanation table
        table = Table(show_header=True, box=box.ROUNDED, title="📚 Annotation Strategies Explained", title_style="bold cyan")
        table.add_column("Type", style="cyan bold", width=28)
        table.add_column("What it does", style="white", width=40)
        table.add_column("Example Sentences & Output", style="green", width=60)

        # Named Entity Extraction
        entity_examples = (
            "[dim]Example sentence:[/dim]\n"
            "[italic]\"Justin Trudeau announced new climate measures\n"
            "in Toronto on Friday.\"[/italic]\n\n"
            "[dim]✅ What we EXTRACT (multiple entities from ONE text):[/dim]\n"
            "• [cyan]person:[/cyan] Justin Trudeau\n"
            "• [cyan]role:[/cyan] Prime Minister\n"
            "• [cyan]action:[/cyan] announced\n"
            "• [cyan]topic:[/cyan] climate measures\n"
            "• [cyan]location:[/cyan] Toronto\n"
            "• [cyan]date:[/cyan] Friday\n\n"
            "[yellow]→ One text = [bold]multiple extractions[/bold][/yellow]"
        )

        table.add_row(
            "1️⃣  Named Entity\n    Extraction",
            "[bold]IDENTIFY and EXTRACT[/bold]\nspecific elements (entities)\nmentioned in the text\n\n"
            "[dim]Focus: WHO, WHAT, WHERE, WHEN[/dim]",
            entity_examples
        )

        # Categorical Annotation
        categorical_examples = (
            "[dim]Example sentence:[/dim]\n"
            "[italic]\"The government announces an ambitious climate plan.\"[/italic]\n\n"
            "[dim]✅ How we CLASSIFY (whole text meaning):[/dim]\n"
            "• [cyan]theme:[/cyan] environment\n"
            "• [cyan]sentiment:[/cyan] positive\n"
            "• [cyan]party:[/cyan] Liberal\n\n"
            "[yellow]→ One text = [bold]one classification[/bold] per category[/yellow]"
        )

        table.add_row(
            "2️⃣  Categorical\n    Annotation",
            "[bold]CLASSIFY the MEANING[/bold]\nof the entire text into\npredefined categories\n\n"
            "[dim]Focus: TOPIC, TONE, MEANING[/dim]",
            categorical_examples
        )

        self.console.print(table)
        self.console.print()

        # Additional guidance
        self.console.print(Panel(
            "[bold yellow]💡 Which one to choose?[/bold yellow]\n\n"
            "• [cyan]Named Entity Extraction[/cyan] → When you want to [underline]identify WHO/WHAT[/underline] is mentioned\n"
            "  [dim]Example uses: Extract all person names, organizations, locations, dates[/dim]\n\n"
            "• [cyan]Categorical Annotation[/cyan] → When you want to [underline]classify the MEANING/TOPIC[/underline]\n"
            "  [dim]Example uses: Is this about health? Is the tone positive? Which political party?[/dim]",
            border_style="yellow",
            title="[bold]Decision Guide[/bold]"
        ))
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

    def _define_categories(self, annotation_type: str, target_language: str = "en") -> List[AnnotationCategory]:
        """Guide user through defining annotation categories"""
        categories = []

        if annotation_type == "entity_extraction":
            # Only entity extraction
            categories = self._define_entity_categories(target_language=target_language)
        else:
            # Start with categorical annotation
            categories = self._define_categorical_categories(target_language=target_language)

            # After categorical, offer to add NER categories
            if categories:  # Only if they added some categorical categories
                if self._offer_additional_ner_categories():
                    ner_categories = self._define_entity_categories(is_additional=True, target_language=target_language)
                    categories.extend(ner_categories)

        return categories

    def _offer_additional_ner_categories(self) -> bool:
        """Ask user if they want to add NER categories after categorical"""
        if not self.console:
            return False

        self.console.print("\n" + "="*80 + "\n")

        self.console.print(Panel(
            "[bold cyan]📎 Additional Option: Named Entity Recognition (NER)[/bold cyan]\n\n"
            "You've defined your [bold]categorical annotations[/bold] (theme, sentiment, etc.).\n\n"
            "[bold yellow]Would you also like to extract specific entities?[/bold yellow]\n\n"
            "[bold]What is NER (Named Entity Recognition)?[/bold]\n"
            "While categorical annotation classifies the [underline]whole text[/underline],\n"
            "NER [underline]extracts specific elements[/underline] mentioned in the text.\n\n"
            "[bold green]Example combining both:[/bold green]\n\n"
            "[dim]Text:[/dim] [italic]\"Justin Trudeau announces climate plan in Toronto.\"[/italic]\n\n"
            "[dim]Categorical (what you just defined):[/dim]\n"
            "• theme: environment\n"
            "• sentiment: positive\n\n"
            "[dim]NER (what you can add now):[/dim]\n"
            "• person: Justin Trudeau\n"
            "• location: Toronto\n"
            "• topic: climate plan\n\n"
            "[bold]Benefits of adding NER:[/bold]\n"
            "• Extract names, organizations, locations automatically\n"
            "• Track which actors are mentioned in which contexts\n"
            "• Richer analysis combining classification + extraction\n\n"
            "[yellow]💡 Common use case:[/yellow] Classify political texts by theme,\n"
            "then extract all politicians, parties, and locations mentioned.",
            border_style="cyan",
            title="[bold]🎯 Combine Categorical + NER[/bold]"
        ))

        return Confirm.ask(
            "\n[cyan]Add Named Entity Recognition categories?[/cyan]",
            default=False
        )

    def _define_entity_categories(self, is_additional: bool = False, target_language: str = "en") -> List[AnnotationCategory]:
        """Define categories for entity extraction

        Args:
            is_additional: If True, this is being called after categorical annotation
            target_language: Target language for the prompt (for translation)
        """
        if not self.console:
            return []

        if is_additional:
            self.console.print(Panel(
                "[bold]Step 4b: Define Named Entity Categories[/bold]\n\n"
                "Now define the types of entities to extract from your texts.\n"
                "[dim]Example: persons, organizations, locations, dates, policy_topics, etc.[/dim]",
                border_style="green"
            ))
        else:
            self.console.print(Panel(
                "[bold]Step 4: Define Entity Categories[/bold]\n\n"
                "You will now define the types of entities to extract.\n"
                "[dim]Example: persons, organizations, policy_topics, etc.[/dim]",
                border_style="green"
            ))

        # Ask about AI assistance ONCE at the beginning
        use_ai_for_entities = False
        if self.llm_client:
            use_ai_for_entities = Confirm.ask(
                "[cyan]🤖 Do you want AI assistance for creating your prompt (wizard mode)?[/cyan]",
                default=True
            )
            if use_ai_for_entities:
                self.console.print("[green]✓[/green] AI will help generate definitions for entity sub-types\n")
            else:
                self.console.print("[yellow]○[/yellow] You will write all definitions manually\n")

        # Explain what comes next
        self.console.print(Panel(
            "[bold cyan]What you'll do next:[/bold cyan]\n\n"
            "For each entity category, you will:\n"
            "1️⃣  Provide a [bold]short entity name[/bold] (e.g., 'persons', 'organizations')\n"
            "2️⃣  Provide a [bold]detailed description[/bold] of what to extract\n"
            "3️⃣  Optionally define [bold]sub-types[/bold] (e.g., politician, scientist)\n"
            "4️⃣  For each sub-type, provide a [bold]definition[/bold]" +
            (" [dim](AI-assisted)[/dim]" if use_ai_for_entities else "") + "\n\n"
            "[dim]Entities are extracted elements (WHO, WHAT, WHERE) mentioned in texts.[/dim]",
            border_style="blue"
        ))
        self.console.print()

        categories = []
        category_count = 0

        while True:
            category_count += 1
            self.console.print(f"\n[bold cyan]━━━ Entity Category #{category_count} ━━━[/bold cyan]")

            # Show naming guidance panel for first category
            if category_count == 1:
                self.console.print(Panel(
                    "[bold yellow]⚠️  Important: Entity Category Naming[/bold yellow]\n\n"
                    "The category [bold]name[/bold] should be a [bold]SHORT keyword[/bold] (1-2 words):\n"
                    "• It becomes the JSON key in your output\n"
                    "• Keep it simple and concise\n"
                    "• The detailed description comes in the NEXT field\n\n"
                    "[bold green]✅ GOOD examples:[/bold green]\n"
                    "• 'persons'\n"
                    "• 'organizations'\n"
                    "• 'locations'\n"
                    "• 'policy_topics'\n\n"
                    "[bold red]❌ BAD examples:[/bold red]\n"
                    "• 'persons_and_their_roles_in_government' [dim](too long - use 'persons')[/dim]\n"
                    "• 'organizations mentioned in the text' [dim](description - use 'organizations')[/dim]\n"
                    "• 'all locations and places' [dim](explanation - use 'locations')[/dim]",
                    border_style="yellow"
                ))
                self.console.print()

            # Get category name (JSON key)
            cat_name = Prompt.ask(
                "[cyan]Entity category name - SHORT keyword (e.g., 'persons', 'organizations')[/cyan]"
            )

            if not cat_name.strip():
                break

            # Sanitize name (accept Unicode letters for accents)
            cat_name = re.sub(r'[^\w]', '_', cat_name.lower(), flags=re.UNICODE)

            # Translate category name to target language if needed
            if self.llm_client:
                translated_name = self._translate_to_target_language(cat_name, target_language)
                if translated_name != cat_name:
                    self.console.print(f"[dim]Translating '{cat_name}' → '{translated_name}'[/dim]")
                    cat_name = translated_name
                    # Re-sanitize after translation to ensure consistency
                    cat_name = re.sub(r'[^\w]', '_', cat_name.lower(), flags=re.UNICODE)

            # Warn if name seems too long
            if len(cat_name) > 30:
                self.console.print(
                    f"\n[yellow]⚠️  Warning: '{cat_name}' seems long for an entity category name.[/yellow]"
                )
                if not Confirm.ask("[cyan]Continue with this name?[/cyan]", default=False):
                    continue

            # Get category description
            cat_desc = Prompt.ask(
                "[cyan]Detailed description of this entity category (explain what to extract)[/cyan]",
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
                # With sub-types, this becomes a CLASSIFICATION category (not extraction)
                self.console.print(Panel(
                    f"[bold cyan]Entity Type Classification:[/bold cyan]\n\n"
                    f"By choosing sub-types, '[cyan]{cat_name}[/cyan]' becomes a [bold]classification category[/bold].\n\n"
                    f"You will [bold]classify the TYPE[/bold] of {cat_name} mentioned (not extract names).\n\n"
                    f"[bold green]Example:[/bold green]\n"
                    f"Text: \"The minister announced reforms.\"\n"
                    f"JSON: \"{cat_name}\": \"politician\"\n\n"
                    f"[dim]Define the possible types (e.g., 'politician', 'scientist', 'activist').[/dim]",
                    border_style="cyan"
                ))

                # Define subtypes - these become the VALUES for this category
                values, value_defs = self._define_category_values(
                    cat_name,
                    f"{cat_desc}. Types of {cat_name} to classify",
                    use_ai_for_entities,
                    target_language,
                    is_entity_type=True  # This is for entity type classification
                )

                # Generate enhanced description that integrates the user's description + types
                if use_ai_for_entities and self.llm_client:
                    types_list = ", ".join([f"'{v}'" for v in values if v != 'null'])
                    enhanced_prompt = f"""Category: {cat_name}
User Description: {cat_desc}
Possible Types: {types_list}

Generate a clear, concise description (1-2 sentences) for this classification category that:
1. Incorporates the user's description context
2. Explains it classifies the TYPE of {cat_name}
3. Lists the possible types
4. Is written in English for the annotation prompt

Description:"""

                    try:
                        enhanced_desc = self.llm_client.generate(enhanced_prompt, temperature=0.3, max_tokens=150)
                        if enhanced_desc:
                            # Clean up
                            enhanced_desc = enhanced_desc.strip()
                            cat_desc = enhanced_desc
                        else:
                            # Fallback: translate if needed
                            translated_desc = self._translate_to_english(cat_desc)
                            types_str = ", ".join([f'"{v}"' for v in values if v != 'null'])
                            cat_desc = f"Classify the type of {cat_name} mentioned. Context: {translated_desc}. Possible types: {types_str}"
                    except:
                        # Fallback: translate if needed
                        translated_desc = self._translate_to_english(cat_desc)
                        types_str = ", ".join([f'"{v}"' for v in values if v != 'null'])
                        cat_desc = f"Classify the type of {cat_name} mentioned. Context: {translated_desc}. Possible types: {types_str}"
                else:
                    # No AI available: translate manually if needed
                    translated_desc = self._translate_to_english(cat_desc) if self.llm_client else cat_desc
                    types_str = ", ".join([f'"{v}"' for v in values if v != 'null'])
                    cat_desc = f"Classify the type of {cat_name} mentioned. Context: {translated_desc}. Possible types: {types_str}"
            else:
                # No sub-types - offer AI enhancement of the entity description
                if use_ai_for_entities and self.llm_client:
                    self.console.print(f"\n[dim]AI will enhance the description for '[cyan]{cat_name}[/cyan]'...[/dim]")

                    # Generate enhanced description with LLM
                    enhanced_desc = self._generate_entity_description_with_llm(cat_name, cat_desc)

                    self.console.print(Panel(
                        f"[bold]AI-Enhanced Description:[/bold]\n\n{enhanced_desc}",
                        border_style="green",
                        title=f"Enhanced Description for '{cat_name}'"
                    ))

                    # Ask if user wants to use it or modify
                    if Confirm.ask("[cyan]Accept this enhanced description?[/cyan]", default=True):
                        cat_desc = enhanced_desc
                    else:
                        # Allow manual edit or refinement
                        self.console.print("\n[bold]What would you like to do?[/bold]")
                        self.console.print("1. ✍️  Edit manually")
                        self.console.print("2. 🔄 Regenerate with additional context")
                        self.console.print("3. ❌ Keep original description")

                        refine_choice = Prompt.ask(
                            "\n[cyan]Your choice[/cyan]",
                            choices=["1", "2", "3"],
                            default="1"
                        )

                        if refine_choice == "1":  # Manual edit
                            cat_desc = Prompt.ask(
                                "[cyan]Enter your custom description[/cyan]",
                                default=enhanced_desc
                            )
                        elif refine_choice == "2":  # Regenerate with context
                            additional_context = Prompt.ask(
                                "[cyan]Provide additional context to guide AI[/cyan]",
                                default=""
                            )
                            enhanced_desc_v2 = f"{cat_desc}. {additional_context}" if additional_context else cat_desc
                            cat_desc = self._generate_entity_description_with_llm(cat_name, enhanced_desc_v2)

                            self.console.print(Panel(
                                f"[bold]Regenerated Description:[/bold]\n\n{cat_desc}",
                                border_style="green"
                            ))

                            if not Confirm.ask("[cyan]Accept this regenerated description?[/cyan]", default=True):
                                cat_desc = Prompt.ask(
                                    "[cyan]Enter your custom description[/cyan]",
                                    default=cat_desc
                                )
                        # elif refine_choice == "3": keep original cat_desc

            # Create entity category
            # If has_subtypes, it's a classification category; otherwise it's free-form extraction
            entity_cat = AnnotationCategory(
                name=cat_name,
                description=cat_desc,
                category_type="categorical" if has_subtypes else "entity",
                values=values if has_subtypes else [],
                value_definitions=value_defs if has_subtypes else {},
                allows_multiple=False if has_subtypes else True,
                allows_null=True,
                parent_category=None  # No parent category in single-key structure
            )

            # Confirm keeping this category
            category_label = "classification category" if has_subtypes else "entity"
            self.console.print(f"\n[bold green]✓ {category_label.capitalize()} '{cat_name}' configured[/bold green]")
            if Confirm.ask(f"[cyan]Keep this {category_label} '{cat_name}'?[/cyan]", default=True):
                categories.append(entity_cat)
                self.console.print(f"[green]✓[/green] {category_label.capitalize()} '{cat_name}' added\n")
            else:
                self.console.print(f"[yellow]⊘[/yellow] {category_label.capitalize()} '{cat_name}' rejected\n")

            # Ask if user wants to add more categories
            if not Confirm.ask("[cyan]Add another entity?[/cyan]", default=False):
                break

        return categories

    def _define_categorical_categories(self, target_language: str = "en") -> List[AnnotationCategory]:
        """Define categories for categorical annotation"""
        if not self.console:
            return []

        self.console.print(Panel(
            "[bold]Step 4: Define Annotation Categories[/bold]\n\n"
            "You will now define the classification categories.\n"
            "[dim]Example: theme → environment/health/economy, sentiment → positive/negative[/dim]",
            border_style="green"
        ))

        # Visual schema explanation for Category → Values system
        explanation_text = (
            "[bold cyan]📖 Understanding the Category → Values System[/bold cyan]\n\n"
            "[bold]Think of it like a questionnaire:[/bold]\n\n"
            "[bold yellow]CATEGORY = The Question[/bold yellow]\n"
            "  What dimension am I analyzing?\n"
            "  Example: [cyan]theme[/cyan] (becomes the JSON key)\n\n"
            "[bold yellow]VALUES = Possible Answers[/bold yellow]\n"
            "  [cyan]theme[/cyan] can be:\n"
            "    • [green]environment[/green] - if text discusses ecology\n"
            "    • [green]health[/green] - if text discusses healthcare  \n"
            "    • [green]economy[/green] - if text discusses economics\n"
            "    • [green]justice[/green] - if text discusses legal system\n\n"
            "[bold green]📝 Final JSON output:[/bold green]\n"
            "{\n"
            '  "[cyan]theme[/cyan]": "[green]environment[/green]",\n'
            '  "[cyan]sentiment[/cyan]": "[green]positive[/green]"\n'
            "}\n\n"
            "[bold]💡 The 4 Steps:[/bold]\n"
            "1. Choose a CATEGORY name (short keyword like 'theme')\n"
            "2. Describe what the category classifies\n"
            "3. Define possible VALUES (like 'environment', 'health')\n"
            "4. Define when to use each value"
        )

        self.console.print(Panel(
            explanation_text,
            border_style="cyan",
            title="[bold]💡 How to Structure Your Categories[/bold]"
        ))
        self.console.print()

        # Ask about AI assistance ONCE at the beginning
        use_ai_for_categories = False
        if self.llm_client:
            use_ai_for_categories = Confirm.ask(
                "[cyan]🤖 Do you want AI assistance to generate category and value definitions?[/cyan]",
                default=True
            )
            if use_ai_for_categories:
                self.console.print("[green]✓[/green] AI will help generate definitions for categories and values\n")
            else:
                self.console.print("[yellow]○[/yellow] You will write all definitions manually\n")

        # Explain what comes next
        self.console.print(Panel(
            "[bold cyan]What you'll do next:[/bold cyan]\n\n"
            "For each category, you will:\n"
            "1️⃣  Provide a [bold]short category name[/bold] (e.g., 'theme', 'sentiment')\n"
            "2️⃣  Provide a [bold]detailed description[/bold] of what it classifies\n"
            "3️⃣  Define [bold]possible values[/bold] (e.g., 'environment', 'health')\n"
            "4️⃣  For each value, provide a [bold]definition[/bold]" +
            (" [dim](AI-assisted)[/dim]" if use_ai_for_categories else "") + "\n\n"
            "[dim]You can add as many categories as needed for your research.[/dim]",
            border_style="blue"
        ))
        self.console.print()

        categories = []
        category_count = 0

        while True:
            category_count += 1
            self.console.print(f"\n[bold cyan]━━━ Category #{category_count} ━━━[/bold cyan]")

            # Show naming guidance panel for first category
            if category_count == 1:
                self.console.print(Panel(
                    "[bold yellow]⚠️  Important: Category Naming[/bold yellow]\n\n"
                    "The category [bold]name[/bold] should be a [bold]SHORT keyword[/bold] (1-2 words):\n"
                    "• It becomes the JSON key in your output\n"
                    "• Keep it simple and concise\n"
                    "• The detailed explanation comes in the NEXT field\n\n"
                    "[bold green]✅ GOOD examples:[/bold green]\n"
                    "• 'theme'\n"
                    "• 'sentiment'\n"
                    "• 'party'\n"
                    "• 'actor'\n\n"
                    "[bold red]❌ BAD examples:[/bold red]\n"
                    "• 'main_topic_discussed_in_text' [dim](too long - use 'theme' or 'topic')[/dim]\n"
                    "• 'positive or negative sentiment' [dim](description - use 'sentiment')[/dim]\n"
                    "• 'political party mentioned' [dim](explanation - use 'party')[/dim]",
                    border_style="yellow"
                ))
                self.console.print()

            # Get category name (JSON key)
            cat_name = Prompt.ask(
                "[cyan]Category name - SHORT keyword (e.g., 'theme', 'sentiment', 'party')[/cyan]"
            )

            if not cat_name.strip():
                break

            # Sanitize name (accept Unicode letters for accents)
            cat_name = re.sub(r'[^\w]', '_', cat_name.lower(), flags=re.UNICODE)

            # Translate category name to target language if needed
            if self.llm_client:
                translated_name = self._translate_to_target_language(cat_name, target_language)
                if translated_name != cat_name:
                    self.console.print(f"[dim]Translating '{cat_name}' → '{translated_name}'[/dim]")
                    cat_name = translated_name
                    # Re-sanitize after translation to ensure consistency
                    cat_name = re.sub(r'[^\w]', '_', cat_name.lower(), flags=re.UNICODE)

            # Warn if name seems too long
            if len(cat_name) > 30:
                self.console.print(
                    f"\n[yellow]⚠️  Warning: '{cat_name}' seems long for a category name.[/yellow]"
                )
                if not Confirm.ask("[cyan]Continue with this name?[/cyan]", default=False):
                    continue

            # Get general description
            cat_desc = Prompt.ask(
                "[cyan]Detailed description of this category (explain what it classifies)[/cyan]",
                default=f"Classification by {cat_name}"
            )

            # Ask if this is an open-ended category or has specific values
            self.console.print(Panel(
                "[bold cyan]Category Type:[/bold cyan]\n\n"
                "[bold]Two types of categories:[/bold]\n\n"
                "1️⃣  [cyan]Closed set[/cyan] - Fixed list of possible values\n"
                "   Example: sentiment → positive, negative, neutral\n"
                "   → You define each possible answer\n\n"
                "2️⃣  [cyan]Open-ended[/cyan] - Free-form extraction\n"
                "   Example: themes → extract all themes mentioned (no fixed list)\n"
                "   → LLM extracts based on your description only\n"
                "   → More flexible, but less controlled",
                border_style="cyan"
            ))

            has_fixed_values = Confirm.ask(
                f"\n[cyan]Does '{cat_name}' have a fixed list of possible values?[/cyan]",
                default=True
            )

            values = []
            value_defs = {}

            if has_fixed_values:
                # Brief explanation of what comes next
                self.console.print(
                    f"\n[dim]Now you'll define the possible values (answers) for '[cyan]{cat_name}[/cyan]'[/dim]"
                )

                # Define possible values
                values, value_defs = self._define_category_values(cat_name, cat_desc, use_ai_for_categories, target_language, is_entity_type=False)
            else:
                # Open-ended category - offer AI enhancement of description
                if use_ai_for_categories and self.llm_client:
                    self.console.print(f"\n[dim]AI will enhance the description for open-ended category '[cyan]{cat_name}[/cyan]'...[/dim]")

                    # Generate enhanced description with LLM
                    enhanced_desc = self._generate_open_category_description_with_llm(cat_name, cat_desc)

                    self.console.print(Panel(
                        f"[bold]AI-Enhanced Description:[/bold]\n\n{enhanced_desc}",
                        border_style="green",
                        title=f"Enhanced Description for '{cat_name}'"
                    ))

                    # Ask if user wants to use it or modify
                    if Confirm.ask("[cyan]Accept this enhanced description?[/cyan]", default=True):
                        cat_desc = enhanced_desc
                    else:
                        # Allow manual edit or refinement
                        self.console.print("\n[bold]What would you like to do?[/bold]")
                        self.console.print("1. ✍️  Edit manually")
                        self.console.print("2. 🔄 Regenerate with additional context")
                        self.console.print("3. ❌ Keep original description")

                        refine_choice = Prompt.ask(
                            "\n[cyan]Your choice[/cyan]",
                            choices=["1", "2", "3"],
                            default="1"
                        )

                        if refine_choice == "1":  # Manual edit
                            cat_desc = Prompt.ask(
                                "[cyan]Enter your custom description[/cyan]",
                                default=enhanced_desc
                            )
                        elif refine_choice == "2":  # Regenerate with context
                            additional_context = Prompt.ask(
                                "[cyan]Provide additional context to guide AI[/cyan]",
                                default=""
                            )
                            enhanced_desc_v2 = f"{cat_desc}. {additional_context}" if additional_context else cat_desc
                            cat_desc = self._generate_open_category_description_with_llm(cat_name, enhanced_desc_v2)

                            self.console.print(Panel(
                                f"[bold]Regenerated Description:[/bold]\n\n{cat_desc}",
                                border_style="green"
                            ))

                            if not Confirm.ask("[cyan]Accept this regenerated description?[/cyan]", default=True):
                                cat_desc = Prompt.ask(
                                    "[cyan]Enter your custom description[/cyan]",
                                    default=cat_desc
                                )
                        # elif refine_choice == "3": keep original cat_desc

                # No fixed values - empty list
                values = []
                value_defs = {}

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

            # Confirm keeping this category
            num_values = len([v for v in values if v != 'null'])
            self.console.print(f"\n[bold green]✓ Category '{cat_name}' configured with {num_values} value(s)[/bold green]")
            if Confirm.ask(f"[cyan]Keep this category '{cat_name}'?[/cyan]", default=True):
                categories.append(category)
                self.console.print(f"[green]✓[/green] Category '{cat_name}' added\n")
            else:
                self.console.print(f"[yellow]⊘[/yellow] Category '{cat_name}' rejected\n")

            # Ask if user wants to add more categories
            if not Confirm.ask("[cyan]Add another category?[/cyan]", default=True):
                break

        return categories

    def _define_category_values(self, cat_name: str, cat_desc: str, use_ai: bool = False, target_language: str = "en", is_entity_type: bool = False) -> Tuple[List[str], Dict[str, str]]:
        """Define values and their definitions for a category

        Args:
            cat_name: Name of the category
            cat_desc: Description of the category
            use_ai: Whether to use AI for generating definitions
            target_language: Target language for translations
            is_entity_type: Whether this is for entity type classification (True) or categorical annotation (False)
        """
        if not self.console:
            return [], {}

        self.console.print(f"\n[bold yellow]Define values for '{cat_name}'[/bold yellow]")
        self.console.print(f"[dim]{cat_desc}[/dim]\n")

        values = []
        value_defs = {}
        value_count = 0

        # Show value naming guidance with context-appropriate examples
        if is_entity_type:
            # Examples for entity type classification (e.g., types of persons)
            good_examples = (
                "[bold green]✅ GOOD value names:[/bold green]\n"
                "• 'politician'\n"
                "• 'scientist'\n"
                "• 'activist'\n"
                "• 'business_leader'"
            )
            bad_examples = (
                "[bold red]❌ BAD value names:[/bold red]\n"
                "• 'professional_politicians_and_career_bureaucrats' [dim](too long - use 'politician')[/dim]\n"
                "• 'people who work in politics' [dim](description not label - use 'politician')[/dim]\n"
                "• 'elected_officials_at_federal_level' [dim](too specific - use 'politician')[/dim]"
            )
            prompt_example = "'politician', 'scientist'"
        else:
            # Examples for categorical annotation (themes, sentiment, etc.)
            good_examples = (
                "[bold green]✅ GOOD value names:[/bold green]\n"
                "• 'environment'\n"
                "• 'positive'\n"
                "• 'liberal'\n"
                "• 'health_care'"
            )
            bad_examples = (
                "[bold red]❌ BAD value names:[/bold red]\n"
                "• 'environmental_protection_and_climate_change' [dim](too long - use 'environment')[/dim]\n"
                "• 'very positive with strong enthusiasm' [dim](description - use 'positive')[/dim]\n"
                "• 'statements made by liberal party' [dim](explanation - use 'liberal')[/dim]"
            )
            prompt_example = "'environment', 'positive'"

        self.console.print(Panel(
            f"[bold cyan]Defining Values for '{cat_name}'[/bold cyan]\n\n"
            "[bold yellow]⚠️  Value Naming Guidelines:[/bold yellow]\n\n"
            "Each value [bold]name[/bold] should be a [bold]SHORT label[/bold] (1-3 words):\n"
            "• It becomes the actual value in your JSON output\n"
            "• Keep it concise and clear\n"
            "• The detailed definition comes in the NEXT step\n\n"
            f"{good_examples}\n\n"
            f"{bad_examples}",
            border_style="cyan"
        ))
        self.console.print()

        # Use the AI setting passed from parent function
        use_llm = use_ai and self.llm_client is not None

        while True:
            value_count += 1

            # Get value name
            value_name = Prompt.ask(
                f"[cyan]Value #{value_count} - SHORT label (e.g., {prompt_example})[/cyan]\n"
                "[dim]Press Enter without input to finish[/dim]"
            )

            if not value_name.strip():
                break

            # Sanitize value (accept Unicode letters for accents)
            value_name = re.sub(r'[^\w]', '_', value_name.lower(), flags=re.UNICODE)

            # Translate value name to target language if needed
            if self.llm_client:
                translated_value = self._translate_to_target_language(value_name, target_language)
                if translated_value != value_name:
                    self.console.print(f"[dim]Translating '{value_name}' → '{translated_value}'[/dim]")
                    value_name = translated_value
                    # Re-sanitize after translation to ensure consistency
                    value_name = re.sub(r'[^\w]', '_', value_name.lower(), flags=re.UNICODE)

            # Warn if value name seems too long
            if len(value_name) > 40:
                self.console.print(
                    f"\n[yellow]⚠️  Warning: '{value_name}' seems long for a value label.[/yellow]"
                )
                if not Confirm.ask("[cyan]Continue with this name?[/cyan]", default=False):
                    continue

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
                    # User rejected - offer options
                    self.console.print("\n[bold yellow]What would you like to do?[/bold yellow]")
                    self.console.print("1. Add context and regenerate with AI")
                    self.console.print("2. Write definition manually")

                    choice = Prompt.ask(
                        "\n[cyan]Your choice[/cyan]",
                        choices=["1", "2"],
                        default="2"
                    )

                    if choice == "1":
                        # Add context and regenerate
                        additional_context = Prompt.ask(
                            "\n[cyan]Additional context for AI (e.g., 'Focus on criminal justice, not social justice')[/cyan]",
                            default=""
                        )

                        # Regenerate with context
                        enhanced_desc = f"{cat_desc}. {additional_context}" if additional_context else cat_desc
                        definition = self._generate_value_definition_with_llm(
                            cat_name, enhanced_desc, value_name
                        )

                        self.console.print(Panel(
                            f"[bold]Regenerated Definition:[/bold]\n\n{definition}",
                            border_style="green"
                        ))

                        if Confirm.ask("[cyan]Accept this new definition?[/cyan]", default=True):
                            value_defs[value_name] = definition
                        else:
                            # Fall back to manual
                            manual_def = Prompt.ask(
                                "[cyan]Enter your definition[/cyan]",
                                default=definition
                            )
                            value_defs[value_name] = manual_def
                    else:
                        # Manual definition
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

        # Show summary and allow deletion
        values, value_defs = self._review_and_edit_values(cat_name, values, value_defs)

        return values, value_defs

    def _review_and_edit_values(self, cat_name: str, values: List[str], value_defs: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
        """Allow user to review and delete values"""
        if not self.console:
            return values, value_defs

        while True:
            # Show summary
            self._show_category_summary(cat_name, values, value_defs)

            # Ask if user wants to delete any values
            if not Confirm.ask(f"\n[cyan]Delete any values from '{cat_name}'?[/cyan]", default=False):
                break

            # Show numbered list
            self.console.print(f"\n[bold yellow]Current values for '{cat_name}':[/bold yellow]")
            for i, val in enumerate(values, 1):
                if val != 'null':  # Don't allow deletion of null
                    self.console.print(f"  {i}. [cyan]{val}[/cyan]")

            # Ask which to delete
            to_delete = Prompt.ask(
                "\n[cyan]Number of value to delete (or 'cancel')[/cyan]",
                default="cancel"
            )

            if to_delete.lower() == 'cancel':
                break

            try:
                idx = int(to_delete) - 1
                if 0 <= idx < len(values) and values[idx] != 'null':
                    deleted_val = values[idx]
                    values.remove(deleted_val)
                    if deleted_val in value_defs:
                        del value_defs[deleted_val]
                    self.console.print(f"[green]✓[/green] Value '{deleted_val}' deleted\n")
                else:
                    self.console.print("[red]Invalid number[/red]\n")
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]\n")

        return values, value_defs

    def _generate_value_definition_with_llm(self, cat_name: str, cat_desc: str, value_name: str) -> str:
        """Use LLM to generate a definition for a category value"""
        # Determine target language
        lang = self.ui_language
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        language_name = lang_names.get(lang, 'English')

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
- Be written ENTIRELY in {language_name}

Output ONLY the definition in {language_name}, without any prefix like "Definition:" or "Définition:".

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

    def _generate_open_category_description_with_llm(self, category_name: str, user_description: str) -> str:
        """Use LLM to generate an enhanced description for an open-ended category"""
        # Determine target language
        lang = self.ui_language
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        language_name = lang_names.get(lang, 'English')

        prompt = f"""You are an expert in social science research methodology and text annotation.

Category: {category_name}
User's Description: {user_description}

This is an OPEN-ENDED category, meaning the LLM will extract values freely rather than choosing from a fixed list.

Generate a precise, clear description for extracting "{category_name}" from texts without predefined values.

The description should:
- Be 1-2 sentences maximum
- Be specific and actionable for annotators performing open-ended extraction
- Clearly define what should be extracted for this category
- Include concrete examples or indicators when possible
- Explain how to identify relevant content
- Be written for annotation guidelines
- Be written ENTIRELY in {language_name}

Output ONLY the enhanced description in {language_name}, without any prefix like "Description:" or "Definition:".

Enhanced Description:"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )

            if response:
                # Clean up response
                description = response.strip()
                # Remove any "Description:" or "Definition:" prefix if present
                description = re.sub(r'^(Description|Definition|Définition):\s*', '', description, flags=re.IGNORECASE)
                return description
            else:
                return user_description

        except Exception as e:
            self.console.print(f"[yellow]⚠️  LLM generation failed: {e}[/yellow]")
            return user_description

    def _generate_entity_description_with_llm(self, entity_name: str, user_description: str) -> str:
        """Use LLM to generate an enhanced description for an entity category"""
        # Determine target language
        lang = self.ui_language
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        language_name = lang_names.get(lang, 'English')

        prompt = f"""You are an expert in social science research methodology and named entity recognition (NER).

Entity Category: {entity_name}
User's Description: {user_description}

Generate a precise, clear description for extracting entities of type "{entity_name}" from texts.

The description should:
- Be 1-2 sentences maximum
- Be specific and actionable for annotators performing named entity extraction
- Clearly define what constitutes an entity of this type
- Include concrete examples or indicators when possible
- Be written for annotation guidelines
- Be written ENTIRELY in {language_name}

Output ONLY the enhanced description in {language_name}, without any prefix like "Description:" or "Definition:".

Enhanced Description:"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )

            if response:
                # Clean up response
                description = response.strip()
                # Remove any "Description:" or "Definition:" prefix if present
                description = re.sub(r'^(Description|Definition|Définition):\s*', '', description, flags=re.IGNORECASE)
                return description
            else:
                return user_description

        except Exception as e:
            self.console.print(f"[yellow]⚠️  LLM generation failed: {e}[/yellow]")
            return user_description

    def _integrate_value_definitions(self, cat: AnnotationCategory, lang: str) -> List[str]:
        """Use AI to create better integrated value definitions"""
        if not self.llm_client:
            return []

        # Get language-specific template
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        target_language = lang_names.get(lang, 'English')

        # Build the values and definitions string
        values_info = []
        for value in cat.values:
            definition = cat.value_definitions.get(value, f"relates to {value}")
            values_info.append(f"- {value}: {definition}")

        values_text = "\n".join(values_info)

        # Get the conditional word (if/si/wenn/etc.) for the target language using LLM
        if_word = self._get_conditional_word_for_language(target_language)

        prompt = f"""You are helping create annotation guidelines in {target_language}.

Category: {cat.name}
Category Description: {cat.description if cat.description else f"Classification by {cat.name}"}

Values and their definitions:
{values_text}

Create grammatically correct value definitions for an annotation prompt that PRESERVE all important information from the original definitions.
Each definition should complete this pattern naturally in {target_language}:
'"{value}" {if_word} [condition that indicates when to use this value]'

Important:
1. Start definitions with lowercase (unless proper noun)
2. Keep definitions clear and complete - preserve key details, examples, policy context, and specific indicators from the original
3. Use natural {target_language} phrasing
4. Include specific subcategories, examples, or indicators mentioned in the original definition
5. Return ONLY the formatted definitions, one per line
6. Maintain the richness and specificity of the original definitions
7. IMPORTANT: Use "{if_word}" as the conditional word in {target_language}

Example transformation for {target_language}:
Original: "Environment encompasses policies related to natural resources (air, water, land, biodiversity), pollution (including climate change, waste management, and toxic substances), and ecological preservation. Annotators should include texts addressing conservation efforts, environmental regulations, sustainability initiatives, or impacts of human activity on natural systems."
Output: "environment" {if_word} the text discusses policies related to natural resources (air, water, land, biodiversity), pollution (including climate change, waste management, toxic substances), ecological preservation, conservation efforts, environmental regulations, sustainability initiatives, or impacts of human activity on natural systems

Generate the definitions in {target_language}:"""

        try:
            response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=1000)
            if response:
                # Parse response into individual definitions
                lines = response.strip().split('\n')
                definitions = []
                for line in lines:
                    line = line.strip()
                    if line and ('"' in line or "'" in line):
                        # Clean up the line and add it
                        definitions.append(line)
                return definitions if definitions else []
            return []
        except:
            return []

    def _get_conditional_word_for_language(self, target_language: str) -> str:
        """Get the conditional word (if/si/wenn/etc.) for the target language using LLM

        Args:
            target_language: Full language name (e.g., 'French', 'Spanish', 'Japanese')

        Returns:
            The conditional word in the target language (e.g., 'si' for French, 'wenn' for German)
        """
        if not self.llm_client:
            # Fallback to template if no LLM
            lang_code = self.spec.language if hasattr(self.spec, 'language') else 'en'
            template = PROMPT_TEMPLATES.get(lang_code, PROMPT_TEMPLATES['en'])
            return template.get("if_condition", "if")

        try:
            prompt = f"""What is the word for "if" (conditional) in {target_language}?

Examples:
- English: if
- French: si
- Spanish: si
- German: wenn
- Italian: se
- Portuguese: se
- Japanese: もし
- Chinese: 如果
- Arabic: إذا
- Russian: если

Return ONLY the single word for "if" in {target_language}, nothing else:"""

            response = self.llm_client.generate(prompt, temperature=0.0, max_tokens=10)
            if response:
                # Clean the response - take only the first word
                word = response.strip().split()[0]
                # Remove any quotes or punctuation
                word = word.strip('"\'.,:;!?')
                return word if word else "if"
            return "if"
        except:
            # Fallback to "if" if translation fails
            return "if"

    def _translate_to_target_language(self, text: str, target_lang: str) -> str:
        """Translate text to target language using LLM if needed"""
        if not self.llm_client:
            return text

        # Skip translation if text is very short
        if len(text) < 3:
            return text

        # Language names for prompts
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }

        target_language = lang_names.get(target_lang, 'English')

        try:
            prompt = f"""Translate the following text to {target_language}. If it's already in {target_language}, return it unchanged.
Important: Return ONLY the translation, nothing else.

Text: {text}

{target_language} translation:"""

            translation = self.llm_client.generate(prompt, temperature=0.3, max_tokens=200)
            if translation:
                return translation.strip()
            return text
        except:
            return text

    def _generate_introduction_with_llm(self, lang: str, template: Dict[str, str]) -> Optional[str]:
        """Generate contextualized prompt introduction using LLM"""

        # Determine language name
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        language_name = lang_names.get(lang, 'English')

        # Determine annotation task type
        task_type = "entity extraction" if self.spec.annotation_type == "entity_extraction" else "categorical classification"

        prompt = f"""You are an expert in prompt engineering for annotation tasks.

Generate a professional, contextualized introduction for an annotation prompt in {language_name}.

Context:
- Domain: {self.spec.domain}
- Project Description: {self.spec.project_description}
- Data Description: {self.spec.data_description}
- Task Type: {task_type}
- Output Language: {language_name}

Requirements:
1. Start with a role statement specifying the annotator specialization (based on domain)
2. Clearly explain the research objective (based on project description)
3. Describe what data will be analyzed (based on data description)
4. Emphasize JSON-only output without explanatory text
5. Write ENTIRELY in {language_name}
6. Be concise (3-4 sentences maximum)
7. Use professional, academic tone appropriate for social science research

Generate the introduction paragraph in {language_name}:"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            )

            if response:
                intro = response.strip()

                # Clean up common prefixes in multiple languages
                prefixes_to_remove = [
                    r'^Voici une introduction possible\s*:\s*',
                    r'^Here is a possible introduction\s*:\s*',
                    r'^Introduction\s*:\s*',
                    r'^Introducción\s*:\s*',
                    r'^Einführung\s*:\s*',
                    r'^Introduzione\s*:\s*',
                    r'^Introdução\s*:\s*'
                ]

                for prefix in prefixes_to_remove:
                    intro = re.sub(prefix, '', intro, flags=re.IGNORECASE)
                intro = intro.strip()

                # Display for review
                if self.console:
                    self.console.print(Panel(
                        f"[bold cyan]🤖 AI-Generated Introduction ({language_name}):[/bold cyan]\n\n{intro}",
                        border_style="cyan"
                    ))

                    if Confirm.ask("[cyan]Use this AI-generated introduction?[/cyan]", default=True):
                        return intro
                    else:
                        if self.console:
                            self.console.print("[yellow]Using template-based introduction instead[/yellow]")
                        return None
                return intro
            return None

        except Exception as e:
            if self.console:
                self.console.print(f"[yellow]⚠️  Introduction generation failed: {e}[/yellow]")
            return None

    def _generate_examples_with_llm(self, categories: List[AnnotationCategory]) -> List[Dict[str, Any]]:
        """Use LLM to generate example annotations"""
        if not self.console:
            return []

        # Check if spec is available
        if not self.spec:
            self.console.print("[yellow]⚠️  Cannot generate AI examples: specification not yet created[/yellow]")
            return []

        self.console.print("\n[cyan]🤖 Generating examples with AI...[/cyan]\n")

        # Build category information for the prompt
        category_info = []
        for cat in categories:
            if cat.category_type == "categorical":
                values_str = ", ".join(cat.values)
                category_info.append(f"- {cat.name}: {cat.description}\n  Possible values: {values_str}")
            else:
                category_info.append(f"- {cat.name}: {cat.description}")

        categories_text = "\n".join(category_info)

        # Ask how many examples to generate
        num_examples = 3  # Default
        if Confirm.ask("[cyan]Generate 3 diverse examples?[/cyan]", default=True):
            num_examples = 3
        else:
            while True:
                try:
                    num_input = Prompt.ask("[cyan]How many examples would you like?[/cyan]", default="3")
                    num_examples = int(num_input)
                    if 1 <= num_examples <= 10:
                        break
                    self.console.print("[yellow]Please enter a number between 1 and 10[/yellow]")
                except ValueError:
                    self.console.print("[yellow]Please enter a valid number[/yellow]")

        # Determine the language for examples (from spec, not UI language)
        dataset_lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        dataset_language_name = lang_names.get(dataset_lang, 'English')

        # Build a comprehensive example showing all categories with non-null values
        all_categories_example = {}
        for cat in categories:
            if cat.values and cat.values != ['null']:
                # Get first non-null value
                non_null_values = [v for v in cat.values if v != 'null']
                if non_null_values:
                    if cat.allows_multiple:
                        # For multiple-value categories, show array with multiple values if possible
                        all_categories_example[cat.name] = non_null_values[:2] if len(non_null_values) >= 2 else non_null_values
                    else:
                        all_categories_example[cat.name] = non_null_values[0]

        all_cats_json = json.dumps(all_categories_example, indent=2)

        prompt = f"""You are an expert in social science research methodology and text annotation.

Project Context: {self.spec.project_description}
Data Description: {self.spec.data_description}

Annotation Categories:
{categories_text}

Generate {num_examples} diverse, realistic example texts that would be annotated using these categories.

CRITICAL: ALL example texts MUST be written in {dataset_language_name}, regardless of the language used in these instructions. This is the language of the actual dataset being annotated.

Requirements:
1. Each example should be realistic and match the data description
2. Examples should cover DIFFERENT scenarios and edge cases
3. Examples should demonstrate the full range of possible category values
4. Keep examples concise (15-50 words)
5. Make examples clear and unambiguous for annotation purposes
6. **ALL example texts MUST be in {dataset_language_name}**

**MANDATORY STRUCTURE:**
- The FIRST example MUST demonstrate ALL categories with non-null values
- This first example should have annotations like this:
{all_cats_json}
- The remaining examples can show edge cases, null values, and other scenarios

For each example, provide:
1. The example text (in {dataset_language_name})
2. The correct annotations for all categories

Output format (JSON):
[
  {{
    "text": "example text here in {dataset_language_name}",
    "annotations": {{
      "category1": "value1",
      "category2": "value2"
    }}
  }},
  ...
]

Generate exactly {num_examples} examples in valid JSON format (with all texts in {dataset_language_name}).
Remember: THE FIRST EXAMPLE MUST HAVE ALL CATEGORIES WITH NON-NULL VALUES:"""

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )

            if response:
                # Try to extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    examples_json = json_match.group(0)
                    examples = json.loads(examples_json)

                    # Display generated examples for review
                    self.console.print(Panel(
                        f"[bold green]✓ Generated {len(examples)} examples[/bold green]",
                        border_style="green"
                    ))

                    for i, ex in enumerate(examples, 1):
                        self.console.print(f"\n[bold cyan]Example {i}:[/bold cyan]")
                        self.console.print(f"[dim]Text:[/dim] {ex['text']}")
                        self.console.print("[dim]Annotations:[/dim]")
                        for key, value in ex['annotations'].items():
                            self.console.print(f"  • {key}: [green]{value}[/green]")

                    # Ask for confirmation
                    if Confirm.ask("\n[cyan]Accept these AI-generated examples?[/cyan]", default=True):
                        return examples
                    else:
                        self.console.print("[yellow]Examples rejected. You can create them manually.[/yellow]")
                        return []
                else:
                    self.console.print("[yellow]⚠️  Could not parse JSON from LLM response[/yellow]")
                    return []
            else:
                self.console.print("[yellow]⚠️  LLM returned empty response[/yellow]")
                return []

        except json.JSONDecodeError as e:
            self.console.print(f"[red]❌ Failed to parse JSON: {e}[/red]")
            self.console.print(f"[dim]Response was: {response[:200]}...[/dim]")
            return []
        except Exception as e:
            self.console.print(f"[red]❌ LLM generation failed: {e}[/red]")
            return []

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
            "You can add example annotations to help guide the LLM.\n\n"
            "[bold yellow]💡 Why examples matter:[/bold yellow]\n"
            "• Examples significantly [bold]improve annotation quality[/bold] (up to 40% accuracy gain)\n"
            "• They help the LLM understand [bold]edge cases[/bold] and ambiguous situations\n"
            "• Good examples ensure [bold]consistency[/bold] across thousands of annotations\n"
            "• They clarify [bold]your expectations[/bold] better than definitions alone\n\n"
            "[dim]Recommended: 2-5 diverse examples covering different scenarios[/dim]",
            border_style="green"
        ))

        # Offer three options: AI-generated, manual, or skip
        self.console.print("\n[bold cyan]How would you like to create examples?[/bold cyan]\n")
        example_options = {
            "1": "🤖 AI-generated (LLM creates examples based on your categories)",
            "2": "✍️  Manual (I'll write examples myself)",
            "3": "⏭️  Skip (no examples)"
        }

        for key, desc in example_options.items():
            self.console.print(f"  {key}. {desc}")

        example_mode = Prompt.ask(
            "\n[cyan]Select option[/cyan]",
            choices=["1", "2", "3"],
            default="1" if self.llm_client else "2"
        )

        if example_mode == "3":
            self.console.print("[yellow]⚠️  Skipping examples - annotation quality may be lower[/yellow]\n")
            return []

        examples = []

        if example_mode == "1":
            # AI-generated examples
            if not self.llm_client:
                self.console.print("[red]❌ AI generation requires an LLM client. Falling back to manual mode.[/red]\n")
                example_mode = "2"
            else:
                examples = self._generate_examples_with_llm(categories)
                if not examples:
                    self.console.print("[yellow]⚠️  AI generation failed. Switch to manual mode?[/yellow]")
                    if Confirm.ask("[cyan]Continue with manual examples?[/cyan]", default=True):
                        example_mode = "2"
                    else:
                        return []

        if example_mode == "2":
            # Manual examples
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
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        # If LLM client available, generate contextualized introduction
        if self.llm_client:
            intro = self._generate_introduction_with_llm(lang, template)
            if intro:
                return intro

        # Fallback to template-based introduction
        you_are = template['you_are'].format(domain=self.spec.domain)
        analyze_data = template['analyze_data'].format(data_description=self.spec.data_description)
        json_format = template['json_format']

        return (
            f"{you_are} "
            f"{self.spec.project_description}\n\n"
            f"{analyze_data}\n\n"
            f"{json_format}"
        )

    def _build_task_description(self) -> str:
        """Build task description"""
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        if self.spec.annotation_type == "entity_extraction":
            return template['task_entity']
        else:
            return template['task_categorical']

    def _build_category_definitions(self) -> str:
        """Build category definitions section"""
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        lines = [template['expected_keys']]

        for cat in self.spec.categories:
            # Build value descriptions with better integration
            value_parts = []

            if cat.values and cat.value_definitions:
                # Use AI to create more natural definition integration if available
                if self.llm_client and len(cat.values) > 1:
                    integrated_defs = self._integrate_value_definitions(cat, lang)
                    if integrated_defs:
                        value_parts = integrated_defs
                    else:
                        # Fallback to original format
                        for value in cat.values:
                            definition = cat.value_definitions.get(value, f"relates to {value}")
                            # Better formatting: make definition lowercase after "if" (except for null)
                            if value != 'null' and definition and definition[0].isupper():
                                definition = definition[0].lower() + definition[1:]
                            value_parts.append(f'"{value}" {template["if_condition"]} {definition}')
                else:
                    # Simple case or no LLM
                    for value in cat.values:
                        definition = cat.value_definitions.get(value, f"relates to {value}")
                        # Better formatting: make definition lowercase after "if" (except for null)
                        if value != 'null' and definition and definition[0].isupper():
                            definition = definition[0].lower() + definition[1:]
                        value_parts.append(f'"{value}" {template["if_condition"]} {definition}')

            # Category line
            multiple_note = " (can be multiple values)" if cat.allows_multiple else ""

            if cat.category_type == "entity":
                # Regular entity extraction (NER) - free-form extraction
                description = cat.description if hasattr(cat, 'description') and cat.description else f"Extract all {cat.name} mentioned in the text"
                cat_line = f'- "{cat.name}"{multiple_note}: {description}.'
            elif value_parts:
                # Categorical annotation with fixed values (including former entity-with-subtypes)
                values_str = ", ".join(value_parts)
                cat_line = f'- "{cat.name}"{multiple_note}: {values_str}.'
            else:
                # Open-ended category without predefined values
                description = cat.description if hasattr(cat, 'description') and cat.description else f"Extract relevant {cat.name} from the text"
                cat_line = f'- "{cat.name}"{multiple_note}: {description}.'

            lines.append(cat_line)

        return "\n".join(lines)

    def _build_instructions(self) -> str:
        """Build instructions section"""
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        instructions = [
            template['instructions'],
            f"- {template['strict_follow']}",
            f"- {template['all_keys_present']}",
            f"- {template['no_extra_keys']}",
            f"- {template['json_only']}"
        ]

        # Add category-specific instructions (translated)
        for cat in self.spec.categories:
            if cat.allows_multiple:
                instructions.append(
                    f"- {template['indicate_multiple'].format(cat_name=cat.name)}"
                )
            else:
                instructions.append(
                    f"- {template['indicate_one'].format(cat_name=cat.name)}"
                )

        return "\n".join(instructions)

    def _build_examples(self) -> str:
        """Build examples section"""
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        lines = []

        for i, example in enumerate(self.spec.examples, 1):
            lines.append(f"{template['examples_intro']}\n")
            lines.append(example["text"])
            lines.append(f"\n{template['example_json']}\n")

            json_str = json.dumps(example["annotations"], indent=2, ensure_ascii=False)
            lines.append(json_str)
            lines.append("")

        lines.append(template['follow_structure'])

        return "\n".join(lines)

    def _build_expected_keys(self) -> str:
        """Build expected JSON keys template with possible values"""
        # Get language template
        lang = self.spec.language if hasattr(self.spec, 'language') else 'en'
        template_lang = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES['en'])

        lines = [template_lang['expected_json_keys']]

        template = {}
        for cat in self.spec.categories:
            if cat.allows_multiple:
                # Show list of possible values for multiple-value categories
                if cat.values:
                    # Include ALL values including 'null' to show complete options
                    template[cat.name] = cat.values
                else:
                    # For entity extraction or open-ended, show empty array
                    template[cat.name] = []
            else:
                # Single value category - show first value (could be any valid value)
                if cat.values:
                    # Show the first value as example (not necessarily 'null')
                    template[cat.name] = cat.values[0] if cat.values else ""
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

        # Display prompt with syntax highlighting - use full width
        syntax = Syntax(prompt_text, "markdown", theme="monokai", line_numbers=True, word_wrap=True)
        self.console.print(Panel(syntax, title="Generated Prompt", border_style="cyan", expand=True))

        # Options
        self.console.print("\n[bold]What would you like to do?[/bold]")
        self.console.print("1. ✅ Accept and use this prompt")
        self.console.print("2. ✏️  Edit the prompt manually")
        self.console.print("3. 🔄 Regenerate with modifications")
        self.console.print("4. 💾 Save prompt to file")
        self.console.print("5. 📄 View full prompt (scrollable)")

        choice = Prompt.ask(
            "\n[cyan]Your choice[/cyan]",
            choices=["1", "2", "3", "4", "5"],
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

        elif choice == "5":
            # View full prompt in pager
            with self.console.pager():
                self.console.print(Syntax(prompt_text, "markdown", theme="monokai", line_numbers=True))
            # After viewing, ask again
            return self._review_and_edit_prompt(prompt_text)

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
