# Benchmark Mode - Implementation Guide

## Vue d'ensemble

Le mode benchmark permet de comparer plusieurs modèles sur des catégories sélectionnées avant l'entraînement complet. Voici l'implémentation complète.

## Architecture

```
1. Question benchmark mode (STEP 2B début)
2. Si activé:
   a. Sélection multiple de modèles (≥2 par langue ou ≥2 multilingues)
   b. Analyse class imbalance des catégories
   c. Sélection des catégories à benchmarker
   d. Exécution du benchmark (quick training 3-5 epochs)
   e. Affichage des résultats et ranking
   f. Choix de la configuration finale
3. Si désactivé: flux normal (sélection d'un seul modèle)
```

## Étape 1: Sélection Multiple de Modèles

### Pour train_by_language (un modèle par langue):

```python
# Pour chaque langue, permettre de sélectionner plusieurs modèles
models_by_language_benchmark = {}  # {lang: [model1, model2, ...]}

for lang in sorted(languages):
    selected_models = []

    while True:
        # Afficher modèles disponibles
        # Permettre sélection
        model = select_model_for_language(lang)
        selected_models.append(model)

        # Demander si ajouter un autre
        if len(selected_models) >= 2:
            add_more = Confirm.ask(f"Add another model for {lang}?", default=False)
            if not add_more:
                break
        else:
            # Force au moins 2 modèles
            continue

    models_by_language_benchmark[lang] = selected_models
```

### Pour multilingual/single model:

```python
selected_models_benchmark = []

while True:
    # Afficher modèles recommandés
    model = select_model()
    selected_models_benchmark.append(model)

    if len(selected_models_benchmark) >= 2:
        add_more = Confirm.ask("Add another model to benchmark?", default=False)
        if not add_more:
            break
```

## Étape 2: Analyse et Sélection des Catégories

```python
from llm_tool.utils.benchmark_utils import (
    analyze_categories_imbalance,
    select_benchmark_categories,
    format_imbalance_summary
)

# Analyser toutes les catégories
imbalance_analysis = analyze_categories_imbalance(
    data=original_dataframe,
    annotation_column='annotation'
)

# Sélectionner 3 catégories représentatives
suggested_categories = select_benchmark_categories(imbalance_analysis)

# Afficher analyse
console.print("\n[bold cyan]📊 Class Imbalance Analysis[/bold cyan]\n")

categories_table = Table(...)
categories_table.add_column("Category")
categories_table.add_column("Profile")
categories_table.add_column("Metrics")

for profile in ['balanced', 'medium', 'imbalanced']:
    for cat in suggested_categories[profile]:
        metrics = imbalance_analysis[cat]
        categories_table.add_row(
            cat,
            profile.capitalize(),
            format_imbalance_summary(metrics)
        )

console.print(categories_table)

# Permettre à l'utilisateur de choisir
console.print("\n[bold]Select categories for benchmark:[/bold]")
console.print("  • Press ENTER to use all 3 suggested categories")
console.print("  • Or enter category names (comma-separated)")
console.print("  • Or enter 'all' to see all available categories\n")

choice = Prompt.ask("Categories", default="suggested")

if choice == "suggested" or choice == "":
    selected_benchmark_categories = [
        cat for cats in suggested_categories.values() for cat in cats
    ]
elif choice == "all":
    # Afficher toutes les catégories avec leurs métriques
    # Permettre sélection multiple
    pass
else:
    selected_benchmark_categories = [c.strip() for c in choice.split(',')]
```

## Étape 3: Exécution du Benchmark

```python
from llm_tool.trainers.model_trainer import ModelTrainer, TrainingConfig

console.print("\n[bold cyan]🚀 Running Benchmark...[/bold cyan]\n")
console.print(f"  • Models: {len(all_models_to_benchmark)}")
console.print(f"  • Categories: {len(selected_benchmark_categories)}")
console.print(f"  • Epochs: 3-5 (quick evaluation)")
console.print(f"  • Estimated time: ~{estimate_time()} minutes\n")

# Créer dataset de benchmark
benchmark_data = create_benchmark_dataset(
    data=original_dataframe,
    annotation_column='annotation',
    selected_categories=selected_benchmark_categories,
    text_column='text'
)

# Entraîner chaque modèle
benchmark_results = {}

for model_id in all_models_to_benchmark:
    console.print(f"\n[yellow]Testing {model_id}...[/yellow]")

    # Configuration légère pour benchmark
    config = TrainingConfig()
    config.num_epochs = 3  # Quick evaluation
    config.batch_size = 16
    config.early_stopping_patience = 2

    trainer = ModelTrainer(config=config)

    # Entraîner
    result = trainer.train({
        'input_file': str(benchmark_data_file),
        'model_name': model_id,
        'num_epochs': 3,
        'text_column': 'text',
        'label_column': 'labels',
        'training_strategy': 'single-label',
        'output_dir': f'benchmark_temp/{model_id}'
    })

    benchmark_results[model_id] = result

    console.print(f"  ✓ F1: {result['best_f1_macro']:.3f}, Acc: {result['accuracy']:.3f}")
```

## Étape 4: Affichage des Résultats

```python
from llm_tool.utils.benchmark_utils import compare_model_results

# Créer tableau de comparaison
comparison_df = compare_model_results(benchmark_results)

console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
console.print("[bold cyan]     📊 BENCHMARK RESULTS      [/bold cyan]")
console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

results_table = Table(...)
results_table.add_column("Rank", style="yellow")
results_table.add_column("Model", style="cyan")
results_table.add_column("F1-Score", style="green")
results_table.add_column("Accuracy", style="green")
results_table.add_column("Training Time", style="blue")

for _, row in comparison_df.iterrows():
    # Ajouter emoji pour top 3
    rank_str = f"🥇 {row['rank']}" if row['rank'] == 1 else f"🥈 {row['rank']}" if row['rank'] == 2 else f"🥉 {row['rank']}" if row['rank'] == 3 else str(row['rank'])

    results_table.add_row(
        rank_str,
        row['model'],
        f"{row['f1_macro']:.3f}",
        f"{row['accuracy']:.3f}",
        f"{row['training_time']:.1f}s"
    )

console.print(results_table)
```

## Étape 5: Choix Final

```python
console.print("\n[bold]🎯 Next Steps:[/bold]\n")
console.print("Based on benchmark results, you can:")
console.print("  [cyan]1.[/cyan] Use top-ranked models (recommended)")
console.print("  [cyan]2.[/cyan] Manually select models")
console.print("  [cyan]3.[/cyan] Stop here (benchmark only)\n")

choice = Prompt.ask(
    "What would you like to do?",
    choices=["1", "2", "3", "top", "manual", "stop"],
    default="1"
)

if choice in ["1", "top"]:
    # Utiliser les meilleurs modèles
    if train_by_language:
        # Sélectionner le meilleur modèle par langue
        final_models_by_language = {}
        for lang in languages:
            # Filtrer les modèles de cette langue
            lang_results = {m: r for m, r in benchmark_results.items() if m in models_by_language_benchmark[lang]}
            # Prendre le meilleur
            best_model = max(lang_results, key=lambda m: lang_results[m]['best_f1_macro'])
            final_models_by_language[lang] = best_model
            console.print(f"  ✓ {lang}: [cyan]{best_model}[/cyan]")

        models_by_language = final_models_by_language
    else:
        # Prendre le meilleur modèle
        best_model = comparison_df.iloc[0]['model']
        model_name = best_model
        console.print(f"  ✓ Selected: [cyan]{best_model}[/cyan]")

elif choice in ["2", "manual"]:
    # Permettre sélection manuelle
    console.print("\n[bold]Manual Selection:[/bold]")

    if train_by_language:
        models_by_language = {}
        for lang in languages:
            # Afficher modèles de cette langue
            lang_models = models_by_language_benchmark[lang]
            console.print(f"\n[yellow]Models for {lang}:[/yellow]")
            for idx, model in enumerate(lang_models, 1):
                result = benchmark_results[model]
                console.print(f"  {idx}. {model} (F1: {result['best_f1_macro']:.3f})")

            choice_idx = IntPrompt.ask(f"Select model for {lang}", default=1)
            models_by_language[lang] = lang_models[choice_idx - 1]
    else:
        console.print("\n[yellow]Available models:[/yellow]")
        for idx, model in enumerate(selected_models_benchmark, 1):
            result = benchmark_results[model]
            console.print(f"  {idx}. {model} (F1: {result['best_f1_macro']:.3f})")

        choice_idx = IntPrompt.ask("Select model", default=1)
        model_name = selected_models_benchmark[choice_idx - 1]

elif choice in ["3", "stop"]:
    console.print("\n[green]✓ Benchmark complete. Exiting without full training.[/green]")
    return None  # Arrêter le flux

# Continuer avec le reste du flux (epochs, reinforced learning, etc.)
```

## Intégration dans le flux existant

Dans `_training_studio_dataset_wizard`, après la question du benchmark mode:

```python
# Ligne 9328 environ
enable_benchmark = Confirm.ask(...)

if enable_benchmark:
    # Appeler la nouvelle méthode
    benchmark_result = self._run_benchmark_mode(
        bundle=bundle,
        languages=languages,
        train_by_language=train_by_language,
        text_length_avg=text_length_avg,
        prefers_long_models=prefers_long_models
    )

    if benchmark_result is None:
        # Utilisateur a choisi de s'arrêter
        return None

    # Extraire les modèles sélectionnés
    model_name = benchmark_result.get('model_name')
    models_by_language = benchmark_result.get('models_by_language')

else:
    # Flux normal existant
    # ...code existant...
```

## Création de la méthode `_run_benchmark_mode`

Ajouter cette méthode dans la classe `AdvancedCLI` (ligne 6360 environ):

```python
def _run_benchmark_mode(
    self,
    bundle: TrainingDataBundle,
    languages: set,
    train_by_language: bool,
    text_length_avg: float,
    prefers_long_models: bool
) -> Optional[Dict[str, Any]]:
    """
    Exécute le mode benchmark complet.

    Returns:
        Dict with 'model_name' and/or 'models_by_language', or None to stop
    """
    # Implémenter toutes les étapes ci-dessus
    # ...
```

## Fichiers modifiés

1. `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/utils/benchmark_utils.py` - ✅ Créé
2. `/Users/antoine/Documents/GitHub/LLM_Tool/llm_tool/cli/advanced_cli.py` - À modifier:
   - Ajouter méthode `_run_benchmark_mode`
   - Modifier flux dans `_training_studio_dataset_wizard` ligne ~9328
   - Appeler conditionnellement la méthode benchmark

## Tests suggérés

1. Tester avec per-language (3 langues, 2 modèles par langue)
2. Tester avec multilingual (3 modèles multilingues)
3. Tester sélection des catégories
4. Vérifier que le benchmark s'exécute correctement
5. Vérifier que les résultats sont bien affichés
6. Tester les 3 options de choix final

## Estimation de temps

- Implémentation complète: ~2-3h
- Tests: ~1h
- Debug: ~1h
Total: 4-5h
