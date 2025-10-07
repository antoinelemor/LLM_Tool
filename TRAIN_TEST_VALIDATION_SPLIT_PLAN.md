# Plan d'Implémentation : Configuration Train/Test/Validation Splits

## Contexte

Actuellement, les ratios de split sont hardcodés dans `TrainingConfig` :
- Train : 70%
- Validation : 20%
- Test : 10%

**Objectif** : Permettre à l'utilisateur de configurer ces ratios dans la Step 6, avec :
- Option pour désactiver le validation set (par défaut)
- Mode uniform : mêmes % pour toutes les clés/valeurs
- Mode custom : % différents par clé ou valeur (selon le mode d'entraînement)

---

## 1. Architecture Proposée

### 1.1 Structure de Configuration

```python
# Stocké dans bundle.metadata['split_config']
split_config = {
    'use_validation': False,  # True/False
    'mode': 'uniform',  # 'uniform' ou 'custom'

    # Mode uniform
    'uniform': {
        'train_ratio': 0.8,
        'test_ratio': 0.2,
        'validation_ratio': 0.0  # Si use_validation=False
    },

    # Mode custom (pour multi-class, one-vs-all, hybrid)
    'custom_by_key': {
        'sentiment': {
            'train_ratio': 0.7,
            'test_ratio': 0.2,
            'validation_ratio': 0.1
        },
        'themes': {
            'train_ratio': 0.8,
            'test_ratio': 0.2,
            'validation_ratio': 0.0
        }
    },

    # Mode custom pour one-vs-all (par valeur)
    'custom_by_value': {
        'sentiment_positive': {
            'train_ratio': 0.7,
            'test_ratio': 0.3,
            'validation_ratio': 0.0
        },
        'sentiment_negative': {
            'train_ratio': 0.8,
            'test_ratio': 0.2,
            'validation_ratio': 0.0
        }
    }
}
```

---

## 2. Modifications dans Step 6 (advanced_cli.py)

### 2.1 Emplacement

Après la section "Training Approach" (ligne ~8680), ajouter une nouvelle section :

```python
# NOUVELLE SECTION : Train/Test/Validation Split Configuration
self.console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
self.console.print("[bold cyan]  STEP 6b:[/bold cyan] [bold white]Data Split Configuration[/bold white]")
self.console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
```

### 2.2 Flux de Questions

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 6b: Data Split Configuration                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │ Use validation set? [Y/n] (default: No) │
        └──────────────────────────────────────┘
                           │
           ┌───────────────┴──────────────┐
           │                              │
           ▼ No                           ▼ Yes
    ┌─────────────┐              ┌──────────────────┐
    │ Train / Test│              │ Train/Valid/Test │
    │  (2 splits) │              │   (3 splits)     │
    └─────────────┘              └──────────────────┘
           │                              │
           └───────────────┬──────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Split mode: [uniform/custom]         │
        │ (default: uniform)                   │
        └──────────────────────────────────────┘
                           │
           ┌───────────────┴──────────────┐
           │                              │
           ▼ Uniform                      ▼ Custom
    ┌─────────────────┐         ┌────────────────────────┐
    │ Same % for all  │         │ Different % per        │
    │ keys/values     │         │ key or value           │
    └─────────────────┘         └────────────────────────┘
           │                              │
           │              ┌───────────────┴────────────────┐
           │              │                                │
           │              ▼ Multi-class                    ▼ One-vs-all
           │    ┌──────────────────┐            ┌──────────────────────┐
           │    │ Choose % per KEY │            │ Choose % per VALUE   │
           │    │ (e.g., sentiment)│            │ (e.g., sentiment_pos)│
           │    └──────────────────┘            └──────────────────────┘
           │              │                                │
           │              ▼ Hybrid/Custom                  │
           │    ┌────────────────────────────────┐         │
           │    │ Mix: some keys, some values    │         │
           │    │ - Multi-class keys → % per key │         │
           │    │ - One-vs-all keys → % per value│         │
           │    └────────────────────────────────┘         │
           │                                               │
           └───────────────┬───────────────────────────────┘
                           ▼
                ┌──────────────────────────┐
                │ Store in bundle.metadata │
                └──────────────────────────┘
```

---

## 3. UI/UX Design

### 3.1 Question Initiale : Validation Set

```python
self.console.print("\n[bold]📊 Data Split Configuration[/bold]\n")
self.console.print("[dim]Configure how your data will be split for training, validation, and testing.[/dim]\n")

# Tableau explicatif
split_table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
split_table.add_column("Set", style="cyan", width=15)
split_table.add_column("Purpose", style="white", width=50)

split_table.add_row(
    "Training",
    "Used to train the model (learn patterns)"
)
split_table.add_row(
    "Validation",
    "Used DURING training to:\n"
    "  • Monitor performance at each epoch\n"
    "  • Select best model checkpoint\n"
    "  • Enable early stopping\n"
    "  • Activate reinforced learning if needed"
)
split_table.add_row(
    "Test",
    "Used AFTER training to:\n"
    "  • Final evaluation on unseen data\n"
    "  • Report unbiased performance metrics"
)

self.console.print(split_table)
self.console.print()

# Question
use_validation = Confirm.ask(
    "[bold yellow]Use a validation set during training?[/bold yellow]",
    default=False
)
```

### 3.2 Mode Uniform (Défaut)

```python
if use_validation:
    self.console.print("\n[bold]📈 Configure Split Ratios (3 sets)[/bold]\n")
    self.console.print("[dim]Ratios must sum to 1.0[/dim]\n")

    train_ratio = FloatPrompt.ask("Training ratio", default=0.7)
    validation_ratio = FloatPrompt.ask("Validation ratio", default=0.2)
    test_ratio = FloatPrompt.ask("Test ratio", default=0.1)

    # Validation
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        self.console.print(f"[red]Error: Ratios sum to {total:.3f}, must be 1.0[/red]")
        # Re-demander ou ajuster automatiquement
else:
    self.console.print("\n[bold]📈 Configure Split Ratios (2 sets)[/bold]\n")
    self.console.print("[dim]Ratios must sum to 1.0[/dim]\n")

    train_ratio = FloatPrompt.ask("Training ratio", default=0.8)
    test_ratio = FloatPrompt.ask("Test ratio", default=0.2)
    validation_ratio = 0.0

    # Validation
    total = train_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        self.console.print(f"[red]Error: Ratios sum to {total:.3f}, must be 1.0[/red]")
```

### 3.3 Mode Custom

#### 3.3.1 Pour Multi-Class (par clé)

```python
self.console.print("\n[bold cyan]⚙️  Custom Split Configuration (per key)[/bold cyan]\n")
self.console.print("[dim]Configure split ratios for each annotation key individually.[/dim]\n")

custom_by_key = {}

# Ratios par défaut (appliqués aux clés non configurées)
self.console.print("[bold]Default ratios[/bold] (applied to keys not configured below):")
default_train = FloatPrompt.ask("  Default train ratio", default=0.7)
default_validation = FloatPrompt.ask("  Default validation ratio", default=0.2) if use_validation else 0.0
default_test = FloatPrompt.ask("  Default test ratio", default=0.1 if use_validation else 0.3)

self.console.print()

# Demander pour chaque clé
for key in keys_to_train:
    self.console.print(f"[bold]{key}[/bold]")

    customize_key = Confirm.ask(
        f"  Customize split for '{key}'?",
        default=False
    )

    if customize_key:
        train = FloatPrompt.ask(f"    Train ratio", default=default_train)

        if use_validation:
            validation = FloatPrompt.ask(f"    Validation ratio", default=default_validation)
            test = FloatPrompt.ask(f"    Test ratio", default=default_test)
        else:
            validation = 0.0
            test = FloatPrompt.ask(f"    Test ratio", default=1.0 - default_train)

        # Validation
        total = train + validation + test
        if abs(total - 1.0) > 0.001:
            self.console.print(f"    [red]Warning: Ratios sum to {total:.3f}, adjusting...[/red]")
            # Auto-adjust
            factor = 1.0 / total
            train *= factor
            validation *= factor
            test *= factor

        custom_by_key[key] = {
            'train_ratio': train,
            'validation_ratio': validation,
            'test_ratio': test
        }

        self.console.print(f"    [green]✓ {key}: {train:.1%} / {validation:.1%} / {test:.1%}[/green]")
    else:
        self.console.print(f"    [dim]Using defaults: {default_train:.1%} / {default_validation:.1%} / {default_test:.1%}[/dim]")

    self.console.print()
```

#### 3.3.2 Pour One-vs-All (par valeur)

```python
self.console.print("\n[bold cyan]⚙️  Custom Split Configuration (per value)[/bold cyan]\n")
self.console.print("[dim]Configure split ratios for each value individually.[/dim]\n")

custom_by_value = {}

# Ratios par défaut
self.console.print("[bold]Default ratios[/bold] (applied to values not configured below):")
default_train = FloatPrompt.ask("  Default train ratio", default=0.7)
default_validation = FloatPrompt.ask("  Default validation ratio", default=0.2) if use_validation else 0.0
default_test = FloatPrompt.ask("  Default test ratio", default=0.1 if use_validation else 0.3)

self.console.print()

# Pour chaque clé, montrer ses valeurs
for key in keys_to_train:
    values = sorted(all_keys_values[key])
    self.console.print(f"[bold cyan]{key}[/bold cyan] ({len(values)} values)")

    customize_key = Confirm.ask(
        f"  Customize splits for values in '{key}'?",
        default=False
    )

    if customize_key:
        # Demander pour chaque valeur
        for value in values:
            full_name = f"{key}_{value}"

            customize_value = Confirm.ask(
                f"    Customize '{value}'?",
                default=False
            )

            if customize_value:
                train = FloatPrompt.ask(f"      Train ratio", default=default_train)

                if use_validation:
                    validation = FloatPrompt.ask(f"      Validation ratio", default=default_validation)
                    test = FloatPrompt.ask(f"      Test ratio", default=default_test)
                else:
                    validation = 0.0
                    test = FloatPrompt.ask(f"      Test ratio", default=1.0 - default_train)

                custom_by_value[full_name] = {
                    'train_ratio': train,
                    'validation_ratio': validation,
                    'test_ratio': test
                }

                self.console.print(f"      [green]✓ {value}: {train:.1%} / {validation:.1%} / {test:.1%}[/green]")
            else:
                self.console.print(f"      [dim]Using defaults for '{value}'[/dim]")

        self.console.print()
```

#### 3.3.3 Pour Hybrid/Custom (mix)

```python
# Pour hybrid/custom, on utilise key_strategies pour déterminer le niveau de customization
custom_by_key = {}
custom_by_value = {}

for key in keys_to_train:
    strategy = key_strategies[key]  # 'multi-class' ou 'one-vs-all'

    if strategy == 'multi-class':
        # Configurer au niveau de la clé
        self.console.print(f"[bold cyan]{key}[/bold cyan] (multi-class)")
        customize = Confirm.ask(f"  Customize split for '{key}'?", default=False)

        if customize:
            # ... (même logique que custom par clé)
            custom_by_key[key] = {...}

    else:  # one-vs-all
        # Configurer au niveau des valeurs
        self.console.print(f"[bold yellow]{key}[/bold yellow] (one-vs-all)")
        customize = Confirm.ask(f"  Customize splits for values in '{key}'?", default=False)

        if customize:
            for value in sorted(all_keys_values[key]):
                # ... (même logique que custom par valeur)
                custom_by_value[f"{key}_{value}"] = {...}
```

---

## 4. Stockage de la Configuration

```python
# À la fin de Step 6, stocker dans metadata
split_config = {
    'use_validation': use_validation,
    'mode': split_mode,  # 'uniform' ou 'custom'
}

if split_mode == 'uniform':
    split_config['uniform'] = {
        'train_ratio': train_ratio,
        'validation_ratio': validation_ratio,
        'test_ratio': test_ratio
    }
else:  # custom
    if custom_by_key:
        split_config['custom_by_key'] = custom_by_key
    if custom_by_value:
        split_config['custom_by_value'] = custom_by_value

    # Toujours stocker les defaults pour les clés/valeurs non configurées
    split_config['defaults'] = {
        'train_ratio': default_train,
        'validation_ratio': default_validation,
        'test_ratio': default_test
    }

# Stocker dans bundle.metadata
if not hasattr(bundle, 'metadata'):
    bundle.metadata = {}
bundle.metadata['split_config'] = split_config
```

---

## 5. Modifications dans les Trainers

### 5.1 ModelTrainer.load_data()

Actuellement utilise `self.config.validation_split` et `self.config.test_split`.

**Nouvelle logique** :

```python
def load_data(self, data_path: str, text_column: str = "text",
              label_column: str = "label",
              split_config: Optional[Dict] = None,
              category_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data for training

    Args:
        split_config: Custom split configuration from bundle.metadata
                      If None, uses default from self.config
        category_name: Name of category/key/value being trained (for custom splits)
    """

    # Determine split ratios
    if split_config:
        mode = split_config.get('mode', 'uniform')

        if mode == 'uniform':
            # Uniform ratios
            ratios = split_config['uniform']
            validation_split = ratios['validation_ratio']
            test_split = ratios['test_ratio']

        else:  # custom
            # Check if this category has custom ratios
            custom_ratios = None

            # Try custom_by_key first
            if 'custom_by_key' in split_config and category_name in split_config['custom_by_key']:
                custom_ratios = split_config['custom_by_key'][category_name]

            # Try custom_by_value
            elif 'custom_by_value' in split_config and category_name in split_config['custom_by_value']:
                custom_ratios = split_config['custom_by_value'][category_name]

            # Use custom or defaults
            if custom_ratios:
                validation_split = custom_ratios['validation_ratio']
                test_split = custom_ratios['test_ratio']
            else:
                defaults = split_config.get('defaults', {})
                validation_split = defaults.get('validation_ratio', 0.2)
                test_split = defaults.get('test_ratio', 0.1)
    else:
        # Use defaults from config
        validation_split = self.config.validation_split
        test_split = self.config.test_split

    # Rest of the function remains the same, but uses calculated splits
    # ...

    # First split: train+val and test
    if test_split > 0:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=self.config.seed
        )
    else:
        # No test set
        X_temp, y_temp = X, y
        X_test, y_test = np.array([]), np.array([])

    # Second split: train and validation
    if validation_split > 0:
        val_size = validation_split / (1 - test_split) if test_split < 1 else validation_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.config.seed
        )
    else:
        # No validation set
        X_train, y_train = X_temp, y_temp
        X_val, y_val = np.array([]), np.array([])

    # ...
```

### 5.2 ModelTrainer.train()

```python
def train(self, config: Dict[str, Any]) -> Dict[str, Any]:
    # ...

    # Extract split_config from config
    split_config = config.get('split_config')
    category_name = config.get('category_name')  # For custom splits

    # Load data with custom splits
    train_df, val_df, test_df = self.load_data(
        data_path,
        text_column=text_column,
        label_column=label_column,
        split_config=split_config,
        category_name=category_name
    )

    # ...
```

### 5.3 MultiLabelTrainer

```python
class MultiLabelTrainer:
    def train(self, ..., split_config: Optional[Dict] = None):
        # Store split_config for later use
        self.split_config = split_config

        # ...

        return self.train_all_models(all_samples, ...)

    def train_all_models(self, samples, train_ratio=0.8, val_ratio=0.1, ...):
        # Check if we have custom split config
        if hasattr(self, 'split_config') and self.split_config:
            # This will be handled per-label during dataset preparation
            pass

        # ...

        label_datasets = self.prepare_label_datasets(samples, train_ratio, val_ratio, ...)

        # ...

    def prepare_label_datasets(self, samples, train_ratio, val_ratio, ...):
        # For each label, check if there's a custom split
        for label_name in unique_labels:
            # Determine split ratios for this label
            if hasattr(self, 'split_config') and self.split_config:
                mode = self.split_config.get('mode', 'uniform')

                if mode == 'uniform':
                    ratios = self.split_config['uniform']
                    label_train_ratio = ratios['train_ratio']
                    label_val_ratio = ratios['validation_ratio']
                else:  # custom
                    # Check for custom ratios
                    custom_ratios = None

                    if 'custom_by_value' in self.split_config:
                        custom_ratios = self.split_config['custom_by_value'].get(label_name)

                    if custom_ratios:
                        label_train_ratio = custom_ratios['train_ratio']
                        label_val_ratio = custom_ratios['validation_ratio']
                    else:
                        defaults = self.split_config.get('defaults', {})
                        label_train_ratio = defaults.get('train_ratio', train_ratio)
                        label_val_ratio = defaults.get('validation_ratio', val_ratio)
            else:
                # Use defaults
                label_train_ratio = train_ratio
                label_val_ratio = val_ratio

            # Use these ratios for splitting
            # ...
```

---

## 6. Propagation dans Tous les Modes

### 6.1 Mode Quick

```python
def _training_studio_run_quick(self, bundle, model_config, quick_params, session_id):
    # ...

    # Extract split_config from bundle
    split_config = bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None

    # Pass to trainer
    if training_approach == 'one-vs-all':
        category_config = {
            # ...
            'split_config': split_config,
            'category_name': category_name  # For custom splits
        }
        category_result = trainer.train(category_config)

    # ...
```

### 6.2 Mode Benchmark

```python
def _training_studio_run_benchmark(self, bundle, model_config, session_id):
    # ...

    split_config = bundle.metadata.get('split_config') if hasattr(bundle, 'metadata') else None

    result = model.run_training(
        # ...
        split_config=split_config,
        category_name=category_name
    )

    # ...
```

### 6.3 Mode Custom / Distributed

Même logique : extraire `split_config` de `bundle.metadata` et le passer aux trainers.

---

## 7. Cas Particuliers

### 7.1 Validation Set = 0

Si l'utilisateur choisit de ne pas avoir de validation set :
- `validation_ratio = 0.0`
- Le code dans `bert_base.py` doit gérer ce cas :
  - Pas d'early stopping
  - Pas de sélection de best model basée sur validation
  - Utiliser le test set pour monitoring ? Ou aucun monitoring ?

**Décision** : Si pas de validation set, utiliser le train set pour monitoring (pas idéal mais mieux que rien).

### 7.2 Test Set = 0

Certains utilisateurs pourraient vouloir 0% test :
- Train : 80%
- Validation : 20%
- Test : 0%

Dans ce cas, l'évaluation finale se fait sur validation set.

### 7.3 Validation des Ratios

```python
def validate_split_ratios(train, validation, test):
    """Validate split ratios sum to 1.0"""
    total = train + validation + test

    if abs(total - 1.0) > 0.001:
        # Auto-adjust
        factor = 1.0 / total
        train *= factor
        validation *= factor
        test *= factor

        self.console.print(f"[yellow]⚠️  Ratios adjusted to sum to 1.0[/yellow]")

    # Minimum values
    if train < 0.5:
        raise ValueError("Training ratio must be at least 50%")

    if validation > 0 and validation < 0.05:
        raise ValueError("Validation ratio must be at least 5% if used")

    if test > 0 and test < 0.05:
        raise ValueError("Test ratio must be at least 5% if used")

    return train, validation, test
```

---

## 8. Résumé Final Affiché

À la fin de la configuration, afficher un résumé :

```python
self.console.print("\n[bold green]✓ Split Configuration Complete[/bold green]\n")

if split_mode == 'uniform':
    self.console.print("[bold]Uniform Split (all keys/values):[/bold]")
    self.console.print(f"  • Train:      {train_ratio:.1%}")
    if use_validation:
        self.console.print(f"  • Validation: {validation_ratio:.1%}")
    self.console.print(f"  • Test:       {test_ratio:.1%}")
else:
    self.console.print(f"[bold]Custom Split:[/bold]")

    if custom_by_key:
        self.console.print(f"\n  Configured keys: {len(custom_by_key)}")
        for key, ratios in custom_by_key.items():
            self.console.print(f"    • {key}: {ratios['train_ratio']:.1%} / {ratios['validation_ratio']:.1%} / {ratios['test_ratio']:.1%}")

    if custom_by_value:
        self.console.print(f"\n  Configured values: {len(custom_by_value)}")
        # Show first 5
        for i, (value, ratios) in enumerate(list(custom_by_value.items())[:5]):
            self.console.print(f"    • {value}: {ratios['train_ratio']:.1%} / {ratios['validation_ratio']:.1%} / {ratios['test_ratio']:.1%}")
        if len(custom_by_value) > 5:
            self.console.print(f"    ... and {len(custom_by_value) - 5} more")

    self.console.print(f"\n  Defaults (for others): {default_train:.1%} / {default_validation:.1%} / {default_test:.1%}")

self.console.print()
```

---

## 9. Tests Requis

Tester avec :

1. **Mode uniform + validation**
   - Train: 0.7, Validation: 0.2, Test: 0.1
   - Vérifier que tous les modèles utilisent ces ratios

2. **Mode uniform sans validation**
   - Train: 0.8, Test: 0.2, Validation: 0.0
   - Vérifier que l'entraînement fonctionne sans validation set

3. **Mode custom multi-class**
   - Configurer 2 clés avec ratios différents
   - Vérifier que chaque clé utilise ses propres ratios

4. **Mode custom one-vs-all**
   - Configurer 2 valeurs avec ratios différents
   - Vérifier que chaque valeur utilise ses propres ratios

5. **Mode custom hybrid**
   - Mélange de clés multi-class et valeurs one-vs-all
   - Vérifier la propagation correcte

6. **Tous les modes d'entraînement**
   - Quick, Benchmark, Custom, Distributed
   - Vérifier que split_config est bien propagé

---

## 10. Migration

Pour les datasets existants sans split_config :
- Utiliser les valeurs par défaut de `TrainingConfig`
- Pas de changement de comportement pour backward compatibility

---

## Auteur

Claude Code (plan créé le 2025-10-07)
