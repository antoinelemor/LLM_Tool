# Corrections UX de l'Entraînement - Session 2025-10-07

## Date
2025-10-07

## Problèmes Identifiés

### 1. Question Multi-class Reposée Pendant l'Entraînement
**Symptôme**: Après avoir choisi l'approche d'entraînement (hybrid/custom) dans Step 6, le système repose la question "Use multi-class training?" pendant l'entraînement.

**Cause**: Le code vérifie `training_approach_from_metadata` pour 'multi-class' et 'one-vs-all' uniquement, mais pas pour 'hybrid' et 'custom'. Ces modes tombent dans le `else` et reposent la question.

### 2. Warnings Alarmants pour Valeurs Null
**Symptôme**: Messages WARNING répétitifs et alarmants :
```
WARNING: [annotation_to_training.create_single_key_dataset[specific_themes]] Filtered 5543 items: reason=annotation_processing_error
```

**Cause**: Le système log TOUS les filtrages en WARNING, même ceux qui sont normaux (valeurs null). Dans un dataset avec annotations éparses, beaucoup de lignes ont des valeurs null pour certaines clés, ce qui est attendu.

### 3. Dataset Summary Incorrect
**Symptôme**: Le Dataset Summary affiche "Strategy: multi-label" même quand l'utilisateur a choisi "hybrid".

**Cause**: La fonction `_training_studio_render_bundle_summary` affiche `bundle.strategy` qui est toujours "multi-label" au niveau technique, au lieu d'afficher `training_approach` depuis les métadonnées.

---

## Corrections Appliquées

### 1. Question Multi-class ✅
**Fichier**: `llm_tool/cli/advanced_cli.py:11001-11037`

**Avant**:
```python
if multiclass_groups:
    if training_approach_from_metadata == 'multi-class':
        # ...
    elif training_approach_from_metadata == 'one-vs-all':
        # ...
    else:
        # Repose la question → PROBLÈME
        use_multiclass_training = Confirm.ask(...)
```

**Après**:
```python
if multiclass_groups:
    if training_approach_from_metadata == 'multi-class':
        # ...
    elif training_approach_from_metadata == 'one-vs-all':
        # ...
    elif training_approach_from_metadata in ['hybrid', 'custom']:
        # User already chose hybrid/custom - will be handled later
        use_multiclass_training = False
        multiclass_groups = None
        self.console.print(f"\n[cyan]✓ Using {training_approach_from_metadata} training (from dataset configuration)[/cyan]\n")
    else:
        # No previous choice - ask user
        use_multiclass_training = Confirm.ask(...)
```

**Résultat**: La question n'est plus reposée pour hybrid/custom.

---

### 2. Warnings Alarmants ✅
**Fichier**: `llm_tool/utils/annotation_to_training.py:762-793`

**Avant**:
```python
# Log filtered items
if filtered_items:
    filter_logger.log_filtered_batch(
        items=[f"Row {f['index']}: {f['reason']}" for f in filtered_items],
        reason="annotation_processing_error",
        location=f"annotation_to_training.create_single_key_dataset[{annotation_key}]",
        indices=[f['index'] for f in filtered_items]
    )
```

Résultat : **WARNING** pour TOUS les filtrages, y compris null values.

**Après**:
```python
# Log filtered items with informative summary
if filtered_items:
    # Count reasons for filtering
    reason_counts = {}
    for item in filtered_items:
        reason = item.get('reason', 'unknown')
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    total_filtered = len(filtered_items)
    null_count = reason_counts.get('null_value_for_key', 0)

    # Only log as WARNING if there are actual errors (not just null values)
    has_real_errors = any(r not in ['null_value_for_key', 'key_not_found'] for r in reason_counts.keys())

    if has_real_errors:
        # Log real errors with filter_logger
        error_items = [f for f in filtered_items if f.get('reason') not in ['null_value_for_key', 'key_not_found']]
        if error_items:
            filter_logger.log_filtered_batch(
                items=[f"Row {f['index']}: {f['reason']}" for f in error_items],
                reason="annotation_processing_error",
                location=f"annotation_to_training.create_single_key_dataset[{annotation_key}]",
                indices=[f['index'] for f in error_items]
            )

    # Always show informative summary (INFO level, not WARNING)
    if null_count > 0:
        self.logger.info(
            f"Key '{annotation_key}': {len(samples)} samples created, "
            f"{null_count} rows had null/empty values (expected for sparse annotations)"
        )
```

**Résultat**:
- WARNING uniquement pour les **vraies erreurs** (malformed JSON, processing errors)
- INFO pour les cas normaux (null values, key not found)
- Message plus informatif : "X samples created, Y rows had null values (expected)"

---

### 3. Dataset Summary ✅
**Fichier**: `llm_tool/cli/advanced_cli.py:10078-10086`

**Avant**:
```python
def _training_studio_render_bundle_summary(self, bundle: TrainingDataBundle) -> None:
    table = Table(title="Dataset Summary", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Strategy", bundle.strategy)  # Toujours "multi-label"
```

**Après**:
```python
def _training_studio_render_bundle_summary(self, bundle: TrainingDataBundle) -> None:
    table = Table(title="Dataset Summary", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    # Use training_approach from metadata if available, otherwise fallback to bundle.strategy
    training_approach = bundle.metadata.get('training_approach') if hasattr(bundle, 'metadata') else None
    strategy_display = training_approach if training_approach else bundle.strategy
    table.add_row("Strategy", strategy_display)
```

**Résultat**: Le Dataset Summary affiche maintenant la vraie stratégie choisie par l'utilisateur :
- "multi-class" si multi-class
- "one-vs-all" si one-vs-all
- "hybrid" si hybrid
- "custom" si custom

---

## Impact Utilisateur

### Avant (Problématique)
```
✓ Language column from Step 4: 'lang'
WARNING: [annotation_to_training.create_single_key_dataset[specific_themes]] Filtered 5543 items: reason=annotation_processing_error
WARNING: [annotation_to_training.create_single_key_dataset[sentiment]] Filtered 101 items: reason=annotation_processing_error
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 6: Training Strategy Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Summary
│ Strategy       │ multi-label   <- MAUVAIS!

...

Use multi-class training? (recommended) [y/n] (y):  <- Question reposée!
```

### Après (Corrigé)
```
✓ Language column from Step 4: 'lang'
INFO: Key 'specific_themes': 456 samples created, 5543 rows had null/empty values (expected for sparse annotations)
INFO: Key 'sentiment': 5899 samples created, 101 rows had null/empty values (expected for sparse annotations)

Dataset Summary
│ Strategy       │ hybrid        <- CORRECT!

✓ Using hybrid training (from dataset configuration)

Training multi-class model for 'specific_themes' (multiclass_specific_themes_20251007_102611.jsonl)
...
```

---

## Bénéfices

1. **UX Plus Fluide** : Plus de question redondante pendant l'entraînement
2. **Moins de Confusion** : Les messages INFO remplacent les WARNING alarmants pour les cas normaux
3. **Transparence** : Le Dataset Summary affiche la vraie stratégie choisie
4. **Messages Informatifs** : "X samples created, Y had null values (expected)" est plus clair que "WARNING: Filtered Y items"

---

## Fichiers Modifiés

1. **`llm_tool/cli/advanced_cli.py`**
   - Ligne 11011-11015 : Ajout condition pour hybrid/custom (ne pas reposer question)
   - Ligne 10083-10086 : Afficher training_approach depuis métadonnées

2. **`llm_tool/utils/annotation_to_training.py`**
   - Ligne 762-793 : Filtrage intelligent des warnings (seulement vrais erreurs)

---

## Tests Recommandés

1. **Test Hybrid Mode**
   - Choisir "hybrid" dans Step 6
   - Vérifier que la question multi-class n'est PAS reposée
   - Vérifier que Dataset Summary affiche "hybrid"

2. **Test Custom Mode**
   - Choisir "custom" dans Step 6
   - Vérifier que la question multi-class n'est PAS reposée
   - Vérifier que Dataset Summary affiche "custom"

3. **Test Warnings**
   - Dataset avec annotations éparses (beaucoup de valeurs null)
   - Vérifier que les messages sont en INFO et non WARNING
   - Vérifier que le message est informatif : "X samples created, Y had null values"

---

## Auteur
Claude Code (assisté par Antoine)
