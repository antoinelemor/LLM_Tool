# Restructuration des Logs d'Entraînement - Session 2025-10-07

## Date
2025-10-07

## Objectif
Réorganiser la structure des logs d'entraînement pour avoir :
- Un dossier de session avec date/heure au niveau racine
- Des sous-dossiers par catégorie (pas par modèle)
- Des CSV généraux qui agrègent tous les modèles

## Structure Actuelle (Problématique)
```
training_logs/
  specific_themes/
    bert-base-uncased/
      specific_themes_training.csv
    camembert-base/
      specific_themes_training.csv
  sentiment/
    bert-base-uncased/
      sentiment_training.csv
    camembert-base/
      sentiment_training.csv
```

**Problèmes**:
- Les logs sont organisés par modèle, pas par catégorie
- Difficile de comparer les modèles pour une même catégorie
- Pas de notion de "session d'entraînement"

## Structure Nouvelle (Souhaitée)
```
training_logs/
  20251007_103025/              <- Session timestamp
    specific_themes/            <- Catégorie
      training.csv              <- Tous les modèles (bert, camembert, etc.)
      best.csv                  <- Tous les modèles
    sentiment/
      training.csv
      best.csv
    political_parties/
      training.csv
      best.csv
```

**Avantages**:
- Une session = un dossier horodaté
- Facile de comparer tous les modèles pour une catégorie donnée
- Les CSV généraux contiennent les métriques de tous les modèles
- Meilleure organisation chronologique

---

## Changements Implémentés

### 1. Ajout du paramètre `session_id` à `run_training()`

**Fichier**: `llm_tool/trainers/bert_base.py:846`

```python
def run_training(
    self,
    # ... autres paramètres ...
    session_id: Optional[str] = None  # NEW: Session timestamp
) -> Tuple[Any, Any, Any, Any]:
```

### 2. Nouvelle logique de création de dossiers

**Fichier**: `llm_tool/trainers/bert_base.py:926-954`

```python
# Create or use session ID (timestamp)
if session_id is None:
    import datetime
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Determine category name
# Priority: label_value > label_key > "default"
if label_value:
    category_name = label_value
elif label_key:
    category_name = label_key
else:
    category_name = "default"

# Clean category name
category_name = category_name.replace("/", "_").replace(" ", "_")

# Build directory structure: training_logs/{session_id}/{category}/
session_dir = os.path.join(metrics_output_dir, session_id)
category_dir = os.path.join(session_dir, category_name)
os.makedirs(category_dir, exist_ok=True)

# CSV files are now general (contain all models for this category)
training_metrics_csv = os.path.join(category_dir, "training.csv")
best_models_csv = os.path.join(category_dir, "best.csv")
```

### 3. Création du `session_id` au niveau CLI

**Fichier**: `llm_tool/cli/advanced_cli.py:6653-6656`

```python
# Create session ID for this training session (shared across all models)
from datetime import datetime
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
self.console.print(f"[dim]Session ID: {session_id}[/dim]\n")
```

### 4. Transmission du `session_id` à toutes les fonctions

**Fichier**: `llm_tool/cli/advanced_cli.py:6661-6672`

Toutes les fonctions `_training_studio_run_*` reçoivent maintenant `session_id`:

```python
if mode == "distributed":
    training_result = self._training_studio_run_distributed(bundle, model_config, session_id)
elif mode == "quick":
    training_result = self._training_studio_run_quick(bundle, model_config, quick_params, session_id)
elif mode == "benchmark":
    training_result = self._training_studio_run_benchmark(bundle, model_config, session_id)
else:
    training_result = self._training_studio_run_custom(bundle, model_config, session_id)
```

---

## Changements À Implémenter

### Fichiers à modifier

1. **`llm_tool/cli/advanced_cli.py`**
   - ✅ Création du `session_id` (ligne 6655)
   - ✅ Passage aux fonctions de training (lignes 6662-6672)
   - ⏳ Modifier signatures de `_training_studio_run_distributed`
   - ⏳ Modifier signatures de `_training_studio_run_quick`
   - ⏳ Modifier signatures de `_training_studio_run_benchmark`
   - ⏳ Modifier signatures de `_training_studio_run_custom`
   - ⏳ Passer `session_id` à tous les appels `model.run_training()`
   - ⏳ Passer `session_id` au `MultiLabelTrainer`

2. **`llm_tool/trainers/bert_base.py`**
   - ✅ Ajout paramètre `session_id` (ligne 846)
   - ✅ Nouvelle structure de dossiers (lignes 926-954)
   - ✅ CSV généraux (training.csv, best.csv)

3. **`llm_tool/trainers/multi_label_trainer.py`**
   - ⏳ Ajouter paramètre `session_id` à `train_single_model()`
   - ⏳ Passer `session_id` à `model.run_training()`

4. **`llm_tool/trainers/model_trainer.py`**
   - ⏳ Ajouter `session_id` aux appels `model.run_training()`

5. **`llm_tool/trainers/benchmarking.py`**
   - ⏳ Ajouter `session_id` aux appels `model.run_training()`

---

## Points d'Attention

1. **Compatibilité Backwards**: Le paramètre `session_id` est optionnel. Si non fourni, un nouveau timestamp est créé automatiquement.

2. **Print Epoch Summary**: Le print après chaque époque (ajouté dans `bert_base.py:1798-1803`) doit être présent pour TOUS les modes d'entraînement.

3. **CSV Headers**: Les headers CSV doivent inclure la colonne `model_name` pour distinguer les modèles dans les fichiers généraux.

4. **Tous les modes**: La nouvelle structure doit fonctionner pour :
   - Mode quick
   - Mode benchmark
   - Mode custom
   - Mode distributed (multi-label)
   - Mode hybrid/custom (multi-class + one-vs-all)
   - Reinforced learning

---

## Exemple de Résultat

Après un entraînement avec 2 modèles (bert-base-uncased et camembert-base) sur 3 catégories :

```
training_logs/
  20251007_105530/
    specific_themes/
      training.csv    <- Contient toutes les époques de bert-base-uncased ET camembert-base
      best.csv        <- Contient le meilleur modèle de chaque type
    sentiment/
      training.csv
      best.csv
    political_parties/
      training.csv
      best.csv
```

Le fichier `training.csv` pour `specific_themes` contiendra :
```csv
# TRAINING METRICS
# ...
model_identifier,model_name,label_key,label_value,language,timestamp,epoch,train_loss,val_loss,...
,bert-base-uncased,,specific_themes,EN,20251007_105531,1,0.5781,0.4536,...
,bert-base-uncased,,specific_themes,EN,20251007_105540,2,0.3859,0.3456,...
...
,camembert-base,,specific_themes,FR,20251007_105630,1,0.5781,0.4536,...
,camembert-base,,specific_themes,FR,20251007_105640,2,0.3859,0.3456,...
```

---

## Tests Recommandés

1. **Test Mode Quick avec Multi-Langue**
   - Entraîner 2 modèles (EN + FR)
   - Vérifier que les logs vont dans `training_logs/{session_id}/{category}/`
   - Vérifier que training.csv contient les 2 modèles

2. **Test Mode Hybrid**
   - Entraîner avec hybrid strategy (6 keys multi-class + 6 keys one-vs-all)
   - Vérifier que toutes les catégories ont leur dossier
   - Vérifier que chaque training.csv contient tous les modèles de la catégorie

3. **Test Reinforced Learning**
   - Activer reinforced learning
   - Vérifier que les logs reinforced vont aussi dans la bonne structure

---

## Auteur
Claude Code (assisté par Antoine)
