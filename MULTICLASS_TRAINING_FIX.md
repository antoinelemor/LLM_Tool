# Correction de la logique d'entraînement Multi-class et One-vs-All

## Problème Identifié

Le système ne gérait pas correctement les modes d'entraînement "multi-class" et "one-vs-all" lorsque l'utilisateur sélectionnait **toutes les clés (ALL)**.

### Comportement Incorrect (Avant)

**Mode Multi-class avec ALL keys:**
- ❌ Créait UN SEUL fichier avec TOUTES les valeurs de TOUTES les clés
- ❌ Essayait d'entraîner UN SEUL modèle pour détecter toutes les valeurs
- ❌ Causait l'erreur "Target 33 is out of bounds"

**Mode One-vs-all avec ALL keys:**
- ⚠️ Créait UN SEUL fichier avec toutes les valeurs
- ⚠️ Mais le comportement n'était pas clairement documenté

---

## Comportement Correct (Après)

### Mode Multi-class

**Définition:** Entraîne **UN modèle PAR CLÉ** (pas un modèle pour toutes les valeurs)

#### Exemple avec 2 clés :
```
Clés disponibles:
- political_party: [BQ, CAQ, CPC, LPC, NDP]
- sentiment: [positive, negative, neutral]

Mode multi-class → 2 modèles:
1. Modèle "political_party": apprend à détecter BQ, CAQ, CPC, LPC, NDP
2. Modèle "sentiment": apprend à détecter positive, negative, neutral
```

#### Implémentation :
- Crée **UN fichier d'entraînement PAR CLÉ**
- Chaque fichier contient uniquement les valeurs de sa clé
- Format: `multiclass_<key_name>_<timestamp>.jsonl`

### Mode One-vs-All

**Définition:** Entraîne **UN modèle PAR VALEUR** (pas un modèle par clé)

#### Exemple avec les mêmes 2 clés :
```
Clés disponibles:
- political_party: [BQ, CAQ, CPC, LPC, NDP]
- sentiment: [positive, negative, neutral]

Mode one-vs-all → 8 modèles binaires:
1. Modèle "political_party_BQ": binaire (BQ vs NOT BQ)
2. Modèle "political_party_CAQ": binaire (CAQ vs NOT CAQ)
3. Modèle "political_party_CPC": binaire (CPC vs NOT CPC)
4. Modèle "political_party_LPC": binaire (LPC vs NOT LPC)
5. Modèle "political_party_NDP": binaire (NDP vs NOT NDP)
6. Modèle "sentiment_positive": binaire (positive vs NOT positive)
7. Modèle "sentiment_negative": binaire (negative vs NOT negative)
8. Modèle "sentiment_neutral": binaire (neutral vs NOT neutral)
```

#### Implémentation :
- Crée **UN fichier global** avec toutes les valeurs de toutes les clés
- Le MultiLabelTrainer détecte automatiquement les groupes multiclass
- Entraîne un modèle binaire par valeur

---

## Fichiers Modifiés

### 1. `llm_tool/trainers/training_data_builder.py`

**Modification:** Méthode `_build_llm_annotations()`

- Détecte si `mode == "single-label"` ET `len(annotation_keys) > 1`
- Si oui → crée UN fichier par clé avec la nouvelle fonction `create_single_key_dataset()`
- Retourne un `TrainingDataBundle` avec `training_files` (un par clé)

### 2. `llm_tool/utils/annotation_to_training.py`

**Ajout:** Nouvelle fonction `create_single_key_dataset()`

- Extrait les données pour UNE SEULE clé spécifique
- Crée un fichier JSONL avec seulement les valeurs de cette clé
- Utilisée par `_build_llm_annotations()` pour le mode multi-class

### 3. `llm_tool/cli/advanced_cli.py`

**Modification:** Messages utilisateur (lignes 8545-8563)

- Clarifie que multi-class = **UN modèle PAR CLÉ** (pas un modèle pour toutes les valeurs)
- Clarifie que one-vs-all = **UN modèle PAR VALEUR** (pas un modèle par clé)
- Ajoute des exemples concrets avec `political_party` et `sentiment`

---

## Messages Utilisateur Mis à Jour

### Mode Multi-class (Plusieurs clés)

```
🎯 Trains ONE model PER KEY (not per value)

• 12 models total (one per annotation key)
• Each model learns ALL values of ITS key
• Example: One model for 'political_party' learns BQ, CAQ, CPC, etc.
• Example: Another model for 'sentiment' learns positive, negative, neutral
• Total: 12 models (one per key)

Best for: Standard classification with mutually exclusive categories per key
```

### Mode One-vs-All (Plusieurs clés)

```
⚡ Trains ONE model PER VALUE (not per key)

• 142 binary models total (one per unique value)
• Each model: 'value X' vs NOT 'value X'
• Example: Separate model for 'political_party_BQ' (binary: BQ or not)
• Example: Separate model for 'sentiment_positive' (binary: positive or not)
• Total: 142 models (one per value)

Best for: Imbalanced data, or when texts can have multiple labels
```

---

## Résolution de l'Erreur "Target 33 is out of bounds"

### Cause
Le système créait un fichier avec toutes les valeurs (34 valeurs uniques), mais extrayait seulement la première valeur de chaque liste, créant des incohérences dans l'encoding des labels.

### Solution
Avec les corrections :
1. Mode multi-class crée maintenant **UN fichier PAR CLÉ**
2. Chaque fichier contient uniquement les valeurs de sa clé
3. Le MultiLabelTrainer entraîne un modèle par clé avec le bon nombre de classes
4. Plus d'erreur "Target out of bounds"

---

## Impact sur les Utilisateurs

### Avant (Incorrect)
- Multi-class avec ALL keys → Échec avec erreur
- Messages confus sur le nombre de modèles
- Comportement imprévisible

### Après (Correct)
- Multi-class avec ALL keys → ✅ Fonctionne correctement
- Messages clairs et précis
- Comportement prévisible et documenté
- Un modèle par clé comme attendu

---

## Tests Recommandés

1. **Test multi-class avec 2 clés:**
   - Sélectionner "all" keys + "multi-class"
   - Vérifier que 2 fichiers sont créés (un par clé)
   - Vérifier que 2 modèles sont entraînés

2. **Test one-vs-all avec 2 clés:**
   - Sélectionner "all" keys + "one-vs-all"
   - Vérifier qu'un fichier global est créé
   - Vérifier que N modèles sont entraînés (N = total des valeurs)

3. **Test multi-class avec 1 clé:**
   - Sélectionner 1 clé + "multi-class"
   - Vérifier qu'un fichier est créé
   - Vérifier qu'un modèle est entraîné

---

---

## Corrections Supplémentaires (même session)

### Problème : Question du mode multi-class reposée pendant l'entraînement

**Cause :** Le code vérifiait `training_approach == 'multi-label'` au lieu de `training_approach == 'multi-class'` pour détecter si l'utilisateur avait déjà choisi le mode.

**Correction :** `llm_tool/cli/advanced_cli.py:10829`
```python
# Avant
if training_approach_from_metadata == 'multi-label':

# Après
if training_approach_from_metadata == 'multi-class':
```

### Problème : Entraînement utilise le mauvais fichier

**Cause :** Même après la création de fichiers par clé, le système chargeait le fichier consolidé `multilabel_all_keys.jsonl` au lieu des fichiers individuels.

**Correction :** `llm_tool/cli/advanced_cli.py:11138-11216`

Ajout d'un nouveau bloc de code qui :
1. Détecte si `training_approach == 'multi-class'` ET `bundle.training_files` existe
2. Extrait les fichiers par clé (en excluant 'multilabel')
3. Entraîne UN modèle PAR fichier (donc un modèle par clé)
4. Agrège les résultats

**Résultat :**
- ✅ Chaque clé a son propre modèle
- ✅ Chaque modèle est entraîné sur les valeurs de SA clé uniquement
- ✅ Plus d'erreur "Target out of bounds"

---

## Date
2025-10-07

## Auteur
Claude Code (assisté par Antoine)
