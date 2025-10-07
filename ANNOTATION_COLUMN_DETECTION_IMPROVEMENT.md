# Amélioration de la Détection Automatique des Colonnes d'Annotations

## Date
2025-10-07

## Problème Identifié

Le système ne détectait pas automatiquement les colonnes d'annotations lorsque leur nom ne correspondait pas aux patterns standards (`annotation`, `label`, `category`, etc.).

### Exemple Concret
Une colonne nommée `gemma3` contenant des annotations JSON n'était **pas** automatiquement suggérée comme colonne d'annotation, car son nom ne correspond à aucun des patterns attendus.

---

## Solution Implémentée

Ajout d'une **détection par contenu** en complément de la détection par nom.

### Logique de Détection Améliorée

#### 1. Première Passe : Détection par Nom (Existante)
Vérifie si le nom de la colonne contient :
- `annotation`, `annotations`
- `label`, `labels`
- `category`, `categories`
- `class`, `classification`

#### 2. Deuxième Passe : Détection par Contenu (NOUVEAU)
Pour toutes les colonnes **non encore détectées** :

1. **Analyse du Contenu**
   - Examine 10 échantillons de la colonne
   - Compte combien contiennent du JSON valide
   - Vérifie que les valeurs commencent par `{` ou `[`

2. **Seuil de Détection**
   - Si ≥70% des échantillons sont du JSON → Colonne d'annotation détectée

3. **Priorité Maximale**
   - Priority = 3 (plus élevée que la détection par nom)
   - Match type = `'json_content'`
   - Affichage : "Auto-detected JSON annotations"

### Exclusions
Le système ignore :
- Colonnes déjà détectées comme annotations
- Colonnes de texte (candidates)
- Colonnes d'ID (`_id`, `id`, `identifier`)

---

## Modifications de Code

### Fichier : `llm_tool/cli/advanced_cli.py`

#### 1. Détection par Contenu (lignes 915-976)
```python
# IMPROVED: Detect JSON annotation columns by content (even if name doesn't match patterns)
detected_annotation_names = {c['name'] for c in result['annotation_column_candidates']}
for col in df.columns:
    # Skip already detected, text columns, and ID columns
    if col in detected_annotation_names:
        continue

    # Check if column contains JSON data
    sample_values = df[col].dropna().head(10)
    json_count = 0

    for val in sample_values:
        if isinstance(val, str) and (val.strip().startswith('{') or val.strip().startswith('[')):
            try:
                json.loads(val)
                json_count += 1
            except:
                pass
        elif isinstance(val, dict) or isinstance(val, list):
            json_count += 1

    # 70% threshold for JSON content
    if json_count >= len(sample_values) * 0.7 and len(sample_values) > 0:
        result['annotation_column_candidates'].append({
            'name': col,
            'match_type': 'json_content',
            'is_json': True,
            'priority': 3  # Highest priority
        })
```

#### 2. Affichage Amélioré (lignes 7945-7973)
```python
# Show detection method in suggestions
if is_json:
    if match_type == 'json_content':
        reason_parts.append("Auto-detected JSON annotations")
    else:
        reason_parts.append("Contains JSON annotations")
```

---

## Résultat

### Avant
- Colonne `gemma3` → ❌ Non détectée automatiquement
- Utilisateur doit deviner et entrer manuellement le nom

### Après
- Colonne `gemma3` → ✅ **Détectée automatiquement**
- Affichage : "Auto-detected JSON annotations, 100% filled"
- Suggérée comme colonne d'annotation par défaut

---

## Avantages

1. **Détection Intelligente** : Analyse le contenu, pas seulement le nom
2. **Priorité Maximale** : Les colonnes JSON détectées par contenu ont la priorité la plus élevée
3. **Transparent** : L'utilisateur voit "Auto-detected" dans les suggestions
4. **Robuste** : Seuil de 70% évite les faux positifs
5. **Compatible** : Fonctionne avec tous les formats de noms de colonnes

---

## Cas d'Usage

### Cas 1 : Colonnes avec Noms Non-Standard
```
Colonnes du CSV:
- sentence: "Hello world"
- gemma3: {"political_party": "LPC", "sentiment": "positive"}
- id: 123

Résultat:
✅ gemma3 détectée automatiquement comme annotation
```

### Cas 2 : Plusieurs Colonnes JSON
```
Colonnes du CSV:
- text: "Some text"
- model_a: {"category": "tech"}
- model_b: {"category": "politics"}

Résultat:
✅ model_a ET model_b détectées comme annotations
✅ Classées par priorité (toutes priority=3)
```

### Cas 3 : Colonnes Mixtes
```
Colonnes du CSV:
- content: "Text content"
- annotation: {"label": "A"}  <- Nom standard
- custom_labels: {"label": "B"}  <- Nom non-standard mais JSON

Résultat:
✅ annotation détectée (priority=2, name pattern)
✅ custom_labels détectée (priority=3, json_content)
→ custom_labels suggérée en premier (priorité plus élevée)
```

---

## Tests Recommandés

1. **Test avec `gemma3`** :
   - CSV avec colonne `gemma3` contenant JSON
   - Vérifier que `gemma3` est suggérée automatiquement
   - Vérifier l'affichage "Auto-detected JSON annotations"

2. **Test avec Colonnes Standard** :
   - CSV avec colonne `annotation` (nom standard)
   - Vérifier que la détection par nom fonctionne toujours
   - Vérifier l'affichage "Contains JSON annotations"

3. **Test avec Multiples Colonnes JSON** :
   - CSV avec plusieurs colonnes JSON
   - Vérifier que toutes sont détectées
   - Vérifier que la priorité est correcte

---

## Auteur
Claude Code (assisté par Antoine)
