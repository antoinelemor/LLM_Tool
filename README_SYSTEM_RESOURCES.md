# System Resource Detection - Visual Improvements

## Overview

Le module de détection des ressources système a été considérablement amélioré avec de nouvelles visualisations attractives et informatives.

## Nouvelles Fonctionnalités

### 1. Panneau Visuel pour la Page d'Accueil

Un grand panneau détaillé avec des barres de progression et des recommandations côte à côte.

```python
from llm_tool.utils import create_visual_resource_panel

panel = create_visual_resource_panel(
    resources,
    show_recommendations=True
)
console.print(panel)
```

**Affiche:**
- GPU avec nom, mémoire et barre de progression
- CPU avec nom, cores/threads et utilisation
- RAM avec barre de progression et mémoire disponible
- Storage avec barre de progression
- Recommandations détaillées (device, batch size, workers, FP16, etc.)
- Notes contextuelles

### 2. Banner Horizontal pour les Pages de Modes

Un banner compact horizontal qui affiche l'essentiel en une ligne.

```python
from llm_tool.utils import create_mode_resource_banner

banner = create_mode_resource_banner(resources)
console.print(banner)
```

**Affiche:**
- GPU type et mémoire
- CPU cores et utilisation
- RAM disponible
- Batch size et workers recommandés

### 3. Panneau Détaillé pour les Modes

Un panneau compact mais informatif pour les pages de modes spécifiques.

```python
from llm_tool.utils import create_detailed_mode_panel

panel = create_detailed_mode_panel(
    resources,
    mode_name="The Annotator"
)
console.print(panel)
```

**Affiche:**
- GPU avec barre de progression
- CPU avec utilisation
- RAM avec barre de progression
- Recommandations principales

## Barres de Progression Visuelles

Toutes les barres de progression sont colorées selon l'utilisation:
- **Vert** (< 70%): Normal
- **Jaune** (70-90%): Attention
- **Rouge** (> 90%): Critique

Exemple:
```
🧠 Memory (RAM)   ████░░░░░░░░░░░░  101 GB free
                  ▲▲▲▲ Utilisé
                      ░░░░░░░░░░░░ Libre
```

## Icônes et Couleurs

### GPU
- 🎮 **NVIDIA CUDA** (vert bright_green)
- 🍎 **Apple Silicon MPS** (vert bright_green)
- 💻 **CPU Only** (jaune)

### Composants
- ⚡ **CPU** (jaune bright_yellow)
- 🧠 **RAM** (magenta bright_magenta)
- 💾 **Storage** (bleu bright_blue)

### Recommandations
- 🎯 **Device**
- 📦 **Batch Size**
- 👷 **Workers**
- ⚡ **FP16**
- 🔄 **Gradient Accumulation**

## Intégration dans le CLI

### Page d'Accueil

La page d'accueil affiche automatiquement le panneau visuel complet après la détection des LLMs et des datasets.

```
╭───────── ⚙️  System Resources & Recommendations ─────────╮
│                                                          │
│  GPU     🍎 Apple M4 Max      🎯 Device   🍎 MPS        │
│          96.0 GB              📦 Batch    16 samples     │
│          ████░░░░░░░░░░       Size                       │
│          75.6 GB free         👷 Workers  8 threads      │
│                                                          │
│  CPU     ⚡ Apple M4 Max      ⚡ FP16     ✗ Disabled     │
│          16 cores / 16                                   │
│          threads                                         │
│          ██░░░░░░░░░░░░                                  │
│          7.7% used                                       │
│                                                          │
│  RAM     🧠 Memory (RAM)      Notes      💡 Apple        │
│          128 GB total                    Silicon         │
│          ████░░░░░░░░░░                  detected: MPS   │
│          101 GB free                     acceleration    │
│                                          enabled         │
│  Storage 💾 Storage (Disk)                               │
│          3722 GB total                                   │
│          ████████████████                                │
│          742 GB free                                     │
│                                                          │
╰──────────────────────────────────────────────────────────╯
```

### Pages de Modes

Chaque mode (The Annotator, The Annotator Factory, Training Arena, etc.) affiche un banner horizontal compact.

```
╭────────────── ⚙️  System Resources ──────────────╮
│   🍎 MPS     ⚡ 16 Cores   🧠 128 GB   💡 Batch: 16│
│   96.0 GB    7.7% used     101 GB free  Workers: 8│
╰──────────────────────────────────────────────────╯
```

## Utilisation Programmatique

### Détection Simple

```python
from llm_tool.utils import detect_resources

resources = detect_resources()

# Accéder aux informations
print(f"GPU: {resources.gpu.device_type}")
print(f"Memory: {resources.memory.total_gb} GB")
```

### Obtenir les Recommandations

```python
recommendations = resources.get_recommendation()

print(f"Device: {recommendations['device']}")
print(f"Batch Size: {recommendations['batch_size']}")
print(f"Workers: {recommendations['num_workers']}")
```

### Fonctions Helper

```python
from llm_tool.utils import (
    get_device_recommendation,
    get_optimal_batch_size,
    get_optimal_workers
)

device = get_device_recommendation()  # "cuda", "mps", ou "cpu"
batch = get_optimal_batch_size()      # Ex: 16
workers = get_optimal_workers()       # Ex: 8
```

### Affichage Personnalisé

```python
from llm_tool.utils import create_visual_resource_panel
from rich.console import Console

console = Console()
resources = detect_resources()

# Afficher le panneau complet
panel = create_visual_resource_panel(resources)
console.print(panel)
```

## Architecture

### Modules

1. **system_resources.py**: Détection des ressources
   - `SystemResourceDetector`: Classe principale de détection
   - `SystemResources`: Container pour toutes les ressources
   - Fonctions helper pour accès rapide

2. **resource_display.py**: Affichage visuel
   - `create_visual_resource_panel()`: Panneau complet pour page d'accueil
   - `create_mode_resource_banner()`: Banner horizontal pour modes
   - `create_detailed_mode_panel()`: Panneau détaillé pour modes
   - Fonctions utilitaires pour barres de progression

### Dataclasses

- `GPUInfo`: Informations GPU (type, mémoire, CUDA version)
- `CPUInfo`: Informations CPU (cores, fréquence, utilisation)
- `MemoryInfo`: Informations RAM (total, disponible, utilisé)
- `StorageInfo`: Informations stockage (total, disponible, utilisé)
- `SystemInfo`: Informations système (OS, version, Python)

## Recommandations Automatiques

Le système génère automatiquement des recommandations optimales basées sur:

### GPU NVIDIA CUDA
- **≥16 GB**: batch_size=32, FP16=True
- **≥8 GB**: batch_size=16, FP16=True
- **<8 GB**: batch_size=8, gradient_accumulation=2

### Apple Silicon MPS
- **Tous**: batch_size=16, device="mps"

### CPU Seulement
- **Tous**: batch_size=8, gradient_accumulation=2

### Workers
- **≥8 cores**: min(8, cores // 2)
- **<8 cores**: max(2, cores // 2)

### Ajustements RAM
- **<8 GB RAM disponible**: batch_size et workers réduits de moitié

## Tests

### Test Complet

```bash
python examples/system_resources_demo.py
```

### Test des Visualisations

```python
from llm_tool.utils import detect_resources, create_visual_resource_panel
from rich.console import Console

console = Console()
resources = detect_resources()

# Test panneau principal
panel = create_visual_resource_panel(resources)
console.print(panel)
```

### Test du CLI

```bash
python -m llm_tool.cli.advanced_cli
```

## Exemples

Voir:
- `examples/system_resources_demo.py`: Démo complète
- `docs/SYSTEM_RESOURCES.md`: Documentation détaillée

## Notes Techniques

### Cache
- Les détections sont cachées pendant 5 minutes (300 secondes)
- Utilisez `force_refresh=True` pour forcer une nouvelle détection

### Performance
- Première détection: ~1-2 secondes
- Détections suivantes (depuis cache): <0.01 seconde
- Affichage: <0.1 seconde

### Compatibilité
- **macOS**: Détection complète (Apple Silicon MPS, CPU, RAM, Storage)
- **Windows**: Détection complète (CUDA, CPU, RAM, Storage)
- **Linux**: Détection complète (CUDA, CPU, RAM, Storage)

### Dépendances
- **torch**: Pour détection GPU (requis)
- **psutil**: Pour informations détaillées CPU/RAM (optionnel mais recommandé)
- **rich**: Pour affichage visuel (requis)

## Changelog

### Version 1.0 - 2025-10-08

**Nouvelles fonctionnalités:**
- ✨ Panneau visuel complet pour page d'accueil
- ✨ Banner horizontal pour pages de modes
- ✨ Barres de progression colorées pour toutes les ressources
- ✨ Icônes et couleurs pour meilleure lisibilité
- ✨ Recommandations automatiques intelligentes
- ✨ Intégration complète dans le CLI

**Améliorations:**
- 🎨 Design plus attractif et moderne
- 📊 Visualisations plus informatives
- ⚡ Performances optimisées avec cache
- 🔧 API simplifiée et intuitive

## Support

Pour toute question ou problème:
1. Consultez la documentation complète: `docs/SYSTEM_RESOURCES.md`
2. Exécutez les démos: `examples/system_resources_demo.py`
3. Vérifiez les tests: `python -m llm_tool.utils.system_resources`

## Auteur

Antoine Lemor - 2025
