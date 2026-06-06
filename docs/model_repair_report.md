# Réparation des modèles YouPol non entraînables — rapport

**Date :** 2026-06-06 · **Auteur :** Lemorphic · **Outil :** LLM Tool

## Contexte et problème

Après les campagnes d'entraînement one-vs-all (CamemBERTav2) sur les 11 thèmes
YouPol, deux catégories restaient **impossibles à entraîner** — F1 de la classe
positive = 0 quelle que soit la base ou la stratégie de loss :

| Thème | aug | no-aug (plain) | no-aug (focal/weighted) |
|---|---|---|---|
| **libertarianism** | 0.000 | 0.000 | 0.000 |
| **fictional_metaphors** | 0.348 | 0.000 | 0.270 |

Symptôme caractéristique : **train_loss figée à ~ln2 (~0.69)** et prédictions
oscillant entre tout-négatif et tout-positif (« flip-flop ») — le modèle ne
décollait jamais de l'initialisation, alors qu'un simple **TF-IDF + régression
logistique** atteignait f1=0.65 (libertarianism) / 0.58 (fictional) : le signal
était donc bien présent dans le texte.

## Démarche scientifique (itérative, une expérience à la fois)

Sur `libertarianism` (no-aug), avec monitoring époque par époque :

| Exp | Configuration | Résultat (f1_pos) | Conclusion |
|---|---|---|---|
| E1 | CamemBERTav2 + float32(MPS) + weighted, lr 2e-5 | 0.00 (loss plate) | le float32 ne suffit pas |
| E2 | CamemBERTav2 + weighted, **lr 5e-5** | 0.40 (dégénéré) | une LR plus forte ne décolle pas |
| E3 | CamemBERTav2 + **linear probe** (encodeur gelé) | 0.40 | la tête linéaire sur embeddings camembertav2 ne sépare pas |
| **probe** | **embedding-probe par backbone** | voir ci-dessous | **le backbone est le goulot** |
| **E4** | **camembert-base** + weighted, lr 2e-5 | **0.865** ✅ | **réparé** |

**Embedding-probe décisif** (mean-pooling → LogReg pondérée, f1_pos) :

| Backbone | libertarianism | fictional_metaphors |
|---|---|---|
| almanach/camembertav2-base (deberta-v2) | 0.41 | 0.48 |
| **camembert-base (RoBERTa fr)** | **0.68** | **0.82** |
| xlm-roberta-base | 0.61 | 0.73 |

→ La représentation **deberta-v2** de CamemBERTav2 ne capte pas ces thèmes
subtils/diffus ; **camembert-base** (RoBERTa français) les sépare nettement —
mieux même que le TF-IDF. La cause racine n'était ni la loss, ni la donnée, ni
le déséquilibre : **c'était le backbone.**

## Résultat : les deux modèles sont réparés

Modèles officiels, reproductibles (session LLM Tool + checkpoint sauvegardé) :

| Thème | Avant (meilleur) | **Après (camembert-base)** | Backbone | Base |
|---|---|---|---|---|
| **libertarianism** | 0.000 | **f1_pos 0.865 / macro 0.909** | camembert-base | no-aug |
| **fictional_metaphors** | 0.348 | **f1_pos 0.863 / macro 0.911** | camembert-base | aug (542 pos) |

Sessions officielles :
- `logs/training_arena/repair-libertarianism-FINAL-camembert_20260606_021727/`
- `logs/training_arena/repair-fictional_metaphors-FINAL-camembert_20260606_024648/`
Checkpoints : `models/.../normal_training/<thème>/FR/model/pytorch_model.bin`.

## Améliorations apportées au package (toutes architectures, défauts inchangés)

Commitées et poussées (identité utilisateur) — 35 tests verts :

1. **Détection d'architecture DeBERTa par `config.model_type`** (et non par le
   nom). Corrige un bug : `camembertav2`/`camemberta` sont des deberta-v2 mais
   leur nom ne contient pas « deberta » → ils tournaient en **float16 sur MPS**
   sans le forçage float32 (précision de l'attention désenchevêtrée) et sans le
   garde-fou de batch. Vérifié : camembertav2/camemberta/deberta-v3/mdeberta →
   True ; bert/roberta/xlm-roberta/camembert-v1/electra → False.
2. **Câblage du `warmup_ratio`** (auparavant loggué mais jamais appliqué :
   `num_warmup_steps=0` en dur). Défaut 0.0 = comportement legacy identique.
3. **Exposition de `learning_rate` + `warmup_ratio` dans le CLI** (quick mode,
   opt-in), threadés jusqu'à `run_training` pour tous les types d'entraînement.
4. **`freeze_encoder` (linear probe)** : option SOTA pour les catégories
   difficiles (geler l'encodeur, n'entraîner que la tête). Défaut off.
5. **Détection auto des `target_modules` LoRA/DoRA** (commit précédent) :
   DoRA fonctionne désormais sur toutes les architectures (camembertav2 inclus).

Aucune logique par architecture existante (multiplicateurs xlm/mdeberta,
reinforced-params) n'a été modifiée.

## Reproductibilité

Harnais non-interactif officiel : `scripts/repair_experiment.py`. Exemple :
```
python scripts/repair_experiment.py --category libertarianism --base noaug \
   --model camembert-base --lr 2e-5 --strategy weighted --epochs 12 --save
```
Produit une session `logs/training_arena/` standard (metadata + training_metrics)
+ le checkpoint, exactement comme un modèle entraîné via le CLI.

## Recommandations

1. **libertarianism & fictional_metaphors → camembert-base** (modèles ci-dessus).
2. **Le choix du backbone est le levier décisif** pour les thèmes français
   subtils. Voir l'annexe (probe large) : envisager camembert-base au-delà de
   ces deux thèmes si l'écart se confirme.
3. **Mise en garde méthodologique** : le split val est row-level (segments d'une
   même vidéo possibles en train ET val). Pour une évaluation plus stricte,
   passer à un split *par video_id* — appliqué uniformément, il ne change pas le
   classement relatif mais donnerait des F1 absolus plus conservateurs.

## Mise à jour — camembert-base étendu aux catégories sous 0.7

Le probe large ayant prédit un gain net, les catégories CamemBERTav2 sous 0.7 ont
été ré-entraînées sur **camembert-base** (no-aug, weighted, sauvegarde). Toutes
progressent fortement :

| Thème | CamemBERTav2 (best-of) | **camembert-base** | Δ |
|---|---|---|---|
| libertarianism | 0.000 | **0.865** | +0.865 |
| fictional_metaphors | 0.348 | **0.863** | +0.515 |
| tradition | 0.679 | **0.847** | +0.168 |
| progress | 0.664 | **0.814** | +0.150 |
| authority | 0.659 | **0.805** | +0.146 |
| technology | 0.795 | **0.891** | +0.096 |

→ **camembert-base améliore CHAQUE catégorie testée (6/6)**, y compris
`technology` qui était la **meilleure** de CamemBERTav2 (0.795 → 0.891). Le
backbone est donc le facteur **dominant et universel** pour ces thèmes politiques
français — pas un effet limité aux thèmes difficiles.

### Recommandation forte
Ré-entraîner **les 11 thèmes sur camembert-base** pour un jeu homogène et
nettement supérieur. Les 5 restants encore sur CamemBERTav2 (ecology 0.826,
immigration 0.812, nationalism 0.792, equality 0.766, democracy 0.705)
devraient eux aussi gagner (probe frozen camembert-base 0.71–0.89 ; et
`technology` vient de passer de 0.795 à 0.891). Commande type (reproductible) :
```
python scripts/repair_experiment.py --category <thème> --base noaug \
   --model camembert-base --lr 2e-5 --strategy weighted --epochs 8 --save
```
Note : `fictional_metaphors` reste meilleur sur la base **aug** (542 positifs vs
~100 en no-aug) ; pour les autres, no-aug suffit.

## Annexe — probe large (camembertav2 vs camembert-base)

Embedding-probe **frozen** (mean-pool → LogReg pondérée, f1_pos, échantillons
plafonnés pour la vitesse) sur 9 thèmes, base no-aug :

| thème | camembertav2 | camembert-base | Δ |
|---|---|---|---|
| ecology | 0.455 | 0.892 | +0.437 |
| technology | 0.451 | 0.826 | +0.375 |
| immigration | 0.491 | 0.782 | +0.291 |
| authority | 0.438 | 0.724 | +0.287 |
| nationalism | 0.448 | 0.723 | +0.275 |
| tradition | 0.445 | 0.718 | +0.273 |
| progress | 0.416 | 0.685 | +0.269 |
| equality | 0.502 | 0.760 | +0.257 |
| democracy | 0.456 | 0.706 | +0.250 |

**Lecture rigoureuse :** il s'agit de la séparabilité **frozen** (sans
fine-tuning). CamemBERTav2 (deberta-v2) expose peu de signal dans ses embeddings
bruts et n'« ouvre » sa représentation qu'**après** fine-tuning — d'où ses bons
scores fine-tunés sur les thèmes faciles (ecology 0.826, immigration 0.812). En
revanche, pour les thèmes **subtils et/ou à faible volume** (libertarianism,
fictional_metaphors), il reste coincé à l'initialisation. camembert-base, lui, a
le signal **immédiatement accessible** (probe 0.69–0.89) → apprentissage stable
dès l'époque 1.

**Implication :** camembert-base est un backbone plus robuste pour ces thèmes
politiques français. Pour les catégories actuellement **sous 0.7** (authority
0.659, progress 0.664, tradition 0.679), le probe suggère un gain net — un
ré-entraînement camembert-base est recommandé (et a été lancé : voir mises à
jour ci-dessous / sessions `repair-<thème>-camembert_*`).

