#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
model_display.py

MAIN OBJECTIVE:
---------------
Centralized utility for displaying all available training models with filtering
and intelligent recommendations based on data characteristics (language, text length, etc.)

Dependencies:
-------------
- rich (table display)
- typing

MAIN FEATURES:
--------------
1) Display all available models in a comprehensive table
2) Filter models by language, text length, and other characteristics
3) Sort models by relevance to user's data
4) Show model characteristics: size, context length, multilingual support
5) Integration with ModelSelector for intelligent scoring

Author:
-------
Antoine Lemor
"""

from typing import List, Set, Optional, Dict, Any
from rich.table import Table
from rich.console import Console
from rich import box


# Complete model metadata with ALL models from the package
# Based on models.py and sota_models.py
MODEL_METADATA = {
    # ========================================================================
    # MULTILINGUAL MODELS
    # ========================================================================
    'xlm-roberta-base': {
        'name': 'xlm-roberta-base',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': True,
        'description': 'XLM-RoBERTa Base: Trained on 100+ languages (2.5TB CommonCrawl data). Best general-purpose multilingual model. Excellent cross-lingual transfer. Use for: multilingual datasets, low-resource languages, cross-lingual tasks. ~270M parameters, balanced speed/accuracy.'
    },
    'xlm-roberta-large': {
        'name': 'xlm-roberta-large',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': True,
        'description': 'XLM-RoBERTa Large: SOTA multilingual model (100+ languages). Superior performance on all multilingual benchmarks. ~550M parameters. 2x slower than base but significantly more accurate. Use when: maximum accuracy needed, sufficient GPU memory, training time not critical.'
    },
    'microsoft/mdeberta-v3-base': {
        'name': 'microsoft/mdeberta-v3-base',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': True,
        'description': 'mDeBERTa-v3 Base: Microsoft\'s SOTA multilingual model (100+ languages). Disentangled attention mechanism for better context modeling. Outperforms XLM-R on many tasks. Best choice for: complex classification tasks, nuanced understanding, balanced performance. ~279M parameters.'
    },
    'bert-base-multilingual-cased': {
        'name': 'bert-base-multilingual-cased',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': True,
        'description': 'mBERT: Original multilingual BERT (104 languages). Well-established, widely used. Good baseline but outperformed by newer models (XLM-R, mDeBERTa). Use when: reproducibility with existing work needed, or as baseline comparison. ~177M parameters. Case-sensitive.'
    },

    # ========================================================================
    # ENGLISH MODELS
    # ========================================================================
    'bert-base-uncased': {
        'name': 'bert-base-uncased',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'BERT Base Uncased: Original BERT model, lowercases all text. Trained on BookCorpus + English Wikipedia. Solid baseline for English tasks. ~110M parameters. Use when: case doesn\'t matter (most classification), established baseline needed. Good balance of speed/accuracy for general English text.'
    },
    'bert-base-cased': {
        'name': 'bert-base-cased',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'BERT Base Cased: Preserves text casing (capitalization). Same architecture as uncased. Use when: case is important (named entities, proper nouns, acronyms). Better for: legal documents, formal text, entity recognition. ~110M parameters. Slightly better for nuanced tasks.'
    },
    'roberta-base': {
        'name': 'roberta-base',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'RoBERTa Base: Robustly Optimized BERT. Trained longer on more data (160GB text). Improved training procedure removes NSP task. Generally outperforms BERT on most tasks. ~125M parameters. Best for: general English classification. Default choice for English-only projects. Fast training.'
    },
    'roberta-large': {
        'name': 'roberta-large',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'RoBERTa Large: Larger version with superior accuracy. ~355M parameters. Trained on same data as base but larger capacity. Best English BERT-family model. Use when: maximum accuracy needed, have GPU resources, can afford longer training. Expect ~2-5% accuracy gain over base.'
    },
    'distilroberta-base': {
        'name': 'distilroberta-base',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Small',
        'multilingual': False,
        'description': 'DistilRoBERTa: Knowledge-distilled RoBERTa, 40% smaller, 60% faster. Retains 97% of RoBERTa performance. ~82M parameters. Best for: production deployment, limited GPU, fast inference needed, edge devices. Excellent speed/accuracy tradeoff. Trains 2-3x faster than RoBERTa-base.'
    },
    'google/electra-base-discriminator': {
        'name': 'google/electra-base-discriminator',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'ELECTRA Base: Efficiently Learns an Encoder that Classifies Token Replacements Accurately. Novel pretraining: discriminator vs generator. More sample-efficient than BERT/RoBERTa. ~110M parameters. Use when: limited training data, need good performance with fewer epochs. Comparable to RoBERTa.'
    },
    'google/electra-small-discriminator': {
        'name': 'google/electra-small-discriminator',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Small',
        'multilingual': False,
        'description': 'ELECTRA Small: Compact model for resource-constrained environments. ~14M parameters. 10x smaller than base models. Use for: edge deployment, mobile devices, CPU inference, very fast training. Good accuracy for size. Best when: speed >> accuracy, memory constrained.'
    },
    'google/electra-large-discriminator': {
        'name': 'google/electra-large-discriminator',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'ELECTRA Large: High-performance discriminator model. ~335M parameters. Matches or exceeds RoBERTa-large on many benchmarks with more efficient training. Use when: SOTA accuracy needed, have GPU capacity. More sample-efficient than RoBERTa-large, often trains faster for similar accuracy.'
    },
    'albert-base-v2': {
        'name': 'albert-base-v2',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'ALBERT Base v2: Parameter-efficient model via factorized embeddings + cross-layer sharing. Only ~12M parameters but performance similar to BERT-base. Slower inference than size suggests. Use when: memory constrained, parameter efficiency important. Good for: multi-task learning, parameter budgets.'
    },
    'albert-large-v2': {
        'name': 'albert-large-v2',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'ALBERT Large v2: Larger parameter-efficient model. ~18M parameters, performance near BERT-large. Superior memory efficiency. Use when: need large-model performance with lower memory footprint. Note: slower than parameter count suggests due to layer sharing. Good accuracy/memory ratio.'
    },
    'albert-xlarge-v2': {
        'name': 'albert-xlarge-v2',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'XLarge',
        'multilingual': False,
        'description': 'ALBERT XLarge v2: Maximum parameter sharing efficiency. ~60M parameters, performance approaching RoBERTa-large. Best parameter efficiency in ALBERT family. Use when: very memory constrained but need high accuracy. Training slower than smaller ALBERTs. Best for: memory-limited high-accuracy scenarios.'
    },
    'microsoft/deberta-v3-base': {
        'name': 'microsoft/deberta-v3-base',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'DeBERTa-v3 Base: Disentangled Attention + Enhanced Mask Decoder. SOTA English model at base size. Disentangled attention encodes content/position separately. Often best-in-class for English. ~183M parameters. Use when: need best possible English accuracy at base size. Top choice for complex English tasks.'
    },
    'microsoft/deberta-v3-large': {
        'name': 'microsoft/deberta-v3-large',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'DeBERTa-v3 Large: Best-in-class English model. Disentangled attention at scale. Consistently tops English NLU benchmarks. ~434M parameters. Use when: absolute best English accuracy needed, have GPU resources, training time acceptable. Expect top-tier performance on English classification tasks.'
    },
    'microsoft/deberta-v3-xsmall': {
        'name': 'microsoft/deberta-v3-xsmall',
        'languages': ['EN'],
        'max_length': 512,
        'size': 'XSmall',
        'multilingual': False,
        'description': 'DeBERTa-v3 XSmall: Smallest DeBERTa variant. ~22M parameters. Significantly faster than base models while retaining strong performance. Use for: fast experimentation, resource-constrained deployment, CPU inference. Better accuracy than similarly-sized competitors. Good for: rapid iteration, edge deployment.'
    },
    'google/bigbird-roberta-base': {
        'name': 'google/bigbird-roberta-base',
        'languages': ['EN'],
        'max_length': 4096,
        'size': 'Base',
        'multilingual': False,
        'description': 'BigBird-RoBERTa Base: Sparse attention for long documents (up to 4096 tokens). Block sparse attention + random + global. Linear complexity vs quadratic. ~127M parameters. Use when: documents 512-4096 tokens, need full-document context, memory limited. Faster than full attention on long docs.'
    },
    'google/bigbird-roberta-large': {
        'name': 'google/bigbird-roberta-large',
        'languages': ['EN'],
        'max_length': 4096,
        'size': 'Large',
        'multilingual': False,
        'description': 'BigBird-RoBERTa Large: High-performance long-document model. Sparse attention up to 4096 tokens. ~355M parameters. Best for: long English documents requiring high accuracy. Better than Longformer on some tasks. Use when: docs 512-4096 tokens, maximum accuracy needed, have GPU capacity.'
    },
    'allenai/longformer-base-4096': {
        'name': 'allenai/longformer-base-4096',
        'languages': ['EN'],
        'max_length': 4096,
        'size': 'Base',
        'multilingual': False,
        'description': 'Longformer Base: Efficient long-document transformer (up to 4096 tokens). Sliding window + dilated + global attention. Linear complexity. ~149M parameters. Best general-purpose long-doc English model. Use for: legal docs, research papers, long articles. Well-tested, production-ready.'
    },
    'allenai/longformer-large-4096': {
        'name': 'allenai/longformer-large-4096',
        'languages': ['EN'],
        'max_length': 4096,
        'size': 'Large',
        'multilingual': False,
        'description': 'Longformer Large: Best long-document English model. Up to 4096 tokens with efficient attention. ~435M parameters. Superior performance on long-document tasks. Use when: need best accuracy on long English docs (legal, academic, technical), have GPU resources. Top choice for long English classification.'
    },
    'allenai/led-base-16384': {
        'name': 'allenai/led-base-16384',
        'languages': ['EN'],
        'max_length': 16384,
        'size': 'Base',
        'multilingual': False,
        'description': 'LED Base: Longformer Encoder-Decoder for VERY long documents (up to 16384 tokens). Encoder-decoder architecture. ~406M parameters. Use for: extremely long documents (books, legal briefs, comprehensive reports). More complex than encoder-only. Best when: docs exceed 4096 tokens regularly.'
    },
    'allenai/led-large-16384': {
        'name': 'allenai/led-large-16384',
        'languages': ['EN'],
        'max_length': 16384,
        'size': 'Large',
        'multilingual': False,
        'description': 'LED Large: Maximum capacity for extremely long documents (up to 16384 tokens). Encoder-decoder with Longformer attention. ~777M parameters. Use when: handling book-length documents, comprehensive analysis needed, very long context critical. Requires significant GPU memory. Best for: research papers, legal documents, books.'
    },

    # ========================================================================
    # FRENCH MODELS
    # ========================================================================
    'camembert-base': {
        'name': 'camembert-base',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'CamemBERT Base: THE standard French model. RoBERTa architecture trained on 138GB French text (OSCAR corpus). Best general-purpose French model. ~110M parameters. Use for: any French text classification. Default choice for French-only. Excellent performance, well-tested, production-ready.'
    },
    'camembert/camembert-large': {
        'name': 'camembert/camembert-large',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'CamemBERT Large: High-performance French model. Same training as base but larger capacity. ~335M parameters. Best-in-class French accuracy. Use when: maximum French accuracy needed, have GPU resources. Expect 2-5% improvement over base. Top choice for complex French NLP tasks.'
    },
    'cmarkea/distilcamembert-base': {
        'name': 'cmarkea/distilcamembert-base',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Small',
        'multilingual': False,
        'description': 'DistilCamemBERT: Knowledge-distilled CamemBERT. 40% smaller, 60% faster, retains 97% performance. ~68M parameters. Best for: fast French inference, production deployment, limited GPU, CPU inference. Excellent speed/accuracy tradeoff. Trains 2-3x faster than CamemBERT-base.'
    },
    'flaubert/flaubert_base_cased': {
        'name': 'flaubert/flaubert_base_cased',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'FlauBERT Base: Alternative French BERT from CNRS. Trained on French books, Wikipedia, web crawl. ~138M parameters. Use when: need diversity from CamemBERT, academic setting. Generally comparable to CamemBERT. Good for: research baselines, ensemble models with CamemBERT.'
    },
    'flaubert/flaubert_large_cased': {
        'name': 'flaubert/flaubert_large_cased',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Large',
        'multilingual': False,
        'description': 'FlauBERT Large: Large French model from CNRS. ~373M parameters. Alternative to CamemBERT-large. Similar performance, different pretraining corpus. Use when: need high French accuracy with different pretraining than CamemBERT, ensemble modeling, academic research.'
    },
    'almanach/camemberta-base': {
        'name': 'almanach/camemberta-base',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'CamemBERTa Base: Modern French RoBERTa (v2). Improved training on updated French corpora. ~110M parameters. Better on recent French text than original CamemBERT. Use for: contemporary French (social media, modern web), when CamemBERT underperforms. Good alternative to standard CamemBERT.'
    },
    'moussaKam/barthez': {
        'name': 'moussaKam/barthez',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'BARThez: French BART (encoder-decoder). Denoising autoencoder trained on French. ~165M parameters. Better for: generation-focused tasks, seq2seq. Use when: task benefits from encoder-decoder (summarization-then-classify). Less common for pure classification, better for hybrid tasks.'
    },
    'cls/fralbert': {
        'name': 'cls/fralbert',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Small',
        'multilingual': False,
        'description': 'FrALBERT: French ALBERT with parameter sharing. Lightweight French model. ~12M parameters. Very memory efficient. Use for: memory-constrained French tasks, multi-task French learning, parameter budgets. Slower inference than size suggests. Good accuracy/memory ratio.'
    },
    'dbmdz/electra-base-french-europeana-cased-discriminator': {
        'name': 'dbmdz/electra-base-french-europeana-cased-discriminator',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'French ELECTRA: Discriminator model trained on Europeana French corpus. Sample-efficient training. ~110M parameters. Use when: limited French training data, need efficient learning. Alternative to CamemBERT with different training approach. Good for: few-shot French tasks.'
    },
    'almanach/camembert-bio-base': {
        'name': 'almanach/camembert-bio-base',
        'languages': ['FR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'CamemBERT-bio: Domain-specific French biomedical model. CamemBERT further trained on French medical/scientific texts. ~110M parameters. ONLY use for: French medical/biomedical/health text. Significantly better than general CamemBERT on medical French. Worse on general French.'
    },

    # ========================================================================
    # GERMAN MODELS
    # ========================================================================
    'dbmdz/bert-base-german-cased': {
        'name': 'dbmdz/bert-base-german-cased',
        'languages': ['DE'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'German BERT: Standard German model from DBmdz. Trained on German Wikipedia, news, web crawl. ~110M parameters. Best general-purpose German model. Use for: any German text classification. Well-established, production-ready. Case-sensitive (preserves capitalization important in German).'
    },
    'distilbert-base-german-cased': {
        'name': 'distilbert-base-german-cased',
        'languages': ['DE'],
        'max_length': 512,
        'size': 'Small',
        'multilingual': False,
        'description': 'DistilBERT German: Distilled German BERT, 40% smaller, faster inference. ~66M parameters. Retains ~95% of BERT performance. Use for: fast German inference, production deployment, limited GPU. Good speed/accuracy tradeoff for German. Trains 2x faster than full German BERT.'
    },
    'deepset/gbert-base': {
        'name': 'deepset/gbert-base',
        'languages': ['DE'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'GBERT: Deepset-optimized German BERT. Trained on large German corpus with careful preprocessing. ~110M parameters. Alternative to dbmdz BERT. Use when: dbmdz underperforms, need different pretraining, Deepset ecosystem. Comparable to dbmdz, slightly different strengths.'
    },

    # ========================================================================
    # SPANISH MODELS
    # ========================================================================
    'dccuchile/bert-base-spanish-wwm-cased': {
        'name': 'dccuchile/bert-base-spanish-wwm-cased',
        'languages': ['ES'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Spanish BERT WWM: Whole-word masking for better Spanish morphology. University of Chile. Trained on Spanish Wikipedia, news. ~110M parameters. Good baseline Spanish model. Use for: general Spanish text. Alternative to RoBERTa-BNE. Case-sensitive.'
    },
    'PlanTL-GOB-ES/roberta-base-bne': {
        'name': 'PlanTL-GOB-ES/roberta-base-bne',
        'languages': ['ES'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'RoBERTa-BNE: SOTA Spanish model from Spanish National Library (Biblioteca Nacional de España). RoBERTa architecture, massive Spanish corpus. ~125M parameters. Best general Spanish model. Use for: any Spanish task. Superior to BETO/Spanish BERT. Default choice for Spanish.'
    },

    # ========================================================================
    # ITALIAN MODELS
    # ========================================================================
    'dbmdz/bert-base-italian-cased': {
        'name': 'dbmdz/bert-base-italian-cased',
        'languages': ['IT'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Italian BERT: Standard Italian model from DBmdz. Trained on Italian Wikipedia, OPUS, OSCAR. ~110M parameters. Best general Italian model. Use for: any Italian text classification. Well-tested, production-ready. Case-sensitive (important for Italian proper nouns).'
    },

    # ========================================================================
    # PORTUGUESE MODELS
    # ========================================================================
    'neuralmind/bert-base-portuguese-cased': {
        'name': 'neuralmind/bert-base-portuguese-cased',
        'languages': ['PT'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'BERTimbau: Best Portuguese BERT from NeuralMind. Trained on brWaC (Brazilian Web as Corpus). ~110M parameters. Optimized for Brazilian Portuguese, works well on European Portuguese. Use for: any Portuguese task. Standard choice for Portuguese NLP. Case-sensitive.'
    },

    # ========================================================================
    # CHINESE MODELS
    # ========================================================================
    'bert-base-chinese': {
        'name': 'bert-base-chinese',
        'languages': ['ZH'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Chinese BERT: Original Google Chinese BERT. Trained on Chinese Wikipedia. ~110M parameters. Character-level tokenization. Baseline Chinese model. Use for: general Chinese, established baselines. Outperformed by newer models (Chinese RoBERTa) but widely used, well-documented.'
    },
    'hfl/chinese-roberta-wwm-ext': {
        'name': 'hfl/chinese-roberta-wwm-ext',
        'languages': ['ZH'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Chinese RoBERTa WWM-ext: SOTA Chinese model from HFL (iFLYTEK). Whole-word masking + extended training. Trained on massive Chinese corpus. ~102M parameters. Best general Chinese model. Use for: any Chinese task. Outperforms BERT-Chinese significantly. Default choice for Chinese.'
    },

    # ========================================================================
    # ARABIC MODELS
    # ========================================================================
    'asafaya/bert-base-arabic': {
        'name': 'asafaya/bert-base-arabic',
        'languages': ['AR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Arabic BERT: General Arabic model trained on Arabic Wikipedia, news, OSCAR. ~110M parameters. Good baseline for Arabic. Use for: general Modern Standard Arabic. Alternative to AraBERT. Works across Arabic dialects but optimized for MSA.'
    },
    'aubmindlab/bert-base-arabertv2': {
        'name': 'aubmindlab/bert-base-arabertv2',
        'languages': ['AR'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'AraBERT v2: SOTA Arabic model from AUB MIND Lab. Improved preprocessing, larger corpus than v1. ~110M parameters. Best general Arabic model. Use for: any Arabic task (MSA, dialects). Superior Arabic diacritization handling. Default choice for Arabic NLP.'
    },

    # ========================================================================
    # RUSSIAN MODELS
    # ========================================================================
    'DeepPavlov/rubert-base-cased': {
        'name': 'DeepPavlov/rubert-base-cased',
        'languages': ['RU'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'RuBERT: Best Russian BERT from DeepPavlov. Trained on Russian Wikipedia, news, social media. ~180M parameters. Handles Russian morphology well. Use for: any Russian text classification. Well-maintained, widely used. Case-sensitive (important for Russian proper nouns).'
    },

    # ========================================================================
    # HINDI MODELS
    # ========================================================================
    'monsoon-nlp/hindi-bert': {
        'name': 'monsoon-nlp/hindi-bert',
        'languages': ['HI'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Hindi BERT: Specialized Hindi model from Monsoon NLP. Trained on Hindi Wikipedia, news, web crawl. ~110M parameters. Best dedicated Hindi model. Use for: Hindi-only tasks. Better than multilingual models on Hindi. Handles Devanagari script well.'
    },

    # ========================================================================
    # SWEDISH MODELS
    # ========================================================================
    'KB/bert-base-swedish-cased': {
        'name': 'KB/bert-base-swedish-cased',
        'languages': ['SV'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Swedish BERT: Official Swedish model from National Library (Kungliga Biblioteket). Trained on Swedish books, articles, web. ~110M parameters. Best Swedish model. Use for: any Swedish text. Authoritative source, high-quality training data. Case-sensitive.'
    },

    # ========================================================================
    # DUTCH MODELS
    # ========================================================================
    'GroNLP/bert-base-dutch-cased': {
        'name': 'GroNLP/bert-base-dutch-cased',
        'languages': ['NL'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Dutch BERT (BERTje): Best Dutch model from University of Groningen. Trained on Dutch Wikipedia, books, news. ~110M parameters. Standard Dutch model. Use for: any Dutch text. Well-tested on Dutch benchmarks. Better than multilingual for Dutch-only. Case-sensitive.'
    },

    # ========================================================================
    # POLISH MODELS
    # ========================================================================
    'allegro/herbert-base-cased': {
        'name': 'allegro/herbert-base-cased',
        'languages': ['PL'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'HerBERT: Best Polish BERT from Allegro. Trained on large Polish corpus (KGR10, NKJP, Wikipedia). ~124M parameters. Optimized for Polish morphology. Use for: any Polish text. Handles Polish inflections well. Default choice for Polish NLP. Case-sensitive.'
    },

    # ========================================================================
    # JAPANESE MODELS
    # ========================================================================
    'cl-tohoku/bert-base-japanese-whole-word-masking': {
        'name': 'cl-tohoku/bert-base-japanese-whole-word-masking',
        'languages': ['JA'],
        'max_length': 512,
        'size': 'Base',
        'multilingual': False,
        'description': 'Japanese BERT WWM: Best Japanese model from Tohoku University. Whole-word masking for Japanese morphemes. MeCab tokenization. ~110M parameters. Trained on Japanese Wikipedia. Use for: any Japanese task. Handles Japanese segmentation well. Default choice for Japanese NLP.'
    },

    # ========================================================================
    # MULTILINGUAL LONG-DOCUMENT MODELS
    # ========================================================================
    'markussagen/xlm-roberta-longformer-base-4096': {
        'name': 'markussagen/xlm-roberta-longformer-base-4096',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 4096,
        'size': 'Base',
        'multilingual': True,
        'description': 'XLM-RoBERTa-Longformer: Best multilingual long-document model. XLM-R with Longformer attention (100+ languages, up to 4096 tokens). ~278M parameters. Use for: long documents in multiple languages, cross-lingual long-text tasks. Combines XLM-R multilingual strength with efficient long-context.'
    },
    'google/long-t5-local-base': {
        'name': 'google/long-t5-local-base',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 4096,
        'size': 'Base',
        'multilingual': True,
        'description': 'LongT5-Local: Multilingual encoder-decoder for very long documents (up to 4096 tokens). Local attention pattern. ~248M parameters. Use for: long multilingual generation tasks, encoder-decoder needed. Less common for pure classification, better for hybrid summarize-then-classify tasks.'
    },
    'google/long-t5-tglobal-base': {
        'name': 'google/long-t5-tglobal-base',
        'languages': ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'NL', 'PL', 'RU', 'ZH', 'JA', 'KO', 'AR', 'HI', 'TR', 'SV', 'DA', 'NO', 'FI', 'CS', 'UK', 'EL', 'RO', 'HU', 'BG', 'HR', 'SR', 'SK', 'SL', 'LT', 'LV', 'ET', 'MT', 'GA', 'EU', 'CA', 'GL', 'CY', 'MK', 'SQ', 'IS', 'KA', 'HY', 'AZ', 'KK', 'UZ', 'TG', 'KY', 'TK', 'MN', 'VI', 'TH', 'ID', 'MS', 'TL', 'SW', 'YO', 'IG', 'XH', 'ZU', 'AF', 'NE', 'ML', 'TA', 'TE', 'KN', 'PA', 'GU', 'OR', 'BN', 'AS', 'MR', 'SI', 'MY', 'KM', 'LO', 'BO', 'DZ', 'FA', 'UR', 'PS', 'SD', 'UG', 'AM', 'SO', 'HA', 'RW', 'SN', 'LG', 'WO', 'FF', 'DY', 'BM', 'AK', 'TW', 'EE', 'FO', 'IK', 'SM', 'MH', 'HT', 'KU', 'HE', 'BE', 'TT', 'GN', 'NV', 'MI', 'FJ', 'HO', 'MG', '+50'],
        'max_length': 4096,
        'size': 'Base',
        'multilingual': True,
        'description': 'LongT5-TGlobal: Multilingual encoder-decoder with transient-global attention (up to 4096 tokens). Better global context than local variant. ~248M parameters. Use for: long multilingual docs needing global context, generation tasks. Encoder-decoder architecture, more complex than encoder-only.'
    },
}


def format_language_display(languages: List[str], max_width: int = 15) -> str:
    """
    Format language list for compact display.

    Args:
        languages: List of language codes
        max_width: Maximum character width for display

    Returns:
        Formatted string with truncation if needed

    Examples:
        ['EN'] -> 'EN'
        ['EN', 'FR'] -> 'EN, FR'
        ['EN', 'FR', 'DE', ...] -> 'EN, FR, DE...'
        [100+ languages] -> 'Multi (100+)'
    """
    if not languages:
        return "?"

    # For very multilingual models (10+ languages), show compact format
    if len(languages) > 10:
        return f"Multi ({len(languages)}+)"

    # For few languages, try to show them all
    lang_str = ', '.join(languages)

    # If it fits, return as-is
    if len(lang_str) <= max_width:
        return lang_str

    # Otherwise, truncate with ellipsis
    # Show as many languages as fit, then add "..."
    shown_langs = []
    current_len = 0
    for lang in languages:
        test_len = current_len + len(lang) + (2 if shown_langs else 0)  # +2 for ", "
        if test_len + 3 > max_width:  # +3 for "..."
            break
        shown_langs.append(lang)
        current_len = test_len

    if shown_langs:
        return ', '.join(shown_langs) + '...'
    else:
        # Even first language doesn't fit, just show count
        return f"{len(languages)} langs"


def calculate_model_relevance_score(
    model_meta: Dict[str, Any],
    user_languages: Set[str],
    avg_text_length: float,
    requires_long_model: bool = False
) -> float:
    """
    Calculate relevance score for a model based on user's data characteristics.

    Args:
        model_meta: Model metadata dictionary
        user_languages: Set of language codes from user's data (e.g., {'FR', 'EN'})
        avg_text_length: Average text length in tokens or characters
        requires_long_model: Whether user needs long-document models

    Returns:
        Relevance score (higher = more relevant)
    """
    score = 0.0

    # Language match (highest priority)
    model_langs = set(model_meta.get('languages', []))
    is_multilingual_model = model_meta.get('multilingual', False) or len(model_langs) > 3

    # Check if user is specifically looking for multilingual models
    if user_languages == {'MULTI'}:
        # User explicitly wants multilingual models only
        if is_multilingual_model:
            score += 100  # Perfect match - multilingual model for multilingual request
        else:
            score += 0  # Monolingual model not relevant for multilingual request
    elif is_multilingual_model:
        # Multilingual models: lower base score for single language requests
        matching_langs = user_languages & model_langs
        if matching_langs:
            # Multilingual model supports the language, but not optimal for single-language use
            if len(user_languages) == 1:
                score += 60  # Decent match, but prefer language-specific models
            else:
                score += 80  # Good match for multiple languages
                score += min(len(matching_langs) * 5, 20)  # Bonus for supporting multiple user languages
        else:
            score += 0  # No language match
    else:
        # Monolingual/few-language models: BEST for single-language tasks
        matching_langs = user_languages & model_langs
        if matching_langs:
            if len(user_languages) == 1:
                # Perfect match: monolingual model for monolingual data
                score += 120  # BONUS for language-specific models when user selects ONE language
            else:
                # Monolingual model but user has multiple languages
                score += 40  # Partial match - not ideal
        else:
            score += 0  # No match, low relevance

    # Text length match
    max_length = model_meta.get('max_length', 512)
    if requires_long_model or avg_text_length > 512:
        # Prefer long-document models
        if max_length > 512:
            score += 80  # Long model for long docs
        else:
            score -= 30  # Short model for long docs (not ideal)
    else:
        # Standard length docs
        if max_length <= 512:
            score += 30  # Standard model for standard docs
        else:
            score += 10  # Long model for standard docs (slower but works)

    # Model size preference (medium preference for base, small penalty for large/small)
    size = model_meta.get('size', 'Base')
    if size == 'Base':
        score += 10
    elif size == 'Small' or size == 'XSmall':
        score += 5  # Faster but slightly less accurate
    elif size == 'Large':
        score += 0  # More accurate but slower

    return score


def display_all_models(
    languages: Optional[Set[str]] = None,
    avg_text_length: Optional[float] = None,
    requires_long_model: bool = False,
    console: Optional[Console] = None
) -> Table:
    """
    Display all available models in a comprehensive table, optionally filtered
    and sorted by relevance to user's data.

    Args:
        languages: Set of language codes from user's data (e.g., {'FR', 'EN'})
        avg_text_length: Average text length in tokens/characters
        requires_long_model: Whether user needs long-document models
        console: Rich console for display (optional)

    Returns:
        Rich Table object with all models
    """
    if console is None:
        console = Console()

    # Calculate relevance scores if filtering criteria provided
    scored_models = []
    for model_id, meta in MODEL_METADATA.items():
        if languages and avg_text_length is not None:
            score = calculate_model_relevance_score(
                meta, languages, avg_text_length, requires_long_model
            )
        else:
            score = 0  # No scoring, alphabetical order

        scored_models.append((model_id, meta, score))

    # Sort by score (descending) if scores were calculated
    if languages and avg_text_length is not None:
        scored_models.sort(key=lambda x: x[2], reverse=True)
    else:
        # Sort alphabetically by name
        scored_models.sort(key=lambda x: x[0])

    # Create table
    title = "📚 All Available Models"
    if languages:
        lang_str = ', '.join(sorted(languages))
        title += f" (filtered for {lang_str})"

    table = Table(title=title, box=box.ROUNDED, show_lines=False)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Model ID", style="white", width=50)
    table.add_column("Languages", style="yellow", width=12)
    table.add_column("Max Length", style="green", width=12, justify="center")
    table.add_column("Size", style="magenta", width=8, justify="center")
    table.add_column("Description", style="dim", width=45)

    # Add rows
    for idx, (model_id, meta, score) in enumerate(scored_models, 1):
        # Determine language display
        langs = meta.get('languages', [])
        if len(langs) > 10:  # Multilingual models with many languages
            # Show key languages and total count
            key_langs = ['EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'ZH', 'JA', 'AR', 'RU']
            shown_langs = [l for l in key_langs if l in langs][:5]  # Show max 5 key languages
            lang_display = ', '.join(shown_langs) + f' +{len(langs)-5}'
        elif 'MULTI' in langs:  # Legacy support
            lang_display = "Multi (100+)"
        else:
            lang_display = ', '.join(langs)

        # Highlight recommended models (score > 70)
        if score > 70:
            style = "bold green"
        elif score > 40:
            style = "bold yellow"
        else:
            style = ""

        table.add_row(
            str(idx),
            model_id,
            lang_display,
            f"{meta.get('max_length', 512)} tokens",
            meta.get('size', 'Base'),
            meta.get('description', ''),
            style=style
        )

    return table


def get_recommended_models(
    languages: Set[str],
    avg_text_length: float,
    requires_long_model: bool = False,
    top_n: int = 10
) -> List[str]:
    """
    Get top N recommended models for user's data.

    Args:
        languages: Set of language codes from user's data
        avg_text_length: Average text length in tokens/characters
        requires_long_model: Whether user needs long-document models
        top_n: Number of top models to return

    Returns:
        List of model IDs, sorted by relevance (best first)
    """
    scored_models = []
    for model_id, meta in MODEL_METADATA.items():
        score = calculate_model_relevance_score(
            meta, languages, avg_text_length, requires_long_model
        )
        scored_models.append((model_id, score))

    # Sort by score descending
    scored_models.sort(key=lambda x: x[1], reverse=True)

    # Filter out models with score 0 (not relevant), then return top N
    # This ensures we only show relevant models, not just filling up to top_n
    relevant_models = [model_id for model_id, score in scored_models if score > 0]

    # Return top N model IDs from relevant models only
    return relevant_models[:top_n]
