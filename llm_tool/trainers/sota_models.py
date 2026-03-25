#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
sota_models.py

MAIN OBJECTIVE:
---------------
Bundle state-of-the-art transformer wrappers (DeBERTa, RoBERTa, ELECTRA,
Longformer, etc.) so the Training Arena can instantiate them through a
unified BertBase-derived interface.

Dependencies:
-------------
- transformers
- llm_tool.trainers.bert_base

MAIN FEATURES:
--------------
1) Provide DeBERTa-v3 variants tuned for disparate resource envelopes
2) Wrap RoBERTa, ELECTRA, and ALBERT families with consistent constructors
3) Expose long-context backbones such as BigBird, Longformer, and LED
4) Include multilingual options like mDeBERTa and XLM-RoBERTa hubs
5) Offer French-specific models tailored to CamemBERT and FlauBERT ecosystems

Author:
-------
Antoine Lemor
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    ElectraTokenizer,
    ElectraForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    LongformerTokenizer,
    LongformerForSequenceClassification,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    T5Tokenizer,
    T5ForSequenceClassification,
)

from llm_tool.trainers.bert_base import BertBase


class DeBERTaV3Base(BertBase):
    """DeBERTa-v3: Decoding-enhanced BERT with disentangled attention.
    Superior performance on most NLU benchmarks."""

    def __init__(
            self,
            model_name='microsoft/deberta-v3-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification,
            **kwargs
        )


class DeBERTaV3Large(BertBase):
    """DeBERTa-v3 Large: High-performance model for demanding tasks."""

    def __init__(
            self,
            model_name='microsoft/deberta-v3-large',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification,
            **kwargs
        )


class DeBERTaV3XSmall(BertBase):
    """DeBERTa-v3 XSmall: Efficient model for resource-constrained environments."""

    def __init__(
            self,
            model_name='microsoft/deberta-v3-xsmall',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification,
            **kwargs
        )


class RoBERTaBase(BertBase):
    """RoBERTa: Robustly optimized BERT pretraining approach."""

    def __init__(
            self,
            model_name='roberta-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification,
            **kwargs
        )


class RoBERTaLarge(BertBase):
    """RoBERTa Large: Enhanced capacity for complex tasks."""

    def __init__(
            self,
            model_name='roberta-large',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification,
            **kwargs
        )


class DistilRoBERTa(BertBase):
    """Distilled RoBERTa: Lighter and faster while maintaining performance."""

    def __init__(
            self,
            model_name='distilroberta-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification,
            **kwargs
        )


class ELECTRABase(BertBase):
    """ELECTRA: Pre-training text encoders as discriminators rather than generators."""

    def __init__(
            self,
            model_name='google/electra-base-discriminator',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification,
            **kwargs
        )


class ELECTRALarge(BertBase):
    """ELECTRA Large: Superior efficiency-performance trade-off."""

    def __init__(
            self,
            model_name='google/electra-large-discriminator',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification,
            **kwargs
        )


class ELECTRASmall(BertBase):
    """ELECTRA Small: Highly efficient for edge deployments."""

    def __init__(
            self,
            model_name='google/electra-small-discriminator',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification,
            **kwargs
        )


class ALBERTBase(BertBase):
    """ALBERT: A Lite BERT for self-supervised learning."""

    def __init__(
            self,
            model_name='albert-base-v2',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification,
            **kwargs
        )


class ALBERTLarge(BertBase):
    """ALBERT Large: Parameter-efficient large model."""

    def __init__(
            self,
            model_name='albert-large-v2',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification,
            **kwargs
        )


class ALBERTXLarge(BertBase):
    """ALBERT XLarge: Maximum capacity with parameter sharing."""

    def __init__(
            self,
            model_name='albert-xlarge-v2',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification,
            **kwargs
        )


class BigBirdBase(BertBase):
    """BigBird: Transformers for longer sequences (up to 4096 tokens)."""

    def __init__(
            self,
            model_name='google/bigbird-roberta-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BigBirdTokenizer,
            device=device,
            model_sequence_classifier=BigBirdForSequenceClassification,
            **kwargs
        )


class BigBirdLarge(BertBase):
    """BigBird Large: Extended context for document-level understanding."""

    def __init__(
            self,
            model_name='google/bigbird-roberta-large',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BigBirdTokenizer,
            device=device,
            model_sequence_classifier=BigBirdForSequenceClassification,
            **kwargs
        )


class LongformerBase(BertBase):
    """Longformer: Efficient attention for long documents (up to 4096 tokens)."""

    def __init__(
            self,
            model_name='allenai/longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=LongformerTokenizer,
            device=device,
            model_sequence_classifier=LongformerForSequenceClassification,
            **kwargs
        )


class LongformerLarge(BertBase):
    """Longformer Large: Maximum performance on long sequences."""

    def __init__(
            self,
            model_name='allenai/longformer-large-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=LongformerTokenizer,
            device=device,
            model_sequence_classifier=LongformerForSequenceClassification,
            **kwargs
        )


# Multilingual SOTA models
class MDeBERTaV3Base(BertBase):
    """Multilingual DeBERTa-v3: State-of-the-art for multilingual tasks."""

    def __init__(
            self,
            model_name='microsoft/mdeberta-v3-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification,
            **kwargs
        )


class XLMRobertaBase(BertBase):
    """XLM-RoBERTa Base: Cross-lingual understanding (100+ languages)."""

    def __init__(
            self,
            model_name='xlm-roberta-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification,
            **kwargs
        )


class XLMRobertaLarge(BertBase):
    """XLM-RoBERTa Large: Superior multilingual performance."""

    def __init__(
            self,
            model_name='xlm-roberta-large',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification,
            **kwargs
        )


# French-specific SOTA models (equivalents to English models)
class CamembertaV2Base(BertBase):
    """CamemBERTa-v2 Base: Modern French equivalent to RoBERTa Base."""

    def __init__(
            self,
            model_name='almanach/camemberta-base',
            device=None,
            **kwargs
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class CamembertLarge(BertBase):
    """CamemBERT Large: French language model (legacy)."""

    def __init__(
            self,
            model_name='camembert/camembert-large',
            device=None,
            **kwargs
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification,
            **kwargs
        )


class FlauBERTBase(BertBase):
    """FlauBERT Base: French BERT with better performance than CamemBERT base."""

    def __init__(
            self,
            model_name='flaubert/flaubert_base_cased',
            device=None,
            **kwargs
    ):
        from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=FlaubertTokenizer,
            device=device,
            model_sequence_classifier=FlaubertForSequenceClassification,
            **kwargs
        )


class FlauBERTLarge(BertBase):
    """FlauBERT Large: Large French language model."""

    def __init__(
            self,
            model_name='flaubert/flaubert_large_cased',
            device=None,
            **kwargs
    ):
        from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=FlaubertTokenizer,
            device=device,
            model_sequence_classifier=FlaubertForSequenceClassification,
            **kwargs
        )


class BARThez(BertBase):
    """BARThez: French BART model for sequence classification."""

    def __init__(
            self,
            model_name='moussaKam/barthez',
            device=None,
            **kwargs
    ):
        from transformers import MBartTokenizer, MBartForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=MBartTokenizer,
            device=device,
            model_sequence_classifier=MBartForSequenceClassification,
            **kwargs
        )


class FrALBERT(BertBase):
    """FrALBERT: French ALBERT model - lightweight but effective."""

    def __init__(
            self,
            model_name='cls/fralbert',
            device=None,
            **kwargs
    ):
        from transformers import AlbertTokenizer, AlbertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification,
            **kwargs
        )


class DistilCamemBERT(BertBase):
    """DistilCamemBERT: Distilled version of CamemBERT - faster inference."""

    def __init__(
            self,
            model_name='cmarkea/distilcamembert-base',
            device=None,
            **kwargs
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification,
            **kwargs
        )


class FrELECTRA(BertBase):
    """French ELECTRA: Discriminator-based French model."""

    def __init__(
            self,
            model_name='dbmdz/electra-base-french-europeana-cased-discriminator',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification,
            **kwargs
        )


class CamembertBioBERT(BertBase):
    """CamemBERT-bio: French biomedical language model."""

    def __init__(
            self,
            model_name='almanach/camembert-bio-base',
            device=None,
            **kwargs
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification,
            **kwargs
        )


# ============================================================================
# LONG-DOCUMENT MODELS (4096+ tokens) - Multilingual and Language-Specific
# ============================================================================

class XLMRobertaLongformer(BertBase):
    """XLM-RoBERTa Longformer: Multilingual long-document model (4096 tokens, 100+ languages).
    Best choice for long documents in multiple languages."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class LongT5Base(BertBase):
    """LongT5: Multilingual long-document model with local attention (4096+ tokens, 100+ languages).
    Efficient for very long multilingual documents."""

    def __init__(
            self,
            model_name='google/long-t5-local-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=T5Tokenizer,
            device=device,
            model_sequence_classifier=T5ForSequenceClassification,
            **kwargs
        )


class LongT5TGlobalBase(BertBase):
    """LongT5 TGlobal: Multilingual long-document model with transient global attention (4096+ tokens).
    Alternative to LongT5Local with different attention mechanism."""

    def __init__(
            self,
            model_name='google/long-t5-tglobal-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=T5Tokenizer,
            device=device,
            model_sequence_classifier=T5ForSequenceClassification,
            **kwargs
        )


class LEDBase(BertBase):
    """LED (Longformer Encoder-Decoder) Base: English long-document model (16384 tokens).
    Best for very long English documents."""

    def __init__(
            self,
            model_name='allenai/led-base-16384',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class LEDLarge(BertBase):
    """LED Large: High-performance English long-document model (16384 tokens)."""

    def __init__(
            self,
            model_name='allenai/led-large-16384',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# French Long-Document Models
class FrenchLongformer(BertBase):
    """French-optimized Longformer: Uses XLM-RoBERTa Longformer with French focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Spanish Long-Document Models
class SpanishLongformer(BertBase):
    """Spanish-optimized Longformer: Uses XLM-RoBERTa Longformer with Spanish focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class SpanishRoBERTaBNE(BertBase):
    """Spanish RoBERTa-BNE: State-of-the-art Spanish model from Spanish National Library (512 tokens)."""

    def __init__(
            self,
            model_name='PlanTL-GOB-ES/roberta-base-bne',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# German Long-Document Models
class GermanLongformer(BertBase):
    """German-optimized Longformer: Uses XLM-RoBERTa Longformer with German focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class GermanGBERT(BertBase):
    """German GBERT: Deepset's optimized German BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='deepset/gbert-base',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Italian Long-Document Models
class ItalianLongformer(BertBase):
    """Italian-optimized Longformer: Uses XLM-RoBERTa Longformer with Italian focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Portuguese Long-Document Models
class PortugueseLongformer(BertBase):
    """Portuguese-optimized Longformer: Uses XLM-RoBERTa Longformer with Portuguese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Dutch Long-Document Models
class DutchLongformer(BertBase):
    """Dutch-optimized Longformer: Uses XLM-RoBERTa Longformer with Dutch focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class DutchBERT(BertBase):
    """Dutch BERT: GroNLP's optimized Dutch BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='GroNLP/bert-base-dutch-cased',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Polish Long-Document Models
class PolishLongformer(BertBase):
    """Polish-optimized Longformer: Uses XLM-RoBERTa Longformer with Polish focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class PolishHerBERT(BertBase):
    """Polish HerBERT: Allegro's optimized Polish BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='allegro/herbert-base-cased',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Chinese Long-Document Models
class ChineseLongformer(BertBase):
    """Chinese-optimized Longformer: Uses XLM-RoBERTa Longformer with Chinese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class ChineseRoBERTa(BertBase):
    """Chinese RoBERTa WWM: HFL's optimized Chinese RoBERTa model (512 tokens)."""

    def __init__(
            self,
            model_name='hfl/chinese-roberta-wwm-ext',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Japanese Long-Document Models
class JapaneseLongformer(BertBase):
    """Japanese-optimized Longformer: Uses XLM-RoBERTa Longformer with Japanese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class JapaneseBERTWWM(BertBase):
    """Japanese BERT WWM: Tohoku's Japanese BERT with Whole Word Masking (512 tokens)."""

    def __init__(
            self,
            model_name='cl-tohoku/bert-base-japanese-whole-word-masking',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Arabic Long-Document Models
class ArabicLongformer(BertBase):
    """Arabic-optimized Longformer: Uses XLM-RoBERTa Longformer with Arabic focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


class ArabicAraBERT(BertBase):
    """Arabic AraBERT v2: AUB MIND Lab's state-of-the-art Arabic BERT (512 tokens)."""

    def __init__(
            self,
            model_name='aubmindlab/bert-base-arabertv2',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# Russian Long-Document Models
class RussianLongformer(BertBase):
    """Russian-optimized Longformer: Uses XLM-RoBERTa Longformer with Russian focus (4096 tokens)."""

    def __init__(
            self,
            model_name='markussagen/xlm-roberta-longformer-base-4096',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )

# ===========================================================================
# AUTO MODEL WRAPPER FOR UNKNOWN MODELS
# ===========================================================================

class AutoModelWrapper(BertBase):
    """
    Universal wrapper that uses AutoTokenizer and AutoModelForSequenceClassification.

    This class provides maximum compatibility with any HuggingFace model by using
    the Auto classes which automatically detect the correct architecture.
    Used as a fallback when the model type cannot be matched to a specific class.
    """

    def __init__(
            self,
            model_name: str = 'bert-base-cased',
            device=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification,
            **kwargs
        )


# ===========================================================================
# MODEL NAME TO CLASS MAPPING
# ===========================================================================

def get_model_class_for_name(model_name: str):
    """
    Get the appropriate model class for a given HuggingFace model name.
    
    This function maps model names to their correct wrapper classes that use
    the appropriate tokenizer and model architecture.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'markussagen/xlm-roberta-longformer-base-4096')
    
    Returns:
        Model class (e.g., XLMRobertaLongformer) or BertBase as fallback
    
    Examples:
        >>> cls = get_model_class_for_name('markussagen/xlm-roberta-longformer-base-4096')
        >>> model = cls()
    """
    # Normalize model name for matching
    name_lower = model_name.lower()
    
    # Long-document models (4096+ tokens)
    if 'xlm-roberta-longformer' in name_lower or 'xlm-long' in name_lower:
        return XLMRobertaLongformer
    if 'longformer' in name_lower and 'allenai' in name_lower:
        if 'large' in name_lower:
            return LongformerLarge
        return LongformerBase
    if 'bigbird' in name_lower:
        if 'large' in name_lower:
            return BigBirdLarge
        return BigBirdBase
    if 'led-' in name_lower or 'longformer-encoder-decoder' in name_lower:
        if 'large' in name_lower:
            return LEDLarge
        return LEDBase
    if 'long-t5' in name_lower:
        if 'tglobal' in name_lower:
            return LongT5TGlobalBase
        return LongT5Base
    
    # DeBERTa models
    if 'deberta-v3' in name_lower or 'deberta-v2' in name_lower:
        if 'xsmall' in name_lower or 'small' in name_lower:
            return DeBERTaV3XSmall
        elif 'large' in name_lower:
            return DeBERTaV3Large
        else:
            return DeBERTaV3Base
    if 'mdeberta' in name_lower:
        return MDeBERTaV3Base
    
    # RoBERTa models
    if 'xlm-roberta' in name_lower:
        if 'large' in name_lower:
            return XLMRobertaLarge
        return XLMRobertaBase
    if 'roberta' in name_lower:
        if 'large' in name_lower:
            return RoBERTaLarge
        elif 'distil' in name_lower:
            return DistilRoBERTa
        return RoBERTaBase
    
    # ELECTRA models
    if 'electra' in name_lower:
        if 'large' in name_lower:
            return ELECTRALarge
        elif 'small' in name_lower:
            return ELECTRASmall
        return ELECTRABase
    
    # ALBERT models
    if 'albert' in name_lower:
        if 'xlarge' in name_lower:
            return ALBERTXLarge
        elif 'large' in name_lower:
            return ALBERTLarge
        return ALBERTBase
    
    # French models
    if 'camembert' in name_lower:
        if 'distil' in name_lower:
            return DistilCamemBERT
        elif 'large' in name_lower:
            return CamembertLarge
        return CamembertaV2Base
    if 'flaubert' in name_lower:
        if 'large' in name_lower:
            return FlauBERTLarge
        return FlauBERTBase
    if 'fralbert' in name_lower:
        return FrALBERT
    if 'barthez' in name_lower:
        return BARThez

    # German models
    if 'gbert' in name_lower or ('german' in name_lower and 'deepset' in name_lower):
        return GermanGBERT

    # Spanish models
    if 'beto' in name_lower or 'bne' in name_lower or 'spanish' in name_lower:
        return AutoModelWrapper

    # Italian models
    if 'umberto' in name_lower or 'italian' in name_lower:
        return AutoModelWrapper

    # Dutch models
    if 'bertje' in name_lower or 'robbert' in name_lower or 'dutch' in name_lower:
        return AutoModelWrapper

    # Polish models
    if 'herbert' in name_lower or 'polish' in name_lower:
        return AutoModelWrapper

    # Arabic models
    if 'arabert' in name_lower or 'marbert' in name_lower or 'arabic' in name_lower:
        return AutoModelWrapper

    # Japanese models
    if 'tohoku' in name_lower or 'japanese' in name_lower:
        return AutoModelWrapper

    # Korean models
    if 'klue' in name_lower or 'korean' in name_lower:
        return AutoModelWrapper

    # Chinese models (beyond bert-base-chinese)
    if 'chinese' in name_lower and ('roberta' in name_lower or 'electra' in name_lower):
        return AutoModelWrapper

    # Russian models (beyond rubert)
    if 'rubert' in name_lower or 'russian' in name_lower:
        return AutoModelWrapper

    # Portuguese models (beyond neuralmind)
    if 'bertimbau' in name_lower or 'portuguese' in name_lower:
        return AutoModelWrapper

    # Turkish models
    if 'turkish' in name_lower or 'berturk' in name_lower:
        return AutoModelWrapper

    # Vietnamese models
    if 'vietnamese' in name_lower or 'phobert' in name_lower:
        return AutoModelWrapper

    # Thai models
    if 'thai' in name_lower or 'wangchanberta' in name_lower:
        return AutoModelWrapper

    # Indonesian models
    if 'indonesian' in name_lower or 'indobert' in name_lower:
        return AutoModelWrapper

    # Nordic models (Danish, Norwegian, Finnish, Swedish)
    if any(lang in name_lower for lang in ['danish', 'norwegian', 'finnish', 'nordic']):
        return AutoModelWrapper

    # Other European models
    if any(lang in name_lower for lang in ['greek', 'hebrew', 'czech', 'romanian', 'bulgarian', 'ukrainian', 'croatian', 'serbian', 'slovak', 'hungarian']):
        return AutoModelWrapper

    # BERT-based multilingual models
    if 'bert-base-multilingual' in name_lower or 'multilingual' in name_lower:
        return AutoModelWrapper

    # DistilBERT variants
    if 'distilbert' in name_lower:
        return AutoModelWrapper

    # Default to AutoModelWrapper for unknown models
    # This uses AutoTokenizer and AutoModelForSequenceClassification for maximum compatibility
    return AutoModelWrapper
