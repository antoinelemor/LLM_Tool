"""
PROJECT:
-------
LLMTool

TITLE:
------
sota_models.py

MAIN OBJECTIVE:
---------------
This script provides state-of-the-art transformer models including DeBERTa-v3, RoBERTa,
ELECTRA, ALBERT, BigBird, Longformer, and French-specific models (CamemBERTa-v2, FlauBERT,
etc.) offering superior performance compared to standard BERT for various NLP tasks.

Dependencies:
-------------
- transformers (HuggingFace model classes and tokenizers)
- LLMTool.bert_base (base implementation)

MAIN FEATURES:
--------------
1) DeBERTa-v3 models (XSmall, Base, Large) with disentangled attention
2) RoBERTa variants (Base, Large, Distilled) with robust pretraining
3) ELECTRA models (Small, Base, Large) for efficient discrimination
4) ALBERT models (Base, Large, XLarge) with parameter sharing
5) Long-context models (BigBird, Longformer) for documents up to 4096 tokens
6) Multilingual models (mDeBERTa-v3, XLM-RoBERTa) for 100+ languages
7) French-specific models (CamemBERTa-v2, FlauBERT, DistilCamemBERT, FrALBERT, FrELECTRA, BARThez)
8) All models inherit full training pipeline from BertBase

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
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification
        )


class DeBERTaV3Large(BertBase):
    """DeBERTa-v3 Large: High-performance model for demanding tasks."""

    def __init__(
            self,
            model_name='microsoft/deberta-v3-large',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification
        )


class DeBERTaV3XSmall(BertBase):
    """DeBERTa-v3 XSmall: Efficient model for resource-constrained environments."""

    def __init__(
            self,
            model_name='microsoft/deberta-v3-xsmall',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification
        )


class RoBERTaBase(BertBase):
    """RoBERTa: Robustly optimized BERT pretraining approach."""

    def __init__(
            self,
            model_name='roberta-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification
        )


class RoBERTaLarge(BertBase):
    """RoBERTa Large: Enhanced capacity for complex tasks."""

    def __init__(
            self,
            model_name='roberta-large',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification
        )


class DistilRoBERTa(BertBase):
    """Distilled RoBERTa: Lighter and faster while maintaining performance."""

    def __init__(
            self,
            model_name='distilroberta-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=RobertaTokenizer,
            device=device,
            model_sequence_classifier=RobertaForSequenceClassification
        )


class ELECTRABase(BertBase):
    """ELECTRA: Pre-training text encoders as discriminators rather than generators."""

    def __init__(
            self,
            model_name='google/electra-base-discriminator',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification
        )


class ELECTRALarge(BertBase):
    """ELECTRA Large: Superior efficiency-performance trade-off."""

    def __init__(
            self,
            model_name='google/electra-large-discriminator',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification
        )


class ELECTRASmall(BertBase):
    """ELECTRA Small: Highly efficient for edge deployments."""

    def __init__(
            self,
            model_name='google/electra-small-discriminator',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification
        )


class ALBERTBase(BertBase):
    """ALBERT: A Lite BERT for self-supervised learning."""

    def __init__(
            self,
            model_name='albert-base-v2',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification
        )


class ALBERTLarge(BertBase):
    """ALBERT Large: Parameter-efficient large model."""

    def __init__(
            self,
            model_name='albert-large-v2',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification
        )


class ALBERTXLarge(BertBase):
    """ALBERT XLarge: Maximum capacity with parameter sharing."""

    def __init__(
            self,
            model_name='albert-xlarge-v2',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification
        )


class BigBirdBase(BertBase):
    """BigBird: Transformers for longer sequences (up to 4096 tokens)."""

    def __init__(
            self,
            model_name='google/bigbird-roberta-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BigBirdTokenizer,
            device=device,
            model_sequence_classifier=BigBirdForSequenceClassification
        )


class BigBirdLarge(BertBase):
    """BigBird Large: Extended context for document-level understanding."""

    def __init__(
            self,
            model_name='google/bigbird-roberta-large',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BigBirdTokenizer,
            device=device,
            model_sequence_classifier=BigBirdForSequenceClassification
        )


class LongformerBase(BertBase):
    """Longformer: Efficient attention for long documents (up to 4096 tokens)."""

    def __init__(
            self,
            model_name='allenai/longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=LongformerTokenizer,
            device=device,
            model_sequence_classifier=LongformerForSequenceClassification
        )


class LongformerLarge(BertBase):
    """Longformer Large: Maximum performance on long sequences."""

    def __init__(
            self,
            model_name='allenai/longformer-large-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=LongformerTokenizer,
            device=device,
            model_sequence_classifier=LongformerForSequenceClassification
        )


# Multilingual SOTA models
class MDeBERTaV3Base(BertBase):
    """Multilingual DeBERTa-v3: State-of-the-art for multilingual tasks."""

    def __init__(
            self,
            model_name='microsoft/mdeberta-v3-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=DebertaV2Tokenizer,
            device=device,
            model_sequence_classifier=DebertaV2ForSequenceClassification
        )


class XLMRobertaBase(BertBase):
    """XLM-RoBERTa Base: Cross-lingual understanding (100+ languages)."""

    def __init__(
            self,
            model_name='xlm-roberta-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification
        )


class XLMRobertaLarge(BertBase):
    """XLM-RoBERTa Large: Superior multilingual performance."""

    def __init__(
            self,
            model_name='xlm-roberta-large',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=XLMRobertaTokenizer,
            device=device,
            model_sequence_classifier=XLMRobertaForSequenceClassification
        )


# French-specific SOTA models (equivalents to English models)
class CamembertaV2Base(BertBase):
    """CamemBERTa-v2 Base: Modern French equivalent to RoBERTa Base."""

    def __init__(
            self,
            model_name='almanach/camemberta-base',
            device=None
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class CamembertLarge(BertBase):
    """CamemBERT Large: French language model (legacy)."""

    def __init__(
            self,
            model_name='camembert/camembert-large',
            device=None
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


class FlauBERTBase(BertBase):
    """FlauBERT Base: French BERT with better performance than CamemBERT base."""

    def __init__(
            self,
            model_name='flaubert/flaubert_base_cased',
            device=None
    ):
        from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=FlaubertTokenizer,
            device=device,
            model_sequence_classifier=FlaubertForSequenceClassification
        )


class FlauBERTLarge(BertBase):
    """FlauBERT Large: Large French language model."""

    def __init__(
            self,
            model_name='flaubert/flaubert_large_cased',
            device=None
    ):
        from transformers import FlaubertTokenizer, FlaubertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=FlaubertTokenizer,
            device=device,
            model_sequence_classifier=FlaubertForSequenceClassification
        )


class BARThez(BertBase):
    """BARThez: French BART model for sequence classification."""

    def __init__(
            self,
            model_name='moussaKam/barthez',
            device=None
    ):
        from transformers import MBartTokenizer, MBartForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=MBartTokenizer,
            device=device,
            model_sequence_classifier=MBartForSequenceClassification
        )


class FrALBERT(BertBase):
    """FrALBERT: French ALBERT model - lightweight but effective."""

    def __init__(
            self,
            model_name='cls/fralbert',
            device=None
    ):
        from transformers import AlbertTokenizer, AlbertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=AlbertTokenizer,
            device=device,
            model_sequence_classifier=AlbertForSequenceClassification
        )


class DistilCamemBERT(BertBase):
    """DistilCamemBERT: Distilled version of CamemBERT - faster inference."""

    def __init__(
            self,
            model_name='cmarkea/distilcamembert-base',
            device=None
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


class FrELECTRA(BertBase):
    """French ELECTRA: Discriminator-based French model."""

    def __init__(
            self,
            model_name='dbmdz/electra-base-french-europeana-cased-discriminator',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=ElectraTokenizer,
            device=device,
            model_sequence_classifier=ElectraForSequenceClassification
        )


class CamembertBioBERT(BertBase):
    """CamemBERT-bio: French biomedical language model."""

    def __init__(
            self,
            model_name='almanach/camembert-bio-base',
            device=None
    ):
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


# ============================================================================
# LONG-DOCUMENT MODELS (4096+ tokens) - Multilingual and Language-Specific
# ============================================================================

class XLMRobertaLongformer(BertBase):
    """XLM-RoBERTa Longformer: Multilingual long-document model (4096 tokens, 100+ languages).
    Best choice for long documents in multiple languages."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class LongT5Base(BertBase):
    """LongT5: Multilingual long-document model with local attention (4096+ tokens, 100+ languages).
    Efficient for very long multilingual documents."""

    def __init__(
            self,
            model_name='google/long-t5-local-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=T5Tokenizer,
            device=device,
            model_sequence_classifier=T5ForSequenceClassification
        )


class LongT5TGlobalBase(BertBase):
    """LongT5 TGlobal: Multilingual long-document model with transient global attention (4096+ tokens).
    Alternative to LongT5Local with different attention mechanism."""

    def __init__(
            self,
            model_name='google/long-t5-tglobal-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=T5Tokenizer,
            device=device,
            model_sequence_classifier=T5ForSequenceClassification
        )


class LEDBase(BertBase):
    """LED (Longformer Encoder-Decoder) Base: English long-document model (16384 tokens).
    Best for very long English documents."""

    def __init__(
            self,
            model_name='allenai/led-base-16384',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class LEDLarge(BertBase):
    """LED Large: High-performance English long-document model (16384 tokens)."""

    def __init__(
            self,
            model_name='allenai/led-large-16384',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# French Long-Document Models
class FrenchLongformer(BertBase):
    """French-optimized Longformer: Uses XLM-RoBERTa Longformer with French focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Spanish Long-Document Models
class SpanishLongformer(BertBase):
    """Spanish-optimized Longformer: Uses XLM-RoBERTa Longformer with Spanish focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class SpanishRoBERTaBNE(BertBase):
    """Spanish RoBERTa-BNE: State-of-the-art Spanish model from Spanish National Library (512 tokens)."""

    def __init__(
            self,
            model_name='PlanTL-GOB-ES/roberta-base-bne',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# German Long-Document Models
class GermanLongformer(BertBase):
    """German-optimized Longformer: Uses XLM-RoBERTa Longformer with German focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class GermanGBERT(BertBase):
    """German GBERT: Deepset's optimized German BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='deepset/gbert-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Italian Long-Document Models
class ItalianLongformer(BertBase):
    """Italian-optimized Longformer: Uses XLM-RoBERTa Longformer with Italian focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Portuguese Long-Document Models
class PortugueseLongformer(BertBase):
    """Portuguese-optimized Longformer: Uses XLM-RoBERTa Longformer with Portuguese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Dutch Long-Document Models
class DutchLongformer(BertBase):
    """Dutch-optimized Longformer: Uses XLM-RoBERTa Longformer with Dutch focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class DutchBERT(BertBase):
    """Dutch BERT: GroNLP's optimized Dutch BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='GroNLP/bert-base-dutch-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Polish Long-Document Models
class PolishLongformer(BertBase):
    """Polish-optimized Longformer: Uses XLM-RoBERTa Longformer with Polish focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class PolishHerBERT(BertBase):
    """Polish HerBERT: Allegro's optimized Polish BERT model (512 tokens)."""

    def __init__(
            self,
            model_name='allegro/herbert-base-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Chinese Long-Document Models
class ChineseLongformer(BertBase):
    """Chinese-optimized Longformer: Uses XLM-RoBERTa Longformer with Chinese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class ChineseRoBERTa(BertBase):
    """Chinese RoBERTa WWM: HFL's optimized Chinese RoBERTa model (512 tokens)."""

    def __init__(
            self,
            model_name='hfl/chinese-roberta-wwm-ext',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Japanese Long-Document Models
class JapaneseLongformer(BertBase):
    """Japanese-optimized Longformer: Uses XLM-RoBERTa Longformer with Japanese focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class JapaneseBERTWWM(BertBase):
    """Japanese BERT WWM: Tohoku's Japanese BERT with Whole Word Masking (512 tokens)."""

    def __init__(
            self,
            model_name='cl-tohoku/bert-base-japanese-whole-word-masking',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Arabic Long-Document Models
class ArabicLongformer(BertBase):
    """Arabic-optimized Longformer: Uses XLM-RoBERTa Longformer with Arabic focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


class ArabicAraBERT(BertBase):
    """Arabic AraBERT v2: AUB MIND Lab's state-of-the-art Arabic BERT (512 tokens)."""

    def __init__(
            self,
            model_name='aubmindlab/bert-base-arabertv2',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )


# Russian Long-Document Models
class RussianLongformer(BertBase):
    """Russian-optimized Longformer: Uses XLM-RoBERTa Longformer with Russian focus (4096 tokens)."""

    def __init__(
            self,
            model_name='facebook/xlm-roberta-longformer-base-4096',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=AutoTokenizer,
            device=device,
            model_sequence_classifier=AutoModelForSequenceClassification
        )