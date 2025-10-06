"""
PROJECT:
-------
LLMTool

TITLE:
------
models.py

MAIN OBJECTIVE:
---------------
This script defines language-specific BERT model variants for multilingual NLP tasks,
providing pre-configured models for Arabic, Chinese, French (CamemBERT), German, Hindi,
Italian, Portuguese, Russian, Spanish, Swedish, and cross-lingual (XLM-RoBERTa) applications.

Dependencies:
-------------
- transformers (HuggingFace model classes)
- LLMTool.bert_base (base implementation)
- LLMTool.sota_models (SOTA model imports)

MAIN FEATURES:
--------------
1) Pre-configured BERT models for 11+ languages
2) Each model uses appropriate tokenizer and pre-trained weights
3) Simple initialization with automatic language-specific defaults
4) Inherits all training capabilities from BertBase
5) Convenient imports of SOTA models (DeBERTa, RoBERTa, ELECTRA, etc.)

Author:
-------
Antoine Lemor
"""

from transformers import BertTokenizer, BertForSequenceClassification, \
                         CamembertTokenizer, CamembertForSequenceClassification, \
                         XLMRobertaForSequenceClassification, XLMRobertaTokenizer


from llm_tool.trainers.bert_base import BertBase

# Import SOTA models for convenient access
from llm_tool.trainers.sota_models import (
    DeBERTaV3Base, DeBERTaV3Large, DeBERTaV3XSmall,
    RoBERTaBase, RoBERTaLarge, DistilRoBERTa,
    ELECTRABase, ELECTRALarge, ELECTRASmall,
    ALBERTBase, ALBERTLarge, ALBERTXLarge,
    BigBirdBase, BigBirdLarge,
    LongformerBase, LongformerLarge,
    MDeBERTaV3Base, XLMRobertaBase, XLMRobertaLarge,
    # Long-document models
    XLMRobertaLongformer, LEDBase, LEDLarge,
    # Language-specific long-document models
    FrenchLongformer, SpanishLongformer, SpanishRoBERTaBNE,
    GermanLongformer, GermanGBERT,
    ItalianLongformer, PortugueseLongformer,
    DutchLongformer, DutchBERT,
    PolishLongformer, PolishHerBERT,
    ChineseLongformer, ChineseRoBERTa,
    JapaneseLongformer, JapaneseBERTWWM,
    ArabicLongformer, ArabicAraBERT,
    RussianLongformer
)


class Bert(BertBase):
    def __init__(
            self,
            model_name='bert-base-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name, 
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ArabicBert(BertBase):
    def __init__(
            self,
            model_name="asafaya/bert-base-arabic",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class Camembert(BertBase):
    def __init__(
            self,
            model_name='camembert-base',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=CamembertTokenizer,
            device=device,
            model_sequence_classifier=CamembertForSequenceClassification
        )


class ChineseBert(BertBase):
    def __init__(
            self,
            model_name="bert-base-chinese",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese"),
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class GermanBert(BertBase):
    def __init__(
            self,
            model_name="dbmdz/bert-base-german-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class HindiBert(BertBase):
    def __init__(
            self,
            model_name="monsoon-nlp/hindi-bert",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class ItalianBert(BertBase):
    def __init__(
            self,
            model_name="dbmdz/bert-base-italian-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class PortugueseBert(BertBase):
    def __init__(
            self,
            model_name='neuralmind/bert-base-portuguese-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class RussianBert(BertBase):
    def __init__(
            self,
            model_name="DeepPavlov/rubert-base-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SpanishBert(BertBase):
    def __init__(
            self,
            model_name="dccuchile/bert-base-spanish-wwm-cased",
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class SwedishBert(BertBase):
    def __init__(
            self,
            model_name='KB/bert-base-swedish-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class MultiBERT(BertBase):
    def __init__(
            self,
            model_name='bert-base-multilingual-cased',
            device=None
    ):
        super().__init__(
            model_name=model_name,
            tokenizer=BertTokenizer,
            device=device,
            model_sequence_classifier=BertForSequenceClassification
        )


class XLMRoberta(BertBase):
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
