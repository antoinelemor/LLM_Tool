#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
models.py

MAIN OBJECTIVE:
---------------
Provide ready-to-train wrappers around language-specific BERT families and
expose state-of-the-art variants for use inside the Training Arena.

Dependencies:
-------------
- transformers
- llm_tool.trainers.bert_base
- llm_tool.trainers.sota_models

MAIN FEATURES:
--------------
1) Offer BertBase-derived classes for major languages with correct tokenizers
2) Keep a canonical Bert wrapper for generic English fine-tuning workflows
3) Integrate CamemBERT, XLM-R, and other multilingual backbones seamlessly
4) Surface SOTA model imports so trainer modules share a unified entrypoint
5) Simplify model instantiation by standardising constructor signatures

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
