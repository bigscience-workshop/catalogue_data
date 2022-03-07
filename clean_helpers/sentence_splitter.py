import subprocess
import os
import torch
import stanza
from stanza_batch import batch
from indicnlp import common
from indicnlp.tokenize import sentence_tokenize
import nltk
from nltk.tokenize import sent_tokenize


def build_nltk_splitter(lang):
    lang_to_punkt = {
        "en": "english",
        "fr": "french",
        "pt": "portuguese",
        "es": "spanish"
    }
    
    def splitter(examples):
        split_texts = ["\n".join(sent_tokenize(text, language=lang_to_punkt[lang])) for text in examples["text"]]
        return {**examples, "text": split_texts }        
    return splitter


def build_stanza_splitter(lang, batch_size=32):
    lang_to_stanza = {"zht": "zh-hant", "zhs": "zh-hans"}
    lang = lang_to_stanza.get(lang, lang)
    tokenizer = stanza.Pipeline(lang, logging_level="WARNING", processors='tokenize',
                          use_gpu=torch.cuda.is_available())
    
    def splitter(examples):
        split_texts = []
        for document in batch(examples["text"], tokenizer, batch_size=batch_size):
            split_texts.append("\n".join([sentence.text for sentence in document.sentences]))
        return {**examples, "text": split_texts }        
    return splitter


def build_indic_splitter(lang):
    lang_to_indic = {
        "indic-bn": "bn",
        "indic-gu": "gu",
        "indic-hi": "hi",
        "indic-kn": "kn",
        "indic-ml": "ml",
        "indic-mr": "mr",
        "indic-pa": "pa",
        "indic-ta": "ta",
        "indic-te": "te"
        }
    def splitter(examples):
        split_texts = ["\n".join(sentence_tokenize.sentence_split(text, lang=lang_to_indic[lang])) for text in examples["text"]]
        return {**examples, "text": split_texts }
    return splitter


def build_sentence_splitter(lang):
    if lang in stanza_list:
        return build_stanza_splitter(lang)
    elif lang in nltk_list:
        return build_nltk_splitter(lang)
    elif lang in indic_list:
        return build_indic_splitter(lang)
    else:
        NotImplementedError(f"Lang '{lang}' has no sentence splitter implemented.")


stanza_list = {"ar", "ca", "eu", "id", "vi", "zhs", "zht"}
nltk_list = {"en", "fr", "pt", "es"}
indic_list = {"indic-bn", "indic-gu", "indic-hi", "indic-kn", "indic-ml", "indic-mr", "indic-pa", "indic-ta",
              "indic-te"}

assert len(stanza_list & nltk_list) == 0
assert len(stanza_list & indic_list) == 0
assert len(indic_list & nltk_list) == 0

sentence_split_langs = stanza_list | nltk_list | indic_list