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
    if len(lang)==3:
        lang.replace("zh", "zh-han") # zhs-> zh-hans, zht -> zh-hant
    tokenizer = stanza.Pipeline(lang, logging_level="WARNING", processors='tokenize',
    nlp = stanza.Pipeline(lang, logging_level="WARNING", processors='tokenize',
                          use_gpu=torch.cuda.is_available())
    
    def splitter(examples):
        split_texts = []
        for document in batch(examples["text"], nlp, batch_size=batch_size):
            split_texts.append("\n".join([sentence.text for sentence in document.sentences]))
        return {**examples, "text": split_texts }        
    return splitter


def build_indic_splitter(lang):
    lang_to_indic = {
        "hindi-bn": "bn",
        "hindi-gu": "gu",
        "hindi-hi": "hi",
        "hindi-kn": "kn",
        "hindi-ml": "ml",
        "hindi-mr": "mr",
        "hindi-pa": "pa",
        "hindi-ta": "ta",
        "hindi-te": "te"
        }
    def splitter(examples):
        split_texts = ["\n".join(sentence_tokenize.sentence_split(text, lang=lang_to_indic[lang])) for text in examples["text"]]
        return {**examples, "text": split_texts }
    return splitter


def build_sentence_splitter(lang):
    stanza_list = {"ar", "ca", "eu", "id", "vi", "zhs", "zht", "zh"}
    nltk_list = {"en", "fr", "pt", "es"}
    indic_list = {"bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te"}
    
    if lang in stanza_list:
        return build_stanza_splitter(lang)
    elif lang in nltk_list:
        return build_nltk_splitter(lang)
    elif lang in indic_list:
        return build_indic_splitter(lang)
    else:
        return lambda x: x


def remove_newlines(examples):
    return {**examples, "text": [text.replace("\n", " ") for text in examples["text"]]}  


sentence_split_langs = {"ar", "ca", "eu", "id", "vi", "zh", "zhs", "zht", "en", "fr", 
                        "pt", "es", "bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te"}