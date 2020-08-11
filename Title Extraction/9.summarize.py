# coding=utf-8
""" Text Processing Classes for processing scripts.
"""
import re
import logging
import itertools
import unicodedata
import requests
from functools import reduce
from transformers import *
import torch
from summarizer import Summarizer
from bs4 import BeautifulSoup
import string
from textrankr import TextRank
from lexrankr import LexRank
import MeCab
import kss
import numpy as np

logger = logging.getLogger(__name__)


def sent_word_tokenize(self, text, residual=True):
    m = MeCab.Tagger('/Users/yoonk/Downloads/mecab-ko-dic-2.1.1-20180720')


    sent = []
    tokens = text.split()
    tokens_it = iter(tokens)

    try:
        token = next(tokens_it)
        index = 0
        yield_sent = False

        for f, pos in m.parse(text):

            if index >= len(token):
                if token:
                    sent.append(token)

                token = next(tokens_it)
                index = 0

                if yield_sent:
                    yield sent
                    yield_sent = False
                    sent = []

            # assert token[index:index + len(f)] == f

            if pos.startswith("S"):
                if index:
                    t, token = token[:index], token[index:]

                    if t:
                        sent.append(t)

                    index = 0

                t, token = token[index:index + len(f)], token[
                                                        index + len(f):]

                if t:
                    sent.append(t)

                index = 0

                if pos == "SF":
                    yield_sent = True
            else:
                index += len(f)

        if token and index:
            sent.append(token[:index])

        if sent and (yield_sent or residual):
            yield sent

    except StopIteration:
        pass

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def _split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>",
                  text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character tab, space, next line."""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def summarize(text):
    url = "https://api.smrzr.io/summarize?ratio=0.15"
    headers = {
        'content-type': 'raw/text',
        'origin': 'https://smrzr.io',
        'referer': 'https://smrzr.io/',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        "user-agent": "Mozilla/5.0"
    }
    resp = requests.post(url, headers=headers, data=text.encode('utf-8'))
    assert resp.status_code == 200

    summary = resp.json()['summary']
    temp = summary.split('\n')
    return temp



def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation or whitespace character."""
    last_char = text[-1]
    return bool(_is_punctuation(last_char) | _is_whitespace(last_char))

def summarizeSlow(text, ratio, model_name='Distil'):
    """keys for kakao translation api service
    002558914cc8caacc923bbb4f9d6a772
    03e207126164816f438e90f73832984c
    18fea080a0db2ee0346353e8db4466e8
    """
    global model
    if model_name == "Distil":
        model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased', output_hidden_states=True)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    elif model_name == "Bert":
        bmodel = AutoModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
        configuration = bmodel.config
        configuration.vocab_size = 32000
        configuration.max_position_embeddings = 384
        configuration.hidden_size = 1024
        configuration.num_attention_heads = 16
        configuration.output_hidden_states = True
        model = BertModel.from_pretrained("/Users/yoonk/Downloads/large_v2.bin", config=configuration)
        tokenizer = BertTokenizer.from_pretrained("/Users/yoonk/Downloads/large_v2_32k_vocab.txt")

    summarizer = Summarizer(custom_model=model, custom_tokenizer=tokenizer)
    url = 'https://kapi.kakao.com/v1/translation/translate'
    before_lang = 'kr'
    after_lang = 'en'
    KEY = '18fea080a0db2ee0346353e8db4466e8'
    header = {
        "Authorization": 'KakaoAK {}'.format(KEY)
    }

    data = {
        "src_lang": before_lang,
        "target_lang": after_lang,
        "query": text
    }

    response = requests.get(url, headers=header, params=data)
    result = response.json()
    translated = result['translated_text'][0]
    result_eng = summarizer(translated, ratio=ratio)

    before_lang = 'en'
    after_lang = 'kr'
    data = {
        "src_lang": before_lang,
        "target_lang": after_lang,
        "query": ''.join(result_eng)
    }
    before_lang = 'en'
    after_lang = 'kr'

    response = requests.get(url, headers=header, params=data)
    result_kor = response.json()
    summarized = result_kor['translated_text'][0]

    return summarized


def summarizeTextRank(text, max=3):
    tr = TextRank(text)
    return tr.summarize(max)

def summarizeLexRank(text,num=3):
    lr = LexRank()
    lr.summarize(text)
    summaries = lr.probe(num)
    return summaries


text = '''
TEXT2PPTX의 구현하기 위해 two track process를 거쳤습니다. 대본을 피피티로 옮기기 전에 요약하고 분석하기 위해 자연어처리의 최신 기술을 다수 사용하였습니다. 또한 파이썬에서 파워포인트의 소스에 접근하기 위해 xml 코드를 심층적으로 분석했습니다. 그 결과 사용자가 자연어로 쓰인 대본을 텍2피에 제공하면 높은 수준의 피피티로 제공할 수 있게 되었습니다.
'''

def ensembleSummarize(text):
    sentences=kss.split_sentences(text)
    n=len(sentences)
    s = [0]*n
    for idx in range(len(sentences)):
        if sentences[idx] in summarize(text):
            s[idx]+=1
        if sentences[idx] in summarizeLexRank(text):
            s[idx]+=1
        if sentences[idx] in summarizeTextRank(text).split("\n"):
            s[idx]+=1
    i = s.index(max(s))
    return sentences[i]

print("BERT",summarize(text))
print("ensemble",ensembleSummarize(text))