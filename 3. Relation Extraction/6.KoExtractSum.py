# encoding="utf-8"


from transformers import BertTokenizer
import argparse
import wandb
# !wandb login 5295808ee2ec2b1fef623a0b1838c5a5c55ae8d1
import kss
import wandb
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
import torch
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")




alphabets = "([A-Za-z가-힣])"
numbers = "^\d*[.,]?"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>",
                  text)
    text = re.sub(numbers + "[.]" + " ", " \\1<prd> ", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets, "\\1<prd>\\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
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

def read_data(PATH):
  df = pd.read_excel(os.path.join(PATH,"testLabel.xlsx"),encoding='utf-sig-8',na_filter=True)
  return df

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
df = read_data()

def clean(object):
  context = object.data
  label = object.label
  sentenceList=[]
  for sentences in context:
    for sentence in kss.split_sentences(sentences):
      sentenceList.append(sentence)
  labels = []
  for i in range(len(df)):
    try:
      sentences = [j for j in split_into_sentences(context[i])]
      df.label[i] = literal_eval(df.label[i])
      target = df.label[i]
      for j in range(len(target)):
        labels.append(j)
    except:
      pass
  print(len(sentences))
  print(len(labels))
  return pd.DataFrame({'sentences':sentences,'label':labels})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--PATH",
        default="/Users/yoonk/Desktop/",
        type=str,
        required=True,
        help="The input data dir. Should contain the .xlsx files (or other data files) for the task.",
    )
    parser.add_argument(
        "--OUTPUT_PATH",
        default="/Users/yoonk/Desktop/",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
        default=None,
    )
