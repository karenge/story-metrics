import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers

df = pd.read_csv('clozeTest2018.csv')
story1 = df.iloc[0]
filter_ = df['AnswerRightEnding'] >= 2
del df['InputStoryid']
firstEndings = df[~filter_].drop(labels='RandomFifthSentenceQuiz2', axis=1)
secondEndings = df[filter_].drop(labels='RandomFifthSentenceQuiz1', axis=1)
print(firstEndings.head())
print(secondEndings.head())

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = df[0:4].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
