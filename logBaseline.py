import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


df = pd.read_csv('clozeTest2018.csv')
df2 = df.copy()
filter_ = df['AnswerRightEnding'] >= 2
del df['InputStoryid']
#all the correct story endings
firstEndings = df[~filter_].drop(labels='RandomFifthSentenceQuiz2', axis=1)
secondEndings = df[filter_].drop(labels='RandomFifthSentenceQuiz1', axis=1)
firstEndings.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz1', 'Ending'), inplace=True)
secondEndings.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz2', 'Ending'), inplace=True)
correct = pd.concat([firstEndings, secondEndings])
correct['Label'] = 1
del correct['AnswerRightEnding']

filter_ = df2['AnswerRightEnding'] >= 2
#all the incorrect story endings
firstEndings2 = df[~filter_].drop(labels='RandomFifthSentenceQuiz1', axis=1)
secondEndings2 = df[filter_].drop(labels='RandomFifthSentenceQuiz2', axis=1)
firstEndings2.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz2', 'Ending'), inplace=True)
secondEndings2.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz1', 'Ending'), inplace=True)
incorrect = pd.concat([firstEndings2, secondEndings2])
incorrect['Label'] = 0
del incorrect['AnswerRightEnding']

data = pd.concat([correct, incorrect])
train, test = train_test_split(data, test_size=0.15)
print(train.shape)
print(test.shape)