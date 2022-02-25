import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from graphs import get_kg
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def get_train_test(csv):
    df = pd.read_csv(csv)
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
    return train, test

def get_embedding(word):
    if word in model.vocab:
        return model[word]
    else:
        return np.zeros(300)

def centroid(words):
    return np.mean([get_embedding(word) for word in words], axis = 0)

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def main():
    train, test = get_train_test('clozeTest2018.csv')
    print(get_embedding("dog"))

    
if __name__ == "__main__":
    main()