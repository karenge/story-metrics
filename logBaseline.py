import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from graphs import get_kg
from gensim import models

#model = models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

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

    filter_ = df2['AnswerRightEnding'] >= 2
    #all the incorrect story endings
    firstEndings2 = df[~filter_].drop(labels='RandomFifthSentenceQuiz1', axis=1)
    secondEndings2 = df[filter_].drop(labels='RandomFifthSentenceQuiz2', axis=1)
    firstEndings2.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz2', 'Ending'), inplace=True)
    secondEndings2.rename(columns=lambda x: x.replace('RandomFifthSentenceQuiz1', 'Ending'), inplace=True)
    incorrect = pd.concat([firstEndings2, secondEndings2])
    incorrect['Label'] = 0

    data = pd.concat([correct, incorrect])
    x = data[['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4']]
    y = data['Label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    return x_train, x_test, y_train, y_test

def get_embedding(word):
    try:
        return model[word]
    except:
        return np.zeros(300)

def centroid(words):
    return np.mean([get_embedding(word) for word in words], axis = 0)

def centroid_feature(story, ending):
    story_centroid = np.mean([centroid[sent] for sent in story], axis = 0)
    ending_centroid = centroid(ending)
    return np.concatenate([story_centroid, ending_centroid])

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def se__similarity_feature(story, ending):
    story_centroid = np.mean([centroid[sent] for sent in story], axis = 0)
    ending_centroid = centroid(ending)
    return cosine_sim(story_centroid, ending_centroid)

def main():
    x_train, x_test, y_train, y_test = get_train_test('clozeTest2018.csv')
    print(x_train.head())
    print(y_train.head())

    
if __name__ == "__main__":
    main()