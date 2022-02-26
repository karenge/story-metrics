import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from graphs import get_kg
from gensim import models

model = models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

def get_xy(csv):
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
    x = data[['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'Ending']]
    y = data['Label']
    return x, y

def get_embedding(word):
    try:
        return model[word]
    except:
        return np.zeros(300)

def centroid(words):
    return np.mean([get_embedding(word) for word in words], axis = 0)

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def se__similarity_feature(story, ending):
    story_centroid = np.mean([centroid(sent) for sent in story], axis = 0)
    ending_centroid = centroid(ending)
    return cosine_sim(story_centroid, ending_centroid)

def concat_features(story, ending):
    story_centroid = np.mean([centroid(sent) for sent in story], axis = 0)
    ending_centroid = centroid(ending)
    return np.concatenate([se__similarity_feature(story, ending)])

def get_features(csv):
    x, y = get_xy(csv)
    x_features = x.apply(lambda story: se__similarity_feature([story.InputSentence1, story.InputSentence2, story.InputSentence3, story.InputSentence4], story.Ending), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.15)
    return x_train, x_test, y_train, y_test

def main():
    x_train, x_test, y_train, y_test = get_features('clozeTest2018.csv')
    print("x_train.shape ", x_train.shape)
    print("y_train.shape ", y_train.shape)
    print("x_test.shape ", x_test.shape)
    print("y_test.shape ", y_test.shape)


if __name__ == "__main__":
    main()