import numpy as np
from numpy.linalg import norm
from numpy import dot
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from graphs import get_kg
from gensim import models

w2v_model = models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
     def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred.squeeze(-1)

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
        return w2v_model[word]
    except:
        return np.zeros(300)

def centroid(words):
    return np.mean([get_embedding(word) for word in words.split()], axis = 0)

def cosine_sim(a, b):
    if np.sum(a**2)*np.sum(b**2) == 0:
        return 0
    else:
        return np.sum(a*b)/np.sqrt(np.sum(a**2)*np.sum(b**2))
        
def story_ending_sim(story, ending):
    ending_centroid = centroid(ending)
    story_centroid = np.mean([centroid(sent) for sent in story], axis = 0)
    return cosine_sim(story_centroid, ending_centroid)

def max_sim(story, ending):
    story_combined = []
    for sent in story:
        for word in sent.split():
            story_combined.append(word)
    ending_centroid = centroid(ending)
    word_sim = [cosine_sim(get_embedding(word), ending_centroid) for word in story_combined]
    word_sim.sort(reverse=True)
    return np.mean(word_sim[:4])

def concat_features(story, ending):
    ending_centroid = centroid(ending)
    return np.array([story_ending_sim(story, ending_centroid), max_sim(story, ending_centroid)])

def get_features(csv):
    x, y = get_xy(csv)
    x_features1 = x.apply(lambda story: story_ending_sim([story.InputSentence1, story.InputSentence2, story.InputSentence3, story.InputSentence4], story.Ending), axis=1)
    x_features2 = x.apply(lambda story: max_sim([story.InputSentence1, story.InputSentence2, story.InputSentence3, story.InputSentence4], story.Ending), axis=1)
    
    x_features = pd.DataFrame()
    x_features['se_sim'] = x_features1
    x_features['max_sim'] = x_features2

    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.15)
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def main():
    x_train, x_test, y_train, y_test = get_features('clozeTest2018.csv')
    print("x_train shape ", x_train.shape)
    x_train=torch.from_numpy(x_train.astype(np.float32))
    x_test=torch.from_numpy(x_test.astype(np.float32))
    y_train=torch.from_numpy(y_train.astype(np.float32))
    y_test=torch.from_numpy(y_test.astype(np.float32))

    model = LogisticRegression()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        y_pred = model(x_train)
        loss=criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch+1)%10 == 0:
            print('epoch:', epoch+1,',loss=',loss.item())

    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_class = y_pred.round()
        accuracy = (y_pred_class.eq(y_test).sum()) / float(y_test.shape[0])
        print("accuracy ", accuracy.item())


if __name__ == "__main__":
    main()