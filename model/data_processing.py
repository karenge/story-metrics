import numpy as np
from numpy.linalg import norm
import pandas as pd
from gensim import models
import re

w2v_model = models.KeyedVectors.load_word2vec_format("~/cs224n/project/GoogleNews-vectors-negative300.bin.gz", binary=True)

def get_embedding(word):
    try:
        return w2v_model[word]
    except:
        return np.zeros(300)

def split_string(text):
    text = re.sub(r'[^\w\d\s\']+', '', text) 
    return text.split()

def centroid(words):
    word_list = split_string(words)
    if (len(word_list) > 0):
        return np.mean([get_embedding(word) for word in word_list], axis = 0)
    return 0

def main():
    df = pd.read_csv("emotions_generated.csv")
    for i in range(1, 6):
        col = 'causes' + str(i)
        df[col] = df[col].map(lambda a: centroid(a))
        col = 'hasprereq' + str(i)
        df[col] = df[col].map(lambda a: centroid(a))
    df = df.iloc[: , 2:]
    df.to_csv("final_data.csv")

if __name__ == "__main__":
    main()