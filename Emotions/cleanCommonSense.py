#import sklearn
import pandas as pd
import numpy as np
import re
import os

path = "story_emotion_data/emotions_data.csv"

df = pd.read_csv(path)

#print(df.head())

#df[['emo1', 'emo2', 'emo3']] = df['emotionlist'].str.split('\',', expand=True)

df['emotionlist'] = df['emotionlist'].str.replace('u\'', '')
df['emotionlist'] = df['emotionlist'].apply(lambda x: re.sub(r"('\])|\[|\]", "", x))

df_idx = df['emotionlist'].str.split("\', ",expand=True)
df_list = df['emotionlist'].str.split("\', ")
print(df_idx.head())


df = df.set_index('text')['emotionlist'].str.get_dummies('\', ').reset_index()
'''
df = df.join(df_idx)

emotions = pd.unique(df.iloc[:,2:10].values.flatten())

emotions = emotions[emotions != '']
emotions = emotions[emotions != None]

for emo in emotions:
    emo_data = df[df['emotionlist'].str.contains(emo)]
    print(emo)
    print(emo_data.head())
    df = pd.concat([df,emo_data])

del df['emotionlist']
'''

#del df.iloc[:,2:9]
print(df.head())

df.to_csv('story_emotion_data/cleanEmotion.csv')
