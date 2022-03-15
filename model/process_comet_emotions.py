import pandas as pd
import numpy as np
import os

df = pd.read_csv("cometdata.txt")

for i in range(5):
    i = i+1
    df_idx = df['sentence'+str(i)].str.split(" —— ",expand=True)
    #print(df_idx)
    del df["sentence"+str(i)]
    df = pd.concat([df,df_idx],axis=1)
    df[1] = df[1].str.replace('HasPrerequisite: ', '')
    df[2] = df[2].str.replace('Causes: ', '')
    df.rename(columns={0: "sentence"+str(i), 1: "hasprereq"+str(i), 2:"causes"+str(i)},inplace=True)

last_column = df.pop('label')
df.insert(len(df.columns), 'label', last_column)

#print(df.head())
#print(df.iloc[4])

df.to_csv('cometClean.csv')
