import os
import numpy as np
import pandas as pd

def load_from_source(source):
    print('Loading SWAG...')

    test = pd.read_csv('SWAG/test.csv')
    train = pd.read_csv('SWAG/train.csv')
    val = pd.read_csv('SWAG/val.csv')
    print('Done.')

    print('Cleaning')

    total = pd.concat([test,train,val])

    df.drop(columns = ['video-id','fold-ind','startphrase','gold-source','ending1','ending2','ending3'],axis =1)
    #left: sent1,sent2,ending0
    return

def clean(story):
    story = story.replace('<newline>', '')
    story = story.replace('*', '')
    story = story.replace('<', '')

    return story

total = []
for s in sources:
    stories = load_from_source(s)
    print(s, len(stories))
    for t in stories:
        total.append(clean(t))

with open("swagClean.txt", "w") as output:
    for elt in total:
        output.write(elt)
