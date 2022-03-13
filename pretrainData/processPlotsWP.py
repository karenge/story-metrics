import os
import numpy as np

sources = ['writingPrompts/test.wp_target','writingPrompts/train.wp_target','writingPrompts/valid.wp_target']

def load_from_source(source):
    print('Loading writing prompts...')
    with open(source, errors='ignore') as ft:
         stories = ft.readlines()
    print('Done.')

    return stories

def clean(story):
    story = story.replace('<newline>', '\n')
    story = story.replace('*', '')
    story = story.replace('<', '')

    if '<EOS>' in story:
        story = "\n"

    return story

total = []
print('Loading plots...')
with open('plots', errors='ignore') as st:
     plots = st.readlines()
print('Done.')

for p in plots:
    total.append(clean(p))

for s in sources:
    stories = load_from_source(s)
    print(s, len(stories))
    for t in stories:
        total.append(clean(t))

with open("PromptsPlotsClean.txt", "w") as output:
    for elt in total:
        output.write(elt)
