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

with open("writingPromptsClean.txt", "w") as output:
    for elt in total:
        output.write(elt)
