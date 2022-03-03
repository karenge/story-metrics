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
    story = story.strip('*')

total = []
for s in sources:
    stories = load_from_source(s)
    total.append(clean(t) for t in stories)

with open("writingPromptsClean.txt", "w") as output:
    output.write(str(total))
