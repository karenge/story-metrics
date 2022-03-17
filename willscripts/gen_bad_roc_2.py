#!/usr/bin/env python3
import json
import os
import pandas as pd

offset = 5000
Boffset = 20001
stories = pd.read_csv('ROCStories_winter2017.csv')

story_num = 0
with open("rocBad-ordering.txt", "w") as output:
    header = "'storid','sentence1','sentence2','sentence3','sentence4','r1','r2'\n"
    output.write(header)
    for i in range(20000):
        row1 = stories.iloc[offset + i]
        row2 = stories.iloc[offset + Boffset + i]
        out_row = '"' + row1['storyid'] + '-r","' + row1['sentence1'] + '","' + row1['sentence2'] + '","' + row1['sentence3'] + '","' + row1['sentence4'] + '","' + row2['sentence4'] + '","' + row2['sentence5'] + '"\n'
        print(out_row)
        output.write(out_row)
        print("num_processed =", i)
