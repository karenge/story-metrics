import sys
import pandas as pd
import numpy as np

with open("test_data.txt",'w') as tgt1:
    header = 'storyid,text,label\n'
    tgt1.write(header)
tgt1.close()

with open("val_data.txt",'w') as tgt2:
    header = 'storyid,text,label\n'
    tgt2.write(header)
tgt2.close()

with open("train_data.txt",'w') as tgt3:
    header = 'storyid,text,label\n'
    tgt3.write(header)
tgt3.close()


with open("projdata.txt", 'r') as src:
    i = 0
    text = ""
    for line in src:
        if i == 0:
            i += 1
            continue
        if i % 6 == 0:
            x = np.random.uniform(0,1)
            if x < 0.05:
                with open("test_data.txt",'a') as tgt:
                    tgt.write(text)
                tgt.close()
            elif x < 0.15:
                with open("val_data.txt", 'a') as tgt1:
                    tgt1.write(text)
                tgt1.close()
            else:
                with open("train_data.txt", 'a') as tgt2:
                    tgt2.write(text)
                tgt2.close()
            text = line
        else:
            text += line
        i += 1
