import pandas as pd

data = pd.read_csv("cometdata.txt", on_bad_lines='skip', quotechar='"', engine='python')
good, bad = 0,0
for index, row in data.iterrows():
    label = int(row['label'])
    if label == 0:
        bad += 1
    if label == 1:
        good += 1

print(good/(good+bad))
