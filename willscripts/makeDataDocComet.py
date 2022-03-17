import sys
import pandas as pd

with open("cometdata.txt", 'w') as tgt:
    header = 'storyid,sentence1,sentence2,sentence3,sentence4,sentence5,label\n'
    tgt.write(header)
    tgt.close()

num_cloze = 3

stories = pd.read_csv("clozeTrain2016_comet.txt", on_bad_lines='skip', quotechar='"', engine='python')
with open("cometdata.txt", 'a') as tgt:
    for index, row in stories.iterrows():
        id,first,second,third,fourth,fifth,sixth,ans = row['InputStoryid'],row['InputSentence1'],row['InputSentence2'],row['InputSentence3'],row['InputSentence4'],row['RandomFifthSentenceQuiz1'],row['RandomFifthSentenceQuiz2'],row['AnswerRightEnding']
        stor1_label = 2 - int(ans)
        stor2_label = - 1 + int(ans)
        line_1 = id + ',"' + first + '","' + second + '","' + third + '","' + fourth + '","' + fifth + '",' + str(stor1_label) + '\n'
        line_2 = id + ',"' + first + '","' + second + '","' + third + '","' + fourth + '","' + fifth + '",' + str(stor2_label) + '\n'
        tgt.write(line_1)
        tgt.write(line_2)
tgt.close()


stories = pd.read_csv("clozeTest2016_comet.txt", on_bad_lines='skip', quotechar='"', engine='python')
with open("cometdata.txt", 'a') as tgt:
    for index, row in stories.iterrows():
        id,first,second,third,fourth,fifth,sixth,ans = row['InputStoryid'],row['InputSentence1'],row['InputSentence2'],row['InputSentence3'],row['InputSentence4'],row['RandomFifthSentenceQuiz1'],row['RandomFifthSentenceQuiz2'],row['AnswerRightEnding']
        stor1_label = 2 - int(ans)
        stor2_label = - 1 + int(ans)
        line_1 = id + ',"' + first + '","' + second + '","' + third + '","' + fourth + '","' + fifth + '",' + str(stor1_label) + '\n'
        line_2 = id + ',"' + first + '","' + second + '","' + third + '","' + fourth + '","' + fifth + '",' + str(stor2_label) + '\n'
        tgt.write(line_1)
        tgt.write(line_2)
tgt.close()


stories = pd.read_csv("rocBad_comet.txt", on_bad_lines='skip', quotechar='"', engine='python')
with open("cometdata.txt", 'a') as tgt:
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        id,first,second,third,fourth,fifth,sixth,label = row['InputStoryid'],row['InputSentence1'],row['InputSentence2'],row['InputSentence3'],row['InputSentence4'],row['Gen1'],row['Gen2'],'0'
        sents = [first,second,third,fourth,fifth,sixth]
        for sent in sents:
            sent = sent.replace('"',"'")
        line_1 = id + ',"' + sents[0] + '","' + sents[1] + '","' + sents[2] + '","' + sents[3] + '","' + sents[4] + '",' + '0' + '\n'
        #line_2 = id + ',"' + sents[0] + '","' + sents[1] + '","' + sents[2] + '","' + sents[3] + '","' + sents[5] + '",' + '0' + '\n'
        tgt.write(line_1)
        #tgt.write(line_2)
tgt.close()



stories = pd.read_csv("rocBad_ordering_comet_2.txt", on_bad_lines='skip', quotechar='"', engine='python')

with open("cometdata.txt", 'a') as tgt:
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        id,first,second,third,fourth,fifth,label = row['storid'],row['sentence1'],row['sentence2'],row['sentence3'],row['sentence4'],row['r1'],'0'
        sents = [first,second,third,fourth,fifth]
        for sent in sents:
            sent = sent.replace('"',"'")
        line = id + ',"' + sents[0] + '","' + sents[1] + '","' + sents[2] + '","' + sents[3] + '","' + sents[4] + '",' + '0' + '\n'
        tgt.write(line)
tgt.close()


stories = pd.read_csv("roc_comet.txt", on_bad_lines='skip', quotechar='"', engine='python')

with open("cometdata.txt", 'a') as tgt:
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        id,first,second,third,fourth,fifth = row['storid'],row['sentence1'],row['sentence2'],row['sentence3'],row['sentence4'],row['sentence5']
        sents = [first,second,third,fourth,fifth]
        for sent in sents:
            sent = sent.replace('"',"'")
        line = id + ',"' + sents[0] + '","' + sents[1] + '","' + sents[2] + '","' + sents[3] + '","' + sents[4] + '",' + '1' + '\n'
        tgt.write(line)
tgt.close()
