from graphs import Graph
from query_comet import COMET
import sys
import pandas as pd
import logBaseline
import threading


beam_k = 3

def enrich_sent_kg(sent, graph, comet):
    kg = graph.get_kg(sent)
    kg_sent = ' '.join(kg)
    #print(kg_sent)
    prediction = comet.expect(kg_sent, "HasPrerequisite")
    #print(prediction)
    ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
    prediction = comet.expect(kg_sent, "Causes")
    ret += ' —— Causes: ' + ', '.join(prediction) + '.'
    return ret


def list_2_line(list):
    line = ''
    for string in list:
        line += '"' + string + '",'
    return line[:-1]


t1_work, t2_work, t3_work, t4_work, t5_work, t1_out, t2_out, t3_out,t4_out, t5_out = [],[],[],[],[],[],[],[],[],[]

def task1():
    comet = COMET()
    graph = Graph()
    i = 0
    while(len(t1_work) > 0):
        vals = t1_work.pop(0)
        sents, id = vals[1:], vals[0]
        enriched = []
        for sent in sents:
            kg = graph.get_kg(sent)
            kg_sent = ' '.join(kg)
            prediction = comet.expect(kg_sent, "HasPrerequisite")
            ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
            prediction = comet.expect(kg_sent, "Causes")
            ret += ' —— Causes: ' + ', '.join(prediction) + '.'
            enriched.append(ret)
        line = list_2_line([id+'-c']+enriched)+'\n'
        t1_out.append(line)
        print("worker 1:", i)
        i += 1


def task2():
    comet = COMET()
    graph = Graph()
    i = 0
    while(len(t2_work) > 0):
        vals = t2_work.pop(0)
        sents, id = vals[1:], vals[0]
        enriched = []
        for sent in sents:
            kg = graph.get_kg(sent)
            kg_sent = ' '.join(kg)
            prediction = comet.expect(kg_sent, "HasPrerequisite")
            ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
            prediction = comet.expect(kg_sent, "Causes")
            ret += ' —— Causes: ' + ', '.join(prediction) + '.'
            enriched.append(ret)
        line = list_2_line([id+'-c']+enriched)+'\n'
        t2_out.append(line)
        print("worker 2:", i)
        i += 1

def task3():
    comet = COMET()
    graph = Graph()
    i = 0
    while(len(t3_work) > 0):
        vals = t3_work.pop(0)
        sents, id = vals[1:], vals[0]
        enriched = []
        for sent in sents:
            kg = graph.get_kg(sent)
            kg_sent = ' '.join(kg)
            prediction = comet.expect(kg_sent, "HasPrerequisite")
            ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
            prediction = comet.expect(kg_sent, "Causes")
            ret += ' —— Causes: ' + ', '.join(prediction) + '.'
            enriched.append(ret)
        line = list_2_line([id+'-c']+enriched)+'\n'
        t3_out.append(line)
        print("worker 3:", i)
        i += 1

def task4():
    comet = COMET()
    graph = Graph()
    i = 0
    while(len(t4_work) > 0):
        vals = t4_work.pop(0)
        sents, id = vals[1:], vals[0]
        enriched = []
        for sent in sents:
            kg = graph.get_kg(sent)
            kg_sent = ' '.join(kg)
            prediction = comet.expect(kg_sent, "HasPrerequisite")
            ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
            prediction = comet.expect(kg_sent, "Causes")
            ret += ' —— Causes: ' + ', '.join(prediction) + '.'
            enriched.append(ret)
        line = list_2_line([id+'-c']+enriched)+'\n'
        t4_out.append(line)
        print("worker 4:", i)
        i += 1

def task5():
    comet = COMET()
    graph = Graph()
    i = 0
    while(len(t5_work) > 0):
        vals = t5_work.pop(0)
        sents, id = vals[1:], vals[0]
        enriched = []
        for sent in sents:
            kg = graph.get_kg(sent)
            kg_sent = ' '.join(kg)
            prediction = comet.expect(kg_sent, "HasPrerequisite")
            ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
            prediction = comet.expect(kg_sent, "Causes")
            ret += ' —— Causes: ' + ', '.join(prediction) + '.'
            enriched.append(ret)
        line = list_2_line([id+'-c']+enriched)+'\n'
        t4_out.append(line)
        print("worker 5:", i)
        i += 1

i = 0
worker_map = {0:t1_work,1:t2_work,2:t3_work,3:t4_work,4:t5_work}
stories = pd.read_csv("../../Desktop/metricsData/ROCStories_winter2017.csv", on_bad_lines='skip', quotechar='"', engine='python')
with open("roc_comet_2.txt", 'w') as output:
    header = "'storid','sentence1','sentence2','sentence3','sentence4','r1','r2'\n"
    output.write(header)
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        if i < 1001:
            i += 1
            continue
        id,first,second,third,fourth,fifth = row['storyid'],row['sentence1'],row['sentence2'],row['sentence3'],row['sentence4'],row['sentence5']
        vals = [id,first,second,third,fourth,fifth]
        worker_map[i%5].append(vals)
        i += 1
        if i > 2001:
            break
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')
    t3 = threading.Thread(target=task3, name='t3')
    t4 = threading.Thread(target=task4, name='t4')
    t5 = threading.Thread(target=task5, name='t5')

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    results = [t1_out, t2_out, t3_out,t4_out,t5_out]
    for list in results:
        for line in list:
            output.write(line)

    output.close()

#print(properties)
