from graphs import Graph
from query_comet import COMET
import sys
import pandas as pd
import logBaseline
import threading

sent_1 = "Karen was assigned a roommate"# her first year of college."
sent_2 =  "The show was absolutely exhilarating."
sent_3 = "He didn’t have a job so he bought everything on his card."
sent_4 =  "They heard a hurricane was coming."


sents = [sent_1, sent_2, sent_3]
common_rels = ["AtLocation", "CapableOf", "Causes", "CausesDesire", "CreatedBy", "DefinedAs", "Desires", "HasA",
"HasFirstSubevent", "HasLastSubevent", "HasPrerequisite", "HasProperty", "HasSubevent", "IsA",
"MadeOf", "MotivatedByGoal", "PartOf", "ReceivesAction", "SymbolOf", "UsedFor"]

max_len = len("Karen was assigned a roommate her first year of")
comet = COMET()
graph = Graph()
beam_k = 3

#print(comet.has_property(sent_1))

#ent_dict = {}

def naive_shorten(sent):
    sent = sent[:max_len]
    return sent

def enrich_sent(sent):
    ents = graph.get_entities(sent)
    ret = [sent]
    for ent in ents:
        if ent == "" : continue
        if ent in ent_dict.keys():
            prop1 = ent_dict[ent]["prop1"]
            prop2 = ent_dict[ent]["prop2"]
            prop3 = ent_dict[ent]["prop3"]
        else:
            prop1 = comet.expect(ent, "DefinedAs")
            prop2 = comet.expect(ent, "Causes")
            prop3 = comet.expect(ent, "ReceivesAction")
            ent_dict[ent] = {"prop1":prop1,"prop2":prop2,"prop3":prop3}
        ret += [ent, "DefinedAs", prop1] + [ent, "Causes", prop2]+ [ent, "ReceivesAction", prop3]


def enrich_sent_kg(sent):
    kg = graph.get_kg(sent)
    kg_sent = ' '.join(kg)
    #print(kg_sent)
    prediction = comet.expect(kg_sent, "HasPrerequisite")
    #print(prediction)
    ret = sent + " —— HasPrerequisite: " + ', '.join(prediction)
    prediction = comet.expect(kg_sent, "Causes")
    ret += ' —— Causes: ' + ', '.join(prediction) + '.'
    return ret

    #["<UNK>" for _ in range(beam_k)]
    #causes = comet.expect(short,"Causes")
    #sub = comet.expect(short,"HasSubevecnt")
    #if causes == None:
    #causes = ["<UNK>" for _ in range(2)]

    #ret += prop1 + prop2 + prop3

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



i = 0
worker_map = {0:t1_work,1:t2_work,2:t3_work,3:t4_work,4:t5_work}
stories = pd.read_csv("rocBad-ordering-1.txt", on_bad_lines='skip', quotechar='"', engine='python')
with open("rocBad-reordered1-comet.txt", 'w') as output:
    header = "'storid','sentence1','sentence2','sentence3','sentence4','r1','r2', 'ans'\n"
    output.write(header)
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        id,first,second,third,fourth,e1,e2 = row['storid'],row['sentence1'],row['sentence2'],row['sentence3'],row['sentence4'],row['r1'],row['r2']
        vals = [id,first,second,third,fourth,e1]
        worker_map[i%3].append(vals)
        i += 1
        if i > 3500: break
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')
    t3 = threading.Thread(target=task3, name='t3')
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    results = [t1_out, t2_out, t3_out,t4_out,t5_out]
    for list in results:
        for line in list:
            output.write(line)
    output.close()

"""
f = open("../clozeTest2016.csv", "r")
f.readline()
i = 0
for line in f:
    print(f.readline().split(","))
    i += 1
    if i > 10: break

"""
"""
properties = dict()
for sent in sents:
    ents = graph.get_entities(sent)
    for ent in ents:
        if ent == "" : continue
        props = comet.has_property(ent)
        print(props)
        properties[ent] = props
"""

#print(properties)
