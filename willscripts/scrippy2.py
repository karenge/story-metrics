from graphs import Graph
from query_comet import COMET
import sys
import pandas as pd
import logBaseline

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
    print(sent)
    print(kg_sent)
    prediction = comet.expect(kg_sent, "HasPrerequisite")
    print(prediction)
    add = ' '
    try:
        add = ', '.join(prediction)
    except:
        pass

    ret = sent + " —— HasPrerequisite: " + add
    prediction = comet.expect(kg_sent, "Causes")

    add = ' '
    try:
        add = ', '.join(prediction)
    except:
        pass
    ret += ' —— Causes: ' + add + '.'
    return ret


def list_2_line(list):
    line = ''
    for string in list:
        line += '"' + string + '",'
    return line[:-1]

stories = pd.read_csv("rocBadClean.txt", on_bad_lines='skip', quotechar="'", engine='python')
i = 0
with open("uhhhhhh.txt", 'w') as output:
    header = 'InputStoryid,InputSentence2,InputSentence3,InputSentence4,GenSentence1,GenSentence2\n'
    output.write(header)
    print("cols",stories.columns)
    for index, row in stories.iterrows():
        id,first,second,third,fourth,g1,g2 = row['storyid'],row['sentence1'],row['sentence2'],row['sentence3'],row['sentence4'],row['gen1'],row['gen2']
        sents = [first,second,third,fourth,g1,g2]
        id = id
        enriched = []
        for sent in sents:
            enriched.append(enrich_sent_kg(sent))
        line = list_2_line([id+'-c']+enriched)+'\n'
        output.write(line)

        print("num_processed=",i)
        i += 1
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
