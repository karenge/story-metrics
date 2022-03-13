import json
import csv

def get_ids():
    with open("storyid_partition.txt") as f:
        story_ids = [line.split()[0] for line in f]
    return story_ids

def get_data():
    with open('annotations.json', 'r') as annotations:
        read_data = annotations.read()
    data = json.loads(read_data)
    return data

def extract(story_ids, data):
    story_data = []
    for id in story_ids:
        story = data[id]
        for lineNum in ["1", "2", "3", "4", "5"]:
            line = story["lines"][lineNum]
            charsline = line["characters"] 
            ps_lists = [v["emotion"][ann]["plutchik"] for k, v in charsline.items() for ann in v["emotion"].keys()]
            ps_lists = [item.partition(":")[0] for sublist in ps_lists for item in sublist]
            ps_lists = list(set(ps_lists))
            story_data.append([line["text"], ps_lists])
    return story_data

def main():
    story_ids = get_ids()
    data = get_data()
    story_data = extract(story_ids, data)
    with open("emotions_data.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(story_data)

if __name__ == "__main__":
    main()