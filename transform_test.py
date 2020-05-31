import json
import os
from os.path import join, abspath, dirname

train_row_data = json.load(open('data/train/train_author_fix.json', 'r', encoding='utf-8'))

test_author = {}
test_label = {}
test_size   = {}

for name, author_data in train_row_data.items():
    pred_data = {}
    for author_id, papers in author_data.items():
        for paper_id in papers:
            pred_data[paper_id] = {"paper": paper_id, "author": author_id}

    if len(pred_data) == 0:
        continue
    
    test_author[name] = [v["paper"] for k, v in pred_data.items()]
    test_label[name] = [v["author"] for k, v in pred_data.items()]
    test_size[name] = len(author_data)

os.makedirs(join("data", "test"), exist_ok=True)

with open(join("data", "test", "test_author_fix.json"), 'w') as f:
    json.dump(test_author, f, indent='\t')

with open(join("data", "test", "test_label.json"), 'w') as f:
    json.dump(test_label, f, indent='\t')

with open(join("data", "test", "test_size.json"), 'w') as f:
    json.dump(test_size, f, indent='\t')