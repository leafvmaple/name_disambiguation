import json
from utility.features import format_name

train_pub_data = json.load(open('data/sna_data/sna_valid_pub.json', 'r', encoding='utf-8'))
train_author_data = json.load(open('data/sna_data/sna_valid_author_raw.json', 'r', encoding='utf-8'))

author_table = {}

f = open("error.txt", 'w', encoding='utf-8')

def get_author_index(paper_id, author_name):
    names = author_name.split("_")
    elements = [
        names,
        [v if i == 0 else v[0] for i, v in enumerate(names)],
        [v if i == len(names) - 1 else v[0] for i, v in enumerate(names)]
    ]

    author_names = []
    for element in elements:
        author_names.append("_".join(element))
        author_names.append("_".join(list(reversed(element))))
        author_names.append("".join(element))
        author_names.append("".join(list(reversed(element))))
    
    for i, v in enumerate(author_table[paper_id]):
        if v in author_names:
            return i

    for i, v in enumerate(author_table[paper_id]):
        if v.replace("_", "") in author_names:
            return i

    f.write("{} {} {} {}\n".format(author_name, paper_id, author_names, author_table[paper_id]))

for paper_id, data in train_pub_data.items():
    author_table[paper_id] = [format_name(v["name"]) for v in data["authors"]]

for name, papers in train_author_data.items():
    for i, paper_id in enumerate(papers):
        idx = get_author_index(paper_id, name)
        if idx is not None:
            papers[i] = "{}-{}".format(paper_id, idx)
            train_pub_data[paper_id]["authors"][idx]["name"] = name
        else:
            papers[i] = "{}-{}".format(paper_id, len(author_table[paper_id]))

with open('data/sna_data/sna_valid_author_raw_fix.json', 'w') as f:
    json.dump(train_author_data, f, indent='\t')

with open('data/sna_data/sna_valid_pub_fix.json', 'w') as f:
    json.dump(train_pub_data, f, indent='\t')