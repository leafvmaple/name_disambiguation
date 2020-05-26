import json
import codecs
import random
import pickle
import math
import gc
import numpy as np
from os.path import join
from gensim.models import Word2Vec
from collections import defaultdict
from utility import get_author_feature, get_feature_emb

train_pub_data_path = 'data/train/train_pub.json'
# train_row_data_path = 'data/train/train_author.json'

class Disambiguation:
    def __init__(self):
        self.paper_pub = {}
        self.paper_idf = {}
        self.paper_emb = {}
        self.model = None

    def transform(self, pub_data):
        print("---------Transfrom------------")

        for paper_id, data in pub_data.items():
            author_features = get_author_feature(paper_id, data)
            for i, author_feature in enumerate(author_features):
                self.paper_pub["{}-{}".format(paper_id, i)] = author_feature

    def word2vec(self):
        print("---------word2vec------------")
        
        paper_data = []

        counter = defaultdict(int)
        feature_cnt = 0

        for _, author_feature in self.paper_pub.items():
            random.shuffle(author_feature)
            paper_data.append(author_feature)

            feature_cnt += len(author_feature)
            for feature_item in author_feature:
                counter[feature_item] += 1

        for item in counter:
            self.paper_idf[item] = math.log(feature_cnt / counter[item])

        self.model = Word2Vec(paper_data, size=100, window=5, min_count=5, workers=20)

    def embedding(self):
        print("---------embedding------------")

        for author_id, author_feature in self.paper_pub.items():
            self.paper_emb[author_id] = get_feature_emb(author_feature, self.paper_idf, self.model)

    def prepare(self, pub_data):
        self.transform(pub_data)
        self.word2vec()
        self.embedding()

    def save(self, output_path):
        with open(output_path + "_pub.json", 'w') as f:
            json.dump(self.paper_pub, f, indent='\t')

        with open(output_path + "_pkl.idf", 'wb') as f:
            pickle.dump(self.paper_idf, f)

        self.model.save(output_path + ".model")
        np.save(output_path, self.paper_emb)


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))

    model = Disambiguation()
    model.prepare(train_pub_data)
    model.save("author_features")
