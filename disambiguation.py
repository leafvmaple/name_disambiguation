import json
import codecs
import random
import pickle
import math
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from utility import get_author_feature

train_pub_data_path = 'data/train/train_pub.json'
# train_row_data_path = 'data/train/train_author.json'

FEATURES_PATH      = "author_features.txt"
FEATURES_JSON_PATH = "author_features.json"
FEATURES_EMB_PATH  = "aminer.emb"
FEATURES_IDF_PATH  = "feature_idf.pkl"

class Disambiguation:
    def __init__(self, pub_out_path=None, idf_out_path=None, emb_out_path=None):
        self.pub_out_path = pub_out_path
        self.idf_out_path = idf_out_path
        self.emb_out_path = emb_out_path

    def transform(self, pub_data):
        paper_pub = {}
        for paper_id, data in pub_data.items():
            author_features = get_author_feature(paper_id, data)
        for i, author_feature in enumerate(author_features):
            paper_pub["{}-{}".format(paper_id, i)] = author_feature

        if self.pub_out_path:
            with open(self.pub_out_path, 'w') as f:
                json.dump(paper_pub, f, indent='\t')
        
        return paper_pub

    def embedding(self, paper_pub):
        paper_idf = {}
        paper_data = []
        paper_emb = []
        counter = defaultdict(int)

        feature_cnt = 0
        sum_weight = 0

        for _, author_feature in paper_pub.items():
            random.shuffle(author_feature)
            paper_data.append(author_feature)

            feature_cnt += len(author_feature)
            for feature_item in author_feature:
                counter[feature_item] += 1

        for item in counter:
            paper_idf[item] = math.log(feature_cnt / counter[item])
        
        if self.idf_out_path:
            with open(self.idf_out_path, 'wb') as f:
                pickle.dump(paper_idf, f)

        model = Word2Vec(paper_data, size=100, window=5, min_count=5, workers=20)
        if self.emb_out_path:
            model.save(self.emb_out_path)

        for _, author_feature in paper_pub.items():
            for item in author_feature:
                if item not in model.wv:
                    continue
                weight = paper_idf[item] if item in paper_idf else 1
                paper_emb.append(model.wv[item] * weight)
                sum_weight += weight
        paper_emb = np.sum(paper_emb, axis=0) / sum_weight

        if self.emb_out_path:
            np.save(self.emb_out_path, paper_emb)

        return paper_emb

    def prepare(self, pub_data):
        self.paper_pub = self.transform(pub_data)
        self.paper_emb = self.embedding(self.paper_pub)


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))

    model = Disambiguation(
        pub_out_path=FEATURES_JSON_PATH,
        idf_out_path=FEATURES_IDF_PATH,
        emb_out_path=FEATURES_EMB_PATH
    )
    model.prepare(train_pub_data)
