import json
import codecs
import random
import pickle
import math
import gc
import numpy as np
import scipy.sparse as sp
from os.path import join
from gensim.models import Word2Vec
from collections import defaultdict
from triplet_model import GlobalTripletModel
from utility.features import get_author_feature, get_author_features, get_feature_emb
from utility.model import get_hidden_output, discretization

EMB_DIM = 100
IDF_THRESHOLD = 32

train_pub_data_path = 'data/train/train_pub.json'
train_row_data_path = 'data/train/train_author.json'

class Disambiguation:
    def __init__(self):
        self.paper_pub = {}
        self.paper_idf = {}
        self.paper_emb = {}

        self.labels   = {}
        self.features = {}

        self.papers = []
        self.triplet_data = {}
        self.model = None

    def transform(self, pub_data):
        print("---------Transfrom------------")

        for paper_id, data in pub_data.items():
            self.paper_pub[paper_id] = get_author_feature(paper_id, data)
            # author_features, author_name = get_author_features(paper_id, data)
            # for i, author_feature in enumerate(author_features):
            #    self.paper_pub["{}-{}".format(paper_id, author_name[i])] = author_feature

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

        self.model = Word2Vec(paper_data, size=EMB_DIM, window=5, min_count=5, workers=20)

    def embedding(self):
        print("---------embedding------------")

        for unique_id, author_feature in self.paper_pub.items():
            self.paper_emb[unique_id] = get_feature_emb(author_feature, self.paper_idf, self.model)

    def prepare(self, pub_data):
        self.transform(pub_data)
        self.word2vec()
        self.embedding()

    def global_data(self, author_data):
        for _, author in author_data.items():
            for _, papers in author.items():
                for paper_id in papers:
                    self.papers.append(paper_id)
        with open("papers.json", 'w') as f:
            json.dump(self.papers, f, indent='\t')
    
    def generate_triplet(self, author_data):
        anchor_data = np.empty([0, EMB_DIM])
        pos_data = np.empty([0, EMB_DIM])
        neg_data = np.empty([0, EMB_DIM])

        for _, author in author_data.items():
            for _, papers in author.items():
                for i, paper_id in enumerate(papers):
                    sample_length = min(6, len(papers))
                    idxs = random.sample(range(len(papers)), sample_length)
                    for idx in idxs:
                        if idx == i:
                            continue
                        pos_paper = papers[idx]
                        neg_paper = self.papers[random.randint(0, len(self.papers) - 1)]
                        while neg_paper in papers:
                            neg_paper = self.papers[random.randint(0, len(self.papers) - 1)]

                        anchor_data = np.concatenate((anchor_data, self.paper_emb[paper_id]))
                        pos_data = np.concatenate((pos_data, self.paper_emb[pos_paper]))
                        neg_data = np.concatenate((neg_data, self.paper_emb[neg_paper]))

        self.triplet_data["anchor_input"] = anchor_data
        self.triplet_data["pos_input"] = pos_data
        self.triplet_data["neg_input"] = neg_data

    def train_global_model(self, author_data):
        # Test
        # with open("author_features_pkl.idf", 'rb') as f:
        #    self.paper_idf = pickle.load(f)

        with open("author_features_pkl.model", 'rb') as f:
            self.paper_emb = pickle.load(f)

        self.global_data(author_data)
        self.generate_triplet(author_data)

        self.triplet_model = GlobalTripletModel()
        self.triplet_model.create(EMB_DIM)
        self.triplet_model.train(self.triplet_data)

    def generate_local(self, raw_data): # Test Data?
        for name, author in raw_data.items():
            self.features[name] = []
            paper_embs = []
            paper_ids = []
            paper_map = {}
            for author_id, papers in author.items():
                if len(papers) < 5:
                    continue
                for paper_id in papers:
                    if self.paper_emb[paper_id] is None:
                        continue
                    paper_embs.append(self.paper_emb[paper_id])
                    paper_ids.append(paper_id)

            paper_embs = np.stack(paper_embs)
            inter_embs = get_hidden_output(self.triplet_model.model, paper_embs)

            idxs = range(0, len(paper_ids))
            random.shuffle(idxs)

            self.labels[name] = discretization([paper_ids[i] for i in idxs])
            for i, idx in enumerate(idxs):
                self.features[name].append(list(map(str, inter_embs[idx])))
                paper_map[paper_id] = i
            
            paper_ids = list(set(paper_ids))
            row_idx = []
            col_idx = []
            for i, paper_id in enumerate(paper_ids):
                author_feature1 = set(self.paper_emb[paper_id])
                
                for j in range(i + 1, len(paper_ids)):
                    author_feature2 = set(self.paper_emb[paper_ids[j]])
                    common_features = author_feature1.intersection(author_feature2)
                    idf_sum = 0
                    for feature in common_features:
                        idf_sum += self.paper_idf[feature]
                        # print(f, idf.get(f, idf_threshold))
                    if idf_sum >= IDF_THRESHOLD:
                        row_idx.append(paper_map[paper_id])
                        col_idx.append(paper_map[paper_ids[j]])

            self.adj = sp.coo_matrix((np.ones(len(row_idx)), (np.array(row_idx), np.array(col_idx))),
                        shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)


    def save(self, output_path):
        with open(output_path + "_pub.json", 'w') as f:
            json.dump(self.paper_pub, f, indent='\t')

        with open(output_path + "_pkl.idf", 'wb') as f:
            pickle.dump(self.paper_idf, f)

        with open(output_path + "_pkl.model", 'wb') as f:
            pickle.dump(self.paper_emb, f)

        self.model.save(output_path + ".model")


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
    train_row_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

    model = Disambiguation()
    model.prepare(train_pub_data)
    model.save("author_features")
    # model.train_global_model(train_row_data)