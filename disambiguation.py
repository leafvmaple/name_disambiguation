import json
import codecs
import random
import pickle
import math
import gc
import os
import numpy as np
import scipy.sparse as sp
from os.path import join, abspath, dirname
from keras.models import Model, model_from_json
from gensim.models import Word2Vec
from collections import defaultdict
from triplet_model import GlobalTripletModel
from gae_model import GraphAutoEncoders
from utility.features import get_author_feature, get_author_features, get_feature_emb
from utility.model import discretization

EMB_DIM = 100
IDF_THRESHOLD = 32
PROJ_DIR = abspath(dirname(__file__))

train_pub_data_path = 'data/train/train_pub.json'
train_row_data_path = 'data/train/train_author.json'

sna_pub_data_path = 'data/sna_data/sna_valid_pub.json'
sna_row_data_path = 'data/sna_data/sna_valid_author_raw.json'

class Disambiguation:
    def __init__(self):
        self.paper_pub = {}
        self.paper_idf = {}
        self.paper_emb = {}

        self.labels   = {}
        self.features = {}
        self.adj      = {}

        self.triplet_data = {}
        self.model = None

    def transform(self, pub_data):
        print("---------Transfrom------------")

        for paper_id, data in pub_data.items():
            author_features = get_author_feature(paper_id, data)
            if author_features is not None:
                self.paper_pub[paper_id] = author_features
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

        for paper_id, author_feature in self.paper_pub.items():
            self.paper_emb[paper_id] = get_feature_emb(paper_id, author_feature, self.paper_idf, self.model)

    def prepare(self, pub_data):
        self.transform(pub_data)
        self.word2vec()
        self.embedding()
    
    def generate_triplet(self, author_data):
        anchor_data = []
        pos_data = []
        neg_data = []
        paper_ids = []

        for _, author in author_data.items():
            for _, papers in author.items():
                for paper_id in papers:
                    paper_ids.append(paper_id)

        with open("papers.json", 'w') as f:
            json.dump(paper_ids, f, indent='\t')

        for _, author in author_data.items():
            for _, papers in author.items():
                for i, paper_id in enumerate(papers):
                    if paper_id not in self.paper_emb:
                        continue
                    sample_length = min(6, len(papers))
                    idxs = random.sample(range(len(papers)), sample_length)
                    for idx in idxs:
                        if idx == i:
                            continue
                        pos_paper = papers[idx]
                        if pos_paper not in self.paper_emb:
                            continue

                        neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]
                        while neg_paper in papers or neg_paper not in self.paper_emb:
                            neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]

                        anchor_data.append(self.paper_emb[paper_id])
                        pos_data.append(self.paper_emb[pos_paper])
                        neg_data.append(self.paper_emb[neg_paper])

        triplet_data = {}
        triplet_data["anchor_input"] = np.stack(anchor_data)
        triplet_data["pos_input"] = np.stack(pos_data)
        triplet_data["neg_input"] = np.stack(neg_data)

        return triplet_data

    def train_global_model(self, author_data):
        # Test
        # with open("author_features_pkl.idf", 'rb') as f:
        #    self.paper_idf = pickle.load(f)

        with open(join(PROJ_DIR, "temp", "prepare_embs.pkl"), 'rb') as f:
            self.paper_emb = pickle.load(f)

        self.triplet_data = self.generate_triplet(author_data)

        self.triplet_model = GlobalTripletModel()
        self.triplet_model.create(EMB_DIM)
        self.triplet_model.train(self.triplet_data)

    def generate_local(self, raw_data): # Test Data?
        self.gae_model = GraphAutoEncoders()

        for name, author in raw_data.items():
            features = []
            self.labels[name] = []

            embs = []
            ids = []
            paper_data = {}
            for author_id, papers in author.items():
                if len(papers) < 5:
                    continue
                for paper_id in papers:
                    if paper_id not in self.paper_emb:
                        continue
                    embs.append(self.paper_emb[paper_id])
                    ids.append(paper_id)

            if len(embs) == 0:
                continue

            embs = np.stack(embs)
            embs = self.triplet_model.get_inter(embs)

            for i, emb in enumerate(embs):
                paper_data[ids[i]] = {"id": ids[i], "emb": emb}

            paper_data = [v for k, v in paper_data.items()]
            random.shuffle(paper_data)

            row_idx = []
            col_idx = []

            for i, data in enumerate(paper_data):
                features.append(data["emb"])
                self.labels[name].append(data["id"])

                author_feature1 = set(self.paper_pub[data["id"]])

                for j in range(i + 1, len(paper_data)):
                    data2 = paper_data[j]
                    author_feature2 = set(self.paper_pub[data2["id"]])
                    common_features = author_feature1.intersection(author_feature2)
                    idf_sum = 0
                    for feature in common_features:
                        idf_sum += self.paper_idf[feature] if feature in self.paper_idf else IDF_THRESHOLD
                    if idf_sum >= IDF_THRESHOLD:
                        row_idx.append(i)
                        col_idx.append(j)

            adj = sp.coo_matrix((np.ones(len(row_idx)), (np.array(row_idx), np.array(col_idx))),
                        shape=(len(paper_data), len(paper_data)), dtype=np.float32)

            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            self.adj[name] = adj
            self.features[name] = np.array(features)

            print('Dataset {} has {} nodes, {} edges, {} features.'.format(name, adj.shape[0], len(row_idx), self.features[name].shape[1]))

            self.gae_model.train(adj, np.array(features), self.labels[name])

    def train_local_model(self, raw_data):
        with open(join(PROJ_DIR, "temp" ,"prepare_pub.json"), 'r') as f:
            self.paper_pub = json.load(f)

        with open(join(PROJ_DIR, "temp", "prepare_embs.pkl"), 'rb') as f:
            self.paper_emb = pickle.load(f)

        self.triplet_model = GlobalTripletModel()

        with open(join(PROJ_DIR, "temp", "global_model.json"), 'r') as f:
            self.triplet_model.model = model_from_json(f.read())
            f.close()
            self.triplet_model.model.load_weights(join(PROJ_DIR, "temp", "global_model-triplets-{}.h5".format(EMB_DIM)))

        self.generate_local(raw_data)
    
    def get_auc_score(self, predict_data):
        predict_triplet = self.generate_triplet(predict_data)
        self.triplet_model.full_auc(predict_triplet)

    def save_prepare(self, path):
        temp_path = join(PROJ_DIR, "temp")
        os.makedirs(temp_path, exist_ok=True)

        with open(join(temp_path, path + "_pub.json"), 'w') as f:
            json.dump(self.paper_pub, f, indent='\t')

        with open(join(temp_path, path + "_idf.pkl"), 'wb') as f:
            pickle.dump(self.paper_idf, f)

        with open(join(temp_path, path + "_embs.pkl"), 'wb') as f:
            pickle.dump(self.paper_emb, f)

        self.model.save(join(temp_path, path + ".model"))

    def save_global(self, path):
        temp_path = join(PROJ_DIR, "temp")
        os.makedirs(temp_path, exist_ok=True)

        with open(join(temp_path, path + "_model.json"), 'w') as f:
            f.write(self.triplet_model.model.to_json())
            f.close()

        self.triplet_model.model.save_weights(join(temp_path, path + "_model-triplets-{}.h5".format(EMB_DIM)))


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
    train_row_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

    model = Disambiguation()
    # model.prepare(train_pub_data)
    # model.save_prepare("prepare")

    # model.train_global_model(train_row_data)
    # model.save_global("global")
    # model.get_auc_score()

    model.train_local_model(train_row_data)
