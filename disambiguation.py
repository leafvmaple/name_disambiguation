import json
import codecs
import random
import pickle
import math
import gc
import os
import time
import numpy as np
import scipy.sparse as sp
from os.path import join, abspath, dirname
from keras.models import Model, model_from_json
from gensim.models import Word2Vec
from collections import defaultdict

from models.triplet import GlobalTripletModel
from models.vgae import GraphAutoEncoders
from models.cluster import ClusterModel
from sklearn.cluster import DBSCAN

from utility.features import get_author_feature, get_author_features, get_feature_emb
from utility.model import discretization, cal_f1, clustering, pairwise_precision_recall_f1
from gae.preprocessing import normalize_vectors

EMB_DIM = 100
IDF_THRESHOLD = 32
PROJ_DIR = abspath(dirname(__file__))

# train_pub_data_path = 'data/train/train_pub.json'
# train_row_data_path = 'data/train/train_author.json'

train_pub_data_path = 'data/train/train_pub_fix.json'
train_row_data_path = 'data/train/train_author_fix.json'

sna_pub_data_path = 'data/sna_data/sna_valid_pub_fix.json'
sna_row_data_path = 'data/sna_data/sna_valid_author_raw_fix.json'

test_pub_data_path = "data/test/test_pub.json"
test_author_data_path = "data/test/test_author_fix.json"
test_label_data_path = "data/test/test_label.json"
test_size_data_path = "data/test/test_size.json"

class Disambiguation:
    def __init__(self):
        self.pub_data = None
        self.author_data = None

        self.paper_embs = {}
        self.global_emb = {}
        self.cur_size = 0

        self.style = "None"

        self.model = None
        self.triplet_model = None

    def __generate_feature(self, pub_data):
        paper_feats = {}
        print("---------Get Teatures------------")
        cnt = 0
        for paper_id, data in pub_data.items():
            #author_features = get_author_feature(paper_id, data)
            #if author_features is not None:
            #    paper_feats[paper_id] = author_features
            if cnt % 1000 == 0:
                print("({}/{})".format(cnt, len(pub_data)))
            cnt += 1

            count = len(data["authors"])
            if count <= 100:
                for i in range(count):
                    author_features = get_author_features(paper_id, data, i)
                    if author_features is not None:
                        paper_feats["{}-{}".format(paper_id, i)] = author_features

        return paper_feats

    def __word2vec(self, paper_feats):
        print("---------word2vec------------")
        paper_idf = {}

        paper_data = []
        features = defaultdict(int)
        feature_cnt = 0

        for _, author_feature in paper_feats.items():
            random_data = author_feature[:]
            random.shuffle(random_data)
            paper_data.append(random_data)

            feature_cnt += len(author_feature)
            for feature_item in author_feature:
                features[feature_item] += 1

        for feature_id in features:
            paper_idf[feature_id] = math.log(feature_cnt / features[feature_id])

        return paper_idf, Word2Vec(paper_data, size=EMB_DIM, window=5, min_count=5, workers=20)

    def __generate_triplet(self, paper_embs, author_data):
        anchor_data = []
        pos_data = []
        neg_data = []
        paper_ids = []

        for _, author in author_data.items():
            for _, papers in author.items():
                for paper_id in papers:
                    paper_ids.append(paper_id)

        for _, author in author_data.items():
            for _, papers in author.items():
                for i, paper_id in enumerate(papers):
                    if paper_id not in paper_embs:
                        continue
                    sample_length = min(6, len(papers))
                    idxs = random.sample(range(len(papers)), sample_length)
                    for idx in idxs:
                        if idx == i:
                            continue
                        pos_paper = papers[idx]
                        if pos_paper not in paper_embs:
                            continue

                        neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]
                        while neg_paper in papers or neg_paper not in paper_embs:
                            neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]

                        anchor_data.append(paper_embs[paper_id])
                        pos_data.append(paper_embs[pos_paper])
                        neg_data.append(paper_embs[neg_paper])

        triplet_data = {}
        triplet_data["anchor_input"] = np.stack(anchor_data)
        triplet_data["pos_input"] = np.stack(pos_data)
        triplet_data["neg_input"] = np.stack(neg_data)

        return triplet_data

    def __generate_adjacency(self, paper_idf, paper_feats, author):
        features = []
        labels = []
        ids = []

        embs = []
        paper_ids = []
        author_ids = []
        paper_data = {}
        for author_id, papers in author.items():
            if len(papers) < 5:
                continue
            for paper_id in papers:
                if paper_id not in self.global_emb:
                    continue
                embs.append(self.global_emb[paper_id])
                paper_ids.append(paper_id)
                author_ids.append(author_id)

        if len(embs) == 0:
            return None, None, None, None

        for i, emb in enumerate(embs):
            paper_data[paper_ids[i]] = {"author": author_ids[i], "paper": paper_ids[i], "emb": emb}

        paper_data = [v for k, v in paper_data.items()]
        random.shuffle(paper_data)

        rows = []
        cols = []

        for i, data in enumerate(paper_data):
            features.append(data["emb"])
            labels.append(data["author"])
            ids.append(data["paper"])

            author_feature1 = set(paper_feats[data["paper"]])

            for j in range(i + 1, len(paper_data)):
                data2 = paper_data[j]
                author_feature2 = set(paper_feats[data2["paper"]])
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for feature in common_features:
                    idf_sum += paper_idf[feature] if feature in paper_idf else IDF_THRESHOLD
                if idf_sum >= IDF_THRESHOLD:
                    rows.append(i)
                    cols.append(j)

        adj = sp.coo_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))), shape=(len(paper_data), len(paper_data)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj, np.array(features), labels, ids

    def __embedding_params(self, paper_embs):
        self.paper_embs = paper_embs
        self.style = "embedding"

    def __global_params(self, paper_embs, triplet_model):
        embs = []
        paper_ids = []
        # Error
        for _, papers in self.author_data.items():
            for paper_id in papers:
                if paper_id not in paper_embs:
                    continue
                embs.append(paper_embs[paper_id])
                paper_ids.append(paper_id)

        embs = np.stack(embs)
        embs = triplet_model.get_inter(embs)

        self.global_emb = {}
        for i, emb in enumerate(embs):
            self.global_emb[paper_ids[i]] = emb

        self.style = "global"

    def __local_params(self, paper_embs, local_emb):
        for paper_id, emb in paper_embs.items():
            self.cur_params[paper_id] = local_emb[paper_id] if paper_id in local_emb else emb

        self.cur_size = 100

    def __get_author_emb(self, papers):
        params = []
        for paper_id in papers:
            if self.style == "embedding":
                param = self.paper_embs[paper_id] if paper_id in self.paper_embs else np.zeros(100)
            elif self.style == "global":
                param = self.global_emb[paper_id] if paper_id in self.global_emb else np.zeros(64)

            params.append(param)
        return np.stack(params)

    def generate_embedding(self, pub_data):
        print("---------embedding------------")
        paper_embs = {}
        paper_feats = self.__generate_feature(pub_data)
        paper_idf, model = self.__word2vec(paper_feats)

        for paper_id, author_feature in paper_feats.items():
            paper_embs[paper_id] = get_feature_emb(paper_id, author_feature, paper_idf, model)

        return paper_embs, paper_feats, paper_idf, model

    def generate_global(self, author_data):
        print("---------global------------")
        _, _, paper_embs = self.load_embedding()
        triplet_data = self.__generate_triplet(paper_embs, author_data)

        triplet_model = GlobalTripletModel(EMB_DIM)
        triplet_model.fit(triplet_data)

        self.__global_params(paper_embs, triplet_model)

        return triplet_model

    def generate_local(self):
        paper_feats, paper_idf, paper_embs = self.load_embedding()
        self.load_global()

        gae_model = GraphAutoEncoders()

        wf = codecs.open(join(PROJ_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
        wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')

        local_embs = {}

        for name, author in self.author_data.items():
            adj, features, labels, ids = self.__generate_adjacency(paper_idf, paper_feats, author)
            if adj is None:
                continue
            print('Dataset {} has {} nodes, {} features.'.format(name, adj.shape[0], features.shape[1]))

            gae_model.fit(adj, features, labels)
            prec, rec, f1 = gae_model.score(labels)
            wf.write('{},{:.5f},{:.5f},{:.5f}\n'.format(name, prec, rec, f1))
            wf.flush()

            emb = gae_model.get_embs()

            for i, paper_id in enumerate(ids):
                local_embs[paper_id] = emb[i]

        wf.close()

        self.__global_params(paper_embs, local_embs)

        return local_embs

    def fit(self, pub_data, author_data):
        self.pub_data = pub_data
        self.author_data = author_data

    def train_embedding(self):
        paper_embs, paper_feats, paper_idf, model = self.generate_embedding(self.pub_data)
        self.__embedding_params(paper_embs)

        os.makedirs(join(PROJ_DIR, "temp"), exist_ok=True)
        with open(join(PROJ_DIR, "temp", "embedding_paper_feats.json"), 'w') as f:
            json.dump(paper_feats, f, indent='\t')

        with open(join(PROJ_DIR, "temp", "embedding_paper_idf.pkl"), 'wb') as f:
            pickle.dump(paper_idf, f)

        with open(join(PROJ_DIR, "temp", "embedding_paper_embs.pkl"), 'wb') as f:
            pickle.dump(paper_embs, f)

        model.save(join(PROJ_DIR, "temp", "embedding_w2v.model"))

    def train_global(self, author_data):
        triplet_model = self.generate_global(author_data)
        triplet_model.save(join(PROJ_DIR, "temp", "global_model"))

    def train_local(self):
        local_embs = self.generate_local()

        with open(join(PROJ_DIR, "temp", "local_paper_embs.pkl"), 'wb') as f:
            pickle.dump(local_embs, f)

    def load_embedding(self):
        with open(join(PROJ_DIR, "temp", "embedding_paper_feats.json"), 'r') as f:
            paper_feats = json.load(f)

        with open(join(PROJ_DIR, "temp", "embedding_paper_idf.pkl"), 'rb') as f:
            paper_idf = pickle.load(f)

        with open(join(PROJ_DIR, "temp", "embedding_paper_embs.pkl"), 'rb') as f:
            paper_embs = pickle.load(f)

        self.__embedding_params(paper_embs)

        return paper_feats, paper_idf, paper_embs

    def load_global(self):
        triplet_model = GlobalTripletModel(EMB_DIM)
        triplet_model.load(join(PROJ_DIR, "temp", "global_model"))

        with open(join(PROJ_DIR, "temp", "embedding_paper_embs.pkl"), 'rb') as f:
            paper_embs = pickle.load(f)

        self.__global_params(paper_embs, triplet_model)

        return triplet_model

    def train_cluster(self, train_row_data):
        model = ClusterModel(dimension=EMB_DIM, k=600)
        model.fit(train_row_data, self.paper_embs)

        model.save(join(PROJ_DIR, "temp", "cluster_model"))

    def simple_cluster(self, author_data):
        for name, papers in author_data.items():
            embs_norm = normalize_vectors(self.__get_author_emb(papers))
            clus = DBSCAN(eps = 0.0002, min_samples = 4).fit_predict(embs_norm)
            print(name, len(set(clus)))

    def predict_cluster(self, pub_data, X):
        model = ClusterModel(dimension=EMB_DIM, k=600)
        model.load(join(PROJ_DIR, "temp", "cluster_model"))

        paper_embs, _, _, _ = self.generate_embedding(pub_data)
        return model.predict(X, paper_embs)

    def predict(self, paper_ids, cluster_size):
        embs_norm = normalize_vectors(self.__get_author_emb(paper_ids))
        return clustering(embs_norm, num_clusters=cluster_size)

    def score(self, author_data, labels, cluster_size, report_path=None):
        prec_cnt, rec_cnt, f1_cnt = 0, 0, 0

        if report_path is not None:
            f = open(join(PROJ_DIR, "report", report_path), "w")

        for name, paper_ids in author_data.items():
            clusters_pred = self.predict(paper_ids, cluster_size[name])
            prec, rec, f1 = pairwise_precision_recall_f1(clusters_pred, labels[name])

            f.write("{} precision {:.5f} recall {:.5f} f1 {:.5f}\n". format(name, prec, rec, f1))

            prec_cnt += prec
            rec_cnt += rec
            f1_cnt += f1

        f.close()

        cnt = len(author_data)
        return prec_cnt / cnt, rec_cnt / cnt, f1_cnt / cnt


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
    train_row_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

    test_author_data = json.load(open(test_author_data_path, 'r', encoding='utf-8'))
    test_label_data = json.load(open(test_label_data_path, 'r', encoding='utf-8'))
    test_size_data = json.load(open(test_size_data_path, 'r', encoding='utf-8'))

    test_pub_data = json.load(open(sna_pub_data_path, 'r', encoding='utf-8'))
    test_row_data = json.load(open(sna_row_data_path, 'r', encoding='utf-8'))
    # test_author_data = json.load(open(test_author_data_path, 'r', encoding='utf-8'))
    # test_label_data = json.load(open(test_label_data_path, 'r', encoding='utf-8'))


    model = Disambiguation()
    model.fit(train_pub_data, test_author_data)

    model.train_embedding()
    # model.load_embedding()
    model.train_cluster(train_row_data)

    prec, rec, f1 = model.score(test_author_data, test_label_data, test_size_data, "embedding_{:.0f}.csv".format(time.time()))
    print("Embedding precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec, rec, f1))

    model.train_global(train_row_data)
    # model.load_global()
    prec, rec, f1 = model.score(test_author_data, test_label_data, test_size_data, "global_{:.0f}.csv".format(time.time()))
    print("Global precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec, rec, f1))

    model.train_local()
    prec, rec, f1 = model.score(test_author_data, test_label_data, test_size_data, "local_{:.0f}.csv".format(time.time()))
    print("Local precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec, rec, f1))

