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
from triplet_model import GlobalTripletModel
from gae_model import GraphAutoEncoders
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

sna_pub_data_path = 'data/sna_data/sna_valid_pub.json'
sna_row_data_path = 'data/sna_data/sna_valid_author_raw.json'

test_pub_data_path = "data/test/test_pub.json"
test_author_data_path = "data/test/test_author_fix.json"
test_label_data_path = "data/test/test_label.json"
test_size_data_path = "data/test/test_size.json"

class Disambiguation:
    def __init__(self):
        self.pub_data = None
        self.author_data = None

        self.cur_params = {}
        self.cur_size = 0

        self.model = None
        self.triplet_model = None

    def __generate_feature(self):
        paper_feats = {}
        print("---------Get Teatures------------")
        cnt = 0
        for paper_id, data in self.pub_data.items():
            #author_features = get_author_feature(paper_id, data)
            #if author_features is not None:
            #    paper_feats[paper_id] = author_features
            if cnt % 1000 == 0:
                print("({}/{})".format(cnt, len(self.pub_data)))
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

    def __generate_triplet(self, paper_embs):
        anchor_data = []
        pos_data = []
        neg_data = []
        paper_ids = []

        for _, author in self.author_data.items():
            for _, papers in author.items():
                for paper_id in papers:
                    paper_ids.append(paper_id)

        for _, author in self.author_data.items():
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

    def __embedding_params(self, paper_embs):
        self.cur_params = paper_embs
        self.cur_size = 100

    def __global_params(self, paper_embs, triplet_model):
        embs = []
        paper_ids = []
        for _, author in self.author_data.items():
            for _, papers in author.items():
                if len(papers) < 5:
                    continue
                for paper_id in papers:
                    if paper_id not in paper_embs:
                        continue
                    embs.append(paper_embs[paper_id])
                    paper_ids.append(paper_id)

        embs = np.stack(embs)
        embs = triplet_model.get_inter(embs)

        self.cur_params = {}
        for i, emb in enumerate(embs):
            self.cur_params[paper_ids[i]] = emb
        self.cur_size = 64

    def generate_embedding(self):
        print("---------embedding------------")
        paper_embs = {}
        paper_feats = self.__generate_feature()
        paper_idf, model = self.__word2vec(paper_feats)

        for paper_id, author_feature in paper_feats.items():
            paper_embs[paper_id] = get_feature_emb(paper_id, author_feature, paper_idf, model)

        self.__embedding_params(paper_embs)

        return paper_embs, paper_feats, paper_idf, model

    def generate_global(self):
        print("---------global------------")
        _, _, paper_embs = self.load_embedding()
        triplet_data = self.__generate_triplet(paper_embs)

        triplet_model = GlobalTripletModel()
        triplet_model.create(EMB_DIM)
        triplet_model.train(triplet_data)

        self.__global_params(paper_embs, triplet_model)

        return triplet_model

    def generate_local(self, name, author): # Test Data?
        features = []
        labels = []

        embs = []
        paper_ids = []
        author_ids = []
        paper_data = {}
        for author_id, papers in author.items():
            if len(papers) < 5:
                continue
            for paper_id in papers:
                if paper_id not in self.paper_embs:
                    continue
                embs.append(self.paper_embs[paper_id])
                paper_ids.append(paper_id)
                author_ids.append(author_id)

        if len(embs) == 0:
            return

        embs = np.stack(embs)
        embs = self.triplet_model.get_inter(embs)

        for i, emb in enumerate(embs):
            paper_data[paper_ids[i]] = {"author": author_ids[i], "paper": paper_ids[i], "emb": emb}
            self.cur_params[paper_ids[i]] = emb

        paper_data = [v for k, v in paper_data.items()]
        random.shuffle(paper_data)

        row_idx = []
        col_idx = []

        for i, data in enumerate(paper_data):
            features.append(data["emb"])
            labels.append(data["author"])

            author_feature1 = set(self.paper_feats[data["paper"]])

            for j in range(i + 1, len(paper_data)):
                data2 = paper_data[j]
                author_feature2 = set(self.paper_feats[data2["paper"]])
                common_features = author_feature1.intersection(author_feature2)
                idf_sum = 0
                for feature in common_features:
                    idf_sum += self.paper_idf[feature] if feature in self.paper_idf else IDF_THRESHOLD
                if idf_sum >= IDF_THRESHOLD:
                    row_idx.append(i)
                    col_idx.append(j)

        adj = sp.coo_matrix((np.ones(len(row_idx)), (np.array(row_idx), np.array(col_idx))),
                    shape=(len(paper_data), len(paper_data)), dtype=np.float32)

        features = np.array(features)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        print('Dataset {} has {} nodes, {} edges, {} features.'.format(name, adj.shape[0], len(row_idx), features.shape[1]))

        return adj, np.array(features), labels

    def fit(self, pub_data, author_data):
        self.pub_data = pub_data
        self.author_data = author_data

    def train_embedding(self):
        paper_embs, paper_feats, paper_idf, model = self.generate_embedding()

        os.makedirs(join(PROJ_DIR, "temp"), exist_ok=True)
        with open(join(PROJ_DIR, "temp", "embedding_paper_feats.json"), 'w') as f:
            json.dump(paper_feats, f, indent='\t')

        with open(join(PROJ_DIR, "temp", "embedding_paper_idf.pkl"), 'wb') as f:
            pickle.dump(paper_idf, f)

        with open(join(PROJ_DIR, "temp", "embedding_paper_embs.pkl"), 'wb') as f:
            pickle.dump(paper_embs, f)

        model.save(join(PROJ_DIR, "temp", "embedding_w2v.model"))

    def train_global(self):
        triplet_model = self.generate_global()

        with open(join(PROJ_DIR, "temp", "global_model.json"), 'w') as f:
            f.write(triplet_model.model.to_json())
            f.close()

        triplet_model.model.save_weights(join(PROJ_DIR, "temp", "global_model-triplets-{}.h5".format(EMB_DIM)))

    def train_local(self, raw_data):
        self.load_global()

        wf = codecs.open(join(PROJ_DIR, 'local_clustering_results.csv'), 'w', encoding='utf-8')
        wf.write('name,n_pubs,n_clusters,precision,recall,f1\n')
        metrics = np.zeros(3)

        cnt = 0

        for name, author in raw_data.items():
            adj, features, labels = self.generate_local(name, author)
            if adj is None:
                continue
            cur_metric, num_nodes, n_clusters = self.gae_model.train(adj, features, labels)
            wf.write('{0},{1},{2},{3:.5f},{4:.5f},{5:.5f}\n'.format(
                name, num_nodes, n_clusters, cur_metric[0], cur_metric[1], cur_metric[2]))
            wf.flush()
            for i, m in enumerate(cur_metric):
                metrics[i] += m
            cnt += 1
            macro_prec = metrics[0] / cnt
            macro_rec = metrics[1] / cnt
            macro_f1 = cal_f1(macro_prec, macro_rec)
            print('average until now', [macro_prec, macro_rec, macro_f1])
        macro_prec = metrics[0] / cnt
        macro_rec = metrics[1] / cnt
        macro_f1 = cal_f1(macro_prec, macro_rec)
        wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
            macro_prec, macro_rec, macro_f1))
        wf.close()

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
        self.gae_model = GraphAutoEncoders()

        with open(join(PROJ_DIR, "temp" ,"prepare_pub.json"), 'r') as f:
            self.paper_feats = json.load(f)

        with open(join(PROJ_DIR, "temp", "prepare_embs.pkl"), 'rb') as f:
            self.paper_embs = pickle.load(f)

        self.triplet_model = GlobalTripletModel()

        with open(join(PROJ_DIR, "temp", "global_model.json"), 'r') as f:
            self.triplet_model.model = model_from_json(f.read())
            f.close()
            self.triplet_model.model.load_weights(join(PROJ_DIR, "temp", "global_model-triplets-{}.h5".format(EMB_DIM)))

        self.__global_params(paper_embs, triplet_model)

    def __get_precision(self, embs, labels, cluster_size):
        embs_norm = normalize_vectors(embs)
        clusters_pred = clustering(embs_norm, num_clusters=cluster_size)
        return  pairwise_precision_recall_f1(clusters_pred, labels)

    def score(self, author_data, labels, cluster_size, report_path=None):
        prec_cnt, rec_cnt, f1_cnt = 0, 0, 0

        if report_path is not None:
            f = open(join(PROJ_DIR, "report", report_path), "w")

        for name, paper_ids in author_data.items():
            params = []
            for paper_id in paper_ids:
                params.append(self.cur_params[paper_id] if paper_id in self.cur_params else np.zeros(self.cur_size))

            params = np.stack(params)

            prec, rec, f1 = self.__get_precision(params, labels[name], cluster_size[name])

            f.write("{} precision {:.5f} recall {:.5f} f1 {:.5f}\n". format(name, prec, rec, f1))

            prec_cnt += prec
            rec_cnt += rec
            f1_cnt += f1

        f.close()

        cnt = len(author_data)
        return prec_cnt / cnt, rec_cnt / cnt, f1_cnt / cnt

        #if self.triplet_model is not None:
        #    embs = self.triplet_model.get_inter(embs)
        #    return self.predict(embs, labels, cluster_size)


if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
    train_row_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

    test_author_data = json.load(open(test_author_data_path, 'r', encoding='utf-8'))
    test_label_data = json.load(open(test_label_data_path, 'r', encoding='utf-8'))
    test_size_data = json.load(open(test_size_data_path, 'r', encoding='utf-8'))

    model = Disambiguation()
    model.fit(train_pub_data, train_row_data)

    model.train_embedding()
    prec, rec, f1 = model.score(test_author_data, test_label_data, test_size_data, "embedding_{:.0f}.txt".format(time.time()))
    print("Embedding precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec, rec, f1))

    model.train_global()
    prec, rec, f1 = model.score(test_author_data, test_label_data, test_size_data, "global_{:.0f}.txt".format(time.time()))
    print("Global precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec, rec, f1))

    #model.train_global_model(train_row_data)
    #model.save_global("global")

    #model.train_local_model(train_row_data)
