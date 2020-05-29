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
from utility.model import discretization, cal_f1, clustering, pairwise_precision_recall_f1
from gae.preprocessing import normalize_vectors

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
        self.paper_embs = {}
        self.paper2author = {}

        self.labels   = {}
        self.features = {}
        self.adj      = {}

        self.triplet_data = {}
        self.model = None
        self.triplet_model = None

    def transform(self, pub_data):
        print("---------Transfrom------------")
        for paper_id, data in pub_data.items():
            author_features = get_author_feature(paper_id, data)
            if author_features is not None:
                self.paper_pub[paper_id] = author_features

    def word2vec(self):
        print("---------word2vec------------")
        paper_data = []
        features = defaultdict(int)
        feature_cnt = 0

        for _, author_feature in self.paper_pub.items():
            random.shuffle(author_feature)
            paper_data.append(author_feature)

            feature_cnt += len(author_feature)
            for feature_item in author_feature:
                features[feature_item] += 1

        for feature_id in features:
            self.paper_idf[feature_id] = math.log(feature_cnt / features[feature_id])

        self.model = Word2Vec(paper_data, size=EMB_DIM, window=5, min_count=5, workers=20)

    def embedding(self):
        print("---------embedding------------")
        for paper_id, author_feature in self.paper_pub.items():
            self.paper_embs[paper_id] = get_feature_emb(paper_id, author_feature, self.paper_idf, self.model)

    def fit(self, pub_data):
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

        for _, author in author_data.items():
            for _, papers in author.items():
                for i, paper_id in enumerate(papers):
                    if paper_id not in self.paper_embs:
                        continue
                    sample_length = min(6, len(papers))
                    idxs = random.sample(range(len(papers)), sample_length)
                    for idx in idxs:
                        if idx == i:
                            continue
                        pos_paper = papers[idx]
                        if pos_paper not in self.paper_embs:
                            continue

                        neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]
                        while neg_paper in papers or neg_paper not in self.paper_embs:
                            neg_paper = paper_ids[random.randint(0, len(paper_ids) - 1)]

                        anchor_data.append(self.paper_embs[paper_id])
                        pos_data.append(self.paper_embs[pos_paper])
                        neg_data.append(self.paper_embs[neg_paper])

        triplet_data = {}
        triplet_data["anchor_input"] = np.stack(anchor_data)
        triplet_data["pos_input"] = np.stack(pos_data)
        triplet_data["neg_input"] = np.stack(neg_data)

        return triplet_data

    def load_prepare(self):
        with open(join(PROJ_DIR, "temp", "prepare_embs.pkl"), 'rb') as f:
            self.paper_embs = pickle.load(f)

    def train_global(self, author_data):
        # Test
        # with open("author_features_pkl.idf", 'rb') as f:
        #    self.paper_idf = pickle.load(f)

        self.load_prepare()

        triplet_data = self.generate_triplet(author_data)

        self.triplet_model = GlobalTripletModel()
        self.triplet_model.create(EMB_DIM)
        self.triplet_model.train(triplet_data)

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

        paper_data = [v for k, v in paper_data.items()]
        random.shuffle(paper_data)

        row_idx = []
        col_idx = []

        for i, data in enumerate(paper_data):
            features.append(data["emb"])
            labels.append(data["author"])

            author_feature1 = set(self.paper_pub[data["paper"]])

            for j in range(i + 1, len(paper_data)):
                data2 = paper_data[j]
                author_feature2 = set(self.paper_pub[data2["paper"]])
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

    def get_auc_score(self, predict_data):
        predict_triplet = self.generate_triplet(predict_data)
        self.triplet_model.full_auc(predict_triplet)

    def load_global(self):
        self.gae_model = GraphAutoEncoders()

        with open(join(PROJ_DIR, "temp" ,"prepare_pub.json"), 'r') as f:
            self.paper_pub = json.load(f)

        with open(join(PROJ_DIR, "temp", "prepare_embs.pkl"), 'rb') as f:
            self.paper_embs = pickle.load(f)

        self.triplet_model = GlobalTripletModel()

        with open(join(PROJ_DIR, "temp", "global_model.json"), 'r') as f:
            self.triplet_model.model = model_from_json(f.read())
            f.close()
            self.triplet_model.model.load_weights(join(PROJ_DIR, "temp", "global_model-triplets-{}.h5".format(EMB_DIM)))

    def save_prepare(self, path):
        temp_path = join(PROJ_DIR, "temp")
        os.makedirs(temp_path, exist_ok=True)

        with open(join(temp_path, path + "_pub.json"), 'w') as f:
            json.dump(self.paper_pub, f, indent='\t')

        with open(join(temp_path, path + "_idf.pkl"), 'wb') as f:
            pickle.dump(self.paper_idf, f)

        with open(join(temp_path, path + "_embs.pkl"), 'wb') as f:
            pickle.dump(self.paper_embs, f)

        self.model.save(join(temp_path, path + ".model"))

    def save_global(self, path):
        temp_path = join(PROJ_DIR, "temp")
        os.makedirs(temp_path, exist_ok=True)

        with open(join(temp_path, path + "_model.json"), 'w') as f:
            f.write(self.triplet_model.model.to_json())
            f.close()

        self.triplet_model.model.save_weights(join(temp_path, path + "_model-triplets-{}.h5".format(EMB_DIM)))

    def predict(self, embs, labels, cluster_size):
        embs_norm = normalize_vectors(embs)
        clusters_pred = clustering(embs_norm, num_clusters=cluster_size)
        return  pairwise_precision_recall_f1(clusters_pred, labels)


    def score(self, papers, labels, cluster_size):
        embs = []
        for paper_id in papers:
            embs.append(self.paper_embs[paper_id] if paper_id in self.paper_embs else np.zeros(100))

        embs = np.stack(embs)

        return self.predict(embs, labels, cluster_size)

        #if self.triplet_model is not None:
        #    embs = self.triplet_model.get_inter(embs)
        #    return self.predict(embs, labels, cluster_size)



if __name__ == '__main__':
    
    train_pub_data = json.load(open(train_pub_data_path, 'r', encoding='utf-8'))
    train_row_data = json.load(open(train_row_data_path, 'r', encoding='utf-8'))

    model = Disambiguation()
    #model.fit(train_pub_data)
    #model.save_prepare("prepare")

    model.load_prepare()
    model.load_global()

    prec_count = 0
    rec_count = 0
    f1_count = 0
    for _, author_data in train_row_data.items():
        pred_data = {}
        cluster_size = len(author_data)
        for author_id, papers in author_data.items():
            for paper_id in papers:
                pred_data[paper_id] = {"paper": paper_id, "author": author_id}
        
        papers = [v["paper"] for k, v in pred_data.items()]
        labels = [v["author"] for k, v in pred_data.items()]

        if len(papers) == 0:
            continue

        prec, rec, f1 = model.score(papers, labels, cluster_size)

        prec_count += prec
        rec_count += rec
        f1_count += f1

    print("word2vec precision {:.5f} recall {:.5f} f1 {:.5f}". format(prec_count / len(train_row_data), rec_count / len(train_row_data), f1_count / len(train_row_data)))


    #model.train_global_model(train_row_data)
    #model.save_global("global")

    #model.train_local_model(train_row_data)
