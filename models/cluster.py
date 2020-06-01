from os.path import join
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

class ClusterModel:
    def __init__(self, dimension=100, k=300):
        self.dimension = dimension
        self.k = k

        self.model = Sequential([
            Bidirectional(LSTM(64), input_shape=(k, dimension)),
            Dropout(0.5),
            Dense(1)
        ])

        self.model.compile(loss="msle", optimizer='rmsprop', metrics=[self.mean_squared_error, "accuracy", "msle", self.mean_log_squared_error])

    def fit(self, author_data, feature_embs):
        self.author_data = author_data
        self.feature_embs = feature_embs
        self.train(author_data, self.k)

    def mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def mean_log_squared_error(self, y_true, y_pred):
        first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
        return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

    def sampler(self, clusters, k=300, batch_size=10, min_range=1, max_range=300):
        sampled_sizes = []
        sampled_embs = []

        for _ in range(batch_size):
            num_clusters = np.random.randint(min_range, max_range)
            cluster_idxs = np.random.choice(len(clusters), num_clusters, replace=False)

            sampled = []
            embs = []

            for idx in cluster_idxs:
                sampled.extend(clusters[idx])

            sampled = [v for v in sampled if v in self.feature_embs]
            if len(sampled) == 0:
                continue
    
            sampled_paper_id = [sampled[v] for v in np.random.choice(len(sampled), k, replace=True)]

            for paper_id in sampled_paper_id:
                embs.append(self.feature_embs[paper_id])

            sampled_embs.append(np.stack(embs))
            sampled_sizes.append(num_clusters)
        
        return np.stack(sampled_embs), np.stack(sampled_sizes)

    def gen_train(self, clusters, k=300, batch_size=1000):
        while True:
            yield self.sampler(clusters, k, batch_size)

    def gen_test(self, test_data, k=300):
        names = []
        X = []
        y = []
        for name, author in test_data.items():
            num_clusters = len(author)
            sampled = []
            embs = []
            for _, papers in author.items():  # one person
                for paper_id in papers:
                    sampled.append(paper_id)

            sampled = [v for v in sampled if v in self.feature_embs]
            if len(sampled) == 0:
                continue

            sampled_points = [sampled[v] for v in np.random.choice(len(sampled), k, replace=True)]
            for paper_id in sampled_points:
                while paper_id not in self.feature_embs:
                    paper_id = sampled[np.random.randint(len(sampled))]
                embs.append(self.feature_embs[paper_id])

            names.append(name)
            X.append(np.stack(embs))
            y.append(num_clusters)

        
        return names, np.stack(X), np.stack(y)


    def train(self, vailidation_data, k=300, seed=1106):
        np.random.seed(seed)
        clusters = []
        for domain in self.author_data.values():
            for cluster in domain.values():
                clusters.append(cluster)

        _, test_X, test_y = self.gen_test(vailidation_data, k)
        self.model.fit_generator(self.gen_train(clusters, k, batch_size=1000), steps_per_epoch=100, epochs=10, validation_data=(test_X, test_y))

    def predict(self, predict_data, path=None):
        pred_names, pred_X, pred_y = self.gen_test(predict_data)
        return self.model.predict(pred_X), pred_names, pred_X, pred_y
