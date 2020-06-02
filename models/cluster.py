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
        self.feature_embs = feature_embs
        self.train(author_data)

    def mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def mean_log_squared_error(self, y_true, y_pred):
        first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
        return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

    def sampler(self, clusters, batch_size=10, min_range=1, max_range=300):
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
    
            sampled_paper_id = [sampled[v] for v in np.random.choice(len(sampled), self.k, replace=True)]
            for paper_id in sampled_paper_id:
                embs.append(self.feature_embs[paper_id])

            sampled_embs.append(np.stack(embs))
            sampled_sizes.append(num_clusters)
        
        return np.stack(sampled_embs), np.stack(sampled_sizes)

    def generate_train(self, clusters, batch_size=1000):
        while True:
            yield self.sampler(clusters, batch_size)

    def generate_valid(self, author_data):
        X = []
        y = []
        for _, author in author_data.items():
            num_clusters = len(author)
            sampled = []
            embs = []
            for _, papers in author.items():  # one person
                for paper_id in papers:
                    sampled.append(paper_id)

            sampled = [v for v in sampled if v in self.feature_embs]
            if len(sampled) == 0:
                continue

            sampled_points = [sampled[v] for v in np.random.choice(len(sampled), self.k, replace=True)]
            for paper_id in sampled_points:
                embs.append(self.feature_embs[paper_id])

            X.append(np.stack(embs))
            y.append(num_clusters)

        return np.stack(X), np.stack(y)

    def generate_predict(self, author_data, feature_embs):
        X = []
        for _, papers in author_data.items():
            sampled = []
            embs = []
            for paper_id in papers:
                sampled.append(paper_id)

            sampled = [v for v in sampled if v in self.feature_embs]
            if len(sampled) == 0:
                continue

            sampled_points = [sampled[v] for v in np.random.choice(len(sampled), self.k, replace=True)]
            for paper_id in sampled_points:
                embs.append(self.feature_embs[paper_id])

            X.append(np.stack(embs))

        return np.stack(X)

    def train(self, author_data, seed=1106):
        np.random.seed(seed)
        clusters = []
        for domain in author_data.values():
            for cluster in domain.values():
                clusters.append(cluster)

        valid_X, valid_y = self.generate_valid(author_data)
        self.model.fit_generator(self.generate_train(clusters, batch_size=1000), steps_per_epoch=100, epochs=1, validation_data=(valid_X, valid_y))

    def predict(self, author_data, feature_embs):
        pred_X = self.generate_predict(author_data, feature_embs)
        return self.model.predict(pred_X)
