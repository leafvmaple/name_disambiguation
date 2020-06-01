from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

class ClusterModel:
    def __init__(self, dimension=64):
        self.dimension = dimension

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(64), input_shape=(300, 100)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

        self.model.compile(loss="msle", optimizer='rmsprop', metrics=[self.mean_squared_error, "accuracy", "msle", self.mean_log_squared_error])

    def fit(self, author_data, feature_embs):
        self.author_data = author_data
        self.feature_embs = feature_embs

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
            names.append(name)
            num_clusters = len(author)
            sampled = []
            embs = []
            for _, papers in author.items():  # one person
                for paper_id in papers:
                    sampled.append(paper_id)
            sampled_points = [sampled[v] for v in np.random.choice(len(sampled), k, replace=True)]
            for paper_id in sampled_points:
                embs.append(self.feature_embs[paper_id])
            X.append(np.stack(embs))
            y.append(num_clusters)

        return names, np.stack(X), np.stack(y)


    def train(self, k=300, seed=1106, vailidation_data=None):
        np.random.seed(seed)
        clusters = []
        for domain in self.author_data.values():
            for cluster in domain.values():
                clusters.append(cluster)

        _, test_X, test_y = self.gen_test(k, vailidation_data)
        self.model.fit_generator(self.gen_train(clusters, k=300, batch_size=1000), steps_per_epoch=100, epochs=1000, validation_data=(test_X, test_y))

    def predict(self, predict_data, path):
        pred_names, pred_X, pred_y = self.gen_test(predict_data)
        pred_res = self.model.predict(pred_X)

        f = open(path, 'w')
        for i, name in enumerate(pred_names):
            f.write('{}\t{}\t{}\n'.format(name, pred_y[i], pred_res[i][0]))
        f.close()