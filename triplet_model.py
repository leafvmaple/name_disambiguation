import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam

from utility.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy

class GlobalTripletModel:
    def __init__(self):
        self.model = None

    def create(self, size):
        emb_anchor = Input(shape=(size, ), name='anchor_input')
        emb_pos = Input(shape=(size, ), name='pos_input')
        emb_neg = Input(shape=(size, ), name='neg_input')

        # shared layers
        layer1 = Dense(128, activation='relu', name='first_emb_layer')
        layer2 = Dense(64, activation='relu', name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        self.model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        self.model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])

        # inter_layer = Model(inputs=self.model.get_input_at(0), outputs=self.model.get_layer('norm_layer').get_output_at(0))

    def train(self, data):
        triplets_count = data["anchor_input"].shape[0]
        self.model.fit(data, np.ones((triplets_count, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)
