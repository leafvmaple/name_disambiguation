from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from utility.model import clustering, pairwise_precision_recall_f1

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, gen_train_edges, normalize_vectors

tf.compat.v1.disable_eager_execution()

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model

class GraphAutoEncoders:
    def train(self, adj, features, labels):

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        adj_train = gen_train_edges(adj)

        adj = adj_train

        # Some preprocessing
        adj_norm = preprocess_graph(adj)
        num_nodes = adj.shape[0]
        input_feature_dim = features.shape[1]
        features = normalize_vectors(features)

        # Define placeholders
        placeholders = {
            'features': tf.compat.v1.placeholder(tf.float32, shape=(None, input_feature_dim)),
            # 'features': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
        }

        # Create model
        model = None
        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, input_feature_dim)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, input_feature_dim, num_nodes)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # negative edges/pos edges
        print('positive edge weight', pos_weight)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.nnz) * 2)

        # Optimizer
        with tf.compat.v1.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                            validate_indices=False), [-1]),
                                pos_weight=pos_weight,
                                norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=pos_weight,
                                norm=norm)

        # Initialize session
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        def get_embs():
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)  # z_mean is better
            return emb

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                            feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
            #     "train_acc=", "{:.5f}".format(avg_accuracy),
            #     "time=", "{:.5f}".format(time.time() - t))

        emb = get_embs()
        n_clusters = len(set(labels))
        emb_norm = normalize_vectors(emb)
        clusters_pred = clustering(emb_norm, num_clusters=n_clusters)
        prec, rec, f1 =  pairwise_precision_recall_f1(clusters_pred, labels)
        print('pairwise precision', '{:.5f}'.format(prec),
            'recall', '{:.5f}'.format(rec),
            'f1', '{:.5f}'.format(f1))
        return [prec, rec, f1], num_nodes, n_clusters
