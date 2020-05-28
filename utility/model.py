from keras import backend as K
import numpy as np

def discretization(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))

def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb - test_embs[0])
    score2 = np.linalg.norm(anchor_emb - test_embs[1])
    return [score1, score2]