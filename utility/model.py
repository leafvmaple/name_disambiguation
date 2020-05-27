from keras import backend as K

def get_hidden_output(model, inp):
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    activations = get_activations([inp, 0])
    return activations[0]

def discretization(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    return list(map(lambda x: classes_dict[x], labels))