
import numpy as np
import math

def subselect(source, select):
    if (isinstance(source, list)):
        return [source[i] for i in select]
    else:
        return source[select]

def single_minibatch(model, train_dataset, shuffle,
                     minibatch_size=128,
                     **kwargs):

    (data, target) = (train_dataset.data, train_dataset.target)

    observations = train_dataset.observations
    minibatches = math.ceil(observations / minibatch_size)

    for i in range(0, minibatches):
        minibatch_start = i * minibatch_size
        minibatch_end = min((i + 1) * minibatch_size, observations)
        minibatch_select = shuffle[minibatch_start:minibatch_end]

        minibatch = train_dataset.select(minibatch_select).astuple()
        model.train(*minibatch, **kwargs)

def minibatch(model, train_dataset, epochs=100, on_epoch=None, **kwargs):
    for epoch_i in range(0, epochs):
        shuffle = np.random.permutation(train_dataset.observations)
        single_minibatch(model, train_dataset, shuffle, **kwargs)

        if (on_epoch is not None): on_epoch(model, epoch_i)
