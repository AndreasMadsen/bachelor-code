
import numpy as np
import math

def single_minibatch(model, train_dataset,
                     minibatch_size=128,
                     **kwargs):
    (data, target) = train_dataset

    observations  = data.shape[0]
    minibatches = math.ceil(observations / minibatch_size)
    for i in range(0, minibatches):
        minibatch_start = i * minibatch_size
        minibatch_end = min((i + 1) * minibatch_size, observations)
        minibatch = (
            data[minibatch_start:minibatch_end],
            target[minibatch_start:minibatch_end]
        )

        model.train(*minibatch, **kwargs)

def minibatch(model, train_dataset, epochs=100, on_epoch=None, **kwargs):
    (data, target) = train_dataset

    for epoch_i in range(0, epochs):
        shuffle = np.random.permutation(data.shape[0])
        shuffle_dataset = (data[shuffle], target[shuffle])
        single_minibatch(model, shuffle_dataset, **kwargs)

        if (on_epoch is not None): on_epoch(model, epoch_i)
