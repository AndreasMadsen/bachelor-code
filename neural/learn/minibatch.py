
import numpy as np
import math
import datetime

def subselect(source, select):
    if (isinstance(source, list)):
        return [source[i] for i in select]
    else:
        return source[select]

def single_minibatch(model, train_dataset, shuffle, epoch_i,
                     minibatch_size=128, on_mini_batch=None,
                     **kwargs):

    (data, target) = (train_dataset.data, train_dataset.target)

    observations = train_dataset.observations
    minibatches = math.ceil(observations / minibatch_size)

    for i in range(0, minibatches):
        minibatch_start = i * minibatch_size
        minibatch_end = min((i + 1) * minibatch_size, observations)
        minibatch_select = shuffle[minibatch_start:minibatch_end]

        minibatch = train_dataset.select(minibatch_select).astuple()
        loss = model.train(*minibatch, **kwargs)
        if (on_mini_batch):
            on_mini_batch(
                model, i,
                (epoch_i - 1) + (minibatch_start + minibatch_end) / (2 * observations),
                loss
            )

def minibatch(model, train_dataset, on_epoch=None, max_time=None, epochs=100, **kwargs):
    if (on_epoch is not None): on_epoch(model, 0)
    epoch_i = 1
    last = start = datetime.datetime.now()

    while (True):
        shuffle = np.random.permutation(train_dataset.observations)
        single_minibatch(model, train_dataset, shuffle, epoch_i, **kwargs)
        if (on_epoch is not None): on_epoch(model, epoch_i)
        epoch_i += 1

        if (max_time is None):
            if (epoch_i > epochs): break
        else:
            now = datetime.datetime.now()
            if ((now - start + (now - last)) >= max_time): break
            last = now
