
import os.path as path
import os
import numpy as np
from sklearn.datasets import fetch_mldata

from dataset._shared import Dataset, index_to_indicator

thisdir = path.dirname(path.realpath(__file__))
data_dir = path.join(thisdir, 'data')

def mnist():
    data = fetch_mldata('MNIST original', data_home=data_dir)
    shuffle = np.random.permutation(data.data.shape[0])

    obs = data.data.shape[0]
    time = data.data.shape[1]

    return Dataset(
        (data.data / 255.0).reshape(obs, 1, time).astype('float32')[shuffle, :],
        data.target.astype('int32')[shuffle],
        np.unique(data.target).size
    )

def mode(items, Tmin=17, Tmax=20):
    maxIndex = 10

    t = np.random.randint(1, maxIndex, size=(items, )).astype('int32')
    X = np.tile(t[:, np.newaxis], (1, Tmax))

    # Substitute random elements
    randsprseq = int(0.4 * Tmax)
    X[
        np.arange(0, items).repeat(randsprseq),
        np.random.randint(0, Tmax, size=(items * randsprseq))
    ] = np.random.randint(1, maxIndex, size=(items * randsprseq))

    # Stop the X sequence
    for i in range(0, items):
        stop = np.random.randint(Tmin, Tmax)
        X[i, stop:] = 0

    return Dataset(index_to_indicator(X, 10), t, maxIndex)
