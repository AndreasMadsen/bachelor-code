
import os.path as path
import os
import numpy as np

from sklearn.datasets import fetch_mldata

thisdir = path.dirname(path.realpath(__file__))
data_dir = path.join(thisdir, 'data')

class Dataset():
    def __init__(self, data, target, classes):
        self.data = data
        self.target = target
        self.n_classes = classes

def _index_to_indicator(matrix, maxIndex):
    shape = matrix.shape
    tensor = np.zeros((shape[0], maxIndex, shape[1]), dtype='float32')
    (obs, time) = np.mgrid[0:shape[0], 0:shape[1]]
    tensor[obs, matrix, time] = 1
    return tensor

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

    return Dataset(_index_to_indicator(X, 10), t, maxIndex)
