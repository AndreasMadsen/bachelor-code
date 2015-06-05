
import os.path as path
import os
import numpy as np

from dataset._shared import Dataset, index_to_indicator

thisdir = path.dirname(path.realpath(__file__))
data_dir = path.join(thisdir, 'data')

def mode(items, Tmin=17, Tmax=20, indexed=False):
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

    if (indexed):
        return Dataset(X, t, maxIndex)
    else:
        return Dataset(index_to_indicator(X, 10), t, maxIndex)

def copy(items, T=9, classes=10):
    X = np.zeros((items, T), dtype='int32')
    X[:, 0:(T - 1)] = np.random.randint(1, classes, size=(items, T - 1))

    t = (np.copy(X[:, 0:(T - 1)]) / (classes - 1)).astype('float32')

    return Dataset(
        index_to_indicator(X, classes).astype('float32'), t, classes
    )
