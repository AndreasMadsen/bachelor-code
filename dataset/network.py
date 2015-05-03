
import numpy as np

from dataset._shared import Dataset, index_to_indicator

def copy(items, T=9, classes=10):
    X = np.zeros((items, T), dtype='int32')
    X[:, 0:(T - 1)] = np.random.randint(1, classes, size=(items, T - 1))

    t = np.zeros((items, T), dtype='int32')
    t[:, 0:(T - 1)] = X[:, 0:(T - 1)]

    return Dataset(
        index_to_indicator(X, classes).astype('float32'), t, classes
    )

def quadrant(items, T=5):
    X = np.random.uniform(low=-1, high=1, size=(items, 2, T)).astype('float32')

    t = np.zeros((items, 5), dtype='int32')
    t += np.all([X[:, 0, :] < 0 , X[:, 1, :] >= 0], axis=0) * 1
    t += np.all([X[:, 0, :] < 0 , X[:, 1, :] < 0] , axis=0) * 2
    t += np.all([X[:, 0, :] >= 0, X[:, 1, :] < 0] , axis=0) * 3
    return Dataset(X, t, 4)

def quadrant_cumsum(items, T=5):
    X = np.random.uniform(low=-1, high=1, size=(items, 2, T)).astype('float32')
    Xsum = np.cumsum(X, axis=2)

    t = np.zeros((items, T), dtype='int32')
    t += np.all([Xsum[:, 0, :] < 0 , Xsum[:, 1, :] >= 0], axis=0) * 1
    t += np.all([Xsum[:, 0, :] < 0 , Xsum[:, 1, :] < 0] , axis=0) * 2
    t += np.all([Xsum[:, 0, :] >= 0, Xsum[:, 1, :] < 0] , axis=0) * 3
    return Dataset(X, t, 4)
