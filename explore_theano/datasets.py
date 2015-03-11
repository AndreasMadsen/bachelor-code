
import numpy as np


def generate_quadrant(items):
    X = np.random.uniform(low=-1, high=1, size=(items, 2)).astype('float32')

    t = np.zeros(items, dtype='int32')
    t += np.all([X[:, 0] < 0, X[:, 1] >= 0], axis=0) * 1
    t += np.all([X[:, 0] < 0, X[:, 1] < 0], axis=0) * 2
    t += np.all([X[:, 0] >= 0, X[:, 1] < 0], axis=0) * 3
    return (X, t)


def generate_accumulated(items, T=5):
    X = np.random.randint(0, 11, size=(items, 2, T))

    t = np.sum(X, axis=1)
    t = np.cumsum(t, axis=1)
    t = np.mod(t, 10)
    return (X.astype('int32'), t.astype('int32'))
