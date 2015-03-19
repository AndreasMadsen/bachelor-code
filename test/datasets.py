
import numpy as np

def quadrant_classify(items, T=5):
    X = np.random.uniform(low=-1, high=1, size=(items, 2, T)).astype('float32')

    t = np.zeros((items, 5), dtype='int32')
    t += np.all([X[:, 0, :] < 0 , X[:, 1, :] >= 0], axis=0) * 1
    t += np.all([X[:, 0, :] < 0 , X[:, 1, :] < 0] , axis=0) * 2
    t += np.all([X[:, 0, :] >= 0, X[:, 1, :] < 0] , axis=0) * 3
    return (X, t)

def quadrant_cumsum_classify(items, T=5):
    X = np.random.uniform(low=-1, high=1, size=(items, 2, T)).astype('float32')
    Xsum = np.cumsum(X, axis=2)

    t = np.zeros((items, 5), dtype='int32')
    t += np.all([Xsum[:, 0, :] < 0 , Xsum[:, 1, :] >= 0], axis=0) * 1
    t += np.all([Xsum[:, 0, :] < 0 , Xsum[:, 1, :] < 0] , axis=0) * 2
    t += np.all([Xsum[:, 0, :] >= 0, Xsum[:, 1, :] < 0] , axis=0) * 3
    return (X, t)
