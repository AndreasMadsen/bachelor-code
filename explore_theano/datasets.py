
import numpy as np

def generate_quadrant(items):
    X = np.random.uniform(low=-1, high=1, size=(items, 2)).astype('float32')

    t = np.zeros(items, dtype='int32')
    t += np.all([X[:, 0] < 0, X[:, 1] >= 0], axis=0) * 1
    t += np.all([X[:, 0] < 0, X[:, 1] < 0], axis=0) * 2
    t += np.all([X[:, 0] >= 0, X[:, 1] < 0], axis=0) * 3
    return (X, t)
