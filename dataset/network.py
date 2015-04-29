
import numpy as np

class Dataset():
    def __init__(self, data, target, classes):
        self.data = data
        self.target = target
        self.n_classes = classes

def count(items, T=8, classes=5):
    # Create initial value
    X = np.random.uniform(0, classes, size=(items, 2, 2))
    X[:, 0, 0] = 0
    X[:, 0, 1] = 1
    X[:, 1, 1] = 0

    # Create targe by incrementing
    inc = np.tile(np.arange(0, T), (items, 1))
    t = np.mod(X[:, 1, 0][:, None] + inc, classes)
    t = np.floor(t)

    # add <EOS>
    t = t + 1
    t = np.hstack([t, np.zeros((items, 1))])

    # Normalize X
    X[:, 1, 0] = X[:, 1, 0] / classes

    return Dataset(X.astype('float32'), t.astype('int32'), classes + 1)
