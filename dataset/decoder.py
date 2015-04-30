
import numpy as np

class Dataset():
    def __init__(self, data, target, classes):
        self.data = data
        self.target = target
        self.n_classes = classes

def count(items, T=8, classes=5):
    # Create initial value
    X = np.random.uniform(0, classes, size=(items, 1))

    # Create targe by incrementing
    inc = np.tile(np.arange(0, T), (items, 1))
    t = np.mod(X + inc, classes)
    t = np.floor(t)

    # add <EOS>
    t = t + 1
    t = np.hstack([t, np.zeros((items, 1))])

    return Dataset(
        (X / classes).astype('float32'),
        t.astype('int32'),
        classes + 1
    )

def floor(items, classes=9):
    # Create initial value
    X = np.random.uniform(1, classes + 1, size=(items, 1))

    # Create targe by incrementing
    t = np.zeros((items, 2))
    t[:, 0] = np.floor(X[:, 0])

    return Dataset(
        X.astype('float32'),
        t.astype('int32'),
        classes + 1
    )

def memorize(items, classes=9):
    # Create initial value
    X = np.random.randint(1, classes + 1, size=(items, 1))

    # Create targe by incrementing
    t = np.zeros((items, 10))
    for i, x_i in enumerate(X[:, 0]):
        t[i, 0:x_i] = np.arange(1, x_i + 1)

    return Dataset(
        X.astype('float32'),
        t.astype('int32'),
        classes + 1
    )
