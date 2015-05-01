
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

def vocal_subset(items, Tmin=17, Tmax=20):
    """
    This will generate a random input and target sequence of letters.

    All letters are used for the input sequence, the taget sequence is
    then the subset of the input where only vocals appear.
    """
    max_letter = 10  # 26 for full alphabet
    vocals = [1, 5, 9, 15, 21, 25]

    # The input sequence length is random between 2 and Tmax
    sequence_length = np.ones(items) * Tmin
    if (Tmin < Tmax):
        sequence_length = np.random.randint(Tmin, Tmax, size=items)

    # Generate a random sequence of letters
    letters = np.random.randint(1, max_letter + 1, size=(items, Tmax)).astype('int32')
    for index, length in enumerate(sequence_length):
        letters[index, length:] = 0

    # Generate a subset of the letter sequences containing only vocals
    vocal_mask = sum([letters == vocal for vocal in vocals])
    vocals = np.zeros((items, Tmax), dtype='int32')
    for index in range(0, items):
        vocal_scan = letters[index, vocal_mask[index, :].nonzero()].ravel()
        vocals[index, 0:vocal_scan.size] = vocal_scan

    return Dataset(
        _index_to_indicator(letters, max_letter + 1),
        vocals,
        max_letter + 1
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
