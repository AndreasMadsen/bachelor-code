
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

def subset_vocal_sequence(items, Tmax=7):
    """
    This will generate a random input and target sequence of letters.

    All letters are used for the input sequence, the taget sequence is
    then the subset of the input where only vocals appear.
    """
    # max_letter = 26
    # vocals = [1, 5, 9, 15, 21, 25]
    max_letter = 10
    vocals = [1, 2, 3, 4, 10]

    # The input sequence length is random between 2 and Tmax
    sequence_length = np.random.randint(2, Tmax, size=items)

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

    def index_to_indicator(matrix, maxIndex):
        shape = matrix.shape
        tensor = np.zeros((shape[0], maxIndex, shape[1]))
        (obs, time) = np.mgrid[0:shape[0], 0:shape[1]]
        tensor[obs, matrix, time] = 1
        return tensor

    return (
        index_to_indicator(letters, max_letter + 1),
        index_to_indicator(vocals, max_letter + 1)
    )
