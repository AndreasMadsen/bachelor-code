
import numpy as np

class Dataset():
    def __init__(self, data, target, classes):
        self.data = data
        self.target = target
        self.n_classes = classes

def index_to_indicator(matrix, maxIndex):
    shape = matrix.shape
    tensor = np.zeros((shape[0], maxIndex, shape[1]), dtype='float32')
    (obs, time) = np.mgrid[0:shape[0], 0:shape[1]]
    tensor[obs, matrix, time] = 1
    return tensor
