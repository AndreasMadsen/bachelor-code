
import os.path as path
import os
import numpy as np

from sklearn.datasets import fetch_mldata

thisdir = path.dirname(path.realpath(__file__))
data_dir = path.join(thisdir, 'data')

class Dataset():
    def __init__(self, data, target, classes):
        self.data = data
        self.target = target
        self.n_classes = classes

def mnist():
    data = fetch_mldata('MNIST original', data_home=data_dir)
    shuffle = np.random.permutation(data.data.shape[0])

    obs = data.data.shape[0]
    time = data.data.shape[1]

    return Dataset(
        (data.data / 255.0).reshape(obs, 1, time).astype('float32')[shuffle, :],
        data.target.astype('int32')[shuffle],
        np.unique(data.target).size
    )
