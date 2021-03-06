
import numpy as np

class Dataset():
    def __init__(self, data, target, n_classes=None, max_output_size=None):
        self.data = data
        self.target = target

        self.n_classes = n_classes
        self.max_output_size = max_output_size
        self.data_type = 'list' if isinstance(target, list) else 'numpy'
        self.observations = len(target) if self.data_type == 'list' else data.shape[0]

    def range(self, start, end):
        return Dataset(
            self.data[start:end], self.target[start:end],
            n_classes=self.n_classes,
            max_output_size=self.max_output_size
        )

    def select(self, selector):
        if (self.data_type == 'list'):
            return Dataset(
                [self.data[i] for i in selector],
                [self.target[i] for i in selector],
                n_classes=self.n_classes,
                max_output_size=self.max_output_size
            )
        else:
            return Dataset(
                self.data[selector], self.target[selector],
                n_classes=self.n_classes,
                max_output_size=self.max_output_size
            )

    def list_to_numpy(self, items):
        # NOTE: currently only indexed data is supported
        t_max = np.max([obs.size for obs in items])
        result = np.zeros((self.observations, t_max), dtype=items[0].dtype)

        for i, obs in enumerate(items):
            result[i, 0:obs.size] = obs

        return result

    def astuple(self):
        if (self.data_type == 'list'):
            return (self.list_to_numpy(self.data), self.list_to_numpy(self.target))
        else:
            return (self.data, self.target)

def index_to_indicator(matrix, maxIndex):
    shape = matrix.shape
    tensor = np.zeros((shape[0], maxIndex, shape[1]), dtype='float32')
    (obs, time) = np.mgrid[0:shape[0], 0:shape[1]]
    tensor[obs, matrix, time] = 1
    return tensor
