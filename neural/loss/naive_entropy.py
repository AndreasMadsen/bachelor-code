
import numpy as np
import theano
import theano.tensor as T

class NaiveEntropy:
    def __init__(self, time=True):
        self._time = time
        pass

    def setup(self, batch_size):
        self.batch_size = batch_size

    def loss(self, y, t):
        # If there is a time dimension, reshape the input such that the
        # features and time appears in the same row.
        if (self._time):
            t = t.ravel()
            y = y.transpose(0, 2, 1).reshape((y.shape[2] * y.shape[0], y.shape[1]))
        return T.mean(T.nnet.categorical_crossentropy(y, t))
