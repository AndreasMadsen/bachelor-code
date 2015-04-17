
import numpy as np
import theano
import theano.tensor as T

class NaiveEntropy:
    def __init__(self, time=True, log=False):
        self._time = time
        self._is_log = log
        pass

    def setup(self, batch_size):
        self.batch_size = batch_size

    def loss(self, y, t):
        # If there is a time dimension, reshape the input such that the
        # observations and time both appears as rows.
        if (self._time):
            t = t.ravel()
            y = y.transpose(0, 2, 1).reshape((y.shape[2] * y.shape[0], y.shape[1]))

        if (self._is_log):
            # y is acutally log(y), likely calculated by Softmax(log=True)
            return - T.mean(y[T.arange(0, y.shape[0]), t])
        else:
            return T.mean(T.nnet.categorical_crossentropy(y, t))
