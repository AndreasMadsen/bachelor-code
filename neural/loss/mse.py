
import numpy as np
import theano
import theano.tensor as T

class MeanSquaredError:
    def __init__(self, time=True, log=False):
        self._time = time
        self._expect_log = log

    def setup(self, batch_size):
        self.batch_size = batch_size

    def loss(self, y, t):
        # If there is a time dimension, reshape the input such that the
        # observations and time both appears as rows.
        if (self._time):
            t = t.ravel()
            y = y.transpose(0, 2, 1).reshape((y.shape[2] * y.shape[0], y.shape[1]))

        return T.mean((y - t)**2)
