
import numpy as np
import theano
import theano.tensor as T

class Softmax:
    def __init__(self, size):
        self.output_size = size
        self.weights = []
        self.outputs_info = []

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size

        self._W_h0_h1 = theano.shared(
            np.random.randn(self.input_size, self.output_size).astype('float32'),
            name="W_h%d_h%d" % (self.layer_index - 1, self.layer_index),
            borrow=True
        )

        self.weights.append(self._W_h0_h1)
        self.outputs_info.append(None)

    def scanner(self, b_h0_t):
        a_h1_t = T.dot(b_h0_t, self._W_h0_h1)
        y_h1_t = T.nnet.softmax(a_h1_t)

        return [y_h1_t]
