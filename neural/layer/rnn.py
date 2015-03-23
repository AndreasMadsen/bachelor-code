
import numpy as np
import theano
import theano.tensor as T

class RNN:
    def __init__(self, size):
        self.output_size = size
        self.weights = []
        self.outputs_info = []

    def _input_unit(self, symbol):
        size = [self.input_size, self.output_size]
        index = self.layer_index

        W01 = theano.shared(
            np.random.randn(size[0], size[1]).astype('float32'),
            name="W_%s%d_%s%d" % (symbol, index - 1, symbol, index),
            borrow=True
        )
        self.weights.append(W01)

        W11 = theano.shared(
            np.random.randn(size[1], size[1]).astype('float32'),
            name="W_%s%d_%s%d" % (symbol, index - 1, symbol, index),
            borrow=True
        )
        self.weights.append(W11)

        return lambda b_h0_t, b_h1_tm1: \
            T.dot(b_h0_t, W01) + T.dot(b_h1_tm1, W11)

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size

        self._forward = self._input_unit('h')

        self.outputs_info.append(
            T.zeros((batch_size, self.output_size), dtype='float32'),  # b_h_tm1 (ops x dims)
        )

    def scanner(self, b_h0_t, b_h1_tm1):
        a_h1_t = self._forward(b_h0_t, b_h1_tm1)
        b_h1_t = T.nnet.sigmoid(a_h1_t)

        return [b_h1_t]
