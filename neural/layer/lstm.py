
import numpy as np
import theano
import theano.tensor as T

class LSTM:
    def __init__(self, size):
        self.output_size = size
        self.weights = []
        self.outputs_info = []

    def _lstm_input_unit(self, symbol):
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

        self._forward_h = self._lstm_input_unit('h')
        self._forward_ρ = self._lstm_input_unit('ρ')
        self._forward_ɸ = self._lstm_input_unit('ɸ')
        self._forward_ω = self._lstm_input_unit('ω')

        self.outputs_info.append(
            T.zeros((batch_size, self.output_size), dtype='float32'),  # s_c_tm1 (ops x dims)
        )
        self.outputs_info.append(
            T.zeros((batch_size, self.output_size), dtype='float32'),  # b_h_tm1 (ops x dims)
        )

    def scanner(self, b_h0_t, s_c1_tm1, b_h1_tm1):
        a_h1_t = self._forward_h(b_h0_t, b_h1_tm1)

        b_ρ1_t = T.nnet.sigmoid(self._forward_ρ(b_h0_t, b_h1_tm1))
        b_ɸ1_t = T.nnet.sigmoid(self._forward_ɸ(b_h0_t, b_h1_tm1))
        b_ω1_t = T.nnet.sigmoid(self._forward_ω(b_h0_t, b_h1_tm1))

        s_c1_t = b_ɸ1_t * s_c1_tm1 + b_ρ1_t * T.nnet.sigmoid(a_h1_t)
        b_h1_t = b_ω1_t * T.nnet.sigmoid(s_c1_t)

        return [s_c1_t, b_h1_t]