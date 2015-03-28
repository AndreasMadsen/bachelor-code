
import numpy as np
import theano
import theano.tensor as T

class LSTM:
    def __init__(self, size):
        self.output_size = size
        self.weights = []
        self.outputs_info = []

        self._splits = [
            i * self.output_size for i in range(0, 5)
        ]

    def _lstm_input_units(self):
        size = [self.input_size, self.output_size]
        index = self.layer_index

        W01 = theano.shared(
            np.random.randn(size[0], size[1] * 4).astype('float32'),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W01)

        W11 = theano.shared(
            np.random.randn(size[1], size[1] * 4).astype('float32'),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W11)

        def forward(b_h0_t, b_h1_tm1):
            a_1_t = T.dot(b_h0_t, W01) + T.dot(b_h1_tm1, W11)
            b_1_t = T.nnet.sigmoid(a_1_t)
            return b_1_t

        return forward

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size

        self._forward = self._lstm_input_units()

        self.outputs_info.append(
            T.zeros((batch_size, self.output_size), dtype='float32'),  # s_c_tm1 (ops x dims)
        )
        self.outputs_info.append(
            T.zeros((batch_size, self.output_size), dtype='float32'),  # b_h_tm1 (ops x dims)
        )

    def scanner(self, b_h0_t, s_c1_tm1, b_h1_tm1, mask=None):
        b_1_t = self._forward(b_h0_t, b_h1_tm1)

        # Select submatrices for each gate / input
        b_i1_t = b_1_t[:, self._splits[0]:self._splits[1]]  # input
        b_ρ1_t = b_1_t[:, self._splits[1]:self._splits[2]]  # input gate
        b_ɸ1_t = b_1_t[:, self._splits[2]:self._splits[3]]  # forget gate
        b_ω1_t = b_1_t[:, self._splits[3]:self._splits[4]]  # output gate

        # Calculate state and block output
        s_c1_t = b_ɸ1_t * s_c1_tm1 + b_ρ1_t * b_i1_t
        b_h1_t = b_ω1_t * T.nnet.sigmoid(s_c1_t)

        # If mask value is 1, return the results from previous iteration
        # TODO: todo consider a more efficent way of doing this
        if (mask is not None):
            s_c1_t = s_c1_t * (1 - mask) + s_c1_tm1 * mask
            b_h1_t = b_h1_t * (1 - mask) + b_h1_tm1 * mask

        return [s_c1_t, b_h1_t]
