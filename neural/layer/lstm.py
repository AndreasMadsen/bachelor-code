
import numpy as np
import theano
import theano.tensor as T

from neural.layer._abstract import LayerAbstract

class LSTM(LayerAbstract):
    def __init__(self, size, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = bias
        self._add_log = False
        self.output_size = size

        self._splits = [
            i * self.output_size for i in range(0, 5)
        ]

    def _lstm_input_units(self):
        size = [self.input_size, self.output_size]
        index = self.layer_index

        W01 = theano.shared(
            self._create_weights(size[0], size[1] * 4),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W01)

        W11 = theano.shared(
            self._create_weights(size[1], size[1] * 4),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W11)

        if (self._use_bias):
            Wb = theano.shared(
                np.zeros(size[1] * 4).astype('float32'),
                name="W_b%d_h%d" % (index - 1, index),
                borrow=True
            )
            self.weights.append(Wb)
        else:
            Wb = 0

        def forward(b_h0_t, b_h1_tm1):
            if (self.indexed_input):
                a_1_t = W01[b_h0_t, :] + T.dot(b_h1_tm1, W11) + Wb
            else:
                a_1_t = T.dot(b_h0_t, W01) + T.dot(b_h1_tm1, W11) + Wb

            b_1_t = T.nnet.sigmoid(a_1_t)
            return b_1_t

        return forward

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size
        self.indexed_input = prev_layer.indexed

        self._forward = self._lstm_input_units()

        s_c_tm1 = T.zeros((batch_size, self.output_size), dtype='float32')
        s_c_tm1.name = "s_c%d" % (self.layer_index)

        b_h_tm1 = T.zeros((batch_size, self.output_size), dtype='float32')
        b_h_tm1.name = "b_h%d" % (self.layer_index)

        self.outputs_info.append(s_c_tm1)
        self.outputs_info.append(b_h_tm1)

    def scanner(self, b_h0_t, s_c1_tm1, b_h1_tm1, mask=None):
        b_1_t = self._forward(b_h0_t, b_h1_tm1)

        # Select submatrices for each gate / input
        b_i1_t = b_1_t[:, self._splits[0]:self._splits[1]]  # input
        b_ρ1_t = b_1_t[:, self._splits[1]:self._splits[2]]  # input gate
        b_ɸ1_t = b_1_t[:, self._splits[2]:self._splits[3]]  # forget gate
        b_ω1_t = b_1_t[:, self._splits[3]:self._splits[4]]  # output gate

        # Calculate state and block output
        s_c1_t = b_ɸ1_t * s_c1_tm1 + b_ρ1_t * b_i1_t
        b_h1_t = b_ω1_t * s_c1_t

        # If mask value is 1, return the results from previous iteration
        # TODO: consider a more efficent way of doing this
        if (mask is not None):
            s_c1_t = s_c1_t * (1 - mask) + s_c1_tm1 * mask
            b_h1_t = b_h1_t * (1 - mask) + b_h1_tm1 * mask

        return [s_c1_t, b_h1_t]
