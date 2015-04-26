
import numpy as np
import theano
import theano.tensor as T

from neural.layer._abstract import LayerAbstract

class RNN(LayerAbstract):
    def __init__(self, size, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = bias
        self.output_size = size

    def _rnn_input_unit(self):
        size = [self.input_size, self.output_size]
        index = self.layer_index

        W01 = theano.shared(
            np.random.randn(size[0], size[1]).astype('float32'),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W01)

        W11 = theano.shared(
            np.random.randn(size[1], size[1]).astype('float32'),
            name="W_h%d_h%d" % (index - 1, index),
            borrow=True
        )
        self.weights.append(W11)

        if (self._use_bias):
            Wb = theano.shared(
                np.zeros(size[1]).astype('float32'),
                name="W_b%d_h%d" % (index - 1, index),
                borrow=True
            )
            self.weights.append(Wb)
        else:
            Wb = 0

        def forward(b_h0_t, b_h1_tm1):
            return T.dot(b_h0_t, W01) + T.dot(b_h1_tm1, W11) + Wb

        return forward

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size

        self._forward = self._rnn_input_unit()

        b_h_tm1 = T.zeros((batch_size, self.output_size), dtype='float32')
        b_h_tm1.name = "b_h%d" % (self.layer_index)

        self.outputs_info.append(b_h_tm1)

    def scanner(self, b_h0_t, b_h1_tm1, mask=None):
        a_h1_t = self._forward(b_h0_t, b_h1_tm1)
        b_h1_t = T.nnet.sigmoid(a_h1_t)

        # If mask value is 1, return the results from previous iteration
        # TODO: todo consider a more efficent way of doing this
        if (mask is not None):
            b_h1_t = b_h1_t * (1 - mask) + b_h1_tm1 * mask

        return [b_h1_t]
