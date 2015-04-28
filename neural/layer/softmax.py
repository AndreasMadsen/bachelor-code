
import numpy as np
import theano
import theano.tensor as T

from neural.layer._abstract import LayerAbstract

class Softmax(LayerAbstract):
    def __init__(self, size, bias=True, log=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_bias = bias
        self._add_log = log
        self.output_size = size

    def setup(self, batch_size, layer_index, prev_layer):
        self.layer_index = layer_index
        self.input_size = prev_layer.output_size

        self._W_h0_h1 = theano.shared(
            (0.5 * np.random.randn(self.input_size, self.output_size)).astype('float32'),
            name="W_h%d_h%d" % (self.layer_index - 1, self.layer_index),
            borrow=True
        )
        self.weights.append(self._W_h0_h1)

        if (self._use_bias):
            self._W_b0_h1 = theano.shared(
                np.zeros(self.output_size).astype('float32'),
                name="W_b%d_h%d" % (self.layer_index - 1, self.layer_index),
                borrow=True
            )
            self.weights.append(self._W_b0_h1)
        else:
            self._W_b0_h1 = 0

        if (self._add_log): self.outputs_info.append(None)
        self.outputs_info.append(None)

    def scanner(self, b_h0_t, mask=None):
        a_h1_t = T.dot(b_h0_t, self._W_h0_h1) + self._W_b0_h1

        if (self._add_log):
            diff = a_h1_t - T.max(a_h1_t, axis=1, keepdims=True)
            divider = T.sum(T.exp(diff), axis=1, keepdims=True)
            y_log = diff - T.log(divider)
            y = T.exp(diff) / divider

            return [y_log, y]
        else:
            y_h1_t = T.nnet.softmax(a_h1_t)

            # It is not nessarry to do something with the mask,
            # as the input values will stay constant.

            return [y_h1_t]
