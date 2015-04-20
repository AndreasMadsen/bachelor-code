
import itertools

import numpy as np
import theano
import theano.tensor as T

from neural.layer.lstm import LSTM
from neural.layer.rnn import RNN

from neural.network._base import BaseAbstraction
from neural.network._optimizer import OptimizerAbstraction


class SutskeverNetwork(OptimizerAbstraction):
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, max_output_size=100, **kwargs):
        super().__init__(**kwargs)

        self._inputs = [T.tensor3('x'), T.ivector('eosi')]
        self._target = T.imatrix('t')
        self._max_output_size = max_output_size

        # encoder -> decoder
        self._encoder = Encoder([self._inputs[0]])
        self._decoder = Decoder(self._inputs)

    def test_value(self, x, t):
        self._inputs[0].tag.test_value = x
        self._inputs[1].tag.test_value = np.argmin(t, axis=1)
        self._target.tag.test_value = t

    def set_encoder_input(self, layer):
        self._encoder.set_input(layer)

    def push_encoder_layer(self, layer):
        self._encoder.push_layer(layer)

    def set_decoder_input(self, layer):
        self._decoder.set_input(layer)

    def push_decoder_layer(self, layer):
        self._decoder.push_layer(layer)
        self._output_layer = layer

    def weight_list(self):
        """
        Create a list containing all the network weights
        """
        return self._encoder.weight_list() + self._decoder.weight_list()

    def forward_pass(self, x, eosi):
        (s_enc, b_enc) = self._encoder.forward_pass(x)
        (log_y, y) = self._decoder.forward_pass(s_enc, b_enc, eosi)

        return (log_y, y)

    def _preloss(self, log_y, y, t):
        return (log_y, t[:, :y.shape[2]])

    def compile(self):
        # Sutskever is numerically unstable unless stable prelog is used
        assert(self._output_layer._add_log)

        # The input decoder much match its softmax output
        assert(self._decoder._layers[+0].output_size == self._decoder._layers[-1].output_size)
        # The hidden encoder output much match the hidden decoder intialization
        assert(self._encoder._layers[-1].output_size == self._decoder._layers[+1].output_size)

        super().compile()

    def train(self, x, t):
        eosi = np.argmin(t, axis=1).astype('int32')
        return super().train(x, eosi, t)

    def test(self, x, t):
        eosi = np.argmin(t, axis=1).astype('int32')
        return super().test(x, eosi, t)

    def predict(self, x):
        eosi = np.ones(x.shape[0], dtype='int32') * (self._max_output_size - 1)
        return super().predict(x, eosi)

class Encoder(BaseAbstraction):
    """
    The encoder is like a normal forward scanner but doesn't have an output
    layer applied to it. It also only outputs the last time iteration. This
    is how the Sutskever network is implemented. Better aproches may exists.

    This implementation also uses a vector, indicate when the sequences starts
    relative to the other sequences in the minibatch.
    """
    def __init__(self, inputs, **kwargs):
        super().__init__(**kwargs)
        # self._input is only used by the layers, to infer the batch size.
        self._inputs = inputs

    def _forward_scanner(self, x_t, *args):
        """
        Defines the forward equations for each time step.
        """
        args = list(args)

        # Intialize loop
        all_outputs = []
        curr = 0
        prev_output = x_t

        # Mask the inputs
        mask = T.eq(x_t[:, 0], 1).dimshuffle((0, 'x'))  # is <EOF>

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = layer.infer_taps()
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps], mask=mask)

            curr += taps
            all_outputs += layer_outputs
            # the last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        return all_outputs

    def forward_pass(self, x):
        """
        Setup equations for the forward pass
        """
        # because scan assmes the iterable is the first tensor dimension, x is
        # transposed into (time, obs, dims).
        # When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
        # this is then transposed back to its original format
        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=[
                x.transpose(2, 0, 1)  # iterate (time), row (observations), col (dims)
            ],
            outputs_info=self._outputs_info_list(),
            name='sutskever_encoder'
        )
        b_enc = self._last_output(outputs)

        # the last output is assumed to be the network output, take the
        # last time iteration. Return value shape is: row (observations), col (dims)
        if (isinstance(self._layers[-1], LSTM)):
            s_enc = outputs[-2]
            return (s_enc[-1, :, :], b_enc[-1, :, :])
        else:
            return (None, b_enc[-1, :, :])

class Decoder(BaseAbstraction):
    """
    The decoder takes the encoder output for the last time iteration and
    passes it intro a forward iteration. The next output iteration is then
    obtained by letting the hidden output of $t$ go into $t + 1$. The <EOF>
    tag signals when to stop.
    """
    def __init__(self, inputs, **kwargs):
        super().__init__(**kwargs)
        # self._input is only used by the layers, to infer the batch size.
        self._inputs = inputs

    def _forward_scanner(self, t, *args):
        """
        Defines the forward equations for each time step.
        """
        # The last argument is the network output (after softmax)
        args = list(args)
        eosi = args.pop()
        y = args.pop()

        # Intialize layer loop
        all_outputs = []
        curr = 0
        prev_output = y
        mask = (t > eosi).dimshuffle((0, 'x'))

        # Loop though each layer and apply send the current layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = layer.infer_taps()
            # It can be assumed that the last value in `layer_outputs` is the
            # actual layer output. The tuple may contain other values, such
            # as the current cell state. They won't be send to the next layer.
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps], mask=mask)
            curr += taps

            # Concatenate all outputs
            all_outputs += layer_outputs
            # The last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        # Stop when all sequences are masked
        return all_outputs

    def _outputs_info_list(self, s_enc, b_enc):
        outputs_info = super()._outputs_info_list()

        # 1) Replace the initial b_{t_0} with b_enc for the first layer
        if (isinstance(self._layers[1], LSTM)):
            outputs_info[1] = b_enc
            if (s_enc is not None):
                outputs_info[0] = s_enc

        elif (isinstance(self._layers[1], RNN)):
            outputs_info[0] = b_enc
            if (s_enc is not None):
                raise TypeError('s_enc could not be transferred to RNN layer')

        else:
            raise NotImplemented

        # 2) Replace the initial y_{t_0} with 0, and add taps = -1
        y = T.zeros((b_enc.shape[0], self._layers[-1].output_size))
        y.name = 'y'
        outputs_info[-1] = y

        # outputs_info will look like [s_enc, b_enc, ..., y]
        return outputs_info

    def forward_pass(self, s_enc, b_enc, eosi):
        """
        Setup equations for the forward pass
        """
        # To create an <EOS> index vector (eosi) the time index is needed.
        # Use `arange` to generate a vector with all the time indexes. The
        # `scan` may finish before because the `until` condition becomes true.
        time_seq = T.arange(0, T.max(eosi) + 1)
        time_seq.name = 'time'

        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=[time_seq],
            outputs_info=self._outputs_info_list(s_enc, b_enc),
            non_sequences=[eosi],
            name='sutskever_decoder'
        )

        # The scan output have the shape (time, dims, observations)
        log_y = outputs[-2].transpose(1, 2, 0)
        y = outputs[-1].transpose(1, 2, 0)

        return (log_y, y)
