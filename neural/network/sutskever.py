
import itertools

import numpy as np
import theano
import theano.tensor as T

from neural.network._base import BaseAbstraction
from neural.network._optimizer import OptimizerAbstraction

class SutskeverNetwork(OptimizerAbstraction):
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, max_output_size=100, **kwargs):
        OptimizerAbstraction.__init__(self, **kwargs)

        self._input = T.tensor3('x')
        self._target = T.imatrix('t')

        # encoder -> decoder
        self._encoder = Encoder(self._input)
        self._decoder = Decoder(self._input, maxlength=max_output_size)

    def test_value(self, x, t):
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def set_input(self, layer):
        self._encoder.set_input(layer)
        self._decoder.set_input(layer)

    def push_encoder_layer(self, layer):
        self._encoder.push_layer(layer)

    def push_decoder_layer(self, layer):
        self._decoder.push_layer(layer)

    def weight_list(self):
        """
        Create a list containing all the network weights
        """
        return self._encoder.weight_list() + self._decoder.weight_list()

    def forward_pass(self, x):
        b_enc = self._encoder.forward_pass(x)
        y = self._decoder.forward_pass(b_enc)

        return y

    def _loss(self, y, t):
        # TODO: improve this by add <EOS> end padding
        t_max = T.max([y.shape[2], t.shape[1]])

        # Pad y vector with an even distribution
        y_pad = T.ones((y.shape[0], y.shape[1], t_max), dtype='float32') * (1 / y.shape[1])
        y_pad = T.set_subtensor(
            y_pad[:, :, 0:y.shape[2]], y
        )

        # Pad t vector with <EOS>
        t_pad = T.zeros((t.shape[0], t_max), dtype='int32')
        t_pad = T.set_subtensor(
            t_pad[:, 0:t.shape[1]], t
        )

        return super()._loss(y_pad, t_pad)

    def compile(self):
        # The input decoder much match its softmax output
        assert(self._decoder._layers[+0].output_size == self._decoder._layers[-1].output_size)
        # The hidden encoder output much match the hidden decoder intialization
        assert(self._encoder._layers[-1].output_size == self._decoder._layers[+1].output_size)

        super().compile()

class Encoder(BaseAbstraction):
    """
    The encoder is like a normal forward scanner but doesn't have an output
    layer applied to it. It also only outputs the last time iteration. This
    is how the Sutskever network is implemented. Better aproches may exists.

    This implementation also uses a vector, indicate when the sequences starts
    relative to the other sequences in the minibatch.
    """
    def __init__(self, x_input, **kwargs):
        super().__init__(**kwargs)
        self._input = x_input

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
            taps = self._infer_taps(layer)
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
            outputs_info=self._outputs_info_list()
        )
        b_enc = self._last_output(outputs)

        # the last output is assumed to be the network output, take the
        # last time iteration. Return value shape is: row (observations), col (dims)
        return b_enc[-1, :, :]

class Decoder(BaseAbstraction):
    """
    The decoder takes the encoder output for the last time iteration and
    passes it intro a forward iteration. The next output iteration is then
    obtained by letting the hidden output of $t$ go into $t + 1$. The <EOF>
    tag signals when to stop.
    """
    def __init__(self, b_input, maxlength=100, **kwargs):
        super().__init__(**kwargs)
        self._input = b_input
        self._maxlength = maxlength

    def _forward_scanner(self, mask, *args):
        """
        Defines the forward equations for each time step.
        """
        args = list(args)
        y = args.pop()

        # Intialize loop
        all_outputs = []
        curr = 0
        prev_output = y
        mask = mask.dimshuffle((0, 'x'))

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = self._infer_taps(layer)
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps], mask=mask)

            curr += taps
            all_outputs += layer_outputs
            # the last output is assumed to be the layer output
            prev_output = layer_outputs[-1]

        # Update the mask, if <EOS> was returned by the last iteration
        mask = T.eq(T.argmax(prev_output, axis=1), 0)
        all_outputs = [mask] + all_outputs

        # Stop when all sequences are masked
        return (all_outputs, theano.scan_module.until(T.all(mask)))

    def _outputs_info_list(self, b_enc):
        outputs_info = super()._outputs_info_list()

        # 1) Replace the initial b_{t_0} with b_enc for the first layer
        outputs_info = [b_enc] + outputs_info[1:]

        # 2) Replace the initial y_{t_0} with 0, and add taps = -1
        y = T.zeros((b_enc.shape[0], self._layers[0].output_size))
        outputs_info = outputs_info[:-1] + [y]

        # 3) Initialize with no mask
        mask = T.zeros((b_enc.shape[0], ), dtype='int8')
        outputs_info = [mask] + outputs_info

        return outputs_info

    def forward_pass(self, b_enc):
        """
        Setup equations for the forward pass
        """
        # because scan assmes the iterable is the first tensor dimension, x is
        # transposed into (time, obs, dims).
        # When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
        # this is then transposed back to its original format
        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            n_steps=self._maxlength,
            outputs_info=self._outputs_info_list(b_enc)
        )

        y = self._last_output(outputs)
        return y.transpose(1, 2, 0)
