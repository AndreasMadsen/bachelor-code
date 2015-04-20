
import itertools

import numpy as np
import theano
import theano.tensor as T

from neural.network._base import BaseAbstraction
from neural.network._optimizer import OptimizerAbstraction

class StdNetwork(OptimizerAbstraction, BaseAbstraction):
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, **kwargs):
        BaseAbstraction.__init__(self, **kwargs)
        OptimizerAbstraction.__init__(self, **kwargs)

        self._inputs = [T.tensor3('x')]
        self._target = T.imatrix('t')

    def test_value(self, x, t):
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def _forward_scanner(self, x_t, *args):
        """
        Defines the forward equations for each time step.
        """
        all_outputs = []
        curr = 0
        prev_output = x_t

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = layer.infer_taps()
            layer_outputs = layer.scanner(prev_output, *args[curr:curr + taps])

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
            sequences=x.transpose(2, 0, 1),  # iterate (time), row (observations), col (dims)
            outputs_info=self._outputs_info_list()
        )
        # the last output is assumed to be the network output
        y = self._last_output(outputs)

        # transpose back to (obs, dims, time)
        y = y.transpose(1, 2, 0)

        return y
