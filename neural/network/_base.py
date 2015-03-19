
import itertools

import numpy as np
import theano
import theano.tensor as T

class BaseNetwork:
    def __init__(self, **kwargs):
        self._layers = []

    def set_input(self, layer):
        self._layers.append(layer)

    def push_layer(self, layer):
        """
        Push a layer to the network. The order of the layers matches
        the order of the `push_layer` calls.
        """
        layer.setup(self._input.shape[0], len(self._layers), self._layers[-1])
        self._layers.append(layer)

    def _weight_list(self):
        """
        Create a list containing all the network weights
        """
        return list(itertools.chain(*[
            layer.weights for layer in self._layers
        ]))

    def _outputs_info_list(self):
        """
        Generate a list of outputs for the forward scanner
        """
        return list(itertools.chain(*[
            layer.outputs_info for layer in self._layers
        ]))

    def _infer_taps(self, layer):
        taps = 0

        for info in layer.outputs_info:
            # If info is dict it can have a taps array, default this is [-1]
            if (isinstance(info, dict)):
                # However of no `inital` property is provided it is treated
                # as None (no taps)
                if ('initial' in info):
                    if ('taps' in info):
                        taps += len(info.taps)
                    else:
                        taps += 1
            # If info is a numpy array or a scalar
            elif (info is not None):
                taps += 1

        return taps

    def forward_pass(self, x):
        raise NotImplemented
