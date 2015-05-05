
import itertools

import numpy as np
import theano
import theano.tensor as T

class BaseAbstraction:
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
        self._output_layer = layer

    def weight_list(self):
        """
        Create a list containing all the network weights
        """
        return list(itertools.chain(*[
            layer.weights for layer in self._layers
        ]))

    def count_weights(self):
        """
        Count how may weight matrices there are in the network.
        """
        return sum([len(layer.weights) for layer in self._layers])

    def reset_weights(self):
        """
        Will reinitialize all weights using a random number generator
        """
        for layer in self._layers:
            layer.reset_weights()

    def set_weights(self, new_weights):
        """
        Set weights using a list of numpy ndarrays
        """
        curr = 0
        for layer in self._layers:
            end = curr + len(layer.weights)
            layer.set_weights(new_weights[curr:end])
            curr += end

    def get_weights(self):
        """
        Return all weights as a list of numpy ndarrays
        """
        return sum([layer.get_weights() for layer in self._layers], [])

    def _outputs_info_list(self):
        """
        Generate a list of outputs for the forward scanner
        """
        return list(itertools.chain(*[
            layer.outputs_info for layer in self._layers
        ]))

    def _last_output(self, outputs):
        if (isinstance(outputs, list)):
            y = outputs[-1]
        else:
            y = outputs
        return y

    def forward_pass(self, x):
        raise NotImplemented
