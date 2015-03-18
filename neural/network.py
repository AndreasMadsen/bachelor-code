
import numpy as np
import theano
import theano.tensor as T

from neural.input_layer import Input as InputLayer

class Network:
    """
    Abstraction for creating recurent neural networks
    """

    def __init__(self, eta=0.1, momentum=0.9):
        self._target = T.imatrix('t')
        self._input = T.tensor3('x')

        self._eta = eta
        self._momentum = momentum

        self._layers = []
        self._loss = None

    def set_input(self, size):
        self._layers.append(InputLayer(size))

    def push_layer(self, layer):
        """
        Push a layer to the network. The order of the layers matches
        the order of the `push_layer` calls.
        """
        layer.setup(self._input.shape[0], len(self._layers), self._layers[-1])
        self._layers.append(layer)

    def set_loss(self, loss):
        """
        Set the loss function.
        """
        loss.setup(self._input.shape[0], self._layers[-1])
        self._loss = loss

    def _weight_list(self):
        """
        Create a list containing all the network weights
        """
        return itertools.chain(*[
            layer.weights for layer in self._layers
        ])

    def _infer_taps(self, outputs_info):
        taps = 0

        for info in outputs_info:
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
            else if (info is not None):
                taps += 1

        return taps

    def _forward_scanner(self, *args):
        """
        Defines the forward equations for each time step.
        """
        all_outputs = []
        curr = 0
        prev_output = self._input

        # Loop though each layer and apply send the previous layers output
        # to the next layer. The layer can have additional paramers, if
        # taps where provided using the `outputs_info` property.
        for layer in self._layers[1:]:
            taps = self._infer_taps(layer.outputs_info)
            outputs = layer.scanner(prev_output, *args[curr:curr + taps])

            curr += taps
            all_outputs += outputs
            # the last output is assumed to be the layer output
            prev_output = outputs[-1]

        return all_outputs

    def _forward_pass(self, x):
        """
        Setup equations for the forward pass
        """
        # Generate a list of outputs for the forward scanner
        outputs_info = itertools.chain(*[
            layer.outputs_info for layer in self._layers
        ])

        # because scan assmes the iterable is the first tensor dimension, x is
        # transposed into (time, obs, dims).
        # When done $b1$ and $y$ will be tensors with the shape (time, obs, dims)
        # this is then transposed back to its original format
        outputs, _ = theano.scan(
            fn=self._forward_scanner,
            sequences=x.transpose(2, 0, 1),  # iterate (time), row (observations), col (dims)
            outputs_info=outputs_info
        )
        # the last output is assumed to be the network output
        y = outputs[-1]
        # transpose back to (obs, dims, time)
        y = y.transpose(1, 2, 0)

        return y

    def _backward_pass(self, L):
        """
        Derive equations for the backward pass
        """
        return T.grad(L, self._weight_list())

    def _momentum_gradient_decent(gWi, Wi):
        """
        The graident decent equation for a single weight matrix.

        This adds a momentum for better generalization. `eta` and `momentum`
        are statically define parameters.
        """
        dWi_tm1 = theano.shared(
            np.zeros_like(Wi.get_value(), dtype='float32'),
            name="d" + Wi.name, borrow=True)

        dWi = - self._momentum * dWi_tm1 - self._eta * gWi
        return ((dWi_tm1, dWi), (Wi, Wi + dWi))

    def _update_functions(self, gW):
        """
        Generate update equations for the weights
        """
        return [
            self_momentum_gradient_decent(gWi, Wi) for (gWi, Wi)
            in zip(gW, self._weight_list())
        ]

    def compile(self):
        """
        Takes the defined layers and compiles train, error and predict functions.

        The functions are set as public properties on the network object.
        """
        #
        # Setup equations
        #

        # Create forward pass equations
        y = self._forward_pass(self._input)

        # Setup loss function
        L = self._loss.loss(y, self._target)

        # Generate backward pass
        gW = self._backward_pass(self, L)

        #
        # Setup functions
        #
        self.train = theano.function(
            inputs=[self._input, self._target],
            outputs=L,
            updates=self._update_functions()
        )
        self.error = theano.function(inputs=[x, t], outputs=L)
        self.predict = theano.function(inputs=[x], outputs=y)
