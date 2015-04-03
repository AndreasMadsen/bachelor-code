
import itertools

import numpy as np
import theano
import theano.tensor as T

class OptimizerAbstraction():
    def __init__(self, eta=0.1, momentum=0.9, **kwargs):
        self._eta = eta
        self._momentum = momentum

        self._loss_layer = None

    def debugprint_list(self):
        return []

    def set_loss(self, loss):
        """
        Set the loss function.
        """
        loss.setup(self._input.shape[0])
        self._loss_layer = loss

    def backward_pass(self, L):
        """
        Derive equations for the backward pass
        """
        return T.grad(L, self.weight_list())

    def _momentum_gradient_decent(self, gWi, Wi):
        """
        The graident decent equation for a single weight matrix.

        This adds a momentum for better generalization. `eta` and `momentum`
        are statically define parameters.
        """
        ΔWi_tm1 = theano.shared(
            np.zeros_like(Wi.get_value(), dtype='float32'),
            name="Δ" + Wi.name, borrow=True)

        ΔWi = - self._momentum * ΔWi_tm1 - self._eta * gWi
        return [(ΔWi_tm1, ΔWi), (Wi, Wi + ΔWi)]

    def _update_functions(self, gW):
        """
        Generate update equations for the weights
        """
        return list(itertools.chain(*[
            self._momentum_gradient_decent(gWi, Wi) for (gWi, Wi)
            in zip(gW, self.weight_list())
        ]))

    def _loss(self, y, t):
        return (y, t)

    def compile(self):
        """
        Takes the defined layers and compiles train, error and predict functions.

        The functions are set as public properties on the network object.
        """
        #
        # Setup equations
        #

        # Create forward pass equations
        forward = self.forward_pass(self._input)
        if (not isinstance(forward, tuple) and not isinstance(forward, list)):
            forward = [forward]
        y = forward[-1]

        # Setup loss function
        L = self._loss_layer.loss(
            *self._loss(*forward, t=self._target)
        )

        # Generate backward pass
        gW = self.backward_pass(L)

        #
        # Setup functions
        #
        self._train = theano.function(
            inputs=[self._input, self._target],
            outputs=[L] + self.debugprint_list(),
            updates=self._update_functions(gW)
        )
        self._test = theano.function(
            inputs=[self._input, self._target],
            outputs=[L] + self.debugprint_list()
        )
        self._predict = theano.function(
            inputs=[self._input],
            outputs=[y] + self.debugprint_list()
        )

    def train(self, *args):
        return list(self._train(*args))[0]

    def test(self, *args):
        return list(self._test(*args))[0]

    def predict(self, *args):
        return list(self._predict(*args))[0]
