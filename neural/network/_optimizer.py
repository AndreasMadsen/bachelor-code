
import itertools

import datetime
import numpy as np
import theano
import theano.tensor as T

def get_tick():
    return datetime.datetime.now()

def get_tock(tick):
    return (datetime.datetime.now() - tick).total_seconds() * 1000

class OptimizerAbstraction():
    def __init__(self, verbose=False, **kwargs):

        self._extra_params = []
        self._verbose = verbose
        self._loss = None
        self._optimizer = None

    def set_loss(self, loss):
        """
        Set the loss function.
        """
        loss.setup(self._input.shape[0])
        self._loss = loss

    def set_optimizer(self, optimizer):
        """
        Set the optimizer logic.
        """
        self._optimizer = optimizer

    def backward_pass(self, L, W):
        """
        Derive equations for the backward pass
        """
        return T.grad(L, W)

    def _preloss(self, y, t):
        return (y, t)

    def compile(self):
        """
        Takes the defined layers and compiles train, error and predict functions.

        The functions are set as public properties on the network object.
        """
        #
        # Setup equations
        #
        if (self._verbose):
            print('compiling network')
            if (theano.config.optimizer != 'fast_run'):
                print('  NOTE: optimizer is disabled')

        # Check output is compatiabel with loss
        if (self._loss._expect_log != self._output_layer._add_log):
            raise ValueError('loss layer and output layer did not agree on the log transform')

        # Create forward pass equations
        tick = get_tick()
        forward = self.forward_pass(self._input)
        if (isinstance(forward, T.TensorVariable)): forward = [forward]
        y = forward[-1]
        if (self._verbose): print('  forward pass generated, took %d ms' % get_tock(tick))

        # Setup loss function
        tick = get_tick()
        L = self._loss.loss(
            *self._preloss(*forward, t=self._target)
        )
        if (self._verbose): print('  loss function generated, took %d ms' % get_tock(tick))

        # Generate backward pass
        tick = get_tick()
        W = self.weight_list()
        gW = self.backward_pass(L, W)
        if (self._verbose): print('  backward pass generated, took %d ms' % get_tock(tick))

        #
        # Setup functions
        #
        tick = get_tick()
        self._train = theano.function(
            inputs=[self._input, self._target] +
            self._extra_params +
            self._optimizer.params,
            outputs=[L],
            updates=self._optimizer.update(W, gW),
            name='train'
        )
        if (self._verbose): print('  compiled train function, took %d ms' % get_tock(tick))
        tick = get_tick()
        self._test = theano.function(
            inputs=[self._input, self._target] + self._extra_params,
            outputs=[L],
            name='test'
        )
        if (self._verbose): print('  compiled error function, took %d ms' % get_tock(tick))
        tick = get_tick()
        self._predict = theano.function(
            inputs=[self._input] + self._extra_params,
            outputs=[y],
            name='predict'
        )
        if (self._verbose): print('  compiled predict function, took %d ms' % get_tock(tick))

    def train(self, *args, **kwargs):
        return list(self._train(*args, **kwargs))[0]

    def test(self, *args, **kwargs):
        return list(self._test(*args, **kwargs))[0]

    def predict(self, *args, **kwargs):
        return list(self._predict(*args, **kwargs))[0]
