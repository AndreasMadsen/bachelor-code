
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

        self._learning_rate = T.scalar('eta')
        self._learning_rate.tag.test_value = 0.1
        self._momentum = T.scalar('m')
        self._momentum.tag.test_value = 0.9

        self._verbose = verbose
        self._loss_layer = None

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

        This adds a momentum for better generalization.
        """
        ΔWi_tm1 = theano.shared(
            np.zeros_like(Wi.get_value(), dtype='float32'),
            name="Δ" + Wi.name, borrow=True)

        ΔWi = - self._momentum * ΔWi_tm1 - self._learning_rate * gWi
        return [(ΔWi_tm1, ΔWi), (Wi, Wi + ΔWi)]

    def _update_functions(self, gW):
        """
        Generate update equations for the weights
        """
        return list(itertools.chain(*[
            self._momentum_gradient_decent(gWi, Wi) for (gWi, Wi)
            in zip(gW, self.weight_list())
        ]))

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
        if (self._loss_layer._expect_log != self._output_layer._add_log):
            raise ValueError('loss layer and output layer did not agree on the log transform')

        # Create forward pass equations
        tick = get_tick()
        forward = self.forward_pass(self._input)
        if (isinstance(forward, T.TensorVariable)): forward = [forward]
        y = forward[-1]
        if (self._verbose): print('  forward pass generated, took %d ms' % get_tock(tick))

        # Setup loss function
        tick = get_tick()
        L = self._loss_layer.loss(
            *self._preloss(*forward, t=self._target)
        )
        if (self._verbose): print('  loss function generated, took %d ms' % get_tock(tick))

        # Generate backward pass
        tick = get_tick()
        gW = self.backward_pass(L)
        if (self._verbose): print('  backward pass generated, took %d ms' % get_tock(tick))

        #
        # Setup functions
        #
        tick = get_tick()
        self._train = theano.function(
            inputs=[
                self._input, self._target,
                theano.Param(self._learning_rate, default=0.1, name='learning_rate'),
                theano.Param(self._momentum, default=0.9, name='momentum')
            ],
            outputs=[L],
            updates=self._update_functions(gW),
            name='train'
        )
        if (self._verbose): print('  compiled train function, took %d ms' % get_tock(tick))
        tick = get_tick()
        self._test = theano.function(
            inputs=[self._input, self._target],
            outputs=[L],
            name='test'
        )
        if (self._verbose): print('  compiled error function, took %d ms' % get_tock(tick))
        tick = get_tick()
        self._predict = theano.function(
            inputs=[self._input],
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
