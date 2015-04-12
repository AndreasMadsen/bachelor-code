
import test
from datasets import mode_encoder_sequence
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural
from neural.network._optimizer import OptimizerAbstraction
from neural.network.sutskever import Encoder

def test_sutskever_encoder_fast():
    # obs: 2, dims: 3, time: 6
    x1 = np.asarray([
        [0, 0, 0, 1, 1, 1],  # <EOS>
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype='int8')
    x2 = np.asarray([
        [0, 0, 0, 0, 0, 1],  # <EOS>
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0]
    ], dtype='int8')
    x = np.asarray([x1, x2])

    x_input = T.tensor3('x', dtype='int8')
    x_input.tag.test_value = x

    # Create encoder
    encoder = Encoder(x_input)

    # Setup layers for a logistic classifier model
    encoder.set_input(neural.layer.Input(3))
    encoder.push_layer(neural.layer.LSTM(4))

    # Enforce weights for consistent results
    weights = encoder.weight_list()
    weights[0].set_value(np.tile(np.asarray([
        [-0.48490700, -0.86166465, -0.68156427, +0.22538510],
        [+0.92648524, +0.27489561, -1.65431046, -0.99628597],
        [+0.79819334, -0.28792551, -0.34098715, +0.58490205]
    ], dtype='float32'), (1, 4)))
    weights[1].set_value(np.tile(np.asarray([
        [+1.90923262, -0.70851284, +1.05009556, +0.38148439],
        [-1.22912812, -0.98054105, +0.43493664, -0.03531076],
        [-0.01739309, -0.03275032, -0.12585467, +1.56854463],
        [-0.20889190, +0.67603844, +1.11195946, +0.08580784]
    ], dtype='float32'), (1, 4)))

    # Perform forward pass
    (s_enc, b_enc) = encoder.forward_pass(x_input)

    # Check that the gradient can be calculated
    T.grad(T.sum(s_enc), weights)
    T.grad(T.sum(b_enc), weights)

    # Assert output
    assert_equal(s_enc.tag.test_value.shape, (2, 4))
    assert_equal(b_enc.tag.test_value.shape, (2, 4))

    assert(np.allclose(s_enc.tag.test_value, [
        [1.54302049, 0.19040981, 0.46381277, 0.64930063],
        [2.52043080, 0.17858852, 0.67638946, 0.93438488]
    ]))

    assert(np.allclose(b_enc.tag.test_value, [
        [0.68185270, 0.16576755, 0.39093050, 0.48691007],
        [0.80372810, 0.15667477, 0.45946684, 0.55524415]
    ]))

class EncoderOptimizer(Encoder, OptimizerAbstraction):
    def __init__(self, **kwargs):
        self._input = T.tensor3('x')
        self._target = T.ivector('t')

        Encoder.__init__(self, self._input, **kwargs)
        OptimizerAbstraction.__init__(self, **kwargs)

    def forward_pass(self, x):
        # Since we can only train on one output tensor use the hidden output
        (s_enc, b_enc) = Encoder.forward_pass(self, x)
        return b_enc

    def test_value(self, x, b_enc):
        self._input.tag.test_value = x
        self._target.tag.test_value = b_enc

def test_sutskever_encoder_train():
    encoder = EncoderOptimizer()
    # Setup theano tap.test_value
    encoder.test_value(*mode_encoder_sequence(10))

    # Setup layers for a logistic classifier model
    encoder.set_input(neural.layer.Input(10))
    encoder.push_layer(neural.layer.LSTM(15))
    encoder.push_layer(neural.layer.Softmax(10))

    # Setup loss function
    encoder.set_loss(neural.loss.NaiveEntropy(time=False))

    # Compile train, test and predict functions
    encoder.compile()

    test.classifier(
        encoder, mode_encoder_sequence,
        y_shape=(100, 10), performance=0.8,
        epochs=1000
    )

    (x, t) = mode_encoder_sequence(10)
    b_enc = encoder.predict(x)
