
import _test
from datasets import quadrant_cumsum_classify
from nose.tools import *

import numpy as np
import neural
import theano
import theano.tensor as T
from neural.network.sutskever import Encoder, Decoder

def test_sutskever_encoder():
    # obs: 2, dims: 3, time: 6
    mask = np.asarray([3, 0], dtype='int32')
    x1 = np.asarray([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype='int8')
    x2 = np.asarray([
        [1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype='int8')
    x = np.asarray([x1, x2])

    x_input = T.tensor3('x', dtype='int8')
    x_input.tag.test_value = x

    # Create encoder
    encoder = Encoder(x_input)

    # Setup layers for a logistic classifier model
    encoder.set_input(neural.layer.Input(3))
    encoder.push_layer(neural.layer.RNN(4))

    # Enforce weights for consistent results
    weights = encoder.weight_list()
    weights[0].set_value(np.asarray([
        [-0.48490700, -0.86166465, -0.68156427, +0.22538510],
        [+0.92648524, +0.27489561, -1.65431046, -0.99628597],
        [+0.79819334, -0.28792551, -0.34098715, +0.58490205]
    ], dtype='float32'))
    weights[1].set_value(np.asarray([
        [+1.90923262, -0.70851284, +1.05009556, +0.38148439],
        [-1.22912812, -0.98054105, +0.43493664, -0.03531076],
        [-0.01739309, -0.03275032, -0.12585467, +1.56854463],
        [-0.20889190, +0.67603844, +1.11195946, +0.08580784]
    ], dtype='float32'))

    # Perform forward pass
    b = encoder.forward_pass(x_input, mask)
    print(b.tag.test_value)

test_sutskever_encoder()
