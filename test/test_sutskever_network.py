
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
    b = encoder.forward_pass(x_input)

    # Assert output
    assert_equal(y.tag.test_value.shape, (2, 4))

    assert(np.allclose(b.tag.test_value, [
        [0.86412394, 0.27515879, 0.74134195, 0.82611161],
        [0.86167115, 0.28139967, 0.78435278, 0.85536474]
    ]))

def test_sutskever_decoder():
    # obs: 2, dims: 4, time: NA
    b_enc = np.asarray([
        [1, 0, 1, 1],
        [0, 2, 2, 0]
    ], dtype='float32')
    b_input = T.matrix('b', dtype='float32')
    b_input.tag.test_value = b_enc

    # Create encoder
    decoder = Decoder(b_input)

    # Setup layers for a logistic classifier model
    decoder.set_input(neural.layer.Input(3))
    decoder.push_layer(neural.layer.RNN(4))
    decoder.push_layer(neural.layer.Softmax(3))

    # Enforce weights for consistent results
    weights = decoder.weight_list()
    weights[0].set_value(np.asarray([
        [+0.68613929, -0.54955703, -0.16661042, +2.01016974],
        [+1.42984617, -0.95113963, -0.46729347, -2.68691254],
        [-2.85298777, -1.53363299, -0.09551691, -0.40909350]
    ], dtype='float32'))

    weights[1].set_value(np.asarray([
        [-0.26722059, +0.17297429, -1.07697213, +0.74177772],
        [-1.65553486, +0.13171023, +0.19016156, +0.09970825],
        [+0.85862929, +0.34413302, -0.90042174, -0.10362423],
        [-0.23016714, -1.02287507, +1.38639832, +1.44669867]
    ], dtype='float32'))

    weights[2].set_value(np.asarray([
        [+0.89090699, +1.07468557, -0.51611972],
        [-0.61338681, +0.67821771, +0.85758942],
        [+0.61945277, -0.86997068, -0.74410045],
        [+1.40485370, -0.73074096, +0.64977872]
    ], dtype='float32'))

    # Perform forward pass
    y = decoder.forward_pass(b_input)

    # Assert output
    assert_equal(y.tag.test_value.shape, (2, 3, 2))

    # The first sequences ends after 1 iteration
    y0 = y[:, :, 0]
    assert(np.allclose(y0.tag.test_value, [
        [0.71534252, 0.11405356, 0.17060389],
        [0.34762833, 0.23523150, 0.41714019]
    ]))

    # The second sequence ends after 2 iterations
    y1 = y[:, :, 1]
    assert(np.allclose(y1.tag.test_value, [
        [0.71534252, 0.11405356, 0.17060389],
        [0.69669789, 0.09191318, 0.21138890]
    ]))
