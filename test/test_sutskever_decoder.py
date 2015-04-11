
import test
from datasets import count_decoder_sequence
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural
from neural.network._optimizer import OptimizerAbstraction
from neural.network.sutskever import Decoder, SutskeverNetwork

def test_sutskever_decoder_fast():
    # obs: 2, dims: 4, time: NA
    b_enc = np.asarray([
        [1, 0, 1, 1],
        [0, 2, 2, 0]
    ], dtype='float32')
    b_input = T.matrix('b_enc', dtype='float32')
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
    (eois, y) = decoder.forward_pass(b_input)

    # Check that the gradient can be calculated
    # TODO: debug errors caused by test_value
    # T.grad(T.sum(y), weights)

    # Assert output
    assert_equal(y.tag.test_value.shape, (2, 3, 2))

    # The first sequences ends after 1 iteration
    y0 = y[:, :, 0]
    assert(np.allclose(y0.tag.test_value, [
        [0.71534252, 0.11405356, 0.17060389],
        [0.34762833, 0.23523150, 0.41714019]
    ]))
    assert_equal(eois[0].tag.test_value, 0)

    # The second sequence ends after 2 iterations
    y1 = y[:, :, 1]
    assert(np.allclose(y1.tag.test_value, [
        [0.71534252, 0.11405356, 0.17060389],
        [0.69669789, 0.09191318, 0.21138890]
    ]))
    assert_equal(eois[1].tag.test_value, 1)

class DecoderOptimizer(Decoder, OptimizerAbstraction):
    def __init__(self, **kwargs):
        self._input = T.matrix('b_enc')
        self._target = T.imatrix('t')

        Decoder.__init__(self, self._input, **kwargs)
        OptimizerAbstraction.__init__(self, **kwargs)

    def test_value(self, x, t):
        self._input.tag.test_value = x
        self._target.tag.test_value = t

    def _loss_scanner(self, *args, **kwargs):
        return SutskeverNetwork._loss_scanner(self, *args, **kwargs)

    def _loss(self, *args, **kwargs):
        return SutskeverNetwork._loss(self, *args, **kwargs)

def _test_sutskever_decoder_train():
    theano.config.compute_test_value = 'off'

    decoder = DecoderOptimizer(eta=0.001, momentum=0.01)
    # Setup theano tap.test_value
    decoder.test_value(*count_decoder_sequence(10))

    # Setup layers for a logistic classifier model
    decoder.set_input(neural.layer.Input(11))  # Should match output
    decoder.push_layer(neural.layer.LSTM(2, bias=True))  # Should match b_enc input
    decoder.push_layer(neural.layer.LSTM(22, bias=True))
    decoder.push_layer(neural.layer.LSTM(22, bias=True))
    decoder.push_layer(neural.layer.Softmax(11, bias=True))

    # Setup loss function
    decoder.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    decoder.compile()

    test.classifier(
        decoder, count_decoder_sequence,
        y_shape=(100, 4, 5), performance=0.6, plot=True, asserts=False,
        epochs=1500
    )

    (b_enc, t) = count_decoder_sequence(10)
    y = decoder.predict(b_enc)

    print(y)
    print(np.argmax(y, axis=1))
    print(t)

    theano.config.compute_test_value = 'warn'
_test_sutskever_decoder_train()
