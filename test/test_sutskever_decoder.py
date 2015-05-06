
import test
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import dataset
import neural

def test_sutskever_decoder_fast():
    # obs: 2, dims: 4, time: NA
    b_enc = np.asarray([
        [1, 0, 1, 1],
        [0, 2, 2, 0]
    ], dtype='float32')
    b_input = T.matrix('b_enc', dtype='float32')
    b_input.tag.test_value = b_enc

    # Create encoder
    decoder = neural.network.SutskeverDecoder(max_output_size=2)
    decoder.test_value(b_enc, np.zeros((2, 2)))

    # Setup layers
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

    weights[3].set_value(np.asarray([
        [+0.89090699, +1.07468557, -0.51611972],
        [-0.61338681, +0.67821771, +0.85758942],
        [+0.61945277, -0.86997068, -0.74410045],
        [+1.40485370, -0.73074096, +0.64977872]
    ], dtype='float32'))

    # Perform forward pass
    y = decoder.forward_pass(b_input)

    # Check that the gradient can be calculated
    T.grad(T.sum(y), weights)

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
        [0.78832114, 0.06528059, 0.14639825],
        [0.69669789, 0.09191318, 0.21138890]
    ]))

def test_sutskever_decoder_train():
    decoder = neural.network.SutskeverDecoder(max_output_size=9)
    # Setup theano tap.test_value
    decoder.test_value(*dataset.decoder.count(10).astuple())

    # Setup layers
    decoder.set_input(neural.layer.Input(6))  # Should match output
    decoder.push_layer(neural.layer.LSTM(1))  # Should match b_enc input
    decoder.push_layer(neural.layer.LSTM(80))
    decoder.push_layer(neural.layer.Softmax(6))

    # Setup loss function
    decoder.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    decoder.compile()

    test.classifier(
        decoder, dataset.decoder.count,
        y_shape=(100, 6, 9), performance=0.8,
        trainer=neural.learn.minibatch, train_size=1280,
        epochs=500, learning_rate=0.07, momentum=0.2
    )
