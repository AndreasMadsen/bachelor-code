
import test
from datasets import subset_vocal_sequence, count_network_sequence
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural


def _test_sutskever_network_count():
    # TODO: debug errors caused by test_value
    theano.config.compute_test_value = 'off'

    sutskever = neural.network.Sutskever(eta=0.2, momentum=0.3, max_output_size=15, verbose=True)
    # Setup theano tap.test_value
    sutskever.test_value(*count_network_sequence(10))

    # Setup layers
    sutskever.set_encoder_input(neural.layer.Input(2))
    sutskever.push_encoder_layer(neural.layer.LSTM(2))

    sutskever.set_decoder_input(neural.layer.Input(6))
    sutskever.push_decoder_layer(neural.layer.LSTM(2))
    sutskever.push_decoder_layer(neural.layer.LSTM(80))
    sutskever.push_decoder_layer(neural.layer.Softmax(6, log=True))

    # Setup loss function
    sutskever.set_loss(neural.loss.NaiveEntropy(log=True))

    # Compile train, test and predict functions
    sutskever.compile()

    test.classifier(
        sutskever, count_network_sequence,
        y_shape=(100, 6, 15), performance=0.6, asserts=False, plot=True, save=True,
        epochs=4000
    )

    (x, t) = count_network_sequence(10)
    y = sutskever.predict(x)

    print(y)
    print(np.argmax(y, axis=1))
    print(t)

    theano.config.compute_test_value = 'warn'

_test_sutskever_network_count()

def _test_sutskever_network_filter():
    # TODO: debug errors caused by test_value
    theano.config.compute_test_value = 'off'

    sutskever = neural.network.Sutskever(eta=0.1, momentum=0.3, max_output_size=20, verbose=True)
    # Setup theano tap.test_value
    test_value = subset_vocal_sequence(10)
    sutskever.test_value(*test_value)

    # Setup layers for a logistic classifier model
    letters = test_value[0].shape[1]
    latent = 100
    sutskever.set_encoder_input(neural.layer.Input(letters))
    sutskever.push_encoder_layer(neural.layer.LSTM(40))
    sutskever.push_encoder_layer(neural.layer.LSTM(latent))

    sutskever.set_decoder_input(neural.layer.Input(letters))
    sutskever.push_decoder_layer(neural.layer.LSTM(latent))
    sutskever.push_decoder_layer(neural.layer.LSTM(40))
    sutskever.push_decoder_layer(neural.layer.Softmax(letters, log=True))

    # Setup loss function
    sutskever.set_loss(neural.loss.NaiveEntropy(log=True))

    # Compile train, test and predict functions
    sutskever.compile()

    test.classifier(
        sutskever, subset_vocal_sequence,
        y_shape=(100, 4, 5), performance=0.6, asserts=False, plot=True, save=True,
        epochs=4000
    )

    def mat2str(mat):
        strs = []
        for row in mat:
            strs.append(
                ''.join([chr(m + ord('A') - 1) for m in row if m != 0])
            )
        return strs

    (x, t) = subset_vocal_sequence(10)
    y = sutskever.predict(x)

    print(y)

    print(mat2str(np.argmax(x, axis=1)))
    print(mat2str(np.argmax(y, axis=1)))
    print(mat2str(t))

    theano.config.compute_test_value = 'warn'
