
import test
import dataset
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural


def _test_sutskever_network_count():
    def generator(items):
        d = dataset.network.count(items)
        return (d.data, d.target)

    sutskever = neural.network.Sutskever(max_output_size=9, verbose=True)
    # Setup theano tap.test_value
    sutskever.test_value(*generator(10))

    # Setup layers
    sutskever.set_encoder_input(neural.layer.Input(2))
    sutskever.push_encoder_layer(neural.layer.LSTM(2))

    sutskever.set_decoder_input(neural.layer.Input(6))
    sutskever.push_decoder_layer(neural.layer.LSTM(2))
    sutskever.push_decoder_layer(neural.layer.LSTM(80))
    sutskever.push_decoder_layer(neural.layer.Softmax(6))

    # Setup loss function
    sutskever.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    sutskever.compile()

    test.classifier(
        sutskever, generator,
        y_shape=(100, 6, 15), performance=0.6, asserts=False, plot=True, save=True,
        epochs=4000, learning_rate=0.2, momentum=0.3
    )

    (x, t) = generator(10)
    y = sutskever.predict(x)

    print(y)
    print(np.argmax(y, axis=1))
    print(t)

def _test_sutskever_network_filter():
    def generator(items):
        d = dataset.network.vocal_subset(items)
        return (d.data, d.target)

    sutskever = neural.network.Sutskever(max_output_size=10)
    # Setup theano tap.test_value
    test_value = generator(10)
    sutskever.test_value(*test_value)

    # Setup layers for a logistic classifier model
    letters = test_value[0].shape[1]
    latent = 40
    sutskever.set_encoder_input(neural.layer.Input(letters))
    sutskever.push_encoder_layer(neural.layer.LSTM(20))
    sutskever.push_encoder_layer(neural.layer.LSTM(latent))

    sutskever.set_decoder_input(neural.layer.Input(letters))
    sutskever.push_decoder_layer(neural.layer.LSTM(latent))
    sutskever.push_decoder_layer(neural.layer.LSTM(20))
    sutskever.push_decoder_layer(neural.layer.Softmax(letters))

    # Setup loss function
    sutskever.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    sutskever.compile()

    test.classifier(
        sutskever, generator,
        y_shape=(100, 4, 5), performance=0.6, asserts=False, plot=True,
        epochs=200, learning_rate=0.1, momentum=0.9
    )

    def mat2str(mat):
        strs = []
        for row in mat:
            strs.append(
                ''.join([chr(m + ord('A') - 1) for m in row if m != 0])
            )
        return strs

    (x, t) = generator(10)
    y = sutskever.predict(x)

    print(y)

    print(mat2str(np.argmax(x, axis=1)))
    print(mat2str(np.argmax(y, axis=1)))
    print(mat2str(t))
