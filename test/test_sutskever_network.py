
import test
import dataset
from nose.tools import *

import numpy as np
import theano
import theano.tensor as T

import neural


def _test_sutskever_network_train():
    sutskever = neural.network.Sutskever(max_output_size=9, verbose=True)
    # Setup theano tap.test_value
    sutskever.test_value(*dataset.network.copy(10).astuple())

    # Setup layers
    sutskever.set_encoder_input(neural.layer.Input(10))
    sutskever.push_encoder_layer(neural.layer.LSTM(40))
    sutskever.push_encoder_layer(neural.layer.LSTM(9))

    sutskever.set_decoder_input(neural.layer.Input(10))
    sutskever.push_decoder_layer(neural.layer.LSTM(9))
    sutskever.push_decoder_layer(neural.layer.LSTM(40))
    sutskever.push_decoder_layer(neural.layer.Softmax(10))

    # Setup loss function
    sutskever.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    sutskever.compile()

    test.classifier(
        sutskever, dataset.network.copy,
        y_shape=(100, 6, 9), performance=0.8,
        trainer=neural.learn.minibatch, train_size=1280, plot=True, asserts=False,
        epochs=500, learning_rate=0.07, momentum=0.2
    )

    (x, t) = generator(10)
    y = sutskever.predict(x)

    print(np.argmax(y, axis=1))
    print(t)
