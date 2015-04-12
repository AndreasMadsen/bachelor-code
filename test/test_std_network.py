
import test
from datasets import quadrant_classify, quadrant_cumsum_classify
from nose.tools import *

import neural

def test_logistic_classifier():
    logistic = neural.network.Std(eta=0.4, momentum=0.9)
    # Setup theano tap.test_value
    logistic.test_value(*quadrant_classify(10))

    # Setup layers for a logistic classifier model
    logistic.set_input(neural.layer.Input(2))
    logistic.push_layer(neural.layer.Softmax(4))

    # Setup loss function
    logistic.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    logistic.compile()

    test.classifier(
        logistic, quadrant_classify,
        y_shape=(100, 4, 5), performance=0.6
    )

def test_rnn_classifier():
    rnn = neural.network.Std(eta=0.2, momentum=0.5)
    # Setup theano tap.test_value
    rnn.test_value(*quadrant_cumsum_classify(10))

    # Setup layers for a logistic classifier model
    rnn.set_input(neural.layer.Input(2))
    rnn.push_layer(neural.layer.RNN(4))
    rnn.push_layer(neural.layer.Softmax(4))

    # Setup loss function
    rnn.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    rnn.compile()

    test.classifier(
        rnn, quadrant_cumsum_classify,
        y_shape=(100, 4, 5), performance=0.6,
        epochs=800
    )

def test_lstm_classifier():
    lstm = neural.network.Std(eta=0.3, momentum=0.5)
    # Setup theano tap.test_value
    lstm.test_value(*quadrant_cumsum_classify(10))

    # Setup layers for a logistic classifier model
    lstm.set_input(neural.layer.Input(2))
    lstm.push_layer(neural.layer.LSTM(4))
    lstm.push_layer(neural.layer.Softmax(4))

    # Setup loss function
    lstm.set_loss(neural.loss.NaiveEntropy())

    # Compile train, test and predict functions
    lstm.compile()

    test.classifier(
        lstm, quadrant_cumsum_classify,
        y_shape=(100, 4, 5), performance=0.6,
        epochs=1000
    )
