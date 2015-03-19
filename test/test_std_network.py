
import _test
from datasets import quadrant_classify, quadrant_cumsum_classify
from nose.tools import *

import matplotlib.pyplot as plt
import numpy as np
import neural

def _classifier_tester(model, generator, epochs=100, plot=False):
    # Setup dataset and train model
    test_dataset = generator(100)

    train_error = np.zeros(epochs)
    test_error = np.zeros(epochs)
    for i in range(0, epochs):
        train_error[i] = model.train(*generator(500))
        test_error[i] = model.test(*test_dataset)

    if (plot):
        plt.plot(np.arange(0, epochs), train_error, label='train', alpha=0.5)
        plt.plot(np.arange(0, epochs), test_error, label='test', alpha=0.5)
        plt.legend()
        plt.ylabel('loss')
        plt.show()

    # Loss function should be improved
    assert(train_error[0] > train_error[-1] > 0)
    assert(test_error[0] > test_error[-1] > 0)

    # Test prediction shape and error rate
    y = model.predict(test_dataset[0])
    assert_equal(y.shape, (100, 4, 5))
    # 0.25 is random guess
    assert(np.mean(np.argmax(y, axis=1) == test_dataset[1]) > 0.40)

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

    _classifier_tester(logistic, quadrant_classify)

def test_lstm_classifier():
    lstm = neural.network.Std(eta=0.6, momentum=0.9)
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

    _classifier_tester(lstm, quadrant_cumsum_classify, epochs=800)
