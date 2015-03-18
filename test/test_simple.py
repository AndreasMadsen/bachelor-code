
import _test
from datasets import quadrant_classify
from nose.tools import *

import matplotlib.pyplot as plt
import numpy as np
import neural

def test_logistic_regression():
    logistic = neural.Network(eta=0.4, momentum=0.9)
    # Setup theano tap.test_value
    logistic.test_value(*quadrant_classify(10))

    # Setup layers for a logistic classifier model
    logistic.set_input(neural.Input(2))
    logistic.push_layer(neural.Softmax(4))
    logistic.set_loss(neural.NaiveEntropy())

    # Compile train, test and predict functions
    logistic.compile()

    # Setup dataset and train model
    epochs = 100
    train_dataset = quadrant_classify(1000)
    test_dataset = quadrant_classify(100)

    train_error = np.zeros(epochs)
    test_error = np.zeros(epochs)
    for i in range(0, epochs):
        train_error[i] = logistic.train(*train_dataset)
        test_error[i] = logistic.test(*test_dataset)

    # Loss function should be improved
    assert(train_error[0] > train_error[-1] > 0)
    assert(test_error[0] > test_error[-1] > 0)

    # Test prediction shape and error rate
    y = logistic.predict(test_dataset[0])
    assert_equal(y.shape, (100, 4, 5))
    assert(np.mean(np.argmax(y, axis=1) == test_dataset[1]) > 0.7)
