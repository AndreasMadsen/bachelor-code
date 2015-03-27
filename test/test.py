
import warnings
import os.path as path
import sys

import matplotlib.pyplot as plt
import numpy as np
import theano

warnings.filterwarnings(
    action='error',
    category=UserWarning
)
warnings.filterwarnings(
    action='ignore',
    message='numpy.ndarray size changed, may indicate binary incompatibility'
)

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'warn'


def classifier(model, generator, y_shape, performance, epochs=100, asserts=True, plot=False):
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
    if (asserts): assert(train_error[0] > train_error[-1] > 0)
    if (asserts): assert(test_error[0] > test_error[-1] > 0)

    # Test prediction shape and error rate
    y = model.predict(test_dataset[0])
    if (asserts): assert_equal(y.shape, y_shape)
    misses = np.mean(np.argmax(y, axis=1) == test_dataset[1])
    if (asserts): assert(misses > performance)
    print(misses)

__all__ = ['classifier']