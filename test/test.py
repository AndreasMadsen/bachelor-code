
from nose.tools import *

import warnings
import os.path as path
import sys
import os

import matplotlib as mpl
if (os.environ.get('DISPLAY') is None): mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano

np.random.seed(42)

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

is_HPC = (os.environ.get('DTU_HPC') is not None)
is_optimize = (os.environ.get('OPTIMIZE') is not None)

run_name = (os.environ.get('OUTNAME')
            if os.environ.get('OUTNAME') is not None
            else str(os.getpid()))

if (not is_HPC):
    theano.config.compute_test_value = 'warn'
if (not is_optimize and not is_HPC):
    theano.config.optimizer = 'None'
    theano.config.linker = 'py'
    theano.config.exception_verbosity = 'high'
if (theano.config.optimizer != 'None'):
    print('Theano optimizer enabled')

def classifier(model, generator, y_shape, performance, epochs=100, asserts=True, plot=False, save=False):
    if (plot): print('testing classifier')

    # Setup dataset and train model
    test_dataset = generator(100)

    train_error = np.zeros(epochs)
    test_error = np.zeros(epochs)
    if (save): test_predict = np.zeros((epochs, ) + y_shape)
    for i in range(0, epochs):
        if (plot): print('  running train epoch %d' % i)
        train_error[i] = model.train(*generator(500))
        test_error[i] = model.test(*test_dataset)
        if (save): test_predict[i] = model.predict(test_dataset[0])

    if (plot):
        plt.figure()
        plt.plot(np.arange(0, epochs), train_error, label='train', alpha=0.5)
        plt.plot(np.arange(0, epochs), test_error, label='test', alpha=0.5)
        plt.legend()
        plt.ylabel('loss')

        if (is_HPC): plt.savefig('loss.png')
        else: plt.show()

    # Loss function should be improved
    if (asserts): assert(train_error[0] > train_error[-1] > 0)
    if (asserts): assert(test_error[0] > test_error[-1] > 0)

    # Test prediction shape and error rate
    y = model.predict(test_dataset[0])
    if (asserts): assert_equal(y.shape, y_shape)
    # TODO: likely wrong in CTC problem. Misses was 0.0
    misses = np.mean(np.argmax(y, axis=1) != test_dataset[1])
    if (plot): print('miss classifications:', misses)
    if (asserts): assert((1 - misses) > performance)

    if (save):
        np.savez(
            path.join(thisdir, '..', 'outputs', run_name + '.npz'),
            train=train_error,
            test=test_error,
            predict=test_predict,
            input=test_dataset[0],
            target=test_dataset[1]
        )

__all__ = ['classifier']
