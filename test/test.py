
from nose.tools import *

import warnings
import os.path as path
import sys
import os

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

import progressbar
import matplotlib as mpl
if (os.environ.get('DISPLAY') is None): mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano

import neural

np.random.seed(42)

warnings.filterwarnings(
    action='error',
    category=UserWarning
)
warnings.filterwarnings(
    action='ignore',
    message='numpy.ndarray size changed, may indicate binary incompatibility'
)

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

def classifier(model, generator, y_shape, performance,
               trainer=neural.learn.batch, train_size=500, test_size=100,
               asserts=True, plot=False, epochs=100, **kwargs):
    if (plot): print('testing classifier')

    # Setup dataset and train model
    train_dataset = generator(train_size)
    test_dataset = generator(test_size)

    train_error = np.zeros(epochs)
    test_error = np.zeros(epochs)

    # Train and show progress
    if (not plot): print()
    pbar = progressbar.ProgressBar(
        widgets=[
            'Training: ', progressbar.Bar(),
            progressbar.Percentage(), ' | ', progressbar.ETA()
        ],
        maxval=epochs
    ).start()

    def on_epoch(model, epoch_i):
        if (plot or epoch_i == 0 or epoch_i == (epochs - 1)):
            train_error[epoch_i] = model.test(*train_dataset.astuple())
            test_error[epoch_i] = model.test(*test_dataset.astuple())

        pbar.update(epoch_i + 1)

    trainer(model, train_dataset, on_epoch=on_epoch, epochs=epochs, **kwargs)
    pbar.finish()

    # Calculate precition and missclassificationrate
    y = model.predict(test_dataset.data)
    misses = np.mean(np.argmax(y, axis=1) != test_dataset.target)

    # Plot
    if (plot):
        print('miss classifications:', misses)

        plt.figure()
        plt.plot(np.arange(0, epochs), train_error, label='train', alpha=0.5)
        plt.plot(np.arange(0, epochs), test_error, label='test', alpha=0.5)
        plt.legend()
        plt.ylabel('loss')

        if (is_HPC): plt.savefig('loss.png')
        else: plt.show()

    # Assert
    if (asserts):
        assert(train_error[0] > train_error[-1] > 0)
        assert(test_error[0] > test_error[-1] > 0)
        assert_equal(y.shape, y_shape)
        assert((1 - misses) > performance)

__all__ = ['classifier']
