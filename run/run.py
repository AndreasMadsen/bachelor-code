
import warnings
import os.path as path
import sys
import os

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

import theano
import numpy as np

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


print('process pid: %d' % (os.getpid()))

if (os.environ.get('DTU_HPC') is not None):
    print('Running on HPC')

if (theano.config.optimizer == 'fast_run'):
    print('Theano optimizer enabled')

#
run_name = (os.environ.get('OUTNAME')
            if os.environ.get('OUTNAME') is not None
            else str(os.getpid()))
output_file = path.join(thisdir, '..', 'outputs', run_name + '.npz')

# Simple batch learning
def missclassification(model, test_dataset):
    (data, target) = test_dataset
    return np.mean(np.argmax(model.predict(data), axis=1) != target)

def batch_learn(model, data, **kwargs):
    return _learn(model, data, neural.learn.batch, **kwargs)

def minibatch_learn(model, data, **kwargs):
    return _learn(model, data, neural.learn.minibatch, **kwargs)

def _learn(model, data, learning_method, test_size=100, epochs=100, **kwargs):
    # Use 1/3 as test data
    test_dataset = (data.data[0:test_size], data.target[0:test_size])
    train_dataset = (data.data[test_size:], data.target[test_size:])

    print('learning model')
    train_size = train_dataset[0].shape[0]

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_miss = np.zeros(epochs)
    test_miss = np.zeros(epochs)

    def on_epoch(model, epoch_i):
        train_loss[epoch_i] = model.test(*train_dataset)
        test_loss[epoch_i] = model.test(*test_dataset)

        train_miss[epoch_i] = missclassification(model, train_dataset)
        test_miss[epoch_i] = missclassification(model, test_dataset)

        print('  train: size %d, epoch %d, train loss %f, test miss: %f' % (
            train_size, epoch_i, train_loss[epoch_i], test_miss[epoch_i]
        ))

    learning_method(model, train_dataset,
                    on_epoch=on_epoch, epochs=epochs,
                    **kwargs)

    return {
        'train_loss': train_loss,
        'test_loss': test_loss,

        'train_miss': train_miss,
        'test_miss': test_miss,

        'n_classes': np.asarray([data.n_classes])
    }
