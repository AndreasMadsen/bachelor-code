
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
output_dir = path.join(thisdir, '..', 'outputs')
output_file = path.join(output_dir, run_name)

# Simple batch learning
def missclassification(model, data):
    (x, t) = data.astuple()
    if (len(t.shape) > 1):
        prediction = model.predict(x, max_output_size=t.shape[1])
    else:
        prediction = model.predict(x)
    return np.mean(np.argmax(prediction, axis=1) != t)

def batch_learn(model, data, **kwargs):
    return _learn(model, data, neural.learn.batch, **kwargs)

def minibatch_learn(model, data, **kwargs):
    return _learn(model, data, neural.learn.minibatch, **kwargs)

def _learn(model, data, learning_method, test_size=100, **kwargs):
    # Use 1/3 as test data
    test_dataset = data.range(0, test_size)
    train_sample = data.range(test_size, 2 * test_size)
    train_dataset = data.range(test_size, None)

    print('learning model')
    train_size = train_dataset.observations

    train_loss = []
    test_loss = []

    train_loss_minibatch = []
    train_loss_minibatch_epoch = []

    train_miss = []
    train_miss_epoch = []
    test_miss = []
    test_miss_epoch = []

    def on_mini_batch(model, index, epoch, loss):
        if (index % 10 == 5):
            train_loss_minibatch.append(loss)
            train_loss_minibatch_epoch.append(epoch)

    def on_epoch(model, epoch_i):
        train_loss.append(model.test(*train_sample.astuple()))
        test_loss.append(model.test(*test_dataset.astuple()))

        train_miss.append(missclassification(model, train_sample))
        test_miss.append(missclassification(model, test_dataset))

        print('  train: size %d, epoch %d, train loss %f, test miss: %f' % (
            train_size, epoch_i, train_loss[-1], test_miss[-1]
        ))

    learning_method(model, train_dataset,
                    on_epoch=on_epoch, on_mini_batch=on_mini_batch,
                    **kwargs)

    return {
        'train_loss': np.asarray(train_loss, dtype='float32'),
        'test_loss': np.asarray(test_loss, dtype='float32'),

        'train_loss_minibatch': np.asarray(train_loss_minibatch, dtype='float32'),
        'train_loss_minibatch_epoch': np.asarray(train_loss_minibatch_epoch, dtype='float32'),

        'train_miss': np.asarray(train_miss, dtype='float32'),
        'test_miss': np.asarray(test_miss, dtype='float32'),

        'n_classes': np.asarray([data.n_classes])
    }
