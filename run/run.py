import warnings
import os.path as path
import sys
import os
import theano
import numpy as np

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

#
def simple_learn(model, data, epochs, missclassification):
    # Use 1/3 as test data
    test_size = data.data.shape[0] // 3
    test_dataset = (data.data[0:test_size, :], data.target[0:test_size])
    train_dataset = (data.data[test_size:300, :], data.target[test_size:300])

    print('learning model')
    train_size = train_dataset[0].shape[0]

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_miss = np.zeros(epochs)
    test_miss = np.zeros(epochs)

    for i in range(0, epochs):
        train_loss[i] = model.train(*train_dataset)
        test_loss[i] = model.test(*test_dataset)

        train_miss[i] = missclassification(model, train_dataset)
        test_miss[i] = missclassification(model, test_dataset)

        print('  train: size %d, epoch %d, train loss %f, test miss: %f' % (
            train_size, i, train_loss[i], test_miss[i]
        ))

    return {
        'train_loss': train_loss,
        'test_loss': test_loss,

        'train_miss': train_miss,
        'test_miss': test_miss,

        'n_classes': np.asarray([data.n_classes])
    }
