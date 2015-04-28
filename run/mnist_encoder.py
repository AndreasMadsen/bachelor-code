
import run

import os
import os.path as path
import theano.tensor as T
import numpy as np

import dataset
import neural

thisdir = path.dirname(path.realpath(__file__))
run_name = (os.environ.get('OUTNAME')
            if os.environ.get('OUTNAME') is not None
            else str(os.getpid()))

mnist = dataset.encoder.mnist()

encoder = neural.network.SutskeverEncoder(
    [T.tensor3('x')], T.ivector('t'),
    eta=0.05, momentum=0.2, verbose=True
)

# Setup theano tap.test_value
encoder.test_value(mnist.data[0:10, :], mnist.target[0:10])

# Setup layers
encoder.set_input(neural.layer.Input(mnist.data.shape[1]))
encoder.push_layer(neural.layer.LSTM(80))
encoder.push_layer(neural.layer.Softmax(mnist.n_classes, log=True))

# Setup loss function
encoder.set_loss(neural.loss.NaiveEntropy(time=False, log=True))

# Compile train, test and predict functions
encoder.compile()

def missclassification(model, test_dateset):
    (data, target) = test_dateset
    return np.mean(model.predict(data) != target)

def simple_learn(model, train_dataset, test_dateset, epochs):
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

        print('  train: size %d, epoch %d, train loss %f' % (train_size, i, train_loss[i]))

    return {
        'train_loss': train_loss,
        'test_loss': test_loss,

        'train_miss': test_miss,
        'test_miss': test_miss
    }

test_dataset = (mnist.data[0:200, :], mnist.target[0:200])
train_dataset = (mnist.data[200:1200, :], mnist.target[200:1200])

results = simple_learn(encoder, train_dataset, test_dataset, 20)

np.savez_compressed(
    path.join(thisdir, '..', 'outputs', run_name + '.npz'),
    **results
)
