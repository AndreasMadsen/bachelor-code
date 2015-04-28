
import run

import dataset
import neural
import theano.tensor as T

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

def simple_learn(model, train_dataset, test_dateset, epochs):
    print('learning model')
    train_size = train_dataset[0].shape[0]

    for i in range(0, epochs):
        train_error = model.train(*train_dataset)
        print('  train: size %d, epoch %d, loss %f' % (train_size, i, train_error))

test_dataset = (mnist.data[0:1000, :], mnist.target[0:1000])
train_dataset = (mnist.data[1001:4000, :], mnist.target[1001:4000])

simple_learn(encoder, train_dataset, test_dataset, 4000)
