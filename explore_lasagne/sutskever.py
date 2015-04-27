
import theano
import theano.tensor as T
import lasagne

import test
from datasets import count_network_sequence

BATCH_SIZE = 100
MAX_LENGTH = 15
N_CLASSES = 6
N_LATENT = 2
N_INPUT = 2
ETA = 0.1
MOMENTUM = 0.3

# Setup Encoder
enc_inp = lasagne.layers.InputLayer((BATCH_SIZE, 2, N_INPUT))
enc_hid = lasagne.layers.LSTMLayer(enc_inp, 2,
                                   return_sequence=False)

# Setup Decoder
dec_inp = lasagne.layers.RepeatLayer(enc_hid, MAX_LENGTH)
dec_hid = lasagne.layers.LSTMLayer(dec_inp, 80)

dec_out1 = lasagne.layers.ReshapeLayer(dec_hid, (BATCH_SIZE * MAX_LENGTH, 80))
dec_out2 = lasagne.layers.DenseLayer(dec_out1, N_CLASSES)
dec_out3 = lasagne.layers.SoftmaxLayer(dec_out2)
dec_out = lasagne.layers.ReshapeLayer(dec_out3, (BATCH_SIZE, MAX_LENGTH, N_CLASSES))

# Cost function
x = T.tensor3('x')
t = T.imatrix('t')
(x.tag.test_value, t.tag.test_value) = count_network_sequence(100)
y = dec_out.get_output(x.dimshuffle(0, 2, 1)).dimshuffle(0, 2, 1)

def entropy(t, y):
    t = t.ravel()
    y = y.transpose(0, 2, 1).reshape((y.shape[2] * y.shape[0], y.shape[1]))
    return T.mean(T.nnet.categorical_crossentropy(y, t))
cost = entropy(t, y)

# Setup update functions
all_params = lasagne.layers.get_all_params(dec_out)
updates = lasagne.updates.nesterov_momentum(cost, all_params, ETA, MOMENTUM)

# Compile
class SutskeverNetwork():
    def __init__(self):
        self.train = theano.function([x, t], cost, updates=updates)
        self.test = theano.function([x, t], cost)
        self.predict = theano.function([x], y)

model = SutskeverNetwork()

# Run
test.classifier(
    model, count_network_sequence,
    y_shape=(100, 6, 15), performance=0.6, asserts=False, plot=True, save=True,
    epochs=8000
)
